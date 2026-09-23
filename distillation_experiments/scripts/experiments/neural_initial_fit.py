"""Initial behavioral-cloning fit of a small ANN student."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from brax import envs
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from distillation.networks.neural_student import StudentPolicy
from distillation.networks.sac_utils import load_q_value_estimator
from distillation.q_dagger import compute_q_dagger_weights
from distillation.rollouts import masked_return, rollout, sanitize_action
from distillation_experiments.scripts.experiments.cgp_expression_then_adam import split_dataset
from distillation_experiments.scripts.experiments.dagger import positive_int, select_rows


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="hopper")
    parser.add_argument("--backend", default="generalized")
    parser.add_argument("--run-name")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=positive_int, default=5)
    parser.add_argument(
        "--state-weighting", choices=("uniform", "q_dagger", "both"),
        default="both",
    )
    parser.add_argument("--expert-samples", type=positive_int, default=10_000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--hidden-sizes", type=positive_int, nargs="+", default=[32, 32])
    parser.add_argument("--epochs", type=positive_int, default=200)
    parser.add_argument("--batch-size", type=positive_int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=positive_int, default=30)
    parser.add_argument("--rollout-steps", type=positive_int, default=1_000)
    parser.add_argument("--evaluation-trajectories", type=positive_int, default=10)
    parser.add_argument("--q-action-grid-size", type=positive_int, default=101)
    parser.add_argument("--q-batch-size", type=positive_int, default=1_000)
    return parser.parse_args()


def weighted_mse(predictions, targets, weights):
    per_sample = jnp.mean(jnp.square(predictions - targets), axis=-1)
    return jnp.sum(per_sample * weights) / jnp.maximum(jnp.sum(weights), 1e-12)


def fit_student(
    model, train_X, train_y, train_weights, validation_X, validation_y,
    validation_weights, *, seed, epochs, batch_size, learning_rate, patience,
):
    params = model.init(jax.random.key(seed), train_X[:1])
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(params, optimizer_state, X, y, weights):
        def loss_fn(current_params):
            return weighted_mse(model.apply(current_params, X), y, weights)

        loss, gradients = jax.value_and_grad(loss_fn)(params)
        updates, optimizer_state = optimizer.update(
            gradients, optimizer_state, params
        )
        return optax.apply_updates(params, updates), optimizer_state, loss

    best_params = params
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    key = jax.random.key(seed + 10_000)
    actual_batch_size = min(batch_size, len(train_X))
    for epoch in range(epochs):
        key, permutation_key = jax.random.split(key)
        permutation = jax.random.permutation(permutation_key, len(train_X))
        batch_losses = []
        for start in range(0, len(train_X), actual_batch_size):
            indices = permutation[start:start + actual_batch_size]
            params, optimizer_state, loss = train_step(
                params,
                optimizer_state,
                train_X[indices],
                train_y[indices],
                train_weights[indices],
            )
            batch_losses.append(loss)
        train_loss = float(weighted_mse(
            model.apply(params, train_X), train_y, train_weights
        ))
        validation_loss = float(weighted_mse(
            model.apply(params, validation_X), validation_y, validation_weights
        ))
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "mean_batch_loss": float(jnp.mean(jnp.asarray(batch_losses))),
        })
        if validation_loss < best_validation_loss:
            best_params = jax.tree.map(lambda value: value.copy(), params)
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 0 or (epoch + 1) % 25 == 0:
            print(
                f"epoch={epoch + 1} train_loss={train_loss:.6g} "
                f"validation_loss={validation_loss:.6g}"
            )
        if epochs_without_improvement >= patience:
            break
    return best_params, history, best_epoch


def evaluate_student(model, params, environment, *, seed, steps, trajectories):
    def evaluate_one(trajectory_seed):
        def action_fn(observation, _key, _step):
            action = sanitize_action(model.apply(params, observation[None])[0])
            return action, action

        _, _, rewards, dones = rollout(
            environment,
            jax.random.key(trajectory_seed),
            action_fn,
            steps,
        )
        return masked_return(rewards, dones)

    seeds = seed + jnp.arange(trajectories)
    return float(jnp.mean(jax.vmap(evaluate_one)(seeds)))


def main():
    args = parse_arguments()
    experiments = Path(__file__).resolve().parents[2]
    run_name = args.run_name or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_directory = (
        experiments / "artifacts" / "repertoires" / f"neural_initial_{args.env}" / run_name
    )
    run_directory.mkdir(parents=True, exist_ok=False)
    dataset_path = args.dataset_path or (
        experiments / "artifacts" / "expert_datasets" / f"expert_{args.env}.npz"
    )
    with (run_directory / "config.json").open("w") as file:
        config = vars(args).copy()
        config["dataset_path"] = str(dataset_path)
        json.dump(config, file, indent=2, sort_keys=True)

    expert = np.load(dataset_path)
    all_X = jnp.asarray(expert["X"], dtype=jnp.float32)
    all_y = jnp.asarray(expert["y"], dtype=jnp.float32)
    checkpoint = experiments / "artifacts" / "expert_models" / args.env / "final"
    modes = (
        ("uniform", "q_dagger")
        if args.state_weighting == "both"
        else (args.state_weighting,)
    )
    q_estimator = (
        load_q_value_estimator(checkpoint)
        if "q_dagger" in modes
        else None
    )
    summaries = []
    print(f"Writing small-ANN results to {run_directory}")
    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        sampled_X, sampled_y = select_rows(
            all_X, all_y, args.expert_samples, jax.random.key(seed)
        )
        q_weights = None
        if q_estimator is not None:
            q_weights, _ = compute_q_dagger_weights(
                sampled_X,
                sampled_y,
                q_estimator,
                args.q_action_grid_size,
                args.q_batch_size,
            )
        environment = envs.get_environment(args.env, backend=args.backend)
        for mode_index, mode in enumerate(modes):
            weights = (
                jnp.ones(len(sampled_X), dtype=jnp.float32)
                if mode == "uniform"
                else q_weights
            )
            train_X, train_y, train_weights, validation_X, validation_y, validation_weights = split_dataset(
                sampled_X,
                sampled_y,
                weights,
                args.validation_fraction,
                jax.random.key(seed + 1_000_000),
            )
            model = StudentPolicy(
                action_size=all_y.shape[1],
                hidden_sizes=tuple(args.hidden_sizes),
            )
            params, history, best_epoch = fit_student(
                model,
                train_X,
                train_y,
                train_weights,
                validation_X,
                validation_y,
                validation_weights,
                seed=seed * 100_000 + mode_index * 10_000,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                patience=args.patience,
            )
            train_loss = float(weighted_mse(
                model.apply(params, train_X), train_y, train_weights
            ))
            validation_loss = float(weighted_mse(
                model.apply(params, validation_X),
                validation_y,
                validation_weights,
            ))
            reward = evaluate_student(
                model,
                params,
                environment,
                seed=seed * 100_000 + 90_000,
                steps=args.rollout_steps,
                trajectories=args.evaluation_trajectories,
            )
            variant_directory = run_directory / f"seed_{seed}" / mode
            variant_directory.mkdir(parents=True)
            with (variant_directory / "history.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=history[0].keys())
                writer.writeheader()
                writer.writerows(history)
            (variant_directory / "policy.msgpack").write_bytes(
                serialization.to_bytes(params)
            )
            metadata = {
                "observation_size": int(all_X.shape[1]),
                "action_size": int(all_y.shape[1]),
                "hidden_sizes": list(args.hidden_sizes),
                "activation": "relu",
                "output_activation": "tanh",
            }
            (variant_directory / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True)
            )
            np.savez_compressed(
                variant_directory / "dataset.npz",
                train_X=np.asarray(train_X),
                train_y=np.asarray(train_y),
                train_weights=np.asarray(train_weights),
                validation_X=np.asarray(validation_X),
                validation_y=np.asarray(validation_y),
                validation_weights=np.asarray(validation_weights),
            )
            summary = {
                "seed": seed,
                "mode": mode,
                "best_epoch": best_epoch,
                "epochs_completed": len(history),
                "reward": reward,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "weight_effective_sample_size": float(
                    jnp.square(jnp.sum(train_weights))
                    / jnp.sum(jnp.square(train_weights))
                ),
            }
            (variant_directory / "summary.json").write_text(
                json.dumps(summary, indent=2)
            )
            summaries.append(summary)
            (run_directory / "aggregate_summary.json").write_text(
                json.dumps({
                    "completed_variants": len(summaries),
                    "requested_variants": args.num_seeds * len(modes),
                    "variants": summaries,
                }, indent=2)
            )
            print(
                f"seed={seed} mode={mode} reward={reward:.3f} "
                f"train_loss={train_loss:.6g} "
                f"validation_loss={validation_loss:.6g}"
            )


if __name__ == "__main__":
    main()
