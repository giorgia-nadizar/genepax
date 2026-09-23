"""Plain DAgger with a small neural-network student.

Fit on a fixed expert bootstrap set, collect states with the current student,
label those states with the ANN expert, aggregate them, and refit.  The held-out
expert split is fixed for the complete run and is never added to training data.
"""

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

from distillation.evaluation import finite_prefix_mask
from distillation.networks.neural_student import StudentPolicy
from distillation.networks.sac_utils import load_q_value_estimator, load_sac_actor
from distillation.q_dagger import compute_q_dagger_weights
from distillation.rollouts import rollout, sanitize_action, valid_transition_mask
from distillation_experiments.scripts.experiments.dagger import positive_int, select_rows
from distillation_experiments.scripts.experiments.dagger_linear import split_train_test
from distillation_experiments.scripts.experiments.neural_initial_fit import (
    evaluate_student,
    fit_student,
    weighted_mse,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="hopper")
    parser.add_argument("--backend", default="generalized")
    parser.add_argument("--run-name")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=positive_int, default=5)
    parser.add_argument("--iterations", type=positive_int, default=10)
    parser.add_argument(
        "--expert-bootstrap-samples", type=positive_int, default=10_000
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--trajectories-per-iteration", type=positive_int, default=20
    )
    parser.add_argument("--rollout-steps", type=positive_int, default=1_000)
    parser.add_argument("--evaluation-trajectories", type=positive_int, default=10)
    parser.add_argument("--target-reward", type=float, default=3_250.0)
    parser.add_argument(
        "--stop-when-solved", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--state-weighting", choices=("uniform", "q_dagger", "both"),
        default="both",
    )
    parser.add_argument("--hidden-sizes", type=positive_int, nargs="+", default=[32, 32])
    parser.add_argument("--epochs", type=positive_int, default=200)
    parser.add_argument("--batch-size", type=positive_int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=positive_int, default=30)
    parser.add_argument("--q-action-grid-size", type=positive_int, default=101)
    parser.add_argument("--q-batch-size", type=positive_int, default=1_000)
    return parser.parse_args()


def collect_student_trajectories(
    model, params, actor, environment, *, num_steps, seed, trajectories
):
    """Collect student-induced states and label them with expert actions."""

    def collect_one(trajectory_seed):
        key = jax.random.key(trajectory_seed)

        def action_fn(observation, action_key, _step):
            expert_action, _ = actor(observation, action_key)
            student_action = sanitize_action(
                model.apply(params, observation[None])[0]
            )
            return student_action, expert_action

        X, y, rewards, dones = rollout(environment, key, action_fn, num_steps)
        episode_mask = valid_transition_mask(dones).astype(bool)
        retained_mask = episode_mask & finite_prefix_mask(X, y)
        safe_rewards = jnp.nan_to_num(
            rewards, nan=0.0, posinf=0.0, neginf=0.0
        )
        episode_return = jnp.sum(jnp.where(retained_mask, safe_rewards, 0.0))
        invalid = jnp.sum(episode_mask & ~finite_prefix_mask(X, y))
        return X, y, retained_mask, episode_return, invalid

    seeds = seed + jnp.arange(trajectories)
    X, y, masks, returns, invalid = jax.vmap(collect_one)(seeds)
    retained_X = [X[index][masks[index]] for index in range(trajectories)]
    retained_y = [y[index][masks[index]] for index in range(trajectories)]
    X_new = jnp.concatenate(retained_X)
    y_new = jnp.concatenate(retained_y)
    if len(X_new) == 0:
        raise RuntimeError("Neural student produced no valid transitions")
    return X_new, y_new, {
        "behavior_reward": float(jnp.mean(returns)),
        "collection_trajectories": trajectories,
        "invalid_transitions": int(jnp.sum(invalid)),
    }


def main():
    args = parse_arguments()
    experiments = Path(__file__).resolve().parents[2]
    checkpoint = experiments / "artifacts" / "expert_models" / args.env / "final"
    dataset_path = experiments / "artifacts" / "expert_datasets" / f"expert_{args.env}.npz"
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Expert model not found: {checkpoint}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Expert dataset not found: {dataset_path}")

    run_name = args.run_name or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_directory = (
        experiments / "artifacts" / "repertoires" / f"dagger_neural_{args.env}" / run_name
    )
    run_directory.mkdir(parents=True, exist_ok=False)
    configuration = vars(args) | {
        "algorithm": "DAgger",
        "student": "StudentPolicy",
        "student_objective": "mean_squared_expert_action_error",
        "refit_strategy": "from_scratch_each_iteration",
        "script": "distillation_experiments.scripts.experiments.dagger_neural",
    }
    (run_directory / "config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True)
    )

    expert = np.load(dataset_path)
    all_X = jnp.asarray(expert["X"], dtype=jnp.float32)
    all_y = jnp.asarray(expert["y"], dtype=jnp.float32)
    actor, _ = load_sac_actor(checkpoint)
    modes = (
        ("uniform", "q_dagger")
        if args.state_weighting == "both"
        else (args.state_weighting,)
    )
    q_estimator = (
        load_q_value_estimator(checkpoint) if "q_dagger" in modes else None
    )
    model = StudentPolicy(
        action_size=all_y.shape[1], hidden_sizes=tuple(args.hidden_sizes)
    )
    summaries = []
    print(f"Writing neural DAgger results to {run_directory}")

    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        bootstrap_X, bootstrap_y = select_rows(
            all_X, all_y, args.expert_bootstrap_samples, jax.random.key(seed)
        )
        initial_q_weights = None
        if q_estimator is not None:
            initial_q_weights, _ = compute_q_dagger_weights(
                bootstrap_X, bootstrap_y, q_estimator,
                args.q_action_grid_size, args.q_batch_size,
            )
        environment = envs.get_environment(args.env, backend=args.backend)

        for mode_index, mode in enumerate(modes):
            bootstrap_weights = (
                jnp.ones(len(bootstrap_X), dtype=jnp.float32)
                if mode == "uniform"
                else initial_q_weights
            )
            X, y, weights, test_X, test_y, test_weights = split_train_test(
                bootstrap_X,
                bootstrap_y,
                bootstrap_weights,
                args.validation_fraction,
                jax.random.key(seed + 1_000_000),
            )
            variant_directory = run_directory / f"seed_{seed}" / mode
            variant_directory.mkdir(parents=True)
            metrics_path = variant_directory / "metrics.csv"
            metric_fields = [
                "iteration", "imitation_loss", "test_imitation_loss",
                "student_reward", "behavior_reward", "dataset_size",
                "new_samples", "collection_trajectories",
                "invalid_transitions", "best_epoch", "epochs_completed",
            ]
            with metrics_path.open("w", newline="") as file:
                csv.DictWriter(file, fieldnames=metric_fields).writeheader()

            best_reward = -float("inf")
            best_iteration = 0
            ever_solved = False
            pending_collection = None
            iteration_histories = []

            for iteration in range(args.iterations):
                params, history, best_epoch = fit_student(
                    model,
                    X,
                    y,
                    weights,
                    test_X,
                    test_y,
                    test_weights,
                    seed=(
                        seed * 1_000_000
                        + mode_index * 100_000
                        + iteration * 1_000
                    ),
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    patience=args.patience,
                )
                loss = float(weighted_mse(model.apply(params, X), y, weights))
                test_loss = float(weighted_mse(
                    model.apply(params, test_X), test_y, test_weights
                ))
                reward = evaluate_student(
                    model,
                    params,
                    environment,
                    seed=seed * 100_000 + 90_000,
                    steps=args.rollout_steps,
                    trajectories=args.evaluation_trajectories,
                )
                if reward > best_reward:
                    best_reward = reward
                    best_iteration = iteration
                solved = reward >= args.target_reward
                ever_solved = ever_solved or solved
                metric = {
                    "iteration": iteration,
                    "imitation_loss": loss,
                    "test_imitation_loss": test_loss,
                    "student_reward": reward,
                    "behavior_reward": "" if pending_collection is None else pending_collection["behavior_reward"],
                    "dataset_size": len(X),
                    "new_samples": args.expert_bootstrap_samples if pending_collection is None else pending_collection["new_samples"],
                    "collection_trajectories": "" if pending_collection is None else pending_collection["collection_trajectories"],
                    "invalid_transitions": "" if pending_collection is None else pending_collection["invalid_transitions"],
                    "best_epoch": best_epoch,
                    "epochs_completed": len(history),
                }
                with metrics_path.open("a", newline="") as file:
                    csv.DictWriter(file, fieldnames=metric_fields).writerow(metric)
                iteration_histories.append({
                    "iteration": iteration,
                    "history": history,
                })
                print(
                    f"seed={seed} mode={mode} iteration={iteration} "
                    f"loss={loss:.6g} test_loss={test_loss:.6g} "
                    f"reward={reward:.3f} samples={len(X)}"
                )
                if args.stop_when_solved and solved:
                    break
                if iteration + 1 == args.iterations:
                    break

                X_new, y_new, pending_collection = collect_student_trajectories(
                    model,
                    params,
                    actor,
                    environment,
                    num_steps=args.rollout_steps,
                    seed=seed * 100_000 + iteration * 10_000,
                    trajectories=args.trajectories_per_iteration,
                )
                pending_collection["new_samples"] = len(X_new)
                X = jnp.concatenate([X, X_new])
                y = jnp.concatenate([y, y_new])
                if mode == "uniform":
                    new_weights = jnp.ones(len(X_new), dtype=jnp.float32)
                else:
                    new_weights, _ = compute_q_dagger_weights(
                        X_new,
                        y_new,
                        q_estimator,
                        args.q_action_grid_size,
                        args.q_batch_size,
                    )
                weights = jnp.concatenate([weights, new_weights])

            (variant_directory / "policy.msgpack").write_bytes(
                serialization.to_bytes(params)
            )
            (variant_directory / "metadata.json").write_text(json.dumps({
                "observation_size": int(all_X.shape[1]),
                "action_size": int(all_y.shape[1]),
                "hidden_sizes": list(args.hidden_sizes),
                "activation": "relu",
                "output_activation": "tanh",
            }, indent=2, sort_keys=True))
            (variant_directory / "training_histories.json").write_text(
                json.dumps(iteration_histories, indent=2)
            )
            np.savez_compressed(
                variant_directory / "final_dataset.npz",
                X=np.asarray(X),
                y=np.asarray(y),
                sample_weights=np.asarray(weights),
                test_X=np.asarray(test_X),
                test_y=np.asarray(test_y),
                test_weights=np.asarray(test_weights),
            )
            summary = {
                "seed": seed,
                "mode": mode,
                "last_iteration": iteration,
                "final_reward": reward,
                "final_imitation_loss": loss,
                "final_test_imitation_loss": test_loss,
                "best_observed_reward": best_reward,
                "best_observed_iteration": best_iteration,
                "solved": ever_solved,
                "dataset_size": len(X),
                "test_dataset_size": len(test_X),
                "weight_effective_sample_size": float(
                    jnp.square(jnp.sum(weights)) / jnp.sum(jnp.square(weights))
                ),
            }
            (variant_directory / "summary.json").write_text(
                json.dumps(summary, indent=2)
            )
            summaries.append(summary)
            (run_directory / "aggregate_summary.json").write_text(json.dumps({
                "completed_variants": len(summaries),
                "requested_variants": args.num_seeds * len(modes),
                "solved_variants": sum(item["solved"] for item in summaries),
                "variants": summaries,
            }, indent=2))


if __name__ == "__main__":
    main()
