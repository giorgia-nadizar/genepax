"""Evolve unweighted CGP expressions, then tune all connection weights.

Stage one searches only graph structure, either with raw outputs or analytic
linear output scaling. Stage two freezes each selected expression and uses
Adam to optimize every CGP input-connection weight. When scaling is enabled,
the affine output calibration is refitted after Adam before Brax validation.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import pickle
from pathlib import Path

from brax import envs
import jax
import jax.numpy as jnp
import numpy as np
import optax

from distillation.evaluation import evaluate_symbolic_policy
from distillation.fit_imitation import (
    apply_linear_scaling,
    fit_imitation_dataset,
)
from distillation.networks.sac_utils import load_q_value_estimator
from distillation.q_dagger import compute_q_dagger_weights
from distillation.rollouts import sanitize_action
from distillation_experiments.scripts.experiments.dagger import positive_int, select_rows
from genepax.gp.cartesian_genetic_programming import CGP


def create_run_directory(root, environment, run_name):
    if run_name is None:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = root / f"cgp_expression_adam_{environment}" / run_name
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def split_dataset(X, y, sample_weights, validation_fraction, key):
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation-fraction must be strictly between 0 and 1")
    permutation = jax.random.permutation(key, len(X))
    validation_size = max(1, round(len(X) * validation_fraction))
    validation_indices = permutation[:validation_size]
    training_indices = permutation[validation_size:]
    return (
        X[training_indices], y[training_indices], sample_weights[training_indices],
        X[validation_indices], y[validation_indices], sample_weights[validation_indices],
    )


def predict_population(genotypes, cgp_structure, X, linear_scaling):
    def predict_genotype(genotype):
        raw = jax.vmap(cgp_structure.apply, in_axes=(None, 0))(genotype, X)
        if linear_scaling:
            raw = apply_linear_scaling(
                raw, genotype["weights"]["custom_weights"]
            )
        return sanitize_action(raw)

    return jax.vmap(predict_genotype)(genotypes)


def population_mse(
    genotypes, cgp_structure, X, y, linear_scaling, sample_weights=None
):
    predictions = predict_population(genotypes, cgp_structure, X, linear_scaling)
    if sample_weights is None:
        sample_weights = jnp.ones(len(X))
    per_sample = jnp.mean(jnp.square(predictions - y[None, :, :]), axis=-1)
    return jnp.sum(per_sample * sample_weights[None, :], axis=1) / jnp.sum(
        sample_weights
    )


def refit_linear_scaling(
    genotypes, cgp_structure, X, y, sample_weights=None
):
    """Refit and store per-output affine scaling for a genotype batch."""
    raw = predict_population(genotypes, cgp_structure, X, linear_scaling=False)
    if sample_weights is None:
        sample_weights = jnp.ones(len(X))
    normalized_weights = sample_weights / jnp.sum(sample_weights)
    mean_prediction = jnp.sum(
        raw * normalized_weights[None, :, None], axis=1
    )
    mean_target = jnp.sum(y * normalized_weights[:, None], axis=0)
    centered_prediction = raw - mean_prediction[:, None, :]
    centered_target = y - mean_target
    covariance = jnp.sum(
        normalized_weights[None, :, None]
        * centered_prediction
        * centered_target[None, :, :],
        axis=1,
    )
    variance = jnp.sum(
        normalized_weights[None, :, None] * jnp.square(centered_prediction),
        axis=1,
    )
    slopes = jnp.where(variance > 1e-12, covariance / variance, 0.0)
    intercepts = mean_target - slopes * mean_prediction
    custom_weights = jnp.concatenate([slopes, intercepts], axis=-1)
    return jax.vmap(
        lambda genotype, weights: cgp_structure.update_weights(
            genotype, {"custom_weights": weights}
        )
    )(genotypes, custom_weights)


def optimize_connection_weights(
    genotypes,
    cgp_structure,
    X,
    y,
    sample_weights,
    *,
    linear_scaling,
    seed,
    learning_rate,
    gradient_steps,
    batch_size,
):
    """Use Adam to tune both input-connection weights for fixed structures."""
    graph_weights = cgp_structure.get_weights(genotypes)
    if set(graph_weights) != {"inputs1", "inputs2"}:
        raise ValueError("Adam stage must expose exactly both connection weights")

    def prediction_fn(X_batch, genotype, graph_weights):
        raw = jax.vmap(cgp_structure.apply, in_axes=(None, 0, None))(
            genotype, X_batch, graph_weights
        )
        if linear_scaling:
            raw = apply_linear_scaling(
                raw, genotype["weights"]["custom_weights"]
            )
        return raw

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(learning_rate)
    )
    optimizer_states = jax.vmap(optimizer.init)(graph_weights)

    def single_step(weights, genotype, optimizer_state, X_batch, y_batch, w_batch):
        def loss_fn(current_weights):
            predictions = prediction_fn(X_batch, genotype, current_weights)
            per_sample = jnp.mean(jnp.square(predictions - y_batch), axis=-1)
            return jnp.sum(per_sample * w_batch) / jnp.maximum(
                jnp.sum(w_batch), 1e-12
            )

        loss, gradients = jax.value_and_grad(loss_fn)(weights)
        gradients = jax.tree.map(
            lambda gradient: jnp.where(jnp.isfinite(gradient), gradient, 0.0),
            gradients,
        )
        updates, optimizer_state = optimizer.update(gradients, optimizer_state)
        weights = optax.apply_updates(weights, updates)
        weights = jax.tree.map(lambda value: jnp.clip(value, -1e4, 1e4), weights)
        return weights, optimizer_state, loss

    batched_step = jax.jit(jax.vmap(single_step, in_axes=(0, 0, 0, None, None, None)))
    key = jax.random.key(seed)
    actual_batch_size = min(batch_size, len(X))
    for _ in range(gradient_steps):
        key, batch_key = jax.random.split(key)
        indices = jax.random.randint(
            batch_key, (actual_batch_size,), 0, len(X)
        )
        graph_weights, optimizer_states, _ = batched_step(
            graph_weights,
            genotypes,
            optimizer_states,
            X[indices],
            y[indices],
            sample_weights[indices],
        )
    optimized_weights = graph_weights
    optimized_genotypes = jax.vmap(
        cgp_structure.update_weights, in_axes=(0, 0)
    )(genotypes, optimized_weights)
    if linear_scaling:
        optimized_genotypes = refit_linear_scaling(
            optimized_genotypes, cgp_structure, X, y, sample_weights
        )
    return optimized_genotypes


def evaluate_population_rewards(
    genotypes,
    cgp_structure,
    environment,
    *,
    linear_scaling,
    rollout_steps,
    evaluation_seed,
    evaluation_trajectories,
):
    return jax.vmap(
        lambda genotype: evaluate_symbolic_policy(
            genotype,
            cgp_structure,
            environment,
            num_steps=rollout_steps,
            seed=evaluation_seed,
            n_seeds=evaluation_trajectories,
            linear_scaling=linear_scaling,
        )
    )(genotypes)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Search unweighted CGP expressions then tune connections with Adam"
    )
    parser.add_argument("--env", default="inverted_pendulum")
    parser.add_argument("--backend", default="generalized")
    parser.add_argument("--run-name")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=positive_int, default=5)
    parser.add_argument(
        "--search-mode", choices=("raw", "scaled", "both"), default="both"
    )
    parser.add_argument("--expert-samples", type=positive_int, default=10_000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--search-generations", type=positive_int, default=50)
    parser.add_argument("--population-size", type=positive_int, default=100)
    parser.add_argument("--dataset-batch-size", type=positive_int, default=4_096)
    parser.add_argument("--adam-top-k", type=positive_int, default=10)
    parser.add_argument("--adam-steps", type=positive_int, default=500)
    parser.add_argument("--adam-learning-rate", type=float, default=1e-3)
    parser.add_argument("--adam-batch-size", type=positive_int, default=256)
    parser.add_argument("--generation-adam-steps", type=int, default=0)
    parser.add_argument(
        "--generation-adam-learning-rate", type=float, default=1e-3
    )
    parser.add_argument(
        "--generation-adam-batch-size", type=positive_int, default=256
    )
    parser.add_argument("--rollout-steps", type=positive_int, default=1_000)
    parser.add_argument("--evaluation-trajectories", type=positive_int, default=10)
    parser.add_argument("--target-reward", type=float, default=999.0)
    parser.add_argument(
        "--state-weighting", choices=("uniform", "q_dagger"), default="uniform"
    )
    parser.add_argument("--q-action-grid-size", type=positive_int, default=101)
    parser.add_argument("--q-batch-size", type=positive_int, default=1_000)
    parser.add_argument(
        "--history-only", action="store_true",
        help="Save CGP generation metrics and skip Adam and Brax validation",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.adam_top_k > args.population_size:
        raise ValueError("adam-top-k cannot exceed population-size")
    if args.adam_learning_rate <= 0:
        raise ValueError("adam-learning-rate must be positive")
    if args.generation_adam_steps < 0:
        raise ValueError("generation-adam-steps cannot be negative")
    if args.generation_adam_learning_rate <= 0:
        raise ValueError("generation-adam-learning-rate must be positive")

    experiments_directory = Path(__file__).resolve().parents[2]
    dataset_path = (
        experiments_directory / "artifacts" / "expert_datasets" / f"expert_{args.env}.npz"
    )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Expert dataset not found: {dataset_path}")
    run_directory = create_run_directory(
        experiments_directory / "artifacts" / "repertoires", args.env, args.run_name
    )
    with (run_directory / "config.json").open("w") as file:
        json.dump(vars(args), file, indent=2, sort_keys=True)

    data = np.load(dataset_path)
    all_X = jnp.asarray(data["X"], dtype=jnp.float32)
    all_y = jnp.asarray(data["y"], dtype=jnp.float32)
    modes = ("raw", "scaled") if args.search_mode == "both" else (args.search_mode,)
    summaries = []
    q_value_estimator = None
    if args.state_weighting == "q_dagger":
        checkpoint_path = (
            experiments_directory / "artifacts" / "expert_models" / args.env / "final"
        )
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(f"Expert model not found: {checkpoint_path}")
        q_value_estimator = load_q_value_estimator(checkpoint_path)
    print(f"Writing expression-plus-Adam results to {run_directory}")

    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        sampled_X, sampled_y = select_rows(
            all_X, all_y, args.expert_samples, jax.random.key(seed)
        )
        if q_value_estimator is None:
            sampled_weights = jnp.ones(len(sampled_X), dtype=jnp.float32)
            raw_q_weights = sampled_weights
        else:
            sampled_weights, raw_q_weights = compute_q_dagger_weights(
                sampled_X,
                sampled_y,
                q_value_estimator,
                args.q_action_grid_size,
                args.q_batch_size,
            )
        (
            train_X,
            train_y,
            train_weights,
            validation_X,
            validation_y,
            validation_weights,
        ) = split_dataset(
            sampled_X,
            sampled_y,
            sampled_weights,
            args.validation_fraction,
            jax.random.key(seed + 50_000),
        )
        environment = envs.get_environment(args.env, backend=args.backend)

        for mode_index, mode in enumerate(modes):
            linear_scaling = mode == "scaled"
            custom_weight_count = 2 * all_y.shape[1] if linear_scaling else 0
            search_cgp = CGP(
                n_inputs=all_X.shape[1],
                n_outputs=all_y.shape[1],
                n_custom_weights=custom_weight_count,
                weighted_inputs=args.generation_adam_steps > 0,
            )
            fit_result = fit_imitation_dataset(
                train_X,
                train_y,
                search_cgp,
                seed=seed * 100_000 + mode_index * 10_000,
                n_gens=args.search_generations,
                n_pop=args.population_size,
                dataset_batch_size=args.dataset_batch_size,
                linear_scaling=linear_scaling,
                sample_weights=train_weights,
                generation_adam_steps=args.generation_adam_steps,
                generation_adam_learning_rate=(
                    args.generation_adam_learning_rate
                ),
                generation_adam_batch_size=(
                    args.generation_adam_batch_size
                ),
            )
            if args.history_only:
                variant_directory = run_directory / f"seed_{seed}" / mode
                variant_directory.mkdir(parents=True)
                with (variant_directory / "search_history.csv").open(
                    "w", newline=""
                ) as file:
                    writer = csv.DictWriter(
                        file, fieldnames=fit_result["history"][0].keys()
                    )
                    writer.writeheader()
                    writer.writerows(fit_result["history"])
                print(
                    f"seed={seed} mode={mode} "
                    f"final_search_loss={fit_result['loss']:.6g}"
                )
                continue
            repertoire = fit_result["repertoire"]
            candidate_indices = jnp.argsort(
                jnp.ravel(repertoire.fitnesses)
            )[-args.adam_top_k:]
            candidates_before = jax.tree.map(
                lambda value: value[candidate_indices], repertoire.genotypes
            )

            weighted_cgp = CGP(
                n_inputs=all_X.shape[1],
                n_outputs=all_y.shape[1],
                n_nodes=search_cgp.n_nodes,
                function_set=search_cgp.function_set,
                n_input_constants=search_cgp.n_input_constants,
                outputs_wrapper=search_cgp.outputs_wrapper,
                weighted_inputs=True,
                n_custom_weights=custom_weight_count,
            )
            train_mse_before = population_mse(
                candidates_before,
                weighted_cgp,
                train_X,
                train_y,
                linear_scaling,
                train_weights,
            )
            validation_mse_before = population_mse(
                candidates_before,
                weighted_cgp,
                validation_X,
                validation_y,
                linear_scaling,
                validation_weights,
            )
            rewards_before = evaluate_population_rewards(
                candidates_before,
                weighted_cgp,
                environment,
                linear_scaling=linear_scaling,
                rollout_steps=args.rollout_steps,
                evaluation_seed=seed * 100_000 + 90_000,
                evaluation_trajectories=args.evaluation_trajectories,
            )
            candidates_after = optimize_connection_weights(
                candidates_before,
                weighted_cgp,
                train_X,
                train_y,
                train_weights,
                linear_scaling=linear_scaling,
                seed=seed * 100_000 + mode_index * 10_000 + 1,
                learning_rate=args.adam_learning_rate,
                gradient_steps=args.adam_steps,
                batch_size=args.adam_batch_size,
            )
            train_mse_after = population_mse(
                candidates_after,
                weighted_cgp,
                train_X,
                train_y,
                linear_scaling,
                train_weights,
            )
            validation_mse_after = population_mse(
                candidates_after,
                weighted_cgp,
                validation_X,
                validation_y,
                linear_scaling,
                validation_weights,
            )
            rewards_after = evaluate_population_rewards(
                candidates_after,
                weighted_cgp,
                environment,
                linear_scaling=linear_scaling,
                rollout_steps=args.rollout_steps,
                evaluation_seed=seed * 100_000 + 90_000,
                evaluation_trajectories=args.evaluation_trajectories,
            )
            selected_position = int(jnp.argmax(rewards_after))
            selected_genotype = jax.tree.map(
                lambda value: value[selected_position], candidates_after
            )

            variant_directory = run_directory / f"seed_{seed}" / mode
            variant_directory.mkdir(parents=True)
            with (variant_directory / "search_history.csv").open(
                "w", newline=""
            ) as file:
                writer = csv.DictWriter(
                    file, fieldnames=fit_result["history"][0].keys()
                )
                writer.writeheader()
                writer.writerows(fit_result["history"])
            with (variant_directory / "search_population.pickle").open("wb") as file:
                pickle.dump(repertoire, file)
            with (variant_directory / "candidates_before_adam.pickle").open("wb") as file:
                pickle.dump(candidates_before, file)
            with (variant_directory / "candidates_after_adam.pickle").open("wb") as file:
                pickle.dump(candidates_after, file)
            with (variant_directory / "best_individual.pickle").open("wb") as file:
                pickle.dump(selected_genotype, file)
            np.savez_compressed(
                variant_directory / "dataset.npz",
                train_X=np.asarray(train_X),
                train_y=np.asarray(train_y),
                validation_X=np.asarray(validation_X),
                validation_y=np.asarray(validation_y),
                train_weights=np.asarray(train_weights),
                validation_weights=np.asarray(validation_weights),
                raw_q_weights=np.asarray(raw_q_weights),
            )

            candidate_records = []
            for position in range(args.adam_top_k):
                candidate_records.append({
                    "position": position,
                    "repertoire_index": int(candidate_indices[position]),
                    "train_mse_before": float(train_mse_before[position]),
                    "validation_mse_before": float(validation_mse_before[position]),
                    "reward_before": float(rewards_before[position]),
                    "train_mse_after": float(train_mse_after[position]),
                    "validation_mse_after": float(validation_mse_after[position]),
                    "reward_after": float(rewards_after[position]),
                    "scaling_weights_after": np.asarray(
                        jax.tree.map(
                            lambda value: value[position], candidates_after
                        )["weights"]["custom_weights"]
                    ).tolist(),
                })
            with (variant_directory / "candidates.json").open("w") as file:
                json.dump(candidate_records, file, indent=2)
            with (variant_directory / "candidates.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=candidate_records[0].keys())
                writer.writeheader()
                writer.writerows(candidate_records)

            summary = {
                "seed": seed,
                "mode": mode,
                "state_weighting": args.state_weighting,
                "selected_position": selected_position,
                "best_reward_before_adam": float(jnp.max(rewards_before)),
                "best_reward_after_adam": float(jnp.max(rewards_after)),
                "best_validation_mse_before_adam": float(
                    jnp.min(validation_mse_before)
                ),
                "best_validation_mse_after_adam": float(
                    jnp.min(validation_mse_after)
                ),
                "solved": float(jnp.max(rewards_after)) >= args.target_reward,
                "weight_effective_sample_size": float(
                    jnp.square(jnp.sum(sampled_weights))
                    / jnp.sum(jnp.square(sampled_weights))
                ),
                "weight_min": float(jnp.min(sampled_weights)),
                "weight_median": float(jnp.median(sampled_weights)),
                "weight_max": float(jnp.max(sampled_weights)),
            }
            with (variant_directory / "summary.json").open("w") as file:
                json.dump(summary, file, indent=2)
            summaries.append(summary)
            with (run_directory / "aggregate_summary.json").open("w") as file:
                json.dump(
                    {
                        "completed_variants": len(summaries),
                        "requested_variants": args.num_seeds * len(modes),
                        "solved_variants": sum(item["solved"] for item in summaries),
                        "variants": summaries,
                    },
                    file,
                    indent=2,
                )
            print(
                f"seed={seed} mode={mode} "
                f"reward_before={float(jnp.max(rewards_before)):.3f} "
                f"reward_after={float(jnp.max(rewards_after)):.3f}"
            )


if __name__ == "__main__":
    main()
