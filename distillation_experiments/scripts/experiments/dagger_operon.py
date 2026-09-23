"""DAgger with a multi-output Operon symbolic-regression policy."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from brax import envs
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pyoperon.sklearn import SymbolicRegressor

from distillation.evaluation import finite_prefix_mask
from distillation.networks.sac_utils import load_q_value_estimator, load_sac_actor
from distillation.q_dagger import compute_q_dagger_weights
from distillation.rollouts import rollout, sanitize_action, valid_transition_mask
from distillation_experiments.scripts.experiments.dagger import positive_int, select_rows
from distillation_experiments.scripts.experiments.dagger_linear import split_train_test
from distillation_experiments.scripts.experiments.operon_imitation import (
    evaluate_expressions,
    expression_to_jax,
    weighted_mse,
)
from distillation_experiments.scripts.experiments.operon_primitives import (
    OPERON_PRIMITIVE_PROFILES,
    operon_allowed_symbols,
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
    parser.add_argument("--fit-samples", type=positive_int, default=10_000)
    parser.add_argument("--generations-per-iteration", type=positive_int, default=10)
    parser.add_argument("--population-size", type=positive_int, default=500)
    parser.add_argument("--max-evaluations", type=positive_int, default=100_000)
    parser.add_argument("--max-length", type=positive_int, default=25)
    parser.add_argument(
        "--primitive-set",
        choices=tuple(OPERON_PRIMITIVE_PROFILES),
        default="compact",
        help="Operon primitive profile; 'cgp' mirrors the numeric CGP functions.",
    )
    parser.add_argument("--threads", type=positive_int, default=8)
    parser.add_argument("--q-action-grid-size", type=positive_int, default=101)
    parser.add_argument("--q-batch-size", type=positive_int, default=1_000)
    return parser.parse_args()


def fit_operon_policy(
    X,
    y,
    weights,
    validation_X,
    validation_y,
    validation_weights,
    *,
    seed,
    fit_samples,
    generations,
    population_size,
    max_evaluations,
    max_length,
    threads,
    primitive_set="compact",
):
    """Fit one expression per action and select each on fixed validation MSE."""
    rng = np.random.default_rng(seed)
    probabilities = np.asarray(weights, dtype=np.float64)
    probabilities /= probabilities.sum()
    fit_size = min(fit_samples, len(X))
    fit_indices = rng.choice(
        len(X), size=fit_size, replace=len(X) < fit_size, p=probabilities
    )
    fit_X = np.asarray(X)[fit_indices]
    search_history = []
    candidates = []
    expressions = []
    positions = []
    per_action_losses = []

    for action_index in range(y.shape[1]):
        fit_y = np.asarray(y[:, action_index])[fit_indices]
        model = SymbolicRegressor(
            allowed_symbols=operon_allowed_symbols(primitive_set),
            generations=1,
            population_size=population_size,
            pool_size=population_size,
            max_evaluations=max_evaluations,
            max_length=max_length,
            max_depth=10,
            initialization_max_length=10,
            objectives=["mse", "length"],
            model_selection_criterion="minimum_description_length",
            optimizer="lm",
            optimizer_iterations=1,
            add_model_scale_term=True,
            add_model_intercept_term=True,
            n_threads=threads,
            random_state=seed * 10 + action_index,
            warm_start=True,
        )
        for generation in range(generations):
            model.fit(fit_X, fit_y)
            losses = np.asarray([
                solution["mean_squared_error"] for solution in model.pareto_front_
            ], dtype=float)
            search_history.append({
                "action_index": action_index,
                "generation": generation,
                "best_loss": float(np.min(losses)),
                "median_loss": float(np.median(losses)),
                "candidate_count": len(losses),
            })

        original_model = model.model_
        action_candidates = []
        for position, solution in enumerate(model.pareto_front_):
            model.model_ = solution["tree"]
            predictions = jnp.asarray(
                model.predict(np.asarray(validation_X)), dtype=jnp.float32
            ).reshape(-1, 1)
            loss = weighted_mse(
                predictions,
                validation_y[:, action_index:action_index + 1],
                validation_weights,
            )
            action_candidates.append({
                "action_index": action_index,
                "position": position,
                "expression": solution["model"],
                "length": int(solution["length"]),
                "operon_mse": float(solution["mean_squared_error"]),
                "validation_mse": loss,
            })
        model.model_ = original_model
        candidates.extend(action_candidates)
        frame = pd.DataFrame(action_candidates)
        best = frame.loc[frame.validation_mse.idxmin()]
        expressions.append(best.expression)
        positions.append(int(best.position))
        per_action_losses.append(float(best.validation_mse))

    return {
        "expressions": expressions,
        "positions": positions,
        "per_action_validation_mse": per_action_losses,
        "validation_mse": float(np.mean(per_action_losses)),
        "candidates": pd.DataFrame(candidates),
        "search_history": pd.DataFrame(search_history),
        "fit_samples": fit_size,
    }


def collect_expression_trajectories(
    callables, actor, environment, *, num_steps, seed, trajectories
):
    """Run the symbolic student and label its visited states with the expert."""

    def collect_one(trajectory_seed):
        key = jax.random.key(trajectory_seed)

        def action_fn(observation, action_key, _step):
            expert_action, _ = actor(observation, action_key)
            raw_action = jnp.stack([
                jnp.asarray(callable_(*observation)).reshape(())
                for callable_ in callables
            ])
            return sanitize_action(raw_action), expert_action

        X, y, rewards, dones = rollout(environment, key, action_fn, num_steps)
        episode_mask = valid_transition_mask(dones).astype(bool)
        finite_mask = finite_prefix_mask(X, y)
        retained_mask = episode_mask & finite_mask
        safe_rewards = jnp.nan_to_num(
            rewards, nan=0.0, posinf=0.0, neginf=0.0
        )
        episode_return = jnp.sum(jnp.where(retained_mask, safe_rewards, 0.0))
        invalid = jnp.sum(episode_mask & ~finite_mask)
        return X, y, retained_mask, episode_return, invalid

    seeds = seed + jnp.arange(trajectories)
    X, y, masks, returns, invalid = jax.vmap(collect_one)(seeds)
    X_new = jnp.concatenate([X[index][masks[index]] for index in range(trajectories)])
    y_new = jnp.concatenate([y[index][masks[index]] for index in range(trajectories)])
    if len(X_new) == 0:
        raise RuntimeError("Operon student produced no valid transitions")
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
    run_name = args.run_name or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_directory = (
        experiments / "artifacts" / "repertoires" / f"dagger_operon_{args.env}" / run_name
    )
    run_directory.mkdir(parents=True, exist_ok=False)
    config = vars(args) | {
        "algorithm": "DAgger",
        "student": "Operon multi-output symbolic regression",
        "refit_strategy": "from_scratch_each_iteration",
        "replay_storage": "all_valid_transitions",
        "fit_sampling": "deterministic_weighted_sample_from_replay",
        "script": "distillation_experiments.scripts.experiments.dagger_operon",
    }
    (run_directory / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True)
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
    summaries = []
    print(f"Writing Operon DAgger results to {run_directory}")

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
                if mode == "uniform" else initial_q_weights
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
                "iteration", "validation_mse", "student_reward",
                "behavior_reward", "dataset_size", "fit_samples", "new_samples",
                "collection_trajectories", "invalid_transitions",
            ]
            with metrics_path.open("w", newline="") as file:
                csv.DictWriter(file, fieldnames=metric_fields).writeheader()

            best_reward = -float("inf")
            best_iteration = 0
            ever_solved = False
            pending_collection = None
            iteration_summaries = []

            for iteration in range(args.iterations):
                fit = fit_operon_policy(
                    X,
                    y,
                    weights,
                    test_X,
                    test_y,
                    test_weights,
                    seed=seed * 100_000 + mode_index * 10_000 + iteration * 100,
                    fit_samples=args.fit_samples,
                    generations=args.generations_per_iteration,
                    population_size=args.population_size,
                    max_evaluations=args.max_evaluations,
                    max_length=args.max_length,
                    threads=args.threads,
                    primitive_set=args.primitive_set,
                )
                callables = [
                    expression_to_jax(expression, X.shape[1])
                    for expression in fit["expressions"]
                ]
                reward = evaluate_expressions(
                    callables,
                    environment,
                    steps=args.rollout_steps,
                    seed=seed * 100_000 + 90_000,
                    trajectories=args.evaluation_trajectories,
                )
                if reward > best_reward:
                    best_reward = reward
                    best_iteration = iteration
                solved = reward >= args.target_reward
                ever_solved = ever_solved or solved
                metric = {
                    "iteration": iteration,
                    "validation_mse": fit["validation_mse"],
                    "student_reward": reward,
                    "behavior_reward": "" if pending_collection is None else pending_collection["behavior_reward"],
                    "dataset_size": len(X),
                    "fit_samples": fit["fit_samples"],
                    "new_samples": args.expert_bootstrap_samples if pending_collection is None else pending_collection["new_samples"],
                    "collection_trajectories": "" if pending_collection is None else pending_collection["collection_trajectories"],
                    "invalid_transitions": "" if pending_collection is None else pending_collection["invalid_transitions"],
                }
                with metrics_path.open("a", newline="") as file:
                    csv.DictWriter(file, fieldnames=metric_fields).writerow(metric)
                iteration_directory = variant_directory / f"iteration_{iteration}"
                iteration_directory.mkdir()
                fit["candidates"].to_csv(
                    iteration_directory / "candidates.csv", index=False
                )
                fit["search_history"].to_csv(
                    iteration_directory / "search_history.csv", index=False
                )
                iteration_summary = {
                    "iteration": iteration,
                    "expressions": fit["expressions"],
                    "selected_positions": fit["positions"],
                    "per_action_validation_mse": fit["per_action_validation_mse"],
                    "validation_mse": fit["validation_mse"],
                    "reward": reward,
                    "dataset_size": len(X),
                    "fit_samples": fit["fit_samples"],
                }
                (iteration_directory / "summary.json").write_text(
                    json.dumps(iteration_summary, indent=2)
                )
                iteration_summaries.append(iteration_summary)
                print(
                    f"seed={seed} mode={mode} iteration={iteration} "
                    f"validation_mse={fit['validation_mse']:.6g} "
                    f"reward={reward:.3f} replay={len(X)}"
                )
                if args.stop_when_solved and solved:
                    break
                if iteration + 1 == args.iterations:
                    break

                X_new, y_new, pending_collection = collect_expression_trajectories(
                    callables,
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
                "final_validation_mse": fit["validation_mse"],
                "best_observed_reward": best_reward,
                "best_observed_iteration": best_iteration,
                "solved": ever_solved,
                "dataset_size": len(X),
                "test_dataset_size": len(test_X),
                "final_expressions": fit["expressions"],
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
