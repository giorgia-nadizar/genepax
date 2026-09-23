"""Distill an ANN expert into CGP with plain dataset aggregation (DAgger).

This follows Algorithm 1 of Kohler et al. (2025): bootstrap from expert
demonstrations, execute the current student thereafter, label every visited
state with the expert action, aggregate the data, and refit using supervised
action error only. Environment reward is logged but never enters training or
behavior-policy selection.
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

from distillation.evaluation import (
    collect_mixed_policy_dataset,
    evaluate_symbolic_policy,
)
from distillation.fit_imitation import fit_imitation_dataset
from distillation.networks.sac_utils import load_sac_actor
from genepax.gp.cartesian_genetic_programming import CGP


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def create_run_directory(root, environment, run_name):
    if run_name is None:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = root / f"dagger_{environment}" / run_name
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def select_rows(X, y, count, key):
    """Select exactly ``count`` aligned rows without replacement."""
    if len(X) < count:
        raise ValueError(
            f"Dataset has {len(X)} samples but {count} were requested"
        )
    indices = jax.random.permutation(key, len(X))[:count]
    return X[indices], y[indices]


def collect_student_trajectories(
    genotype,
    cgp_structure,
    actor,
    environment,
    rollout_steps,
    trajectories,
    seed,
):
    """Collect one fixed batch of student trajectories for aggregation."""
    X, y, reward, diagnostics = collect_mixed_policy_dataset(
        genotype,
        cgp_structure,
        actor,
        environment,
        expert_weight=0.0,
        num_steps=rollout_steps,
        n_seeds=trajectories,
        seed=seed,
    )
    if len(X) == 0:
        raise RuntimeError("Student rollout produced no valid transitions")
    return X, y, {
        "behavior_reward": float(reward),
        "collection_trajectories": trajectories,
        "invalid_transitions": int(diagnostics["invalid_transitions"]),
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run plain DAgger distillation from an ANN expert to CGP"
    )
    parser.add_argument("--env", default="inverted_pendulum")
    parser.add_argument("--backend", default="generalized")
    parser.add_argument("--run-name")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=positive_int, default=1)
    parser.add_argument("--iterations", type=positive_int, default=10)
    parser.add_argument(
        "--expert-bootstrap-samples", type=positive_int, default=10_000
    )
    parser.add_argument("--sr-generations", type=positive_int, default=50)
    parser.add_argument("--population-size", type=positive_int, default=100)
    parser.add_argument("--dataset-batch-size", type=positive_int, default=4_096)
    parser.add_argument("--rollout-steps", type=positive_int, default=1_000)
    parser.add_argument(
        "--trajectories-per-iteration", type=positive_int, default=20
    )
    parser.add_argument("--evaluation-trajectories", type=positive_int, default=10)
    parser.add_argument("--target-reward", type=float, default=999.0)
    parser.add_argument(
        "--linear-scaling",
        action="store_true",
        help="Fit and store an affine calibration for every CGP output.",
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Initialize each refit from the preceding CGP population.",
    )
    parser.add_argument(
        "--stop-when-solved",
        action="store_true",
        help="Stop aggregation early once the current student reaches the target.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    experiments_directory = Path(__file__).resolve().parents[2]
    checkpoint_path = experiments_directory / "artifacts" / "expert_models" / args.env / "final"
    dataset_path = (
        experiments_directory / "artifacts" / "expert_datasets" / f"expert_{args.env}.npz"
    )
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Expert model not found: {checkpoint_path}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Expert dataset not found: {dataset_path}")

    run_directory = create_run_directory(
        experiments_directory / "artifacts" / "repertoires", args.env, args.run_name
    )
    configuration = vars(args) | {
        "algorithm": "DAgger",
        "student_objective": "mean_squared_expert_action_error",
        "script": "distillation_experiments.scripts.experiments.dagger",
    }
    with (run_directory / "config.json").open("w") as file:
        json.dump(configuration, file, indent=2, sort_keys=True)
    print(f"Writing DAgger results to {run_directory}")

    expert_data = np.load(dataset_path)
    all_expert_X = jnp.asarray(expert_data["X"], dtype=jnp.float32)
    all_expert_y = jnp.asarray(expert_data["y"], dtype=jnp.float32)
    summaries = []

    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        environment = envs.get_environment(
            env_name=args.env, backend=args.backend
        )
        actor = None
        if args.iterations > 1:
            actor, _ = load_sac_actor(checkpoint_path)
        cgp_structure = CGP(
            n_inputs=all_expert_X.shape[1],
            n_outputs=all_expert_y.shape[1],
            n_custom_weights=(
                2 * all_expert_y.shape[1] if args.linear_scaling else 0
            ),
        )
        seed_directory = run_directory / f"seed_{seed}"
        seed_directory.mkdir()
        metrics_path = seed_directory / "metrics.csv"
        with metrics_path.open("w", newline="") as file:
            csv.writer(file).writerow([
                "iteration",
                "imitation_loss",
                "raw_imitation_loss",
                "student_reward",
                "behavior_policy",
                "behavior_reward",
                "dataset_size",
                "new_samples",
                "collection_trajectories",
                "invalid_transitions",
                "finite_fitness_fraction",
                "scaling_weights",
            ])

        X, y = select_rows(
            all_expert_X,
            all_expert_y,
            args.expert_bootstrap_samples,
            jax.random.key(seed),
        )
        repertoire = None
        best_reward = -float("inf")
        best_iteration = 0
        last_iteration = 0
        solved = False
        ever_solved = False
        pending_collection = None

        for iteration in range(args.iterations):
            fit_result = fit_imitation_dataset(
                X,
                y,
                cgp_structure,
                seed=seed * 100_000 + iteration * 100,
                n_gens=args.sr_generations,
                n_pop=args.population_size,
                dataset_batch_size=args.dataset_batch_size,
                bootstrap_repertoire=(repertoire if args.warm_start else None),
                linear_scaling=args.linear_scaling,
            )
            repertoire = fit_result["repertoire"]
            genotype = fit_result["genotype"]
            student_reward = float(evaluate_symbolic_policy(
                genotype,
                cgp_structure,
                environment,
                num_steps=args.rollout_steps,
                seed=seed * 100_000 + 90_000,
                n_seeds=args.evaluation_trajectories,
                linear_scaling=args.linear_scaling,
            ))
            behavior_policy = "expert" if iteration == 0 else "student"
            behavior_reward = (
                "" if pending_collection is None
                else pending_collection["behavior_reward"]
            )
            collection_trajectories = (
                "" if pending_collection is None
                else pending_collection["collection_trajectories"]
            )
            invalid_transitions = (
                "" if pending_collection is None
                else pending_collection["invalid_transitions"]
            )
            new_samples = (
                args.expert_bootstrap_samples
                if iteration == 0
                else len(X) - previous_dataset_size
            )
            if student_reward > best_reward:
                best_reward = student_reward
                best_iteration = iteration
            solved = student_reward >= args.target_reward
            ever_solved = ever_solved or solved
            with metrics_path.open("a", newline="") as file:
                csv.writer(file).writerow([
                    iteration,
                    fit_result["loss"],
                    fit_result.get("raw_loss", ""),
                    student_reward,
                    behavior_policy,
                    behavior_reward,
                    len(X),
                    new_samples,
                    collection_trajectories,
                    invalid_transitions,
                    fit_result["finite_fitness_fraction"],
                    json.dumps(fit_result.get("scaling_weights", [])),
                ])
            print(
                f"seed={seed} iteration={iteration} "
                f"loss={fit_result['loss']:.6g} reward={student_reward:.3f} "
                f"samples={len(X)}"
            )
            last_iteration = iteration
            if args.stop_when_solved and solved:
                break
            if iteration + 1 == args.iterations:
                break

            X_new, y_new, collection = collect_student_trajectories(
                genotype,
                cgp_structure,
                actor,
                environment,
                args.rollout_steps,
                args.trajectories_per_iteration,
                seed=seed * 100_000 + iteration * 10_000,
                linear_scaling=args.linear_scaling,
            )
            previous_dataset_size = len(X)
            X = jnp.concatenate([X, X_new])
            y = jnp.concatenate([y, y_new])
            # These statistics describe the new data used by the next fit.
            pending_collection = collection

        with (seed_directory / "final_population.pickle").open("wb") as file:
            pickle.dump(repertoire, file)
        with (seed_directory / "final_individual.pickle").open("wb") as file:
            pickle.dump(genotype, file)
        np.savez_compressed(
            seed_directory / "final_dataset.npz",
            X=np.asarray(X),
            y=np.asarray(y),
        )
        summary = {
            "seed": seed,
            "last_iteration": last_iteration,
            "final_reward": student_reward,
            "final_imitation_loss": fit_result["loss"],
            "final_raw_imitation_loss": fit_result.get("raw_loss"),
            "linear_scaling": args.linear_scaling,
            "scaling_weights": fit_result.get("scaling_weights"),
            "best_observed_reward": best_reward,
            "best_observed_iteration": best_iteration,
            "solved": ever_solved,
            "dataset_size": len(X),
        }
        with (seed_directory / "summary.json").open("w") as file:
            json.dump(summary, file, indent=2)
        summaries.append(summary)
        with (run_directory / "aggregate_summary.json").open("w") as file:
            json.dump(
                {
                    "completed_seeds": len(summaries),
                    "requested_seeds": args.num_seeds,
                    "solved_seeds": sum(item["solved"] for item in summaries),
                    "seeds": summaries,
                },
                file,
                indent=2,
            )


if __name__ == "__main__":
    main()
