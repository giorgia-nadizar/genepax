import argparse
import csv
from datetime import datetime, timezone
import json
import pickle
from pathlib import Path

import jax

from distillation.evaluation import collect_mixed_policy_dataset, \
    evaluate_symbolic_policy
from distillation.fit_dataset import fit_dataset
import jax.numpy as jnp
from brax import envs

from distillation.networks.sac_utils import load_sac_actor, load_q_value_estimator
from genepax.gp.cartesian_genetic_programming import CGP


def mixture_weight(iteration, n_iterations, start, end):
    """Returns a linearly scheduled expert weight, including both endpoints."""
    if not 0.0 <= start <= 1.0 or not 0.0 <= end <= 1.0:
        raise ValueError("Expert weights must be between zero and one")
    progress = iteration / max(n_iterations - 1, 1)
    return start + progress * (end - start)


def bounded_unique_dataset(X, y, max_size, key):
    """Deduplicates aligned samples and uniformly caps their total count."""
    if max_size <= 0:
        return X[:0], y[:0]
    X, indices = jnp.unique(X, axis=0, return_index=True)
    y = y[indices]
    if len(X) > max_size:
        indices = jax.random.permutation(key, len(X))[:max_size]
        X, y = X[indices], y[indices]
    return X, y


def combine_datasets(expert_X, expert_y, dagger_X, dagger_y):
    """Combines the fixed expert anchor with the evolving DAgger reservoir."""
    if len(dagger_X) == 0:
        return expert_X, expert_y
    return (
        jnp.concatenate([expert_X, dagger_X]),
        jnp.concatenate([expert_y, dagger_y]),
    )


def update_reservoir(X, y, priorities, X_new, y_new, max_size, key):
    """Updates a uniform, fixed-size reservoir using persistent priorities."""
    if max_size <= 0:
        return X[:0], y[:0], priorities[:0]

    new_priorities = jax.random.uniform(key, shape=(len(X_new),))
    X = jnp.concatenate([X, X_new])
    y = jnp.concatenate([y, y_new])
    priorities = jnp.concatenate([priorities, new_priorities])

    X, unique_indices = jnp.unique(X, axis=0, return_index=True)
    y = y[unique_indices]
    priorities = priorities[unique_indices]
    if len(X) > max_size:
        retained = jnp.argsort(priorities)[-max_size:]
        X, y, priorities = X[retained], y[retained], priorities[retained]
    return X, y, priorities


def positive_int(value):
    """Argparse converter for strictly positive integer parameters."""
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def create_run_directory(repertoires_dir, env_name, run_name=None):
    """Creates an isolated result directory and refuses to overwrite runs."""
    if run_name is None:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = repertoires_dir / f"spid_{env_name}" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run CGP-based SPID distillation")
    parser.add_argument("--env", default="inverted_pendulum")
    parser.add_argument("--backend", default="generalized")
    parser.add_argument("--run-name")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--search-seed-base",
        type=int,
        help=(
            "Optional CGP/reservoir RNG base for the first training seed. "
            "Useful for exactly replaying an older run."
        ),
    )
    parser.add_argument(
        "--search-seed-stride",
        type=positive_int,
        default=10_000,
        help="Distance between CGP/reservoir RNG streams for consecutive seeds.",
    )
    parser.add_argument("--num-seeds", type=positive_int, default=10)
    parser.add_argument("--iterations", type=positive_int, default=200)
    parser.add_argument("--sr-generations", type=positive_int, default=50)
    parser.add_argument("--population-size", type=positive_int, default=50)
    parser.add_argument("--rollout-steps", type=positive_int, default=1_000)
    parser.add_argument("--evaluation-trajectories", type=positive_int, default=10)
    parser.add_argument("--dataset-batch-size", type=positive_int, default=4_096)
    parser.add_argument("--target-reward", type=float, default=999.0)
    parser.add_argument("--gm-alpha", type=float, default=0.1)
    parser.add_argument("--gm-epsilon", type=float, default=0.1)
    parser.add_argument(
        "--expert-weight-start",
        type=float,
        default=0.5,
        help="Expert action weight at the first iteration (default: 0.5).",
    )
    parser.add_argument(
        "--expert-weight-end",
        type=float,
        help="Expert weight at the last iteration; omit for a fixed weight.",
    )
    parser.add_argument("--max-dataset-size", type=positive_int, default=100_000)
    parser.add_argument(
        "--expert-fraction",
        type=float,
        default=0.25,
        help="Fraction reserved for initial expert demonstrations.",
    )
    parser.add_argument(
        "--trajectories-per-iteration",
        type=positive_int,
        default=20,
    )
    args = parser.parse_args()
    expert_weight_end = (
        args.expert_weight_start
        if args.expert_weight_end is None
        else args.expert_weight_end
    )
    # Validate before loading models or starting an expensive experiment.
    mixture_weight(
        0, 1, args.expert_weight_start, expert_weight_end
    )
    if args.max_dataset_size < 2:
        raise ValueError("max-dataset-size must be at least two")
    if not 0.0 < args.expert_fraction < 1.0:
        raise ValueError("expert-fraction must be strictly between zero and one")
    if args.trajectories_per_iteration <= 0:
        raise ValueError("trajectories-per-iteration must be positive")
    if args.gm_alpha <= 0 or args.gm_epsilon <= 0:
        raise ValueError("gm-alpha and gm-epsilon must be positive")

    experiments_dir = Path(__file__).resolve().parents[1]
    ENV_NAME = args.env
    checkpoint_path = experiments_dir / "expert_models" / ENV_NAME / "final"
    dataset_path = (
        experiments_dir / "expert_datasets" / f"expert_{ENV_NAME}.npz"
    )
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Expert model not found: {checkpoint_path}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Expert dataset not found: {dataset_path}")

    run_dir = create_run_directory(
        experiments_dir / "repertoires",
        ENV_NAME,
        args.run_name,
    )
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    print(f"Writing SPID results to {run_dir}")

    # ENV_NAME = "inverted_double_pendulum"
    # n_iterations = 500
    # sr_generations = 50
    # max_dataset_size = 20_000
    # target_reward = 9000

    for SEED in range(args.seed_start, args.seed_start + args.num_seeds):
        search_seed_base = (
            SEED * args.search_seed_stride
            if args.search_seed_base is None
            else args.search_seed_base
            + (SEED - args.seed_start) * args.search_seed_stride
        )
        eval_env = envs.get_environment(
            env_name=ENV_NAME,
            backend=args.backend,
        )
        seed_dir = run_dir / f"seed_{SEED}"
        seed_dir.mkdir()
        CSV_PATH = seed_dir / "metrics.csv"

        # we have already collected the initial dataset at expert_dataset.npz
        data = jnp.load(dataset_path)

        expert_X = jnp.asarray(data["X"], dtype=jnp.float32)
        expert_y = jnp.asarray(data["y"], dtype=jnp.float32)

        # Init the CGP policy graph with default values
        cgp_structure = CGP(
            n_inputs=expert_X.shape[1],
            n_outputs=data["y"].shape[1],
        )
        repertoire = None
        # SAC is the sole teacher.  The CGP is trained against SAC actions on
        # the initial demonstrations and on every subsequent DAgger rollout.
        sac_actor, _ = load_sac_actor(checkpoint_path)

        expert_budget = min(
            args.max_dataset_size - 1,
            max(1, round(args.max_dataset_size * args.expert_fraction)),
        )
        dagger_budget = args.max_dataset_size - expert_budget
        expert_X, expert_y = bounded_unique_dataset(
            expert_X,
            expert_y,
            expert_budget,
            jax.random.key(SEED),
        )
        dagger_X = jnp.empty((0, expert_X.shape[1]), dtype=expert_X.dtype)
        dagger_y = jnp.empty((0, expert_y.shape[1]), dtype=expert_y.dtype)
        dagger_priorities = jnp.empty((0,), dtype=jnp.float32)
        X, y = combine_datasets(expert_X, expert_y, dagger_X, dagger_y)

        # The critic gives an additional value-regression penalty; action
        # labels themselves always come from SAC.
        q_value_estimator = load_q_value_estimator(checkpoint_path)

        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "iteration", "gm_loss", "symbolic_reward",
                "mixed_reward", "expert_weight", "dataset_size",
                "expert_samples", "dagger_samples",
                "finite_fitness_fraction", "performance_gap",
                "fidelity_gap", "invalid_collected_transitions",
            ])

        # Dataset initialization: fit pi_hat_0 only on teacher demonstrations.
        repertoire, loss, fit_diagnostics = fit_dataset(
            X,
            y,
            cgp_structure,
            bootstrap_repertoire=None,
            n_pop=args.population_size,
            q_value_estimator=q_value_estimator,
            n_gens=args.sr_generations,
            dataset_batch_size=args.dataset_batch_size,
            alpha=args.gm_alpha,
            epsilon=args.gm_epsilon,
            seed=search_seed_base,
        )
        best_idx = jnp.argmax(repertoire.fitnesses)
        best_genotype = jax.tree.map(
            lambda value: value[best_idx], repertoire.genotypes
        )
        evaluation_result = evaluate_symbolic_policy(
            best_genotype,
            cgp_structure,
            eval_env,
            num_steps=args.rollout_steps,
            seed=SEED * 10_000 + 9_000,
            n_seeds=args.evaluation_trajectories,
        )
        best_validation_reward = float(evaluation_result)
        best_iteration = 0
        best_repertoire = repertoire
        best_validation_genotype = best_genotype
        with open(CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow([
                0, loss, evaluation_result, "", "",
                len(X), len(expert_X), len(dagger_X),
                fit_diagnostics["finite_fitness_fraction"],
                fit_diagnostics["best_performance_gap"],
                fit_diagnostics["best_fidelity_gap"],
                "",
            ])
        print(f"0 gm_loss {loss} evaluation {evaluation_result}")

        # GM-DAGGER: collect -> aggregate -> refit -> evaluate.
        iteration = 0
        for iteration in range(1, args.iterations):
            if evaluation_result >= args.target_reward:
                print("ENV SOLVED")
                break
            expert_weight = mixture_weight(
                iteration - 1,
                args.iterations - 1,
                args.expert_weight_start,
                expert_weight_end,
            )
            X_new, y_new, mixed_evaluation_reward, collection_diagnostics = (
                collect_mixed_policy_dataset(
                    best_genotype,
                    cgp_structure,
                    sac_actor,
                    eval_env,
                    expert_weight=expert_weight,
                    num_steps=args.rollout_steps,
                    n_seeds=args.trajectories_per_iteration,
                    seed=SEED * 10_000 + iteration * 100,
                )
            )

            # Persistent random priorities form a uniform reservoir over all
            # DAgger states observed so far. The initial demonstrations retain
            # a separate fixed quota, so neither distribution crowds out the
            # other.
            dagger_X, dagger_y, dagger_priorities = update_reservoir(
                dagger_X,
                dagger_y,
                dagger_priorities,
                X_new,
                y_new,
                dagger_budget,
                jax.random.key(search_seed_base + iteration),
            )
            X, y = combine_datasets(
                expert_X, expert_y, dagger_X, dagger_y
            )

            repertoire, loss, fit_diagnostics = fit_dataset(
                X,
                y,
                cgp_structure,
                bootstrap_repertoire=repertoire,
                n_pop=args.population_size,
                q_value_estimator=q_value_estimator,
                n_gens=args.sr_generations,
                dataset_batch_size=args.dataset_batch_size,
                alpha=args.gm_alpha,
                epsilon=args.gm_epsilon,
                seed=search_seed_base + iteration,
            )
            best_idx = jnp.argmax(repertoire.fitnesses)
            best_genotype = jax.tree.map(
                lambda value: value[best_idx], repertoire.genotypes
            )
            evaluation_result = evaluate_symbolic_policy(
                best_genotype,
                cgp_structure,
                eval_env,
                num_steps=args.rollout_steps,
                seed=SEED * 10_000 + 9_000,
                n_seeds=args.evaluation_trajectories,
            )
            if float(evaluation_result) > best_validation_reward:
                best_validation_reward = float(evaluation_result)
                best_iteration = iteration
                best_repertoire = repertoire
                best_validation_genotype = best_genotype

            print(
                f"{iteration} gm_loss {loss} "
                f"evaluation {evaluation_result} "
                f"mixed-policy reward {mixed_evaluation_reward}"
            )
            with open(CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerow([
                    iteration,
                    loss,
                    evaluation_result,
                    mixed_evaluation_reward,
                    expert_weight,
                    len(X),
                    len(expert_X),
                    len(dagger_X),
                    fit_diagnostics["finite_fitness_fraction"],
                    fit_diagnostics["best_performance_gap"],
                    fit_diagnostics["best_fidelity_gap"],
                    collection_diagnostics["invalid_transitions"],
                ])
            if evaluation_result >= args.target_reward:
                print("ENV SOLVED")
                break

        with open(seed_dir / "best_repertoire.pickle", "wb") as file:
            pickle.dump(best_repertoire, file)
        with open(seed_dir / "best_genotype.pickle", "wb") as file:
            pickle.dump(best_validation_genotype, file)
        with open(seed_dir / "summary.json", "w") as f:
            json.dump(
                {
                    "best_iteration": best_iteration,
                    "best_validation_reward": best_validation_reward,
                    "last_iteration": iteration,
                },
                f,
                indent=2,
            )
