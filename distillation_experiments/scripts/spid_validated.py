"""SPID adaptation with rollout validation and guarded aggregation."""

import argparse
import csv
from datetime import datetime, timezone
import json
import pickle
from pathlib import Path

import jax
import numpy as np

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


def scheduled_expert_weight(iteration, anneal_iterations, start, end):
    """Returns a bounded schedule for a one-based aggregation iteration."""
    if iteration <= 0:
        raise ValueError("aggregation iteration must be positive")
    schedule_index = min(iteration - 1, anneal_iterations - 1)
    return mixture_weight(schedule_index, anneal_iterations, start, end)


def select_aggregation_policy(
    mode,
    current_genotype,
    current_iteration,
    current_reward,
    best_genotype,
    best_iteration,
    best_reward,
):
    """Selects the student used to collect the next DAgger trajectories."""
    if mode == "current":
        return current_genotype, current_iteration, current_reward
    if mode == "best":
        return best_genotype, best_iteration, best_reward
    raise ValueError(f"Unknown aggregation policy: {mode}")


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


def combine_dataset_parts(*parts):
    """Combines any non-empty aligned dataset partitions."""
    non_empty = [(X, y) for X, y in parts if len(X)]
    if not non_empty:
        raise ValueError("At least one dataset partition must be non-empty")
    return (
        jnp.concatenate([X for X, _ in non_empty]),
        jnp.concatenate([y for _, y in non_empty]),
    )


def recovery_state_mask(X, observation_index, threshold):
    """Identifies off-balance states reserved for recovery supervision."""
    if observation_index < 0 or observation_index >= X.shape[1]:
        raise ValueError("recovery observation index is outside the observation")
    if threshold < 0:
        raise ValueError("recovery threshold must be non-negative")
    return jnp.abs(X[:, observation_index]) >= threshold


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


def positive_int_list(value):
    """Argparse converter for an increasing comma-separated integer list."""
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a comma-separated list of integers"
        ) from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all checkpoints must be positive")
    if tuple(sorted(set(values))) != values:
        raise argparse.ArgumentTypeError(
            "checkpoints must be strictly increasing"
        )
    return values


def create_run_directory(repertoires_dir, env_name, run_name=None):
    """Creates an isolated result directory and refuses to overwrite runs."""
    if run_name is None:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = repertoires_dir / f"spid_{env_name}" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def select_validated_candidate(
    repertoire,
    top_k,
    cgp_structure,
    environment,
    num_steps,
    seed,
    n_seeds,
):
    """Validates the top loss-ranked candidates and returns the best rollout."""
    fitnesses = jnp.ravel(repertoire.fitnesses)
    top_k = min(top_k, len(fitnesses))
    candidate_indices = jnp.argsort(fitnesses)[-top_k:]
    candidate_genotypes = jax.tree.map(
        lambda value: value[candidate_indices], repertoire.genotypes
    )
    rewards = jax.vmap(
        lambda genotype: evaluate_symbolic_policy(
            genotype,
            cgp_structure,
            environment,
            num_steps=num_steps,
            seed=seed,
            n_seeds=n_seeds,
        )
    )(candidate_genotypes)
    selected_position = int(jnp.argmax(rewards))
    selected_index = int(candidate_indices[selected_position])
    selected_genotype = jax.tree.map(
        lambda value: value[selected_position], candidate_genotypes
    )
    selected_fitness_rank = 1 + int(
        jnp.sum(fitnesses > fitnesses[selected_index])
    )
    return {
        "genotype": selected_genotype,
        "repertoire_index": selected_index,
        "fitness_rank": selected_fitness_rank,
        "reward": float(rewards[selected_position]),
        "candidate_indices": np.asarray(candidate_indices).tolist(),
        "candidate_rewards": np.asarray(rewards).tolist(),
    }


def fit_and_validate_blocks(
    X,
    y,
    cgp_structure,
    environment,
    q_value_estimator,
    bootstrap_repertoire,
    n_pop,
    n_gens,
    validation_interval,
    validation_checkpoints,
    validation_top_k,
    dataset_batch_size,
    alpha,
    epsilon,
    search_seed,
    evaluation_seed,
    evaluation_trajectories,
    rollout_steps,
):
    """Fits in blocks and retains the best rollout-validated candidate."""
    block_validations = []
    best_validated = None
    best_validated_repertoire = None
    repertoire = bootstrap_repertoire
    completed_generations = 0
    block_index = 0

    checkpoints = (
        tuple(validation_checkpoints)
        if validation_checkpoints
        else tuple(range(validation_interval, n_gens, validation_interval))
        + (n_gens,)
    )
    for checkpoint in checkpoints:
        logical_block_size = checkpoint - completed_generations
        # A bootstrapped fit spends its first reported generation rescoring
        # the population. Add one so repeated blocks retain the same number
        # of mutation updates as one uninterrupted fit.
        fit_generations = logical_block_size + int(block_index > 0)
        repertoire, loss, diagnostics = fit_dataset(
            X,
            y,
            cgp_structure,
            bootstrap_repertoire=repertoire,
            n_pop=n_pop,
            q_value_estimator=q_value_estimator,
            n_gens=fit_generations,
            dataset_batch_size=dataset_batch_size,
            alpha=alpha,
            epsilon=epsilon,
            seed=search_seed + block_index,
        )
        completed_generations += logical_block_size
        validated = select_validated_candidate(
            repertoire,
            validation_top_k,
            cgp_structure,
            environment,
            rollout_steps,
            seed=evaluation_seed,
            n_seeds=evaluation_trajectories,
        )
        validation_record = {
            "generation": completed_generations,
            "loss": float(loss),
            "reward": validated["reward"],
            "selected_fitness_rank": validated["fitness_rank"],
            "candidate_indices": validated["candidate_indices"],
            "candidate_rewards": validated["candidate_rewards"],
        }
        block_validations.append(validation_record)
        if (
            best_validated is None
            or validated["reward"] > best_validated["reward"]
        ):
            best_validated = validated
            best_validated_repertoire = repertoire
            best_validated_generation = completed_generations
            best_loss = loss
            best_diagnostics = diagnostics

        # Continue evolution from the repertoire associated with the best
        # rollout checkpoint, rather than allowing a lower-loss but weaker
        # later block to dictate the next search region.
        repertoire = best_validated_repertoire

        block_index += 1

    return {
        "repertoire": repertoire,
        "loss": best_loss,
        "diagnostics": best_diagnostics,
        "validated": best_validated,
        "validated_repertoire": best_validated_repertoire,
        "best_generation": best_validated_generation,
        "block_validations": block_validations,
    }


def save_iteration(
    seed_dir,
    iteration,
    genotype,
    expert_X,
    expert_y,
    dagger_X,
    dagger_y,
    recovery_X,
    recovery_y,
    metadata,
):
    """Persists the selected controller and exact training dataset."""
    iteration_dir = seed_dir / "iterations" / f"iteration_{iteration:03d}"
    iteration_dir.mkdir(parents=True)
    with (iteration_dir / "selected_genotype.pickle").open("wb") as file:
        pickle.dump(genotype, file)
    np.savez_compressed(
        iteration_dir / "dataset.npz",
        expert_X=np.asarray(expert_X),
        expert_y=np.asarray(expert_y),
        dagger_X=np.asarray(dagger_X),
        dagger_y=np.asarray(dagger_y),
        recovery_X=np.asarray(recovery_X),
        recovery_y=np.asarray(recovery_y),
    )
    with (iteration_dir / "metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2)


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
    parser.add_argument(
        "--validation-top-k",
        type=positive_int,
        default=10,
        help="Number of top loss-ranked candidates validated in Brax.",
    )
    parser.add_argument(
        "--validation-interval",
        type=positive_int,
        default=10,
        help="CGP generations between rollout-validation checkpoints.",
    )
    parser.add_argument(
        "--validation-checkpoints",
        type=positive_int_list,
        help=(
            "Optional comma-separated CGP generations to validate, e.g. "
            "10,30,50. Overrides validation-interval."
        ),
    )
    parser.add_argument("--dataset-batch-size", type=positive_int, default=4_096)
    parser.add_argument(
        "--stop-after-first-solved",
        action="store_true",
        help="Stop the multi-seed experiment after its first solved seed.",
    )
    parser.add_argument("--target-reward", type=float, default=999.0)
    parser.add_argument("--gm-alpha", type=float, default=0.1)
    parser.add_argument("--gm-epsilon", type=float, default=0.1)
    parser.add_argument(
        "--expert-weight-start",
        type=float,
        default=0.8,
        help="Expert action weight at the first aggregation iteration.",
    )
    parser.add_argument(
        "--expert-weight-end",
        type=float,
        default=0.0,
        help="Expert action weight at the last aggregation iteration.",
    )
    parser.add_argument(
        "--expert-anneal-iterations",
        type=positive_int,
        default=9,
        help=(
            "Aggregation iterations used to reach expert-weight-end; the "
            "end weight is held constant afterward."
        ),
    )
    parser.add_argument(
        "--aggregation-policy",
        choices=("best", "current"),
        default="best",
        help=(
            "Student used for DAgger collection: the best rollout-validated "
            "student seen so far, or the student selected in the last fit."
        ),
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
    parser.add_argument(
        "--recovery-fraction",
        type=float,
        default=0.25,
        help="Fraction of the DAgger budget reserved for recovery states.",
    )
    parser.add_argument("--recovery-observation-index", type=int, default=1)
    parser.add_argument("--recovery-threshold", type=float, default=0.1)
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
    if not 0.0 <= args.recovery_fraction < 1.0:
        raise ValueError("recovery-fraction must be in [0, 1)")
    if args.recovery_threshold < 0:
        raise ValueError("recovery-threshold must be non-negative")
    if args.validation_top_k > args.population_size:
        raise ValueError("validation-top-k cannot exceed population-size")
    if args.validation_interval > args.sr_generations:
        raise ValueError("validation-interval cannot exceed sr-generations")
    if (
        args.validation_checkpoints
        and args.validation_checkpoints[-1] != args.sr_generations
    ):
        raise ValueError(
            "the last validation checkpoint must equal sr-generations"
        )

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

    seed_summaries = []
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
        recovery_budget = round(dagger_budget * args.recovery_fraction)
        ordinary_dagger_budget = dagger_budget - recovery_budget
        expert_X, expert_y = bounded_unique_dataset(
            expert_X,
            expert_y,
            expert_budget,
            jax.random.key(SEED),
        )
        dagger_X = jnp.empty((0, expert_X.shape[1]), dtype=expert_X.dtype)
        dagger_y = jnp.empty((0, expert_y.shape[1]), dtype=expert_y.dtype)
        dagger_priorities = jnp.empty((0,), dtype=jnp.float32)
        recovery_X = jnp.empty((0, expert_X.shape[1]), dtype=expert_X.dtype)
        recovery_y = jnp.empty((0, expert_y.shape[1]), dtype=expert_y.dtype)
        recovery_priorities = jnp.empty((0,), dtype=jnp.float32)
        X, y = combine_dataset_parts(
            (expert_X, expert_y),
            (dagger_X, dagger_y),
            (recovery_X, recovery_y),
        )

        # The critic gives an additional value-regression penalty; action
        # labels themselves always come from SAC.
        q_value_estimator = load_q_value_estimator(checkpoint_path)

        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "iteration", "gm_loss", "symbolic_reward",
                "mixed_reward", "expert_weight", "dataset_size",
                "expert_samples", "dagger_samples",
                "recovery_samples",
                "finite_fitness_fraction", "performance_gap",
                "fidelity_gap", "invalid_collected_transitions",
                "selected_fitness_rank", "validation_candidates",
                "behavior_policy_iteration", "behavior_policy_reward",
                "realized_expert_contribution",
            ])

        # Dataset initialization: fit pi_hat_0 only on teacher demonstrations.
        block_result = fit_and_validate_blocks(
            X,
            y,
            cgp_structure,
            eval_env,
            q_value_estimator,
            bootstrap_repertoire=None,
            n_pop=args.population_size,
            n_gens=args.sr_generations,
            validation_interval=args.validation_interval,
            validation_checkpoints=args.validation_checkpoints,
            validation_top_k=args.validation_top_k,
            dataset_batch_size=args.dataset_batch_size,
            alpha=args.gm_alpha,
            epsilon=args.gm_epsilon,
            search_seed=search_seed_base * 100,
            evaluation_seed=SEED * 10_000 + 9_000,
            evaluation_trajectories=args.evaluation_trajectories,
            rollout_steps=args.rollout_steps,
        )
        repertoire = block_result["repertoire"]
        loss = block_result["loss"]
        fit_diagnostics = block_result["diagnostics"]
        validated = block_result["validated"]
        best_genotype = validated["genotype"]
        evaluation_result = validated["reward"]
        best_validation_reward = evaluation_result
        best_iteration = 0
        best_repertoire = block_result["validated_repertoire"]
        best_validation_genotype = best_genotype
        with open(CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow([
                0, loss, evaluation_result, "", "",
                len(X), len(expert_X), len(dagger_X), len(recovery_X),
                fit_diagnostics["finite_fitness_fraction"],
                fit_diagnostics["best_performance_gap"],
                fit_diagnostics["best_fidelity_gap"],
                "", validated["fitness_rank"], args.validation_top_k,
                "", "", "",
            ])
        save_iteration(
            seed_dir,
            0,
            best_genotype,
            expert_X,
            expert_y,
            dagger_X,
            dagger_y,
            recovery_X,
            recovery_y,
            {
                "iteration": 0,
                "validation_reward": evaluation_result,
                "selected_fitness_rank": validated["fitness_rank"],
                "candidate_indices": validated["candidate_indices"],
                "candidate_rewards": validated["candidate_rewards"],
                "best_validation_generation": block_result["best_generation"],
                "block_validations": block_result["block_validations"],
                "expert_weight": None,
            },
        )
        print(f"0 gm_loss {loss} evaluation {evaluation_result}")

        # GM-DAGGER: collect -> aggregate -> refit -> evaluate.
        iteration = 0
        for iteration in range(1, args.iterations):
            if evaluation_result >= args.target_reward:
                print("ENV SOLVED")
                break
            expert_weight = scheduled_expert_weight(
                iteration,
                args.expert_anneal_iterations,
                args.expert_weight_start,
                expert_weight_end,
            )
            (
                behavior_genotype,
                behavior_policy_iteration,
                behavior_policy_reward,
            ) = select_aggregation_policy(
                args.aggregation_policy,
                best_genotype,
                iteration - 1,
                evaluation_result,
                best_validation_genotype,
                best_iteration,
                best_validation_reward,
            )
            X_new, y_new, mixed_evaluation_reward, collection_diagnostics = (
                collect_mixed_policy_dataset(
                    behavior_genotype,
                    cgp_structure,
                    sac_actor,
                    eval_env,
                    expert_weight=expert_weight,
                    num_steps=args.rollout_steps,
                    n_seeds=args.trajectories_per_iteration,
                    seed=SEED * 10_000 + iteration * 100,
                )
            )

            recovery_mask = recovery_state_mask(
                X_new,
                args.recovery_observation_index,
                args.recovery_threshold,
            )
            ordinary_X_new, ordinary_y_new = (
                X_new[~recovery_mask], y_new[~recovery_mask]
            )
            recovery_X_new, recovery_y_new = (
                X_new[recovery_mask], y_new[recovery_mask]
            )
            reservoir_key = jax.random.key(search_seed_base + iteration)
            ordinary_key, recovery_key = jax.random.split(reservoir_key)
            # Separate persistent reservoirs prevent ordinary near-expert
            # states from crowding recovery supervision out of the dataset.
            dagger_X, dagger_y, dagger_priorities = update_reservoir(
                dagger_X,
                dagger_y,
                dagger_priorities,
                ordinary_X_new,
                ordinary_y_new,
                ordinary_dagger_budget,
                ordinary_key,
            )
            recovery_X, recovery_y, recovery_priorities = update_reservoir(
                recovery_X,
                recovery_y,
                recovery_priorities,
                recovery_X_new,
                recovery_y_new,
                recovery_budget,
                recovery_key,
            )
            X, y = combine_dataset_parts(
                (expert_X, expert_y),
                (dagger_X, dagger_y),
                (recovery_X, recovery_y),
            )

            block_result = fit_and_validate_blocks(
                X,
                y,
                cgp_structure,
                eval_env,
                q_value_estimator,
                bootstrap_repertoire=repertoire,
                n_pop=args.population_size,
                n_gens=args.sr_generations,
                validation_interval=args.validation_interval,
                validation_checkpoints=args.validation_checkpoints,
                validation_top_k=args.validation_top_k,
                dataset_batch_size=args.dataset_batch_size,
                alpha=args.gm_alpha,
                epsilon=args.gm_epsilon,
                search_seed=(search_seed_base + iteration) * 100,
                evaluation_seed=SEED * 10_000 + 9_000,
                evaluation_trajectories=args.evaluation_trajectories,
                rollout_steps=args.rollout_steps,
            )
            repertoire = block_result["repertoire"]
            loss = block_result["loss"]
            fit_diagnostics = block_result["diagnostics"]
            validated = block_result["validated"]
            best_genotype = validated["genotype"]
            evaluation_result = validated["reward"]
            if evaluation_result > best_validation_reward:
                best_validation_reward = evaluation_result
                best_iteration = iteration
                best_repertoire = block_result["validated_repertoire"]
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
                    len(recovery_X),
                    fit_diagnostics["finite_fitness_fraction"],
                    fit_diagnostics["best_performance_gap"],
                    fit_diagnostics["best_fidelity_gap"],
                    collection_diagnostics["invalid_transitions"],
                    validated["fitness_rank"],
                    args.validation_top_k,
                    behavior_policy_iteration,
                    behavior_policy_reward,
                    expert_weight,
                ])
            save_iteration(
                seed_dir,
                iteration,
                best_genotype,
                expert_X,
                expert_y,
                dagger_X,
                dagger_y,
                recovery_X,
                recovery_y,
                {
                    "iteration": iteration,
                    "validation_reward": evaluation_result,
                    "selected_fitness_rank": validated["fitness_rank"],
                    "candidate_indices": validated["candidate_indices"],
                    "candidate_rewards": validated["candidate_rewards"],
                    "best_validation_generation": block_result[
                        "best_generation"
                    ],
                    "block_validations": block_result["block_validations"],
                    "expert_weight": expert_weight,
                    "realized_expert_contribution": expert_weight,
                    "aggregation_policy": args.aggregation_policy,
                    "behavior_policy_iteration": behavior_policy_iteration,
                    "behavior_policy_reward": behavior_policy_reward,
                    "new_recovery_samples": int(jnp.sum(recovery_mask)),
                    "new_ordinary_samples": int(jnp.sum(~recovery_mask)),
                },
            )
            if evaluation_result >= args.target_reward:
                print("ENV SOLVED")
                break

        with open(seed_dir / "best_repertoire.pickle", "wb") as file:
            pickle.dump(best_repertoire, file)
        with open(seed_dir / "best_genotype.pickle", "wb") as file:
            pickle.dump(best_validation_genotype, file)
        seed_summary = {
            "seed": SEED,
            "best_iteration": best_iteration,
            "best_validation_reward": best_validation_reward,
            "last_iteration": iteration,
            "solved": best_validation_reward >= args.target_reward,
            "expert_samples": len(expert_X),
            "dagger_samples": len(dagger_X),
            "recovery_samples": len(recovery_X),
        }
        with open(seed_dir / "summary.json", "w") as f:
            json.dump(seed_summary, f, indent=2)
        seed_summaries.append(seed_summary)
        with open(run_dir / "aggregate_summary.json", "w") as f:
            json.dump(
                {
                    "completed_seeds": len(seed_summaries),
                    "requested_seeds": args.num_seeds,
                    "solved_seeds": sum(
                        summary["solved"] for summary in seed_summaries
                    ),
                    "seeds": seed_summaries,
                },
                f,
                indent=2,
            )
        if (
            args.stop_after_first_solved
            and best_validation_reward >= args.target_reward
        ):
            print(f"Stopping after solved seed {SEED}")
            break
