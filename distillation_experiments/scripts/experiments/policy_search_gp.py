"""Search directly for a symbolic Brax policy with Cartesian GP."""

import argparse
import csv
from datetime import datetime, timezone
from functools import partial
import json
import pickle
from pathlib import Path
import time

import jax
import jax.numpy as jnp
from jax import random, vmap
from qdax.core.containers import GARepertoire
import qdax.tasks.brax as environments

from distillation.policy_search.search_scoring import evaluate_genome, scoring_fn_maker
from genepax.evolution.elite_selector import EliteSelector
from genepax.evolution.tournament_selector import TournamentSelector
from genepax.gp.cartesian_genetic_programming import CGP


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def create_run_directory(output_root, env_name, run_name=None):
    if run_name is None:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / env_name / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="inverted_pendulum")
    parser.add_argument("--backend", default="generalized")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=positive_int, default=1000)
    parser.add_argument("--episode-length", type=positive_int, default=1000)
    parser.add_argument("--evaluations", type=positive_int, default=5)
    parser.add_argument("--population-size", type=positive_int, default=100)
    parser.add_argument("--elite-size", type=positive_int, default=10)
    parser.add_argument("--tournament-size", type=positive_int, default=3)
    parser.add_argument("--target-reward", type=float)
    parser.add_argument("--validation-trajectories", type=positive_int, default=50)
    parser.add_argument("--validation-seed", type=int, default=1_000_000)
    parser.add_argument("--validation-mean-reward", type=float)
    parser.add_argument("--validation-min-reward", type=float)
    parser.add_argument("--run-name")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--fixed-budget", action="store_true", help="Run every generation; select by search fitness without validation-based stopping")
    parser.add_argument("--unwrapped-env", action="store_true", help="Use the same raw Brax environment as distillation rollouts")
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.require_gpu and jax.default_backend() != 'gpu':
        raise RuntimeError('GPU required; refusing CPU fallback')
    if args.elite_size >= args.population_size:
        raise ValueError("elite-size must be smaller than population-size")
    target_reward = (
        float(args.episode_length)
        if args.target_reward is None
        else args.target_reward
    )
    validation_mean_reward = (
        target_reward if args.validation_mean_reward is None
        else args.validation_mean_reward
    )
    validation_min_reward = (
        validation_mean_reward if args.validation_min_reward is None
        else args.validation_min_reward
    )

    experiments_dir = Path(__file__).resolve().parents[2]
    output_root = args.output_root or experiments_dir / "artifacts" / "policy_search" / "baselines"
    run_dir = create_run_directory(output_root, args.env, args.run_name)
    config = vars(args).copy()
    config["output_root"] = str(output_root)
    config["target_reward"] = target_reward
    config["validation_mean_reward"] = validation_mean_reward
    config["validation_min_reward"] = validation_min_reward
    config["devices"] = [str(device) for device in jax.devices()]
    config["n_nodes"] = 50
    config["selection"] = 'best observed search fitness' if args.fixed_budget else 'best validated candidate when available'
    with (run_dir / "config.json").open("w") as file:
        json.dump(config, file, indent=2, sort_keys=True)

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as file:
        csv.writer(file).writerow(
            ["generation", "max_fitness", "mean_fitness", "evaluation_time"]
        )

    n_offsprings = args.population_size - args.elite_size
    key = random.PRNGKey(args.seed)
    if args.unwrapped_env:
        from brax import envs
        environment = envs.get_environment(args.env, backend=args.backend)
    else:
        environment = environments.create(
            args.env, episode_length=args.episode_length, backend=args.backend,
        )
    cgp = CGP(
        n_inputs=environment.observation_size,
        n_outputs=environment.action_size,
    )
    # The scorer sanitizes every symbolic action before environment execution;
    # the shared rollout applies the same boundary a second time.
    scoring_fn = scoring_fn_maker(
        environment,
        cgp,
        n_evals=args.evaluations,
        episode_length=args.episode_length,
    )

    tournament_selector = TournamentSelector(args.tournament_size)
    parent_selection = partial(
        tournament_selector.select, num_samples=n_offsprings
    )
    elite_selection = partial(
        EliteSelector().select, num_samples=args.elite_size
    )
    mutate_population = jax.jit(jax.vmap(cgp.mutate, in_axes=(0, 0)))

    key, init_key = random.split(key)
    genomes = vmap(cgp.init)(random.split(init_key, args.population_size))
    best_fitness = -jnp.inf
    best_generation = None
    best_individual = None
    best_validated_individual = None
    final_repertoire = None
    solved = False
    best_validation_mean = -float("inf")
    best_validation_min = -float("inf")
    best_validation_generation = None
    validation_history = []

    validation_fn = jax.jit(partial(
        evaluate_genome,
        environment=environment,
        cgp_structure=cgp,
        n_evals=args.validation_trajectories,
        episode_length=args.episode_length,
    ))

    print(f"Writing policy-search results to {run_dir}")
    for generation in range(args.generations):
        started = time.perf_counter()
        key, eval_key = random.split(key)
        eval_keys = random.split(eval_key, args.population_size)
        outcomes = scoring_fn(genomes, eval_keys)
        outcomes = jnp.nan_to_num(
            outcomes, nan=-jnp.inf, posinf=-jnp.inf, neginf=-jnp.inf
        )
        outcomes.block_until_ready()
        evaluation_time = time.perf_counter() - started
        fitnesses = jnp.mean(outcomes, axis=1)
        max_fitness = float(jnp.max(fitnesses))
        mean_fitness = float(jnp.mean(fitnesses))
        final_repertoire = GARepertoire(
            genomes,
            jnp.expand_dims(fitnesses, axis=1),
            None,
            None,
        )

        generation_best_idx = int(jnp.argmax(fitnesses))
        if max_fitness > float(best_fitness):
            best_fitness = max_fitness
            best_generation = generation
            best_individual = jax.tree.map(
                lambda value: value[generation_best_idx], genomes
            )
            with (run_dir / "best_training_individual.pickle").open("wb") as file:
                pickle.dump(best_individual, file)

        with metrics_path.open("a", newline="") as file:
            csv.writer(file).writerow(
                [generation, max_fitness, mean_fitness, evaluation_time]
            )
        print(
            f"{generation} FITNESS: {max_fitness:.3f} "
            f"MEAN: {mean_fitness:.3f} "
            f"EVALUATION TIME: {evaluation_time:.3f}"
        )

        # Training fitness is deliberately noisy.  A candidate can stop the
        # search only after evaluation on a fixed, disjoint bank of seeds.
        if not args.fixed_budget and max_fitness >= target_reward:
            validation_key = random.PRNGKey(args.validation_seed)
            validation_returns = validation_fn(
                jax.tree.map(lambda value: value[generation_best_idx], genomes),
                validation_key,
            )
            validation_returns.block_until_ready()
            validation_returns = jnp.nan_to_num(
                validation_returns, nan=-jnp.inf, posinf=-jnp.inf,
                neginf=-jnp.inf,
            )
            validation_mean = float(jnp.mean(validation_returns))
            validation_min = float(jnp.min(validation_returns))
            validation_median = float(jnp.median(validation_returns))
            validation_max = float(jnp.max(validation_returns))
            validation_record = {
                "generation": generation,
                "training_fitness": max_fitness,
                "mean_return": validation_mean,
                "median_return": validation_median,
                "min_return": validation_min,
                "max_return": validation_max,
            }
            validation_history.append(validation_record)
            (run_dir / "validation_history.json").write_text(
                json.dumps(validation_history, indent=2)
            )
            print(
                f"HELDOUT: mean={validation_mean:.3f} "
                f"median={validation_median:.3f} min={validation_min:.3f} "
                f"max={validation_max:.3f}"
            )
            if validation_mean > best_validation_mean:
                best_validation_mean = validation_mean
                best_validation_min = validation_min
                best_validation_generation = generation
                best_validated_individual = jax.tree.map(
                    lambda value: value[generation_best_idx], genomes
                )
                with (run_dir / "best_individual.pickle").open("wb") as file:
                    pickle.dump(best_validated_individual, file)
                (run_dir / "heldout_evaluation.json").write_text(json.dumps(
                    validation_record | {
                        "evaluation_seed": args.validation_seed,
                        "trajectories": args.validation_trajectories,
                    }, indent=2
                ))
            solved = (
                validation_mean >= validation_mean_reward
                and validation_min >= validation_min_reward
            )
        if solved or generation == args.generations - 1:
            break

        key, tournament_key, elite_key = random.split(key, 3)
        parents = parent_selection(final_repertoire, tournament_key).genotypes
        elite = elite_selection(final_repertoire, elite_key).genotypes
        key, mutation_key = random.split(key)
        offspring = mutate_population(
            parents, random.split(mutation_key, n_offsprings)
        )
        genomes = jax.tree.map(
            lambda elite_values, offspring_values: jnp.concatenate(
                (elite_values, offspring_values), axis=0
            ),
            elite,
            offspring,
        )

    with (run_dir / "final_population.pickle").open("wb") as file:
        pickle.dump(final_repertoire, file)
    with (run_dir / "final_individual.pickle").open("wb") as file:
        pickle.dump(jax.tree.map(lambda value: value[generation_best_idx], genomes), file)
    saved_individual = (
        best_validated_individual
        if best_validated_individual is not None else best_individual
    )
    with (run_dir / "best_individual.pickle").open("wb") as file:
        pickle.dump(saved_individual, file)
    with (run_dir / "summary.json").open("w") as file:
        json.dump(
            {
                "best_fitness": float(best_fitness),
                "best_generation": best_generation,
                "final_generation": generation,
                "solved": None if args.fixed_budget else solved,
                "target_reward": target_reward,
                "best_validation_mean": best_validation_mean if validation_history else None,
                "best_validation_min": best_validation_min if validation_history else None,
                "best_validation_generation": best_validation_generation,
                "validation_mean_reward": validation_mean_reward,
                "validation_min_reward": validation_min_reward,
                "validation_trajectories": args.validation_trajectories,
            },
            file,
            indent=2,
        )


if __name__ == "__main__":
    main()
