import csv
import pickle
import time
from functools import partial
from pathlib import Path
import jax
import qdax.tasks.brax as environments
from jax import vmap, random
import jax.numpy as jnp
from qdax.core.containers import GARepertoire

from distillation.networks.sac_utils import load_sac_actor
from genepax.evolution.elite_selector import EliteSelector
from genepax.evolution.tournament_selector import TournamentSelector
from genepax.gp.cartesian_genetic_programming import CGP
from distillation.policy_search.search_scoring import (
    mixed_scoring_fn_maker,
    scoring_fn_maker,
)
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    seed = args.seed

    for env_name in [
        # "swimmer",
        "hopper",
        "walker2d",
        "inverted_double_pendulum",
        # "halfcheetah",
        # "ant"
    ]:
        experiments_dir = Path(__file__).resolve().parents[1]
        output_dir = experiments_dir / "policy_search" / "mixed_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"{env_name}_{seed}.csv"
        print(filename)
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["iteration", "best_fitness", "best_cumulative_reward",
                 "cumulative_reward_of_best", "beta", "evaluation_time"])

        if env_name in ["walker2d", "hopper"]:
            n_generations = 1500
        elif env_name == "ant":
            n_generations = 2500
        else:
            n_generations = 1000

        # n_generations = n_generations / 2
        episode_length = 1000
        n_pop = 100
        elite_size = 10
        n_offsprings = n_pop - elite_size

        if env_name in ["hopper", "walker2d"]:
            beta = .6
        else:
            beta = .8

        checkpoint_path = experiments_dir / "expert_models" / env_name / "final"
        actor_policy_fn, _ = load_sac_actor(checkpoint_path)

        rnd_key = jax.random.PRNGKey(seed)
        env = environments.create(env_name, episode_length=episode_length, backend='generalized')
        cgp_structure = CGP(
            n_inputs=env.observation_size,
            n_outputs=env.action_size,
        )
        scoring_fn = scoring_fn_maker(
            env, cgp_structure, episode_length=episode_length
        )
        mixed_scoring_fn = mixed_scoring_fn_maker(
            env, cgp_structure, actor_policy_fn,
            episode_length=episode_length,
        )
        tournament_selector = TournamentSelector(3)
        parent_selection_fn = partial(tournament_selector.select, num_samples=n_offsprings)
        elite_selector = EliteSelector()
        elite_selection_fn = partial(elite_selector.select, num_samples=elite_size)
        pop_mutation_fn = jax.jit(jax.vmap(cgp_structure.mutate, in_axes=(0, 0)))
        rnd_key, init_key = random.split(rnd_key)
        init_keys = random.split(init_key, n_pop)
        genomes = vmap(cgp_structure.init)(init_keys)

        n_steps = ((1.1 - beta) * 10)
        step_size = int(n_generations / n_steps)

        for _generation in range(n_generations):
            if (_generation + 1) % step_size == 0:
                beta = beta + 0.1
                beta = min(beta, 1)

            start_eval = time.process_time()
            rnd_key, *eval_keys = random.split(rnd_key, n_pop + 1)
            evaluation_outcomes = scoring_fn(genomes, jnp.array(eval_keys))
            end_eval = time.process_time()
            evaluation_outcomes = jnp.nan_to_num(evaluation_outcomes, nan=-100000)
            cumulative_rewards = jnp.mean(evaluation_outcomes, axis=1)
            evaluation_time = end_eval - start_eval

            if beta < .99:
                mixed_start_eval = time.process_time()
                rnd_key, *eval_keys = random.split(rnd_key, n_pop + 1)
                mixed_evaluation_outcomes = mixed_scoring_fn(genomes, jnp.array(eval_keys), beta)
                mixed_end_eval = time.process_time()
                mixed_evaluation_outcomes = jnp.nan_to_num(mixed_evaluation_outcomes, nan=-100000)
                fitness_values = jnp.mean(mixed_evaluation_outcomes, axis=1)
                mixed_evaluation_time = mixed_end_eval - mixed_start_eval
            else:
                fitness_values = cumulative_rewards

            best_idx = jnp.argmax(fitness_values)
            best_fitness = fitness_values[best_idx]
            cumulative_reward_of_best = cumulative_rewards[best_idx]
            best_cumulative_reward = jnp.max(cumulative_rewards)

            start_selection = time.process_time()
            rnd_key, tournament_key, elite_key = random.split(rnd_key, 3)
            repertoire = GARepertoire(
                genomes,
                jnp.expand_dims(fitness_values, axis=1),
                None,
                None
            )
            parents = parent_selection_fn(repertoire, tournament_key).genotypes

            elite = elite_selection_fn(repertoire, elite_key).genotypes

            end_selection = time.process_time()
            selection_time = end_selection - start_selection

            start_mutation = time.process_time()
            rnd_key, *mutation_keys = random.split(rnd_key, n_offsprings + 1)
            offspring = pop_mutation_fn(parents, jnp.array(mutation_keys))
            end_mutation = time.process_time()
            mutation_time = end_mutation - start_mutation

            old_genomes = genomes
            genomes = jax.tree.map(
                lambda x, y: jnp.concatenate((x, y), axis=0),
                elite,
                offspring,
            )

            print(
                f"{_generation} \t"
                # f"P: {parents_size:.2f} \t"
                # f"O: {offspring_size:.2f} \t"
                # f"G: {avg_pop_size:.2f} \t"
                f"REWARD OF BEST: {cumulative_reward_of_best} \t"
                f"BEST REWARD: {best_cumulative_reward} \t"
                f"BEST FITNESS: {best_fitness} \t"
                f"BETA: {beta} \t"
                f"EVALUATION TIME: {evaluation_time}"
            )
            with open(filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [_generation, best_fitness, best_cumulative_reward, cumulative_reward_of_best, beta,
                     evaluation_time])

        repertoire_to_store = GARepertoire.init(
            genotypes=old_genomes,
            fitnesses=jnp.expand_dims(fitness_values, axis=1),
            population_size=n_pop,
        )
        path = filename.with_suffix(".pickle")
        with open(path, "wb") as file:
            pickle.dump(repertoire_to_store, file)
