import csv
import time
from functools import partial
from typing import Tuple, Any

import jax
import qdax.tasks.brax as environments
from brax import State
from jax import vmap, jit, random, lax
import jax.numpy as jnp
from qdax.core.containers import GARepertoire

from genepax.evolution.elite_selector import EliteSelector
from genepax.evolution.tournament_selector import TournamentSelector
from genepax.gp.cartesian_genetic_programming import CGP
import argparse


def single_genome_single_seed_scoring_fn(
        genome,
        rnd_key,
        env,
        cgp_structure: CGP,
):
    initial_env_state = jit(env.reset)(rnd_key)

    def _rollout_loop(carry: Tuple[State, jnp.ndarray, float, Tuple[int, int, int], int],
                      unused_arg: Any) -> Tuple[Tuple[State, jnp.ndarray, int], Any]:
        env_state, cum_reward, active_episode = carry
        inputs = env_state.obs
        actions = cgp_structure.apply(genome, inputs)
        new_state = jit(env.step)(env_state, actions)
        corrected_reward = new_state.reward * active_episode
        new_active_episode = (active_episode * (1 - new_state.done)).astype(int)
        new_carry = new_state, cum_reward + corrected_reward, new_active_episode
        return new_carry, corrected_reward

    (final_env_state, cum_reward, _), _ = lax.scan(
        f=_rollout_loop,
        init=(initial_env_state, initial_env_state.reward, 1),
        xs=(),
        length=episode_length,
    )
    return cum_reward


def single_genome_scoring_fn(
        genome,
        rnd_key,
        environment,
        cgp_structure,
        n_evals=5
):
    rnd_key, *subkeys = random.split(rnd_key, n_evals + 1)
    subkeys_array = jnp.array(subkeys)
    partial_single_eval = partial(
        single_genome_single_seed_scoring_fn, env=environment, cgp_structure=cgp_structure
    )
    vmap_evaluate_genome = vmap(partial_single_eval, in_axes=(None, 0))
    return vmap_evaluate_genome(genome, subkeys_array)


def scoring_fn_maker(
        environment,
        cgp_structure,
        n_evals=5
):
    single_scoring_fn = partial(single_genome_scoring_fn, environment=environment, cgp_structure=cgp_structure,
                                n_evals=n_evals)
    vmapped_scoring_fn = vmap(single_scoring_fn, in_axes=(0, 0))
    return jit(vmapped_scoring_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    seed = args.seed

    for env_name in [
        "inverted_double_pendulum",
        # "swimmer",
        "hopper",
        "walker2d",
        "halfcheetah",
        "ant"
    ]:
        filename = f"baselines/{env_name}_{seed}.csv"
        print(filename)
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "max_fitness", "evaluation_time"])

        if env_name in ["hopper", "walker2d"]:
            n_generations = 1500
        elif env_name == "ant":
            n_generations = 2500
        else:
            n_generations = 1000

        episode_length = 1000
        n_pop = 100
        elite_size = 10
        n_offsprings = n_pop - elite_size

        rnd_key = jax.random.PRNGKey(seed)
        env = environments.create(env_name, episode_length=episode_length, backend='generalized')
        cgp_structure = CGP(
            n_inputs=env.observation_size,
            n_outputs=env.action_size,
        )
        scoring_fn = scoring_fn_maker(env, cgp_structure)
        tournament_selector = TournamentSelector(3)
        parent_selection_fn = partial(tournament_selector.select, num_samples=n_offsprings)
        elite_selector = EliteSelector()
        elite_selection_fn = partial(elite_selector.select, num_samples=elite_size)
        pop_mutation_fn = jax.jit(jax.vmap(cgp_structure.mutate, in_axes=(0, 0)))
        rnd_key, init_key = random.split(rnd_key)
        init_keys = random.split(init_key, n_pop)
        genomes = vmap(cgp_structure.init)(init_keys)
        # rnd_key, *init_keys = random.split(rnd_key, n_pop + 1)
        # genomes = vmap(cgp_structure.init)(jnp.asarray(init_keys))

        for _generation in range(n_generations):
            start_eval = time.process_time()
            rnd_key, *eval_keys = random.split(rnd_key, n_pop + 1)
            evaluation_outcomes = scoring_fn(genomes, jnp.array(eval_keys))
            end_eval = time.process_time()
            evaluation_outcomes = jnp.nan_to_num(evaluation_outcomes, nan=-100000)
            # with jnp.printoptions(suppress=True, precision=2):
            #     print(evaluation_outcomes)
            fitness_values = jnp.mean(evaluation_outcomes, axis=1)
            evaluation_time = end_eval - start_eval
            max_fitness = jnp.max(fitness_values)

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

            # elite_sizes = jit(vmap(cgp_structure.size))(elite)
            # print(elite_sizes)

            # parents_size = jnp.mean(jit(vmap(cgp_structure.size))(parents))
            # offspring_size = jnp.mean(jit(vmap(cgp_structure.size))(offspring))
            # avg_pop_size = jnp.mean(jit(vmap(cgp_structure.size))(genomes))

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
                f"FITNESS: {max_fitness}"
            )
            with open(filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([_generation, max_fitness, evaluation_time])

            # print(
            #     f"{_generation} \t"
            #     f"E: {evaluation_time:.2f} \t"
            #     f"S: {selection_time:.2f} \t"
            #     f"M: {mutation_time:.2f} \t"
            #     f"SIZE: {avg_pop_size} \t"
            #     f"FITNESS: {max_fitness}"
            # )
