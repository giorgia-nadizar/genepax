import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

from qdax.baselines.genetic_algorithm import GeneticAlgorithm

from genepax.evolution.custom_emitters import CustomMixingEmitter
from genepax.evolution.tournament_selector import TournamentSelector

import functools
import time

import jax
import jax.numpy as jnp

import qdax.tasks.brax as environments
from qdax.core.neuroevolution.mdp_utils import generate_unroll
from qdax.tasks.brax.env_creators import get_mask_from_transitions
from qdax.core.neuroevolution.buffers.buffer import Transition

from qdax.utils.metrics import default_ga_metrics

from genepax.gp.cartesian_genetic_programming import CGP


def policy_search():
    env_name = "hopper"
    episode_length = 1_000
    seed = 0
    batch_size = 50
    n_evaluation_seeds = 5
    n_iterations = 1_500

    # Init environment
    env = environments.create(env_name, episode_length=episode_length)
    reset_fn = jax.jit(env.reset)

    # Init a random key
    key = jax.random.key(seed)

    # Init policy network
    cgp_structure = CGP(
        n_inputs=env.observation_size,
        n_outputs=env.action_size,
        n_nodes=50,
        fixed_outputs=True
    )

    # Init population of controllers
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=batch_size)
    init_population = jax.vmap(cgp_structure.init)(keys)

    def play_step_fn(
            env_state,
            policy_params,
            key,
    ):
        actions = cgp_structure.apply(policy_params, env_state.obs)
        next_state = env.step(env_state, actions)

        transition = Transition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            truncations=next_state.info["truncation"],
            actions=actions
        )

        return next_state, policy_params, key, transition

    unroll_fn = functools.partial(
        generate_unroll,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
    )

    def single_seed_scoring_function(
            policies_params,
            key,
    ):
        # Reset environments
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, jax.tree.leaves(policies_params)[0].shape[0])
        init_states = jax.vmap(reset_fn)(keys)

        # Step environments
        keys = jax.random.split(key, jax.tree.leaves(policies_params)[0].shape[0])
        _, data = jax.vmap(unroll_fn)(init_states, policies_params, keys)

        # Create a mask to extract data properly
        mask = get_mask_from_transitions(data)

        # Evaluate
        fitnesses = jnp.sum(data.rewards * (1.0 - mask), axis=1)
        fitnesses = jnp.nan_to_num(fitnesses, nan=-jnp.inf)

        return jnp.expand_dims(fitnesses, axis=1), {"transitions": data}

    def scoring_function(
            policies_params,
            key,
    ):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, n_evaluation_seeds)
        multi_fitnesses, multi_data = jax.vmap(single_seed_scoring_function, in_axes=(None, 0))(policies_params, keys)
        averaged_fitnesses = jnp.mean(multi_fitnesses, axis=0)
        return averaged_fitnesses, multi_data

    metrics_function = functools.partial(default_ga_metrics)
    cgp_mutation = functools.partial(cgp_structure.mutate, p_mut_inputs=.2, p_mut_functions=.2)

    mutation_fn = jax.jit(jax.vmap(cgp_mutation, in_axes=(0, 0)))
    tournament_selector = TournamentSelector()
    mixing_emitter = CustomMixingEmitter(
        mutation_fn=mutation_fn,
        variation_fn=None,
        variation_percentage=0,
        batch_size=batch_size,
        selector=tournament_selector,
    )

    ga = GeneticAlgorithm(
        scoring_function=scoring_function,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
    )

    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = ga.init(
        genotypes=init_population, population_size=batch_size, key=subkey
    )

    for iteration in range(n_iterations):
        start_time = time.time()

        repertoire, emitter_state, current_metrics = ga.update(
            repertoire=repertoire,
            emitter_state=emitter_state,
            key=subkey,
        )
        timelapse = time.time() - start_time
        print(iteration, timelapse, current_metrics)


if __name__ == '__main__':
    policy_search()
