import functools

import jax
import jax.numpy as jnp
import pytest
import qdax.tasks.brax as environments
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.tasks.brax.env_creators import scoring_function_brax_envs as scoring_function
from qdax.utils.metrics import default_qd_metrics

from genepax.gp.cartesian_genetic_programming import CGP
from genepax.gp.ensemble_genetic_programming import EnsembleGP


def test_ensemble_with_me() -> None:
    """Test that ensemble GP can be used with ME and is jit safe."""

    batch_size = 10
    env_name = "walker2d_uni"
    episode_length = 100
    num_iterations = 20
    seed = 42
    num_init_cvt_samples = 5_000
    num_centroids = 1024
    min_descriptor = 0.0
    max_descriptor = 1.0

    # Init environment
    env = environments.create(env_name, episode_length=episode_length)
    reset_fn = jax.jit(env.reset)

    # Init a random key
    key = jax.random.key(seed)

    # Init the CGP policy graph with default values
    inner_cgp = CGP(
        n_inputs=env.observation_size,
        n_outputs=1,
    )
    policy_graph = EnsembleGP(n_outputs=env.action_size, base_gp_model=inner_cgp)

    # Init the population of CGP genomes
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=batch_size)
    init_cgp_genomes = jax.jit(jax.vmap(policy_graph.init))(keys)

    # Define the play step fn for CGP to interact with the env
    def gp_play_step_fn(
        env_state,
        policy_params,
        key,
    ):
        """
        Play an environment step and return the updated state and the transition.
        """

        actions = policy_graph.apply(policy_params, env_state.obs)

        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
        )

        return next_state, policy_params, key, transition

    # Prepare the scoring function
    descriptor_extraction_fn = environments.descriptor_extractor[env_name]
    scoring_fn_cgp = functools.partial(
        scoring_function,
        episode_length=episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=gp_play_step_fn,
        descriptor_extractor=descriptor_extraction_fn,
    )

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[env_name]

    # Define a metrics function
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * episode_length,
    )

    # Define emitter
    def vmap_mutate(population_of_genotypes, mutation_key):
        n_pop = jax.tree.leaves(population_of_genotypes)[0].shape[0]
        multi_mutate_keys = jax.random.split(mutation_key, n_pop)
        return jax.jit(jax.vmap(policy_graph.mutate))(
            population_of_genotypes, multi_mutate_keys
        )

    gp_mutation_fn = functools.partial(
        vmap_mutate  # , mutation_probabilities={"inputs" : .2}
    )
    mixing_emitter = MixingEmitter(
        mutation_fn=gp_mutation_fn,
        variation_fn=None,
        variation_percentage=0.0,  # we define the ensemble with mutation only
        batch_size=batch_size,
    )

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn_cgp,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
    )

    # Compute the centroids
    key, subkey = jax.random.split(key)
    centroids = compute_cvt_centroids(
        num_descriptors=env.descriptor_length,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=min_descriptor,
        maxval=max_descriptor,
        key=subkey,
    )

    # Compute initial repertoire and emitter state
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = map_elites.init(
        init_cgp_genomes, centroids, subkey
    )

    # Check repertoire is not empty
    pytest.assume(jnp.any(repertoire.fitnesses > -jnp.inf))

    # Initial elements in repertoire
    n_initial_individuals = jnp.sum(repertoire.fitnesses > -jnp.inf)

    log_period = 3
    num_loops = num_iterations // log_period

    # Main loop
    map_elites_scan_update = map_elites.scan_update
    for _ in range(num_loops):
        (
            repertoire,
            emitter_state,
            key,
        ), current_metrics = jax.lax.scan(
            map_elites_scan_update,
            (repertoire, emitter_state, key),
            (),
            length=log_period,
        )

    # Initial elements in repertoire
    n_final_individuals = jnp.sum(repertoire.fitnesses > -jnp.inf)

    # Check coverage did not decrease
    pytest.assume(n_final_individuals >= n_initial_individuals)
