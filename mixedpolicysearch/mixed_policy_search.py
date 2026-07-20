# import os
#
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
import pickle

from mpmath import beta
from qdax.baselines.genetic_algorithm import GeneticAlgorithm

from distillation.fit_dataset import fit_dataset
from distillation.networks.jax_actor import JaxActor
from distillation.networks.jax_critic import JaxCritic
from distillation.networks.jax_q_value import JaxQValueEstimator
from genepax.evolution.custom_emitters import CustomMixingEmitter
from genepax.evolution.tournament_selector import TournamentSelector

import functools
import time

import jax
import jax.numpy as jnp

import qdax.tasks.brax as environments
from qdax.core.neuroevolution.mdp_utils import generate_unroll, init_population_controllers
from qdax.tasks.brax.env_creators import get_mask_from_transitions
from qdax.core.neuroevolution.buffers.buffer import Transition

from qdax.utils.metrics import default_ga_metrics

from genepax.gp.cartesian_genetic_programming import CGP


def mixed_policy_scoring_fn(
        genotypes,
        key,
        cgp_structure,
        actor,
        actor_params,
        env,
        beta: float = 0.5,
        num_steps: int = 1000,
        n_reps: int = 5
):
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, jax.tree.leaves(genotypes)[0].shape[0])
    scoring_fn = functools.partial(
        single_mixed_policy_multi_seed_scoring_fn,
        cgp_structure=cgp_structure,
        actor=actor,
        actor_params=actor_params,
        env=env,
        beta=beta,
        num_steps=num_steps,
        n_reps=n_reps
    )
    rewards = jax.vmap(scoring_fn, in_axes=(0, 0))(genotypes, keys)
    symbolic_scoring_fn = functools.partial(
        single_mixed_policy_multi_seed_scoring_fn,
        cgp_structure=cgp_structure,
        actor=actor,
        actor_params=actor_params,
        env=env,
        beta=0,
        num_steps=num_steps,
        n_reps=n_reps
    )
    symbolic_rewards = jax.vmap(symbolic_scoring_fn, in_axes=(0, 0))(genotypes, keys)

    return rewards, {"test_accuracy": symbolic_rewards}


def single_mixed_policy_multi_seed_scoring_fn(
        genotype,
        key,
        cgp_structure,
        actor,
        actor_params,
        env,
        beta: float = 0.5,
        num_steps: int = 1000,
        n_reps: int = 5
):
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, n_reps)
    scoring_fn = functools.partial(
        single_mixed_policy_scoring_fn,
        cgp_structure=cgp_structure,
        actor=actor,
        actor_params=actor_params,
        env=env,
        beta=beta,
        num_steps=num_steps,
    )
    rewards = jax.vmap(scoring_fn, in_axes=(None, 0))(genotype, keys)
    return jnp.asarray([jnp.mean(rewards)])


def single_mixed_policy_scoring_fn(
        genotype,
        key,
        cgp_structure,
        actor,
        actor_params,
        env,
        beta: float = 0.5,
        num_steps: int = 1000,
):
    state = env.reset(key)

    def step_fn(carry, _):
        state, key = carry
        key, action_key = jax.random.split(key)
        neural_action = actor.apply(
            actor_params,
            state.obs,
        )
        symbolic_action = cgp_structure.apply(
            genotype,
            state.obs,
        )
        mixed_action = (
                (1.0 - beta) * symbolic_action
                +
                beta * neural_action
        )
        next_state = env.step(
            state,
            mixed_action,
        )
        return (
            next_state,
            key,
        ), (
            state.obs,
            neural_action,
            next_state.reward,
            next_state.done,
        )

    (_, _), data = jax.lax.scan(
        step_fn,
        (state, key),
        None,
        length=num_steps,
    )
    X, y, rewards, dones = data

    alive = 1.0 - jnp.concatenate(
        [jnp.array([0]), jnp.cumsum(dones[:-1])]
    )
    alive = jnp.clip(alive, 0.0, 1.0)

    return jnp.sum(rewards * alive)


def policy_search():
    env_name = "inverted_double_pendulum"
    episode_length = 1000
    seed = 0
    batch_size = 50
    n_evaluation_seeds = 5
    n_iterations = 1_500
    sr_generations = 100

    neural_model_path = f"../distillation/sac_jax_{env_name}.pkl"
    # Load SAC actor and critic
    with open(neural_model_path, "rb") as f:
        sac_data = pickle.load(f)

    actor_data = sac_data["actor"]
    actor_architecture = actor_data["architecture"]
    actor_params = actor_data["params"]
    actor = JaxActor(
        hidden_sizes=tuple(
            actor_architecture["hidden_sizes"]
        ),
        action_dim=actor_architecture["action_dim"],
    )

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

    # Bootstrap population with imitation learning
    # we have already collected the initial dataset at expert_dataset.npz
    dataset_path = f"../distillation/expert_dataset_{env_name}.npz"
    data = jnp.load(dataset_path)
    critic_params = sac_data["critic"]
    critic_architecture = critic_params["architecture"]
    q1 = JaxCritic(
        hidden_sizes=tuple(
            critic_architecture["hidden_sizes"]
        )
    )
    q2 = JaxCritic(
        hidden_sizes=tuple(
            critic_architecture["hidden_sizes"]
        )
    )
    q_value_estimator = JaxQValueEstimator(q1, q2, critic_params["params"]["q1"], critic_params["params"]["q2"])
    repertoire, loss = fit_dataset(data["X"], data["y"], cgp_structure,
                                   # bootstrap_repertoire=repertoire,
                                   q_value_estimator=q_value_estimator,
                                   n_gens=sr_generations, seed=seed, n_pop=batch_size)
    print(f"Bootstrapping finished, with loss {loss}")
    init_population = repertoire.genotypes

    for beta in [1, .9, .8, .7, .6, .5, .4, .3, .2, .1]:
        trial_scoring_fn = functools.partial(
            mixed_policy_scoring_fn,
            cgp_structure=cgp_structure,
            actor=actor,
            actor_params=actor_params,
            env=env,
            n_reps=n_evaluation_seeds,
            beta=beta,
        )
        eval_key, key = jax.random.split(key)
        first_result = trial_scoring_fn(init_population, key)
        print(beta)
        print(first_result)
    exit(5)

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
