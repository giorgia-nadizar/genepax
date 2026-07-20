# import os
#
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
import csv
import pickle
import numpy as np

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

from brax import envs

from qdax.utils.metrics import default_ga_metrics

from genepax.gp.cartesian_genetic_programming import CGP
from mixedpolicysearch.intervalled_policy_scoring import intervalled_policy_scoring_fn
from mixedpolicysearch.mixed_policy_scoring import mixed_policy_scoring_fn


def policy_search():
    env_name = "inverted_double_pendulum"
    seed = 0
    batch_size = 50
    n_evaluation_seeds = 5
    n_iterations = 1_500
    sr_generations = 100
    bootstrapping = True

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
    env = envs.create(
        env_name=env_name,
        backend="generalized",
    )

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
    if bootstrapping:
        repertoire, loss = fit_dataset(data["X"], data["y"], cgp_structure,
                                       # bootstrap_repertoire=repertoire,
                                       q_value_estimator=q_value_estimator,
                                       n_gens=sr_generations, seed=seed, n_pop=batch_size)
        print(f"Bootstrapping finished, with loss {loss}")
        init_population = repertoire.genotypes
    else:
        key, subkey = jax.random.split(key)
        init_keys = jax.random.split(subkey, batch_size)
        init_population = jax.vmap(cgp_structure.init)(init_keys)

    filename = f"{env_name}_{'b' if bootstrapping else 'nb'}_n.csv"
    arr = np.concatenate((["param"], [f"i_{i}" for i in range(batch_size)]))
    print(arr)
    with open(filename, "a") as f:
        f.write(",".join(arr) + "\n")
    for int_beta in range(2, 11):
        trial_scoring_fn = functools.partial(
            intervalled_policy_scoring_fn,
            cgp_structure=cgp_structure,
            actor=actor,
            actor_params=actor_params,
            env=env,
            n_reps=n_evaluation_seeds,
            n=int_beta,
        )
        eval_key, key = jax.random.split(key)
        rewards, extra_info = trial_scoring_fn(init_population, key)
        arr = np.array(jnp.ravel(rewards))
        arr = np.concatenate(([1 / int_beta], arr))
        print(arr)
        with open(filename, "a") as f:
            np.savetxt(f, arr.reshape(1, -1), delimiter=",",)

        print(int_beta)
        # print(rewards)
    arr = np.array(jnp.ravel(extra_info["test_accuracy"]))
    arr = np.concatenate(([0], arr))
    with open(filename, "a") as f:
        np.savetxt(f, arr.reshape(1, -1), delimiter=",")



    filename = f"{env_name}_{'b' if bootstrapping else 'nb'}_beta.csv"
    arr = np.concatenate((["param"], [f"i_{i}" for i in range(batch_size)]))
    with open(filename, "a") as f:
        f.write(",".join(arr) + "\n")
    for int_beta in range(11):
        beta = (10 - int_beta) / 10
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
        rewards, extra_info = trial_scoring_fn(init_population, key)
        arr = np.array(jnp.ravel(rewards))
        arr = np.concatenate(([beta], arr))
        with open(filename, "a") as f:
            np.savetxt(f, arr.reshape(1, -1), delimiter=",")

        print(beta)
        # print(rewards)
    arr = np.array(jnp.ravel(extra_info["test_accuracy"]))
    arr = np.concatenate(([0], arr))
    with open(filename, "a") as f:
        np.savetxt(f, arr.reshape(1, -1), delimiter=",")
    exit(5)
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
