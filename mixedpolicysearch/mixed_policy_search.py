# import os
#
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
import csv
import pickle

from qdax.core.containers import GARepertoire

from distillation.fit_dataset import fit_dataset
from distillation.networks.jax_actor import JaxActor
from distillation.networks.jax_critic import JaxCritic
from distillation.networks.jax_q_value import JaxQValueEstimator
from genepax.evolution.custom_emitters import CustomMixingEmitter
from genepax.evolution.evolution_metrics import custom_ga_metrics
from genepax.evolution.genetic_algorithm_extra_scores import GeneticAlgorithmWithExtraScores
from genepax.evolution.tournament_selector import TournamentSelector

import functools
import time

import jax
import jax.numpy as jnp

from brax import envs

from genepax.gp.cartesian_genetic_programming import CGP
from mixedpolicysearch.mixed_policy_scoring import mixed_policy_scoring_fn, symbolic_policy_scoring_fn


def policy_search():
    env_name = "inverted_double_pendulum"
    seed = 0
    batch_size = 100
    n_evaluation_seeds = 5
    sr_generations = 100
    bootstrapping = True
    beta_init_value = .2
    n_iterations = 500

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

    beta_schedule_idx = 0
    scoring_function = functools.partial(
        mixed_policy_scoring_fn,
        cgp_structure=cgp_structure,
        actor=actor,
        actor_params=actor_params,
        env=env,
        n_reps=n_evaluation_seeds,
        beta=beta_init_value,
    )
    eval_key, key = jax.random.split(key)

    metrics_function = functools.partial(
        custom_ga_metrics, extra_scores_metrics={"test_accuracy": jnp.ravel}
    )
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

    ga = GeneticAlgorithmWithExtraScores(
        scoring_function=scoring_function,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
    )

    filename = f"results/mixed_policy_beta_{env_name}_{seed}.csv"
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["beta", "iteration", "max_fitness", "max_evaluation"])

    key, subkey = jax.random.split(key)
    repertoire, emitter_state, metrics = ga.init(
        genotypes=init_population, population_size=batch_size, key=subkey
    )
    print(metrics)
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([beta_init_value, 0, metrics["max_fitness"][0], metrics["test_accuracy"][0]])

    beta = beta_init_value
    for iteration in range(1, n_iterations):
        start_time = time.time()

        # if the easier task is solved, decay beta
        if metrics["max_fitness"][0] > 9350:
            beta = beta / 2
            print(f"new beta: {beta}")
            scoring_function = functools.partial(
                mixed_policy_scoring_fn,
                cgp_structure=cgp_structure,
                actor=actor,
                actor_params=actor_params,
                env=env,
                n_reps=n_evaluation_seeds,
                beta=beta,
            )
            ga = GeneticAlgorithmWithExtraScores(
                scoring_function=scoring_function,
                emitter=mixing_emitter,
                metrics_function=metrics_function,
            )
            repertoire, emitter_state, metrics = ga.init(
                genotypes=repertoire.genotypes, population_size=batch_size, key=subkey
            )
        else:
            repertoire, emitter_state, metrics = ga.update(
                repertoire=repertoire,
                emitter_state=emitter_state,
                key=subkey,
            )
        timelapse = time.time() - start_time
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [beta, iteration, metrics["max_fitness"][0], metrics["test_accuracy"][0]])
        print(iteration, timelapse, metrics)
        if metrics["test_accuracy"][0] > 9350:
            break

    # change scoring fn to only assess the policy
    scoring_function = functools.partial(
        symbolic_policy_scoring_fn,
        cgp_structure=cgp_structure,
        env=env,
        n_reps=n_evaluation_seeds,
    )
    ga = GeneticAlgorithmWithExtraScores(
        scoring_function=scoring_function,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
    )
    repertoire, emitter_state, metrics = ga.init(
        genotypes=repertoire.genotypes, population_size=batch_size, key=subkey
    )

    for iteration in range(1, n_iterations):
        start_time = time.time()

        repertoire, emitter_state, metrics = ga.update(
            repertoire=repertoire,
            emitter_state=emitter_state,
            key=subkey,
        )
        timelapse = time.time() - start_time
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [beta, iteration, metrics["max_fitness"][0], metrics["test_accuracy"][0]])
        print(iteration, timelapse, metrics)
        if metrics["test_accuracy"][0] > 9350:
            break

    repertoire_to_store = GARepertoire.init(
        genotypes=repertoire.genotypes,
        fitnesses=repertoire.fitnesses,
        population_size=len(repertoire.fitnesses),
    )
    with open(filename.replace("csv", "pickle"), "wb") as file:
        pickle.dump(repertoire_to_store, file)


if __name__ == '__main__':
    policy_search()
