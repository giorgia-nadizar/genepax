import csv
import pickle

import jax
import jax.numpy as jnp

import functools
import time

from qdax.core.containers import GARepertoire

from genepax.evolution.custom_emitters import CustomMixingEmitter
from genepax.evolution.evolution_metrics import custom_ga_metrics
from genepax.evolution.genetic_algorithm_extra_scores import GeneticAlgorithmWithExtraScores
from genepax.evolution.tournament_selector import TournamentSelector
from genepax.gp.cartesian_genetic_programming import CGP
from imitationlearning.dataset import load_d4rl_dataset
from scoring import prepare_bc_scoring_fn

for seed in range(2, 10):

    config = {
        "n_pop": 100,
        "tournament_size": 3,
        "n_gens": 1000
    }

    dataset_id = "mujoco/invertedpendulum/expert-v0"
    obs, actions = load_d4rl_dataset(dataset_id)

    obs = jnp.array(obs)
    actions = jnp.array(actions)

    obs_mean = jnp.mean(obs, axis=0)
    obs_std = jnp.std(obs, axis=0) + 1e-8

    obs = (obs - obs_mean) / obs_std

    cgp_structure = CGP(
        n_inputs=obs.shape[1],
        n_outputs=1,
        n_nodes=50,
        outputs_wrapper=lambda x: jnp.tanh(x),
    )

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)

    init_keys = jax.random.split(subkey, config["n_pop"])
    init_population = jax.jit(jax.vmap(cgp_structure.init))(init_keys)

    mutation_fn = jax.jit(
        jax.vmap(cgp_structure.mutate, in_axes=(0, 0))
    )

    selector = TournamentSelector(
        tournament_size=config["tournament_size"]
    )

    emitter = CustomMixingEmitter(
        mutation_fn=mutation_fn,
        variation_fn=None,
        variation_percentage=0,
        batch_size=config["n_pop"],
        selector=selector,
    )

    scoring_fn = prepare_bc_scoring_fn(
        obs,
        actions,
        cgp_structure,
    )

    metrics_function = functools.partial(
        custom_ga_metrics, extra_scores_metrics={"test_accuracy": jnp.ravel}
    )

    ga = GeneticAlgorithmWithExtraScores(
        scoring_function=scoring_fn,
        emitter=emitter,
        metrics_function=metrics_function
    )

    key, subkey = jax.random.split(key)
    repertoire, emitter_state, metrics = ga.init(init_population, config["n_pop"], subkey)

    csv_path = f"training_log_pearson_{seed}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "generation",
            "best_fitness",
            "elapsed",
        ])

    for generation in range(config["n_gens"]):
        start = time.time()
        key, subkey = jax.random.split(key)

        repertoire, emitter_state, metrics = ga.update(
            repertoire=repertoire,
            emitter_state=emitter_state,
            key=subkey,
        )

        elapsed = time.time() - start

        print(
            f"Gen {generation:03d} | "
            f"max fitness: {metrics['max_fitness']} | "
            f"time: {elapsed:.3f}s"
        )

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                f"{generation:03d}",
                metrics['max_fitness'],
                f"{elapsed:.3f}",
            ])

    repertoire_to_store = GARepertoire.init(
        genotypes=repertoire.genotypes,
        fitnesses=repertoire.fitnesses,
        population_size=len(repertoire.fitnesses),
    )
    path = f"repertoire_pearson_{seed}.pickle"
    with open(path, "wb") as file:
        pickle.dump(repertoire_to_store, file)
