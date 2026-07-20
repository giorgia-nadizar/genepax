import functools
import pickle
import time
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.custom_types import Genotype, RNGKey

from genepax.evolution.custom_emitters import CustomMixingEmitter
from genepax.evolution.evolution_metrics import custom_ga_metrics
from genepax.evolution.genetic_algorithm_extra_scores import (
    GeneticAlgorithmWithExtraScores,
)
from genepax.evolution.tournament_selector import TournamentSelector
from genepax.gp.cartesian_genetic_programming import CGP


def single_genome_scoring_fn(
        genotype: Genotype,
        X: jnp.ndarray,
        y: jnp.ndarray,
        cgp_structure: CGP,
        q_value_estimator
) -> Tuple:
    # Construct features
    y_pred = jax.jit(
        jax.vmap(cgp_structure.apply, in_axes=(None, 0))
    )(genotype, X)

    # Sanitization
    y_pred = jnp.nan_to_num(
        y_pred,
        nan=0.0,
        posinf=1e3,
        neginf=-1e3,
    )
    # # MSE imitation loss
    mse = jnp.mean(
        jnp.square(
            y - y_pred
        )
    )

    # DAGGER loss
    expert_q = q_value_estimator.batched_q(
        X,
        y,
    )
    symbolic_q = q_value_estimator.batched_q(
        X,
        y_pred,
    )
    # print("finite expert:",
    #       jnp.isfinite(expert_q).all())
    # print("finite symbolic:",
    #       jnp.isfinite(symbolic_q).all())
    q_values_difference = expert_q - symbolic_q
    q_values_difference = jnp.nan_to_num(
        q_values_difference,
        nan=0.0,
        posinf=1e3,
        neginf=-1e3,
    )
    q_loss = jnp.mean(
        q_values_difference
    )

    eps = .1
    alpha = 1
    # geometric mean of losses
    loss = jnp.sqrt((mse + eps) * (q_loss + alpha))
    loss = jnp.nan_to_num(loss, nan=-jnp.inf)

    # GA maximizes fitness
    fitness = -loss

    return jnp.asarray([fitness]), {
        "test_accuracy": fitness,
        "updated_params": genotype,
    }


def feature_construction_scoring_fn(genotypes: Genotype, key: RNGKey, X: jnp.ndarray, y: jnp.ndarray,
                                    cgp_structure: CGP, q_value_estimator
                                    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    sng = partial(single_genome_scoring_fn, X=X, y=y, cgp_structure=cgp_structure, q_value_estimator=q_value_estimator)
    return jax.jit(jax.vmap(sng))(genotypes)


# def process_metrics_mtr(metrics: Dict, headers: List) -> Dict:
#     test_accuracy_values = metrics.pop("test_accuracy")
#     for idx, header in enumerate(headers):
#         metrics[header] = test_accuracy_values[idx]
#     return metrics


def fit_dataset(X, y, cgp_structure, seed=0, n_gens=100,
                q_value_estimator=None, bootstrap_repertoire=None, n_pop = 100):
    key = jax.random.key(seed)

    # Init the population
    key, subkey = jax.random.split(key)
    init_keys = jax.random.split(key, n_pop)
    init_population = jax.jit(jax.vmap(cgp_structure.init))(init_keys)

    # Define a metrics function
    metrics_function = functools.partial(
        custom_ga_metrics, extra_scores_metrics={"test_accuracy": jnp.ravel}
    )

    # Define emitter
    mutation_fn = jax.jit(jax.vmap(cgp_structure.mutate, in_axes=(0, 0)))
    tournament_selector = TournamentSelector()
    mixing_emitter = CustomMixingEmitter(
        mutation_fn=mutation_fn,
        variation_fn=None,
        variation_percentage=0,
        batch_size=n_pop,
        selector=tournament_selector,
    )

    # Prepare the scoring function
    scoring_fn = partial(
        feature_construction_scoring_fn,
        X=X, y=y, cgp_structure=cgp_structure,
        q_value_estimator=q_value_estimator
    )
    # Instantiate GA
    ga = GeneticAlgorithmWithExtraScores(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
    )

    # Evaluate the initial population
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = ga.init(
        genotypes=init_population, population_size=n_pop, key=subkey
    )
    if bootstrap_repertoire is not None:
        repertoire = bootstrap_repertoire

    metrics = {
        k: jnp.array([])
        for k in ["iteration", "max_fitness", "time"]
    }

    # Set up init metrics
    # init_metrics = jax.tree.map(lambda x: jnp.array([x]) if x.shape == () else x, init_metrics)
    init_metrics["iteration"] = 0
    init_metrics["max_fitness"] = init_metrics["max_fitness"][0]
    init_metrics["time"] = 0.0  # No time recorded for initialization

    # Convert init_metrics to match the metrics dictionary structure
    # metrics = jax.tree.map(lambda metric, init_metric: jnp.concatenate([metric, init_metric], axis=0), metrics,
    #                        init_metrics)
    # csv_logger = CSVLogger(
    #     f'../results/{config["run_name"]}.csv', header=list(metrics.keys())
    # )

    # Log initial metrics
    # csv_logger.log(jax.tree.map(lambda x: x[-1], init_metrics))
    # csv_logger.log(process_metrics_mtr(init_metrics, test_accuracy_header))

    # Iterations
    for iteration in range(1, n_gens):
        key, subkey, sample_key = jax.random.split(key, 3)

        start_time = time.time()

        repertoire, emitter_state, current_metrics = ga.update(
            repertoire=repertoire,
            emitter_state=emitter_state,
            key=subkey,
        )
        timelapse = time.time() - start_time

        # Metrics
        unwrapped_metrics = jax.tree.map(lambda x: jnp.ravel(x), current_metrics)
        unwrapped_metrics["iteration"] = iteration
        unwrapped_metrics["time"] = timelapse
        unwrapped_metrics["max_fitness"] = unwrapped_metrics["max_fitness"][0]

        # print(unwrapped_metrics)

        # Log
        # csv_logger.log(unwrapped_metrics)

    repertoire_to_store = GARepertoire.init(
        genotypes=repertoire.genotypes,
        fitnesses=repertoire.fitnesses,
        population_size=len(repertoire.fitnesses),
    )
    return repertoire_to_store, unwrapped_metrics["max_fitness"]
