import functools
import os.path
import pickle
import sys
import time
from typing import Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.custom_types import ExtraScores, Fitness, Genotype, RNGKey
from qdax.utils.metrics import CSVLogger

from genepax.evolution.custom_emitters import CustomMixingEmitter
from genepax.evolution.evolution_metrics import custom_ga_metrics
from genepax.evolution.genetic_algorithm_extra_scores import (
    GeneticAlgorithmWithExtraScores,
)
from genepax.evolution.tournament_selector import TournamentSelector
from genepax.gp.cartesian_genetic_programming import CGP
from genepax.gp.tree_genetic_programming import TreeGP
from genepax.supervised_learning.dataset_utils import downsample_dataset, load_dataset
from genepax.supervised_learning.utils import prepare_rescoring_fn, prepare_scoring_fn


def process_metrics_mtr(metrics: Dict, headers: List) -> Dict:
    test_accuracy_values = metrics.pop("test_accuracy")
    for idx, header in enumerate(headers):
        metrics[header] = test_accuracy_values[idx]
    return metrics


def run_sym_reg_ga(config: Dict):
    task = "classification"

    X_train, X_test, y_train, y_test = load_dataset(
        config["problem"],
        scale_x=config.get("scale_x", False),
        scale_y=config.get("scale_y", False),
        random_state=config["seed"],
    )
    key = jax.random.key(config["seed"])
    sample_key, key = jax.random.split(key)
    rescoring = len(X_train) > 2048

    if rescoring:
        downsample_fn = functools.partial(
            downsample_dataset, size=config.get("dataset_size", 1024)
        )
        X_train_sub, y_train_sub = downsample_fn(X_train, y_train, sample_key)
    else:
        X_train_sub, y_train_sub = X_train, y_train

    # Init the CGP policy graph with default values
    cgp_structure = CGP(
        n_inputs=X_train.shape[1],
        n_outputs=2,
        n_nodes=config["solver"]["n_nodes"],
        outputs_wrapper=lambda x: x,
    )

    # Init the population
    key, subkey = jax.random.split(key)
    init_keys = jax.random.split(key, config["n_pop"])
    init_population = jax.jit(jax.vmap(cgp_structure.init))(init_keys)

    # Define a metrics function
    metrics_function = functools.partial(
        custom_ga_metrics, extra_scores_metrics={"test_accuracy": jnp.ravel}
    )

    # Define emitter
    mutation_fn = jax.jit(jax.vmap(cgp_structure.mutate, in_axes=(0, 0)))
    tournament_selector = TournamentSelector(tournament_size=config["tournament_size"])
    mixing_emitter = CustomMixingEmitter(
        mutation_fn=mutation_fn,
        variation_fn=None,
        variation_percentage=0,
        batch_size=config["n_offspring"],
        selector=tournament_selector,
    )

    # Prepare the scoring function
    scoring_fn = prepare_scoring_fn(
        X_train_sub, y_train_sub, X_test, y_test, cgp_structure, task=task
    )
    rescoring_fn = prepare_rescoring_fn(
        X_train_sub, y_train_sub, cgp_structure, task=task
    )
    # Instantiate GA
    ga = GeneticAlgorithmWithExtraScores(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
        lamarckian=False,
        rescoring_function=rescoring_fn,
    )

    # Evaluate the initial population
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = ga.init(
        genotypes=init_population, population_size=config["n_pop"], key=subkey
    )

    # Initialize metrics
    n_targets = y_test.shape[1]
    test_accuracy_header = ["test_accuracy"]
    metrics = {
        key: jnp.array([])
        for key in ["iteration", "max_fitness", "time"] + test_accuracy_header
    }

    # Set up init metrics
    # init_metrics = jax.tree.map(lambda x: jnp.array([x]) if x.shape == () else x, init_metrics)
    init_metrics["iteration"] = 0
    init_metrics["max_fitness"] = init_metrics["max_fitness"][0]
    init_metrics["time"] = 0.0  # No time recorded for initialization

    # Convert init_metrics to match the metrics dictionary structure
    # metrics = jax.tree.map(lambda metric, init_metric: jnp.concatenate([metric, init_metric], axis=0), metrics,
    #                        init_metrics)
    csv_logger = CSVLogger(
        f'../results/{config["run_name"]}.csv', header=list(metrics.keys())
    )

    # Log initial metrics
    # csv_logger.log(jax.tree.map(lambda x: x[-1], init_metrics))
    csv_logger.log(process_metrics_mtr(init_metrics, test_accuracy_header))

    # Iterations
    for iteration in range(1, config["n_gens"]):
        key, subkey, sample_key = jax.random.split(key, 3)

        if rescoring:
            # change batch of the dataset to evaluate upon
            X_train_sub, y_train_sub = downsample_fn(X_train, y_train, sample_key)
            scoring_fn = prepare_scoring_fn(
                X_train_sub, y_train_sub, X_test, y_test, cgp_structure, task=task
            )
            rescoring_fn = prepare_rescoring_fn(
                X_train_sub, y_train_sub, cgp_structure, task=task
            )
            ga = ga.replace_scoring_fns(
                scoring_fn,
                rescoring_fn,
            )

        start_time = time.time()

        repertoire, emitter_state, current_metrics = ga.update(
            repertoire=repertoire,
            emitter_state=emitter_state,
            key=subkey,
            rescore_repertoire=rescoring,
        )
        timelapse = time.time() - start_time

        # Metrics
        unwrapped_metrics = jax.tree.map(lambda x: jnp.ravel(x), current_metrics)
        unwrapped_metrics["iteration"] = iteration
        unwrapped_metrics["time"] = timelapse
        unwrapped_metrics["max_fitness"] = unwrapped_metrics["max_fitness"][0]
        if len(test_accuracy_header) > 1:
            unwrapped_metrics = process_metrics_mtr(
                unwrapped_metrics, test_accuracy_header
            )

        print(unwrapped_metrics)

        # Log
        csv_logger.log(unwrapped_metrics)

    repertoire_to_store = GARepertoire.init(
        genotypes=repertoire.genotypes,
        fitnesses=repertoire.fitnesses,
        population_size=len(repertoire.fitnesses),
    )
    path = f"../results/{conf['run_name']}.pickle"
    with open(path, "wb") as file:
        pickle.dump(repertoire_to_store, file)


if __name__ == "__main__":
    n_gens = 1500
    n_pop = 100
    conf = {
        "solver": {"n_nodes": 50},
        "n_offspring": n_pop,
        "n_pop": n_pop,
        "seed": 0,
        "tournament_size": 3,
        "problem": "chemical_2_competition",
        "scale_x": False,
        "scale_y": False,
    }

    problems = [
        'Hill_Valley_with_noise', 'Hill_Valley_without_noise', 'clean1'
        # "breast_cancer",
        # "diabetes_classification",
    ]

    args = sys.argv[1:]
    for arg in args:
        key, value = arg.split("=")
        if key == "problem":
            conf["problem"] = value
        elif key == "seed":
            conf["seed"] = int(value)
        elif key == "problem_id":
            conf["problem"] = problems[int(value)]

    for seed in range(10):
        for problem in problems:
            conf["problem"] = problem
            conf["seed"] = seed
            conf["n_gens"] = n_gens
            conf["run_name"] = (
                    "CGP_baseline_" + conf["problem"].replace("/", "_") + "_" + str(conf["seed"])
            )
            if conf["solver"]["n_nodes"] == 100:
                conf["run_name"] = conf["run_name"].replace("CGP", "CGP_100")
            print(conf["run_name"])
            if os.path.exists(f"../results/{conf['run_name']}.pickle"):
                print("run already done!")
            else:
                print("running")
                run_sym_reg_ga(conf)
