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
from genepax.gp.tree_genetic_programming import TreeGP
from genepax.supervised_learning.dataset_utils import downsample_dataset, load_dataset
from genepax.supervised_learning.utils import prepare_rescoring_fn, prepare_scoring_fn


def process_metrics_mtr(metrics: Dict, headers: List) -> Dict:
    test_accuracy_values = metrics.pop("test_accuracy")
    for idx, header in enumerate(headers):
        metrics[header] = test_accuracy_values[idx]
    return metrics


def run_sym_reg_ga(config: Dict):
    task = "regression" if "mtr" not in config["problem"] else "multiregression"

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
    tree_structure = TreeGP(
        n_inputs=X_train.shape[1],
        max_depth=config["solver"]["max_depth"],
        outputs_wrapper=lambda x: x,
    )

    # print(tree_structure)

    def get_scoring_fn_wrapper(
        original_scoring_fn: Callable[[Genotype, RNGKey], Tuple[Fitness, ExtraScores]],
    ) -> Callable[[Genotype, RNGKey], Tuple[Fitness, ExtraScores]]:

        def _wrapped(geno: Genotype, the_k: RNGKey) -> Tuple[Fitness, ExtraScores]:
            fit, extras = original_scoring_fn(geno, the_k)
            tree_sizes = jax.jit(jax.vmap(tree_structure.size))(geno)
            mask = (tree_sizes <= config["solver"]["max_size"]).reshape(-1, 1)
            wrapped_fits = jnp.where(mask, fit, -jnp.inf)
            return wrapped_fits, extras

        return _wrapped

    def get_rescoring_fn_wrapper(
        original_rescoring_fn: Callable[[Genotype, RNGKey], Fitness],
    ) -> Callable[[Genotype, RNGKey], Fitness]:

        def _rewrapped(geno: Genotype, the_k: RNGKey) -> Fitness:
            fit = original_rescoring_fn(geno, the_k)
            tree_sizes = jax.jit(jax.vmap(tree_structure.size))(geno)
            mask = (tree_sizes <= config["solver"]["max_size"]).reshape(-1, 1)
            return jnp.where(mask, fit, -jnp.inf)

        return _rewrapped

    # Init the population of trees
    key, subkey = jax.random.split(key)
    init_population = tree_structure.init_ramped_half_and_half(subkey, config["n_pop"])

    # Define a metrics function
    metrics_function = functools.partial(
        custom_ga_metrics, extra_scores_metrics={"test_accuracy": jnp.ravel}
    )

    # Define emitter
    mutation_fn = jax.jit(jax.vmap(tree_structure.mutate, in_axes=(0, 0)))
    variation_fn = jax.jit(jax.vmap(tree_structure.crossover, in_axes=(0, 0, 0)))
    tournament_selector = TournamentSelector(tournament_size=config["tournament_size"])
    mixing_emitter = CustomMixingEmitter(
        mutation_fn=mutation_fn,
        variation_fn=variation_fn,
        variation_percentage=0.8,
        batch_size=config["n_offspring"],
        selector=tournament_selector,
    )

    # Prepare the scoring function
    scoring_fn = prepare_scoring_fn(
        X_train_sub, y_train_sub, X_test, y_test, tree_structure, task=task
    )
    rescoring_fn_gp = prepare_rescoring_fn(
        X_train_sub, y_train_sub, tree_structure, task=task
    )
    # Instantiate GA
    ga = GeneticAlgorithmWithExtraScores(
        scoring_function=get_scoring_fn_wrapper(scoring_fn),
        emitter=mixing_emitter,
        metrics_function=metrics_function,
        lamarckian=False,
        rescoring_function=get_rescoring_fn_wrapper(rescoring_fn_gp),
    )

    # Evaluate the initial population
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = ga.init(
        genotypes=init_population, population_size=config["n_pop"], key=subkey
    )

    # Initialize metrics
    n_targets = y_test.shape[1]
    test_accuracy_header = (
        ["test_accuracy"]
        if n_targets == 1
        else [f"rrmse_{i}" for i in range(n_targets)]
    )
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
                X_train_sub, y_train_sub, X_test, y_test, tree_structure, task=task
            )
            rescoring_fn = prepare_rescoring_fn(
                X_train_sub, y_train_sub, tree_structure, task=task
            )
            ga = ga.replace_scoring_fns(
                get_scoring_fn_wrapper(scoring_fn),
                get_rescoring_fn_wrapper(rescoring_fn),
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
    n_gens = 300
    n_pop = 500
    conf = {
        "solver": {"max_depth": 10, "max_size": 50},
        "n_offspring": n_pop,
        "n_pop": n_pop,
        "seed": 0,
        "tournament_size": 3,
        "problem": "chemical_2_competition",
        "scale_x": False,
        "scale_y": False,
    }

    problems = [
        "chemical_2_competition",
        "friction_dyn_one-hot",
        "friction_stat_one-hot",
        "nasa_battery_1_10min",
        "nasa_battery_2_20min",
        "nikuradse_1",
        "nikuradse_2",
        "chemical_1_tower",
        "flow_stress_phip0.1",
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
        conf["seed"] = seed
        conf["n_gens"] = n_gens
        # extra += f"_wpgs" if w_pgs else ""
        conf["run_name"] = (
            "GP_deep_" + conf["problem"].replace("/", "_") + "_" + str(conf["seed"])
        )
        print(conf["run_name"])
        if os.path.exists(f"../results/{conf['run_name']}.pickle"):
            print("run already done!")
        else:
            print("running")
            run_sym_reg_ga(conf)
