import functools
import os.path
import pickle
import sys
import time
from functools import partial
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import skdim
from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.custom_types import Genotype, RNGKey
from qdax.utils.metrics import CSVLogger

from genepax.evolution.custom_emitters import CustomMixingEmitter
from genepax.evolution.evolution_metrics import custom_ga_metrics
from genepax.evolution.genetic_algorithm_extra_scores import (
    GeneticAlgorithmWithExtraScores,
)
from genepax.evolution.tournament_selector import TournamentSelector
from genepax.gp.cartesian_genetic_programming import CGP
from genepax.supervised_learning.dataset_utils import downsample_dataset, load_dataset
from genepax.supervised_learning.metrics import r2_score


def single_genome_feature_construction_scoring_fn(genotype: Genotype, X_train: jnp.ndarray, y_train: jnp.ndarray,
                                                  X_test: jnp.ndarray, y_test: jnp.ndarray, cgp_structure: CGP
                                                  ) -> Tuple:
    features = jax.jit(jax.vmap(cgp_structure.apply, in_axes=(None, 0)))(genotype, X_train)
    # sanitization step and ridge regression
    features = jnp.nan_to_num(features, nan=0.0, posinf=1e3, neginf=-1e3)
    lam = 1e-5
    XtX = features.T @ features
    Xty = features.T @ y_train
    train_weights = jnp.linalg.solve(
        XtX + lam * jnp.eye(features.shape[1]),
        Xty
    )
    # train_weights, _, _, _ = jnp.linalg.lstsq(features, y_train)
    test_features = jax.jit(jax.vmap(cgp_structure.apply, in_axes=(None, 0)))(genotype, X_test)
    pred_y_train = features @ train_weights
    pred_y_test = test_features @ train_weights
    r2_train = r2_score(y_train, pred_y_train)
    r2_test = r2_score(y_test, pred_y_test)
    return jnp.asarray([r2_train]), {
        "test_accuracy": r2_test,
        "updated_params": genotype,
    }


def feature_construction_scoring_fn(genotypes: Genotype, key: RNGKey, X_train: jnp.ndarray, y_train: jnp.ndarray,
                                    X_test: jnp.ndarray, y_test: jnp.ndarray, cgp_structure: CGP
                                    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    sng = partial(single_genome_feature_construction_scoring_fn, X_train=X_train, y_train=y_train, X_test=X_test,
                  y_test=y_test, cgp_structure=cgp_structure)
    return jax.jit(jax.vmap(sng))(genotypes)


def process_metrics_mtr(metrics: Dict, headers: List) -> Dict:
    test_accuracy_values = metrics.pop("test_accuracy")
    for idx, header in enumerate(headers):
        metrics[header] = test_accuracy_values[idx]
    return metrics


def run_sym_reg_ga(config: Dict):
    X_train, X_test, y_train, y_test = load_dataset(
        config["problem"],
        scale_x=config.get("scale_x", False),
        scale_y=config.get("scale_y", False),
        random_state=config["seed"],
    )
    key = jax.random.key(config["seed"])
    sample_key, key = jax.random.split(key)
    rescoring = len(X_train) > 2048

    # danco_id = skdim.id.DANCo(fractal=False).fit(X_train)
    # n_features = danco_id.dimension_
    # print(n_features)
    n_features = jnp.round(jnp.sqrt(X_train.shape[1])).astype(int)

    # Init the CGP policy graph with default values
    cgp_structure = CGP(
        n_inputs=X_train.shape[1],
        n_outputs=n_features,
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
    scoring_fn = partial(
        feature_construction_scoring_fn,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test, cgp_structure=cgp_structure
    )
    # Instantiate GA
    ga = GeneticAlgorithmWithExtraScores(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
        lamarckian=False,
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
        k: jnp.array([])
        for k in ["iteration", "max_fitness", "time"] + test_accuracy_header
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
        "chemical_2_competition",
        "friction_dyn_one-hot",
        "friction_stat_one-hot",
        "nasa_battery_1_10min",
        "nasa_battery_2_20min",
        "nikuradse_1",
        "nikuradse_2",
        # "chemical_1_tower",
        # "flow_stress_phip0.1",
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
            # extra += f"_wpgs" if w_pgs else ""
            conf["run_name"] = (
                    "CGP_feats_" + conf["problem"].replace("/", "_") + "_" + str(conf["seed"])
            )
            print(conf["run_name"])
            if os.path.exists(f"../results/{conf['run_name']}.pickle"):
                print("run already done!")
            else:
                print("running")
                run_sym_reg_ga(conf)
