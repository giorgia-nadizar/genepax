import functools
import os.path
import pickle
import sys
import time
from functools import partial
from typing import Dict, List

import jax
import jax.numpy as jnp
from jax import grad
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


def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))


def predict_proba(params, features):
    w, b = params
    return sigmoid(jnp.dot(features, w) + b)


def loss_fn(params, features, y):
    w, b = params
    logits = jnp.dot(features, w) + b

    # Stable binary cross entropy
    loss = jnp.mean(
        jnp.maximum(logits, 0)
        - logits * y
        + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    )

    return loss


def accuracy(params, features, y):
    probs = predict_proba(params, features)
    preds = (probs >= 0.5).astype(jnp.float32)
    return jnp.mean(preds == y)


def _single_genome_assessment(single_genotype, X, y, cgp_structure: CGP):
    custom_weights = single_genotype["weights"]["custom_weights"]
    features = jax.jit(jax.vmap(cgp_structure.apply, in_axes=(None, 0)))(single_genotype, X)
    features = jnp.nan_to_num(
        features,
        nan=0.0,
        posinf=100.0,
        neginf=-100.0,
    )
    features = jnp.clip(features, -100.0, 100.0)
    weights, bias = jnp.split(custom_weights, [cgp_structure.n_outputs])
    params = weights, bias
    return accuracy(params, features, y)


def _single_genome_update(single_genotype, X, y, cgp_structure: CGP, lr):
    custom_weights = single_genotype["weights"]["custom_weights"]
    features = jax.jit(jax.vmap(cgp_structure.apply, in_axes=(None, 0)))(single_genotype, X)
    features = jnp.nan_to_num(features, nan=0.0, posinf=1e3, neginf=-1e3)
    weights, bias = jnp.split(custom_weights, [cgp_structure.n_outputs])
    params = weights, bias
    grads = grad(loss_fn)(params, features, y)
    w, b = params
    dw, db = grads
    w = w - lr * dw
    b = b - lr * db
    updated_weights = {
        "custom_weights": jnp.concatenate([w, b])
    }
    return cgp_structure.update_weights(single_genotype, updated_weights)


def feature_construction_scoring_fn(genotypes: Genotype, key: RNGKey, X_train: jnp.ndarray, y_train: jnp.ndarray,
                                    X_test: jnp.ndarray, y_test: jnp.ndarray, cgp_structure: CGP, batch_size: int = 32,
                                    n_updates: int = 100, lr: float = 0.001,
                                    ):
    # single_genotype, X, y,
    partial_updated_fn = jax.jit(partial(_single_genome_update, cgp_structure=cgp_structure, lr=lr))
    step_fn = jax.vmap(
        jax.jit(partial_updated_fn), in_axes=(0, None, None)
    )
    for _ in range(n_updates):
        key, subkey = jax.random.split(key)
        # sample a mini-batch
        X_batch, y_batch = downsample_dataset(X_train, y_train, random_key=subkey, size=batch_size)
        genotypes = step_fn(genotypes, X_batch, y_batch)

    # assess accuracy
    train_assessment_fn = partial(_single_genome_assessment, X=X_train, y=y_train, cgp_structure=cgp_structure)
    test_assessment_fn = partial(_single_genome_assessment, X=X_test, y=y_test, cgp_structure=cgp_structure)
    train_accuracies = jax.vmap(jax.jit(train_assessment_fn))(genotypes)
    train_accuracies_reshaped = jnp.expand_dims(train_accuracies, axis=1)
    test_accuracies = jax.vmap(jax.jit(test_assessment_fn))(genotypes)
    return train_accuracies_reshaped, {
        "test_accuracy": test_accuracies,
        "updated_params": genotypes,
    }


def process_metrics_mtr(metrics: Dict, headers: List) -> Dict:
    test_accuracy_values = metrics.pop("test_accuracy")
    for idx, header in enumerate(headers):
        metrics[header] = test_accuracy_values[idx]
    return metrics


def run_classification_ga(config: Dict):
    X_train, X_test, y_train, y_test = load_dataset(
        config["problem"],
        scale_x=config.get("scale_x", False),
        scale_y=config.get("scale_y", False),
        random_state=config["seed"],
    )
    key = jax.random.key(config["seed"])
    sample_key, key = jax.random.split(key)
    rescoring = len(X_train) > 2048
    y_train = jnp.argmax(y_train, axis=1).astype(jnp.float32)
    y_test = jnp.argmax(y_test, axis=1).astype(jnp.float32)
    y_train = jnp.expand_dims(y_train, axis=1)
    y_test = jnp.expand_dims(y_test, axis=1)
    config["n_gens"] = int((config["n_gens"] * min(2048, len(X_train))) / (min(2048, len(X_train)) + 3200))

    if rescoring:
        downsample_fn = functools.partial(
            downsample_dataset, size=config.get("dataset_size", 1024)
        )
        X_train_sub, y_train_sub = downsample_fn(X_train, y_train, sample_key)
    else:
        X_train_sub, y_train_sub = X_train, y_train

    # danco_id = skdim.id.DANCo(fractal=False).fit(X_train)
    # n_features = danco_id.dimension_
    # print(n_features)
    n_features = jnp.round(jnp.sqrt(X_train.shape[1])).astype(int)
    n_custom_weights = n_features + 1

    # Init the CGP policy graph with default values
    cgp_structure = CGP(
        n_inputs=X_train.shape[1],
        n_outputs=n_features,
        n_nodes=config["solver"]["n_nodes"],
        outputs_wrapper=lambda x: x,
        n_custom_weights=n_custom_weights
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

    scoring_fn = partial(
        feature_construction_scoring_fn,
        X_train=X_train_sub, y_train=y_train_sub,
        X_test=X_test, y_test=y_test, cgp_structure=cgp_structure,
    )
    rescoring_fn = partial(
        feature_construction_scoring_fn,
        X_train=X_train_sub, y_train=y_train_sub,
        X_test=X_test, y_test=y_test, cgp_structure=cgp_structure,
    )
    # Instantiate GA
    ga = GeneticAlgorithmWithExtraScores(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
        lamarckian=True,
        rescoring_function=rescoring_fn,
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
        # if rescoring:
        #     # change batch of the dataset to evaluate upon
        #     X_train_sub, y_train_sub = downsample_fn(X_train, y_train, sample_key)
        #     scoring_fn = partial(
        #         feature_construction_scoring_fn,
        #         X_train=X_train_sub, y_train=y_train_sub,
        #         X_test=X_test, y_test=y_test, cgp_structure=cgp_structure,
        #         inner_fn=inner_scoring_fn
        #     )
        #     rescoring_fn = partial(
        #         feature_construction_scoring_fn,
        #         X_train=X_train_sub, y_train=y_train_sub,
        #         X_test=X_test, y_test=y_test, cgp_structure=cgp_structure,
        #         inner_fn=inner_rescoring_fn
        #     )
        #     ga = ga.replace_scoring_fns(
        #         scoring_fn,
        #         rescoring_fn,
        #     )

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
    n_pop = 7
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
        "diabetes_classification",
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
                    f"CGP_feats_" + conf["problem"].replace("/", "_") + "_" + str(conf["seed"])
            )
            print(conf["run_name"])
            if os.path.exists(f"../results/{conf['run_name']}.pickle"):
                print("run already done!")
            else:
                print("running")
                run_classification_ga(conf)
