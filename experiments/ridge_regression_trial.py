import functools
import os.path
import pickle
import sys
import time
from functools import partial
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
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


def single_genome_feature_construction_scoring_fn(
        genotype: Genotype,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        X_test: jnp.ndarray,
        y_test: jnp.ndarray,
        cgp_structure: CGP,
) -> Tuple:
    # Construct features
    features = jax.jit(
        jax.vmap(cgp_structure.apply, in_axes=(None, 0))
    )(genotype, X_train)

    # Sanitization
    features = jnp.nan_to_num(
        features,
        nan=0.0,
        posinf=1e3,
        neginf=-1e3,
    )

    # Add intercept column
    features_aug = jnp.concatenate(
        [features, jnp.ones((features.shape[0], 1))],
        axis=1,
    )

    # Ridge regression
    lam = 1e-5

    XtX = features_aug.T @ features_aug
    Xty = features_aug.T @ y_train

    # Do not regularize intercept
    reg = lam * jnp.eye(features_aug.shape[1])
    reg = reg.at[-1, -1].set(0.0)

    train_weights = jnp.linalg.solve(
        XtX + reg,
        Xty,
    )

    # Test features
    test_features = jax.jit(
        jax.vmap(cgp_structure.apply, in_axes=(None, 0))
    )(genotype, X_test)

    test_features = jnp.nan_to_num(
        test_features,
        nan=0.0,
        posinf=1e3,
        neginf=-1e3,
    )

    test_features_aug = jnp.concatenate(
        [test_features, jnp.ones((test_features.shape[0], 1))],
        axis=1,
    )

    # Predictions
    pred_y_train = features_aug @ train_weights
    pred_y_test = test_features_aug @ train_weights

    r2_train = r2_score(y_train, pred_y_train)
    r2_test = r2_score(y_test, pred_y_test)

    return jnp.asarray([r2_train]), {
        "test_accuracy": r2_test,
        "updated_params": genotype,
        "linear_weights": train_weights[:-1],
        "intercept": train_weights[-1],
    }


# def single_genome_feature_construction_scoring_fn(genotype: Genotype, X_train: jnp.ndarray, y_train: jnp.ndarray,
#                                                   X_test: jnp.ndarray, y_test: jnp.ndarray, cgp_structure: CGP
#                                                   ) -> Tuple:
#     features = jax.jit(jax.vmap(cgp_structure.apply, in_axes=(None, 0)))(genotype, X_train)
#     # sanitization step and ridge regression
#     features = jnp.nan_to_num(features, nan=0.0, posinf=1e3, neginf=-1e3)
#     lam = 1e-5
#     XtX = features.T @ features
#     Xty = features.T @ y_train
#     train_weights = jnp.linalg.solve(
#         XtX + lam * jnp.eye(features.shape[1]),
#         Xty
#     )
#     # train_weights, _, _, _ = jnp.linalg.lstsq(features, y_train)
#     test_features = jax.jit(jax.vmap(cgp_structure.apply, in_axes=(None, 0)))(genotype, X_test)
#     pred_y_train = features @ train_weights
#     pred_y_test = test_features @ train_weights
#     r2_train = r2_score(y_train, pred_y_train)
#     r2_test = r2_score(y_test, pred_y_test)
#     return jnp.asarray([r2_train]), {
#         "test_accuracy": r2_test,
#         "updated_params": genotype,
#     }



def run_ridge_regression(probl: str):
    X_train, X_test, y_train, y_test = load_dataset(
        probl,
        scale_x=False,
        scale_y=False,
        random_state=0,
    )
    features = X_train

    # Sanitization
    features = jnp.nan_to_num(
        features,
        nan=0.0,
        posinf=1e3,
        neginf=-1e3,
    )

    # Add intercept column
    features_aug = jnp.concatenate(
        [features, jnp.ones((features.shape[0], 1))],
        axis=1,
    )

    # Ridge regression
    lam = 1e-5

    XtX = features_aug.T @ features_aug
    Xty = features_aug.T @ y_train

    # Do not regularize intercept
    reg = lam * jnp.eye(features_aug.shape[1])
    reg = reg.at[-1, -1].set(0.0)

    train_weights = jnp.linalg.solve(
        XtX + reg,
        Xty,
    )

    # Test features
    test_features = X_test

    test_features = jnp.nan_to_num(
        test_features,
        nan=0.0,
        posinf=1e3,
        neginf=-1e3,
    )

    test_features_aug = jnp.concatenate(
        [test_features, jnp.ones((test_features.shape[0], 1))],
        axis=1,
    )

    # Predictions
    pred_y_train = features_aug @ train_weights
    pred_y_test = test_features_aug @ train_weights

    r2_train = r2_score(y_train, pred_y_train)
    r2_test = r2_score(y_test, pred_y_test)
    header_string = "iteration,max_fitness,time,test_accuracy"
    result_string = f"1499,{r2_train},0.0,[{r2_test}]"
    print(probl)
    with open(f"../results/ridge_{probl}.csv", "a") as f:
        f.write(header_string + "\n")
        f.write(result_string + "\n")

if __name__ == "__main__":

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
    for problem in problems:
        run_ridge_regression(problem)
