from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey

from genepax.gp.graph_genetic_programming import GGP
from genepax.supervised_learning.constants_optimization import (
    optimize_constants_with_sgd,
)
from genepax.supervised_learning.metrics import r2_score


def compute_model_predictions(
    X: jnp.ndarray,
    genotype: Genotype,
    graph_structure: GGP,
    graph_weights: Optional[Dict[str, jnp.ndarray]] = None,
    max_norm: float = 1e6,
) -> jnp.ndarray:
    """
    Compute model predictions for a batch of inputs (in a supervised
    learning setting).
    The batch of input is processed in parallel via vectorization.

    Parameters
    ----------
    X : jnp.ndarray
        Input data of shape (batch_size, input_dim), where each row is a
        separate sample (i.e., data points) to evaluate.
    genotype : Genotype
        The genotype parameters to be evaluated.
    graph_structure : GGP
        The structure defining how a genotype is encoded into a program.
    graph_weights : jnp.ndarray
        Optional weighting factors for the graph.
    max_norm: float
        Value to clip the output norm to avoid nans and large gradients.

    Returns
    -------
    jnp.ndarray
        The predicted regression outputs for each input sample, of shape
        (batch_size,) or (batch_size, output_dim).
    """
    parallel_apply = jax.vmap(jax.jit(graph_structure.apply), in_axes=(None, 0, None))
    prediction = parallel_apply(genotype, X, graph_weights)
    pred = jnp.nan_to_num(prediction, nan=0.0, posinf=max_norm, neginf=-max_norm)
    return jnp.clip(pred, -max_norm, max_norm)


def supervised_learning_accuracy_evaluation(
    genotype: Genotype,
    key: RNGKey,
    graph_structure: GGP,
    X: jnp.ndarray,
    y: jnp.ndarray,
    accuracy_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = r2_score,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Evaluate supervised learning accuracy for a batch of genotypes on a dataset.

    This function computes predictions for each genotype in the batch on
    the input data `X`, and then evaluates the predictions against the target
    outputs `y` using the provided `accuracy_fn`. All computations
    are vectorized and JIT-compiled.

    Parameters
    ----------
    genotype : Genotype
        A batch of genotypes to evaluate. Shape should allow vectorization
        over genotypes (e.g., (n_genotypes, ...)).
    key : RNGKey
        JAX random key for any stochastic operations during evaluation.
    graph_structure : GGP
        The computational graph defining how genotypes map inputs to outputs.
    X : jnp.ndarray
        Input features of shape (n_samples, n_features) to evaluate.
    y : jnp.ndarray
        Ground-truth target values corresponding to `X`.
    accuracy_fn : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], optional
        Function to compute accuracy between predictions and targets.
        Defaults to `r2_score`.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        accuracies : jnp.ndarray
            Accuracy values for each genotype, shape (n_genotypes, ...)
            depending on `accuracy_fn`.
        genotype : Genotype
            The batch of genotypes that were evaluated, returned for convenience.
    """
    prediction_fn = jax.jit(
        partial(compute_model_predictions, graph_structure=graph_structure)
    )

    def _accuracy_fn(single_genotype: Genotype) -> jnp.ndarray:
        prediction = prediction_fn(X, single_genotype)
        return accuracy_fn(y, prediction)

    accuracies = jax.vmap(_accuracy_fn)(genotype)

    return jnp.expand_dims(accuracies, 1), genotype


def supervised_learning_accuracy_evaluation_with_constants_optimization(
    genotype: Genotype,
    key: RNGKey,
    graph_structure: GGP,
    X: jnp.ndarray,
    y: jnp.ndarray,
    accuracy_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = r2_score,
    constants_optimization_fn: Callable = optimize_constants_with_sgd,
    reset_weights: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Evaluate supervised learning accuracy after optimizing the constants of a
    genotype's computational graph.

    This function optionally re-initializes the weights of the given genotype(s),
    performs constant optimization (e.g., via Adam/SGD), updates the genotype(s)
    with the optimized weights, and finally computes regression accuracy.

    Parameters
    ----------
    genotype : Genotype
        A batch of genotypes whose constant parameters will be optimized.
    key : RNGKey
        JAX PRNG key used for weight initialization and any stochastic operations
        in the optimization routine.
    graph_structure : GGP
        The graph representation/model used to interpret the genotype and its weights.
    X : jnp.ndarray
        Input feature data. Usually shaped (n_samples, n_features).
    y : jnp.ndarray
        Target regression values. Usually shaped (n_samples,) or (n_samples, n_outputs).
    accuracy_fn : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], optional
        Function that computes regression accuracy between predictions and targets.
        Defaults to `r2_score`.
    constants_optimization_fn : Callable, optional
        Optimization function responsible for updating the constant parameters
        (weights) inside the computational graph. Defaults to
        `optimize_constants_with_adam_sgd`.
    reset_weights : bool, optional
        If True, reinitialize all genotype weights before optimization. Defaults to False.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        accuracies : jnp.ndarray
            Accuracy values computed using `accuracy_fn` for each genotype after
            constant optimization.
        updated_genotype : Genotype
            The genotype(s) with updated/optimized weight parameters.p
    """
    graph_weights = graph_structure.get_weights(genotype)
    if reset_weights:
        n_genomes = jax.tree.leaves(genotype)[0].shape[0]
        key, subkey = jax.random.split(key)
        weights_keys = jax.random.split(subkey, n_genomes)
        new_weights = jax.vmap(jax.jit(graph_structure.init_weights))(weights_keys)
        graph_weights = {
            k: new_weights[k] for k in graph_weights.keys() if k in new_weights
        }

    prediction_fn = jax.jit(
        partial(compute_model_predictions, graph_structure=graph_structure)
    )
    graph_weights = constants_optimization_fn(
        graph_weights, genotype, key, X, y, prediction_fn
    )

    # update the weights in the genomes
    updated_genotype = jax.vmap(
        jax.jit(graph_structure.update_weights), in_axes=(0, 0)
    )(genotype, graph_weights)

    return supervised_learning_accuracy_evaluation(
        updated_genotype, key, graph_structure, X, y, accuracy_fn
    )


def supervised_learning_scoring_fn(
    functions_params: Genotype,
    key: RNGKey,
    train_set_evaluation_fn: Callable[
        [Params, RNGKey], Tuple[jnp.ndarray, jnp.ndarray]
    ],
    test_set_evaluation_fn: Callable[[Params, RNGKey], Tuple[jnp.ndarray, jnp.ndarray]],
    descriptor_extractor: Optional[Callable[[Params], Descriptor]] = None,
) -> Union[Tuple[Fitness, ExtraScores], Tuple[Fitness, Descriptor, ExtraScores]]:
    """
    Evaluate a set of genotypes on training and test sets and optionally extract descriptors.

    This function computes a fitness score for a batch of genotypes by first evaluating them on a training set
    using `train_set_evaluation_fn`, which may also perform updates to the parameters (e.g., gradient steps).
    It then evaluates the updated parameters on a test set using `test_set_evaluation_fn`. Optionally, a
    `descriptor_extractor` can be provided to compute descriptors from the original genotype parameters.

    Parameters
    ----------
    functions_params : Genotype
        The genotype batch to evaluate.
    key : RNGKey
        JAX random key used to seed any stochastic operations in evaluation.
    train_set_evaluation_fn : Callable[[Params, RNGKey], Tuple[jnp.ndarray, jnp.ndarray]]
        Function that evaluates the genotype on the training set. Returns a tuple of
        `(train_accuracy, updated_params)`, where `updated_params` may correspond to new genotypes
        updated during the evaluation on the train set, e.g., via gradient descent.
    test_set_evaluation_fn : Callable[[Params, RNGKey], Tuple[jnp.ndarray, jnp.ndarray]]
        Function that evaluates the (possibly updated) genotypes on the test set.
        Returns `(test_accuracy, updated_params)`.
    descriptor_extractor : Optional[Callable[[Params], Descriptor]], optional
        A function to compute descriptors from the original parameters. If `None`, no descriptor is computed.

    Returns
    -------
    Union[Tuple[Fitness, ExtraScores], Tuple[Fitness, Descriptor, ExtraScores]]
        If `descriptor_extractor` is provided:
            - train_accuracy : Fitness
                Fitness score(s) computed on the training set.
            - descriptor : Descriptor
                Descriptor extracted from the original genotype parameters.
            - extra_scores : ExtraScores
                Dictionary containing additional metrics, including:
                - `"test_accuracy"`: test set accuracy computed from updated parameters.
                - `"updated_params"`: parameters after training evaluation.
        If `descriptor_extractor` is None:
            - train_accuracy : Fitness
                Fitness score(s) computed on the training set.
            - extra_scores : ExtraScores
                Dictionary containing additional metrics as above, but no descriptor is returned.

    Notes
    -----
    - `train_set_evaluation_fn` can optionally perform updates on the parameters.
    - Descriptors are only returned when `descriptor_extractor` is not None.
    """
    train_key, test_key = jax.random.split(key)
    # it can be a simple accuracy computation, but it can also include gradient steps
    train_accuracy, updated_params = train_set_evaluation_fn(
        functions_params, train_key
    )
    test_accuracy, _ = test_set_evaluation_fn(updated_params, test_key)
    if descriptor_extractor is not None:
        descriptor = descriptor_extractor(functions_params)
        return (
            train_accuracy,
            descriptor,
            {"test_accuracy": test_accuracy, "updated_params": updated_params},
        )
    else:
        return train_accuracy, {
            "test_accuracy": test_accuracy,
            "updated_params": updated_params,
        }
