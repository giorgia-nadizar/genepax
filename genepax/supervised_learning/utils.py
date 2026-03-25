import functools
from typing import Callable, Dict, Optional, Tuple

import jax.numpy as jnp
import optax

from genepax.gp.graph_genetic_programming import GGP
from genepax.supervised_learning.constants_optimization import (
    optimize_constants_with_cmaes,
    optimize_constants_with_lbfgs,
    optimize_constants_with_sgd,
)
from genepax.supervised_learning.metrics import (
    categorical_cross_entropy,
    classification_accuracy,
    r2_score,
    rmse,
    rrmse_per_target,
)
from genepax.supervised_learning.scoring_functions import (
    supervised_learning_accuracy_evaluation,
    supervised_learning_accuracy_evaluation_with_constants_optimization,
    supervised_learning_scoring_fn,
)


def prepare_train_test_evaluation_fns(
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_test: jnp.ndarray,
    y_test: jnp.ndarray,
    graph_structure: GGP,
    const_optimizer: Optional[str] = None,
    long_const_optimization: bool = False,
    task: str = "regression",
) -> Tuple[Callable, Callable]:
    """
    Prepare training and testing evaluation functions for symbolic regression models.

    This utility constructs two callable evaluation functions, one for training data
    and one for test data, based on the provided dataset and configuration. Depending
    on the selected constant-optimization strategy, the training function may also
    include an inner optimization routine to refine numeric constants within the
    model’s computational graph.

    Parameters
    ----------
    X_train : jnp.ndarray
        Input features for the training set.
    y_train : jnp.ndarray
        Target values corresponding to `X_train`.
    X_test : jnp.ndarray
        Input features for the test set.
    y_test : jnp.ndarray
        Target values corresponding to `X_test`.
    graph_structure : GGP
        A graph-based program (GGP) structure representing the symbolic model.
        This is passed through to the evaluation functions.
    const_optimizer : str, optional
        Specifies the strategy for optimizing constants within the symbolic model.
        Supported values include:
        - ``"adam"``: optimize constants via SGD with Adam-like settings.
        - ``"lbfgs"``: optimize using L-BFGS.
        - ``"rmsprop"``: optimize using RMSProp.
        - ``"cmaes"``: optimize using CMA-ES.
        Any unrecognized value defaults to no constants optimization.
    long_const_optimization : bool, optional
        Whether the constants are optimizer for longer or not.
    task : str, optional
        The supervised learning task considered, either `regression`, `classification`,
        or `multi-regression`

    Returns
    -------
    (train_fn, test_fn) : tuple of callables
        `train_fn` is a partially applied evaluation function for the training set,
        optionally including a constant-optimization subroutine.
        `test_fn` is a partially applied evaluation function for evaluating regression
        accuracy on the test set.

    Notes
    -----
    - The returned functions are already bound to the provided data and model
      structure and can be called directly with a candidate program representation.
    - If a constant optimizer is specified, training proceeds by first optimizing the
      constants of the program and then evaluating accuracy.
    """
    # Avoid misspelling of multi-regression
    task = task.replace("-", "").replace("_", "")
    if task not in ["regression", "classification", "multiregression"]:
        raise NotImplementedError("Task not supported.")

    # Accuracy fns per task
    train_accuracy_fns = {
        "regression": r2_score,
        "classification": classification_accuracy,
        "multiregression": r2_score,
    }
    test_accuracy_fns: Dict[str, Callable] = {
        "regression": r2_score,
        "classification": classification_accuracy,
        "multiregression": functools.partial(rrmse_per_target, y_train=y_train),
    }
    loss_fns: Dict[str, Callable] = {
        "regression": rmse,
        "classification": categorical_cross_entropy,
        "multiregression": rmse,
    }
    train_accuracy_fn = train_accuracy_fns[task]
    test_accuracy_fn = test_accuracy_fns[task]
    loss_fn = loss_fns[task]

    multiplier = 10 if long_const_optimization else 1
    if const_optimizer == "adam":
        constants_optimizer = functools.partial(
            optimize_constants_with_sgd,
            batch_size=32,
            n_gradient_steps=100 * multiplier,
            loss_fn=loss_fn,
        )
    elif const_optimizer == "rmsprop":
        constants_optimizer = functools.partial(
            optimize_constants_with_sgd,
            batch_size=32,
            n_gradient_steps=120 * multiplier,
            loss_fn=loss_fn,
            optimizer=optax.rmsprop(1e-3, momentum=0.9),
        )
    elif const_optimizer == "cmaes":
        constants_optimizer = functools.partial(
            optimize_constants_with_cmaes, max_iter=20 * multiplier, loss_fn=loss_fn
        )
    elif const_optimizer == "lbfgs":
        constants_optimizer = functools.partial(
            optimize_constants_with_lbfgs, max_iter=5 * multiplier, loss_fn=loss_fn
        )
    else:
        constants_optimizer = None

    if constants_optimizer:
        train_fn = functools.partial(
            supervised_learning_accuracy_evaluation_with_constants_optimization,
            graph_structure=graph_structure,
            X=X_train,
            y=y_train,
            reset_weights=False,
            constants_optimization_fn=constants_optimizer,
            accuracy_fn=train_accuracy_fn,
        )
    else:
        train_fn = functools.partial(
            supervised_learning_accuracy_evaluation,
            graph_structure=graph_structure,
            X=X_train,
            y=y_train,
            accuracy_fn=train_accuracy_fn,
        )

    test_fn = functools.partial(
        supervised_learning_accuracy_evaluation,
        graph_structure=graph_structure,
        X=X_test,
        y=y_test,
        accuracy_fn=test_accuracy_fn,
    )
    return train_fn, test_fn


def prepare_scoring_fn(
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_test: jnp.ndarray,
    y_test: jnp.ndarray,
    graph_structure: GGP,
    const_optimizer: Optional[str] = None,
    long_const_optimization: bool = False,
    task: str = "regression",
) -> Callable:
    """
    Create a scoring function for symbolic regression model evaluation.

    This function constructs a combined scoring function that internally uses
    training and testing evaluation functions produced by
    `prepare_train_test_evaluation_fns`.
    The returned function is pre-configured with the provided datasets,
    graph structure, and optional constant-optimization strategy. It can be
    applied directly to a candidate program genotype (graph representation).

    Parameters
    ----------
    X_train : jnp.ndarray
        Training input features.
    y_train : jnp.ndarray
        Training target values.
    X_test : jnp.ndarray
        Test input features.
    y_test : jnp.ndarray
        Test target values.
    graph_structure : GGP
        A graph-based symbolic program structure to be used when evaluating
        candidate models.
    const_optimizer : str, optional
        Name of the constant-optimization strategy to use during training
        evaluation. See `prepare_train_test_evaluation_fns` for supported values.
    long_const_optimization : bool, optional
        Whether the constants are optimizer for longer or not.
    task : str, optional
        The task the scoring fn will be used for, either `regression`, `multi-regression` or `classification`.

    Returns
    -------
    Callable
        A partially applied scoring function that evaluates both training and
        test performance using internally prepared evaluation functions.

    Notes
    -----
    - The returned scoring function delegates actual evaluation to
      `regression_scoring_fn`.
    - The train/test evaluation functions used by the scorer are already bound
      to the supplied datasets and graph structure.
    """
    train_fn, test_fn = prepare_train_test_evaluation_fns(
        X_train,
        y_train,
        X_test,
        y_test,
        graph_structure,
        const_optimizer,
        long_const_optimization,
        task=task,
    )
    return functools.partial(
        supervised_learning_scoring_fn,
        train_set_evaluation_fn=train_fn,
        test_set_evaluation_fn=test_fn,
    )


def prepare_rescoring_fn(
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    graph_structure: GGP,
    task: str = "regression",
) -> Callable:
    """
    Create a scoring function for symbolic regression model evaluation.

    This function constructs a combined scoring function that internally uses
    training and testing evaluation functions produced by
    `prepare_train_test_evaluation_fns`.
    The returned function is pre-configured with the provided datasets,
    graph structure, and optional constant-optimization strategy. It can be
    applied directly to a candidate program genotype (graph representation).

    Parameters
    ----------
    X_train : jnp.ndarray
        Training input features.
    y_train : jnp.ndarray
        Training target values.
    graph_structure : GGP
        A graph-based symbolic program structure to be used when evaluating
        candidate models.
    task: the supervised learning task considered.

    Returns
    -------
    Callable
        A partially applied scoring function that evaluates only training
        performance using internally prepared evaluation functions.

    Notes
    -----
    - The returned scoring function delegates actual evaluation to
      `regression_scoring_fn`.
    - The train/test evaluation functions used by the scorer are already bound
      to the supplied datasets and graph structure.
    """
    train_fn, _ = prepare_train_test_evaluation_fns(
        X_train, y_train, None, None, graph_structure, task=task
    )
    test_fn = lambda x, y: (None, None)  # noqa: E731
    rescoring_fn = functools.partial(
        supervised_learning_scoring_fn,
        train_set_evaluation_fn=train_fn,
        test_set_evaluation_fn=test_fn,
    )
    return lambda x, y: rescoring_fn(x, y)[0]
