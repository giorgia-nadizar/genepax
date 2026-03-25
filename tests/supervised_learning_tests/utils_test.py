import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from genepax.gp.cartesian_genetic_programming import CGP
from genepax.supervised_learning.constants_optimization import (
    optimize_constants_with_cmaes,
    optimize_constants_with_lbfgs,
    optimize_constants_with_sgd,
)
from genepax.supervised_learning.dataset_utils import load_dataset
from genepax.supervised_learning.metrics import (
    classification_accuracy,
    rrmse_per_target,
)
from genepax.supervised_learning.scoring_functions import (
    supervised_learning_accuracy_evaluation,
    supervised_learning_accuracy_evaluation_with_constants_optimization,
    supervised_learning_scoring_fn,
)
from genepax.supervised_learning.utils import (
    prepare_rescoring_fn,
    prepare_scoring_fn,
    prepare_train_test_evaluation_fns,
)


@pytest.fixture
def sample_data():
    X_train = jnp.ones((4, 3))
    y_train = jnp.ones((4, 1))
    X_test = jnp.ones((2, 3))
    y_test = jnp.ones((2, 1))
    return X_train, y_train, X_test, y_test


@pytest.fixture
def sample_data_mtr():
    X_train = jnp.ones((4, 3))
    y_train = jnp.ones((4, 2))
    X_test = jnp.ones((2, 3))
    y_test = jnp.ones((2, 2))
    return X_train, y_train, X_test, y_test


def test_prepare_eval_fns_supported_tasks(sample_data):
    X_train, y_train, X_test, y_test = sample_data

    for task in [
        "regression",
        "classification",
        "multi-regression",
        "multi_regression",
        "multiregression",
    ]:
        prepare_train_test_evaluation_fns(
            X_train, y_train, X_test, y_test, graph_structure=None, task=task
        )

    with pytest.raises(NotImplementedError):
        prepare_train_test_evaluation_fns(
            X_train, y_train, X_test, y_test, graph_structure=None, task="fake_task"
        )


def test_prepare_eval_fns_default_behavior(sample_data):
    X_train, y_train, X_test, y_test = sample_data

    train_fn, test_fn = prepare_train_test_evaluation_fns(
        X_train, y_train, X_test, y_test, graph_structure=None
    )

    assert train_fn.func is supervised_learning_accuracy_evaluation
    assert test_fn.func is supervised_learning_accuracy_evaluation


@pytest.mark.parametrize("opt", ["automl0", "mutation"])
def test_prepare_eval_fns_no_optimizer_aliases(sample_data, opt):
    X_train, y_train, X_test, y_test = sample_data

    train_fn, _ = prepare_train_test_evaluation_fns(
        X_train, y_train, X_test, y_test, const_optimizer=opt, graph_structure=None
    )

    assert train_fn.func is supervised_learning_accuracy_evaluation


def test_prepare_eval_fns_adam_optimizer(sample_data):
    X_train, y_train, X_test, y_test = sample_data

    train_fn, _ = prepare_train_test_evaluation_fns(
        X_train, y_train, X_test, y_test, const_optimizer="adam", graph_structure=None
    )

    assert (
        train_fn.func
        is supervised_learning_accuracy_evaluation_with_constants_optimization
    )
    opt_fn = train_fn.keywords["constants_optimization_fn"]
    assert opt_fn.func is optimize_constants_with_sgd
    assert opt_fn.keywords["n_gradient_steps"] == 100


def test_prepare_eval_fns_rmsprop_optimizer(sample_data):
    X_train, y_train, X_test, y_test = sample_data

    train_fn, _ = prepare_train_test_evaluation_fns(
        X_train,
        y_train,
        X_test,
        y_test,
        const_optimizer="rmsprop",
        graph_structure=None,
    )

    assert (
        train_fn.func
        is supervised_learning_accuracy_evaluation_with_constants_optimization
    )
    opt_fn = train_fn.keywords["constants_optimization_fn"]
    assert opt_fn.func is optimize_constants_with_sgd
    assert opt_fn.keywords["n_gradient_steps"] == 120


def test_prepare_eval_fns_cmaes_optimizer(sample_data):
    X_train, y_train, X_test, y_test = sample_data

    train_fn, _ = prepare_train_test_evaluation_fns(
        X_train, y_train, X_test, y_test, const_optimizer="cmaes", graph_structure=None
    )

    assert (
        train_fn.func
        is supervised_learning_accuracy_evaluation_with_constants_optimization
    )
    opt_fn = train_fn.keywords["constants_optimization_fn"]
    assert opt_fn.func is optimize_constants_with_cmaes
    assert opt_fn.keywords["max_iter"] == 20


def test_prepare_eval_fns_lbfgs_optimizer(sample_data):
    X_train, y_train, X_test, y_test = sample_data

    train_fn, _ = prepare_train_test_evaluation_fns(
        X_train, y_train, X_test, y_test, const_optimizer="lbfgs", graph_structure=None
    )

    assert (
        train_fn.func
        is supervised_learning_accuracy_evaluation_with_constants_optimization
    )
    opt_fn = train_fn.keywords["constants_optimization_fn"]
    assert opt_fn.func is optimize_constants_with_lbfgs
    assert opt_fn.keywords["max_iter"] == 5


def test_prepare_eval_fns_test_fn_always_simple(sample_data):
    X_train, y_train, X_test, y_test = sample_data

    _, test_fn = prepare_train_test_evaluation_fns(
        X_train, y_train, X_test, y_test, const_optimizer="adam", graph_structure=None
    )

    assert test_fn.func is supervised_learning_accuracy_evaluation


def test_prepare_eval_fns_test_fn_mtr(sample_data_mtr):
    X_train, y_train, X_test, y_test = sample_data_mtr

    _, test_fn = prepare_train_test_evaluation_fns(
        X_train,
        y_train,
        X_test,
        y_test,
        const_optimizer="adam",
        graph_structure=None,
        task="multi-regression",
    )

    assert test_fn.func is supervised_learning_accuracy_evaluation
    assert test_fn.keywords["accuracy_fn"].func is rrmse_per_target
    assert jnp.all(test_fn.keywords["accuracy_fn"].keywords["y_train"] == y_train)


def test_prepare_eval_fns_test_fn_classification(sample_data):
    X_train, y_train, X_test, y_test = sample_data

    _, test_fn = prepare_train_test_evaluation_fns(
        X_train,
        y_train,
        X_test,
        y_test,
        const_optimizer="adam",
        graph_structure=None,
        task="classification",
    )

    assert test_fn.func is supervised_learning_accuracy_evaluation
    assert test_fn.keywords["accuracy_fn"] is classification_accuracy


def test_prepare_scoring_fn_returns_partial(sample_data):
    X_train, y_train, X_test, y_test = sample_data

    scoring_fn = prepare_scoring_fn(
        X_train, y_train, X_test, y_test, graph_structure=None
    )

    assert isinstance(scoring_fn, functools.partial)
    assert scoring_fn.func is supervised_learning_scoring_fn


def test_prepare_scoring_fn_partial_contains_train_and_test_fns(sample_data):
    X_train, y_train, X_test, y_test = sample_data

    scoring_fn = prepare_scoring_fn(
        X_train, y_train, X_test, y_test, graph_structure=None
    )

    train_fn = scoring_fn.keywords["train_set_evaluation_fn"]
    test_fn = scoring_fn.keywords["test_set_evaluation_fn"]

    # default: no constants optimizer → simple evaluation
    assert train_fn.func is supervised_learning_accuracy_evaluation
    assert test_fn.func is supervised_learning_accuracy_evaluation


def test_prepare_scoring_fn_constants_optimizer_flow(sample_data):
    X_train, y_train, X_test, y_test = sample_data

    scoring_fn = prepare_scoring_fn(
        X_train,
        y_train,
        X_test,
        y_test,
        graph_structure=None,
        const_optimizer="adam",
    )

    train_fn = scoring_fn.keywords["train_set_evaluation_fn"]
    test_fn = scoring_fn.keywords["test_set_evaluation_fn"]

    assert (
        train_fn.func
        is supervised_learning_accuracy_evaluation_with_constants_optimization
    )
    assert test_fn.func is supervised_learning_accuracy_evaluation  # always simple


def test_prepare_rescoring_fn(sample_data):
    X_train, y_train, _, _ = sample_data

    cgp = CGP(n_inputs=X_train.shape[1], n_outputs=1)
    rescoring_fn = prepare_rescoring_fn(
        X_train,
        y_train,
        graph_structure=cgp,
    )

    key = jax.random.key(0)
    key, pop_key = jax.random.split(key)
    pop_keys = jax.random.split(pop_key, 10)
    population = jax.vmap(cgp.init)(pop_keys)
    population["genes"]["outputs"] = jnp.zeros_like(population["genes"]["outputs"])

    resulting_fitness = rescoring_fn(population, key)
    assert jnp.allclose(resulting_fitness, 0.0)


@pytest.mark.parametrize(
    "dataset_name, expected_targets",
    [
        ("diabetes", 1),
        ("feynman_I_6_2", 1),
        ("nikuradse_1", 1),
        ("mtr/edm", 2),  # edm has 2 targets
    ],
)
@pytest.mark.parametrize("scale_x", [True, False])
@pytest.mark.parametrize("scale_y", [True, False])
def test_load_dataset_shapes(
    dataset_name, expected_targets, scale_x, scale_y, monkeypatch
):
    """
    Test that load_dataset returns arrays of correct shape.
    """

    # ---- Mock feynman TSV ----
    if "feynman" in dataset_name:
        import pandas as pd

        df_mock = pd.DataFrame(
            np.random.rand(100, 5),
            columns=[f"x{i}" for i in range(4)] + ["y"],
        )
        monkeypatch.setattr("pandas.read_csv", lambda *args, **kwargs: df_mock)

    # ---- Mock custom train/test CSV ----
    elif "nikuradse" in dataset_name:
        import pandas as pd

        df_train = pd.DataFrame(
            np.random.rand(80, 5),
            columns=[f"x{i}" for i in range(4)] + ["target"],
        )
        df_test = pd.DataFrame(
            np.random.rand(20, 5),
            columns=[f"x{i}" for i in range(4)] + ["target"],
        )

        def mock_read_csv(path, *args, **kwargs):
            return df_train if "train" in path else df_test

        monkeypatch.setattr("pandas.read_csv", mock_read_csv)

    # ---- Mock MTR dataset ----
    elif "mtr" in dataset_name:
        import pandas as pd

        # statistics.csv
        statistics = pd.DataFrame(
            {
                "name": ["edm"],
                "targets": [2],
            }
        )
        monkeypatch.setattr("pandas.read_csv", lambda *args, **kwargs: statistics)

        # ARFF loader
        X = np.random.rand(100, 16)
        y = np.random.rand(100, 2)
        data = np.hstack([X, y])

        def mock_loadarff(*args, **kwargs):
            return (data, None)

        monkeypatch.setattr(
            "scipy.io.arff.loadarff",
            mock_loadarff,
        )

    # ---- Call loader ----
    X_train, X_test, y_train, y_test = load_dataset(
        dataset_name=dataset_name,
        scale_x=scale_x,
        scale_y=scale_y,
        test_split=0.2,
        random_state=42,
    )

    # ---- Shape checks ----
    assert X_train.ndim == 2
    assert X_test.ndim == 2
    assert y_train.ndim == 2
    assert y_test.ndim == 2

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train.shape[1] == X_test.shape[1]

    assert y_train.shape == (X_train.shape[0], expected_targets)
    assert y_test.shape == (X_test.shape[0], expected_targets)

    # ---- Scaling checks ----
    if scale_x:
        assert np.allclose(X_train.mean(axis=0), 0, atol=1e-6)
        assert np.allclose(X_train.std(axis=0), 1, atol=1e-6)

    if scale_y:
        assert np.allclose(y_train.mean(axis=0), 0, atol=1e-6)
        assert np.allclose(y_train.std(axis=0), 1, atol=1e-6)


def test_invalid_dataset():
    """
    Test that an invalid dataset raises an error (FileNotFoundError or similar)
    """
    with pytest.raises(FileNotFoundError):
        load_dataset("nonexistent_dataset", scale_x=False, scale_y=False)
