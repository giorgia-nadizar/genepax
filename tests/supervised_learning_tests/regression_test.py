from functools import partial

import jax
import jax.numpy as jnp
import optax
from sklearn.model_selection import train_test_split

from genepax.gp.cartesian_genetic_programming import CGP
from genepax.supervised_learning.constants_optimization import (
    optimize_constants_with_cmaes,
    optimize_constants_with_lbfgs,
    optimize_constants_with_sgd,
)
from genepax.supervised_learning.scoring_functions import (
    compute_model_predictions,
    supervised_learning_accuracy_evaluation,
    supervised_learning_accuracy_evaluation_with_constants_optimization,
    supervised_learning_scoring_fn,
)


def test_prediction_shape():
    """
    Ensure that `_predict_regression_output` returns predictions with the
    correct shape.
    """
    n_inputs = 2
    n_data_points = 10
    n_outputs = 1
    cgp = CGP(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_nodes=5,
        weighted_functions=True,
        weighted_inputs=False,
    )
    key = jax.random.key(42)

    # init genome
    key, init_key = jax.random.split(key)
    cgp_genome = cgp.init(init_key)

    # generate some random points
    random_X = jax.random.uniform(key, (n_data_points, n_inputs))

    # simulate prediction
    jit_predict = jax.jit(partial(compute_model_predictions, graph_structure=cgp))
    prediction = jit_predict(random_X, cgp_genome)
    assert prediction.shape == (n_data_points, n_outputs)

    # simulate prediction with random weights
    key, weights_key = jax.random.split(key)
    cgp_weights = jax.random.uniform(key=weights_key, shape=(cgp.n_nodes,)) * 2 - 1
    weighted_prediction = jit_predict(
        random_X, cgp_genome, graph_weights={"functions": cgp_weights}
    )
    assert weighted_prediction.shape == (n_data_points, n_outputs)


def test_regression_accuracy_evaluation_shape():
    """Test that function returns accuracies with shape matching number of genotypes."""
    n_genotypes = 3
    n_samples = 5
    n_inputs = 2

    cgp = CGP(
        n_inputs=n_inputs,
        n_outputs=1,
        n_nodes=5,
    )
    key = jax.random.key(42)

    # init genomes
    init_key, key = jax.random.split(key)
    init_keys = jax.random.split(init_key, n_genotypes)
    genotypes = jax.vmap(jax.jit(cgp.init))(init_keys)

    # generate random data points
    x_key, y_key, key = jax.random.split(key, 3)
    X = jax.random.uniform(x_key, (n_samples, n_inputs))
    y = jax.random.uniform(y_key, (n_samples, 1))

    accuracies, returned_genotypes = supervised_learning_accuracy_evaluation(
        genotype=genotypes,
        key=key,
        graph_structure=cgp,
        X=X,
        y=y,
    )

    assert accuracies.shape[0] == n_genotypes
    assert returned_genotypes == genotypes


def test_regression_accuracy_evaluation_with_sgd_shape():
    """Test that function returns accuracies with shape matching number of genotypes."""
    n_genotypes = 3
    n_samples = 10
    n_inputs = 2

    cgp = CGP(
        n_inputs=n_inputs,
        n_outputs=1,
        n_nodes=10,
        weighted_functions=True,
        weighted_inputs=False,
    )
    key = jax.random.key(42)

    # init genomes
    init_key, key = jax.random.split(key)
    init_keys = jax.random.split(init_key, n_genotypes)
    genotypes = jax.vmap(jax.jit(cgp.init))(init_keys)

    # generate random data points
    x_key, y_key, key = jax.random.split(key, 3)
    X = jax.random.uniform(x_key, (n_samples, n_inputs))
    y = jax.random.uniform(y_key, (n_samples, 1))

    non_sgd_accuracies, non_sgd_returned_genotypes = (
        supervised_learning_accuracy_evaluation(
            genotype=genotypes,
            key=key,
            graph_structure=cgp,
            X=X,
            y=y,
        )
    )
    for optimizer in [optax.adam(1e-3), optax.rmsprop(1e-3, momentum=0.9)]:
        constants_opt_fn = partial(
            optimize_constants_with_sgd, batch_size=4, optimizer=optimizer
        )
        accuracies, returned_genotypes = (
            supervised_learning_accuracy_evaluation_with_constants_optimization(
                genotype=genotypes,
                key=key,
                graph_structure=cgp,
                X=X,
                y=y,
                constants_optimization_fn=constants_opt_fn,
            )
        )

        assert accuracies.shape[0] == n_genotypes
        assert all(
            jax.tree.leaves(
                jax.tree.map(
                    lambda x, y: jnp.allclose(x, y),
                    returned_genotypes["genes"],
                    genotypes["genes"],
                )
            )
        )
        assert all(accuracies > non_sgd_accuracies)


def test_regression_accuracy_evaluation_with_constants_optimization():
    """Test that function returns accuracies with shape matching number of genotypes."""
    n_genotypes = 3
    n_samples = 10
    n_inputs = 2

    cgp = CGP(
        n_inputs=n_inputs,
        n_outputs=1,
        n_nodes=10,
        weighted_functions=True,
        weighted_inputs=False,
    )
    key = jax.random.key(42)

    # init genomes
    init_key, key = jax.random.split(key)
    init_keys = jax.random.split(init_key, n_genotypes)
    genotypes = jax.vmap(jax.jit(cgp.init))(init_keys)

    # generate random data points
    x_key, y_key, key = jax.random.split(key, 3)
    X = jax.random.uniform(x_key, (n_samples, n_inputs))
    y = jax.random.uniform(y_key, (n_samples, 1))

    constants_opt_fn_1 = partial(
        optimize_constants_with_sgd, batch_size=4, optimizer=optax.adam(1e-3)
    )
    constants_opt_fn_2 = partial(
        optimize_constants_with_sgd,
        batch_size=4,
        optimizer=optax.rmsprop(1e-3, momentum=0.9),
    )
    constants_opt_fn_3 = optimize_constants_with_lbfgs
    constants_opt_fn_4 = optimize_constants_with_cmaes

    for constants_opt_fn in [
        constants_opt_fn_1,
        constants_opt_fn_2,
        constants_opt_fn_3,
        constants_opt_fn_4,
    ]:
        accuracies, returned_genotypes = (
            supervised_learning_accuracy_evaluation_with_constants_optimization(
                genotype=genotypes,
                key=key,
                graph_structure=cgp,
                X=X,
                y=y,
                constants_optimization_fn=constants_opt_fn,
            )
        )

        non_opt_accuracies, _ = supervised_learning_accuracy_evaluation(
            genotype=genotypes,
            key=key,
            graph_structure=cgp,
            X=X,
            y=y,
        )

        assert accuracies.shape[0] == n_genotypes
        assert all(
            jax.tree.leaves(
                jax.tree.map(
                    lambda x, y: jnp.allclose(x, y),
                    returned_genotypes["genes"],
                    genotypes["genes"],
                )
            )
        )
        assert all(accuracies > non_opt_accuracies)


def test_regression_scoring_fn():
    """Test that the regression scoring function works."""
    n_genotypes = 3
    n_samples = 5
    n_inputs = 2

    cgp = CGP(
        n_inputs=n_inputs,
        n_outputs=1,
        n_nodes=5,
    )
    key = jax.random.key(42)

    # init genomes
    init_key, key = jax.random.split(key)
    init_keys = jax.random.split(init_key, n_genotypes)
    genotypes = jax.vmap(jax.jit(cgp.init))(init_keys)

    # generate random data points
    x_key, y_key, key = jax.random.split(key, 3)
    X = jax.random.uniform(x_key, (n_samples, n_inputs))
    y = jax.random.uniform(y_key, (n_samples, 1))

    # define train and test fn
    train_fn = partial(
        supervised_learning_accuracy_evaluation, graph_structure=cgp, X=X, y=y
    )
    test_fn = train_fn

    # compute scoring fn
    fitness, extra_scores = supervised_learning_scoring_fn(
        genotypes, key, train_fn, test_fn
    )
    assert len(fitness) == n_genotypes
    assert jnp.array_equal(fitness, extra_scores["test_accuracy"])
    assert genotypes == extra_scores["updated_params"]


def test_regression_scoring_fn_with_sgd():
    """Test that the regression scoring function works with SGD."""
    n_genotypes = 3
    n_samples = 10
    n_inputs = 2

    cgp = CGP(
        n_inputs=n_inputs,
        n_outputs=1,
        n_nodes=10,
        weighted_functions=True,
    )
    key = jax.random.key(42)

    # init genomes
    init_key, key = jax.random.split(key)
    init_keys = jax.random.split(init_key, n_genotypes)
    genotypes = jax.vmap(jax.jit(cgp.init))(init_keys)

    # generate random data points
    x_key, y_key, key = jax.random.split(key, 3)
    X = jax.random.uniform(x_key, (n_samples, n_inputs))
    y = jax.random.uniform(y_key, (n_samples, 1))

    # define train and test fn
    for reset_weights in [True, False]:
        for batch_size in [4, None]:
            constants_opt_fn = partial(
                optimize_constants_with_sgd, batch_size=batch_size
            )
            train_fn = partial(
                supervised_learning_accuracy_evaluation_with_constants_optimization,
                graph_structure=cgp,
                X=X,
                y=y,
                constants_optimization_fn=constants_opt_fn,
                reset_weights=reset_weights,
            )
            test_fn = partial(
                supervised_learning_accuracy_evaluation, graph_structure=cgp, X=X, y=y
            )

            # compute scoring fn
            fitness, extra_scores = supervised_learning_scoring_fn(
                genotypes, key, train_fn, test_fn
            )
            assert len(fitness) == n_genotypes
            assert jnp.array_equal(fitness, extra_scores["test_accuracy"])
            assert not all(
                jax.tree.leaves(
                    jax.tree.map(
                        lambda x, y: jnp.allclose(x, y),
                        genotypes,
                        extra_scores["updated_params"],
                    )
                )
            )


def test_train_and_test_set_accuracy():
    """Test that the train and test set accuracy works for data generated by the same
    genome we test on."""
    for n_inputs in range(2, 10):
        n_total_samples = 100
        cgp = CGP(
            n_inputs=n_inputs,
            n_outputs=1,
            weighted_inputs=True,
            weighted_functions=True,
        )
        key = jax.random.PRNGKey(42)
        init_key, key = jax.random.split(key)
        random_graph = jax.vmap(cgp.init)(init_key[None, :])

        data_key, key = jax.random.split(key)
        X = jax.random.uniform(data_key, (n_total_samples, n_inputs))
        y = jax.vmap(compute_model_predictions, in_axes=(None, 0, None))(
            X, random_graph, cgp
        )[1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        train_fn = partial(
            supervised_learning_accuracy_evaluation,
            graph_structure=cgp,
            X=X_train,
            y=y_train,
        )
        test_fn = partial(
            supervised_learning_accuracy_evaluation,
            graph_structure=cgp,
            X=X_test,
            y=y_test,
        )

        fitness, extra_scores = supervised_learning_scoring_fn(
            random_graph, key, train_fn, test_fn
        )
        assert jnp.isclose(fitness, 1.0)
        assert jnp.isclose(extra_scores["test_accuracy"], 1.0)
