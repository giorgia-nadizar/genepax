from functools import partial

import jax
import jax.numpy as jnp
import optax

from genepax.gp.cartesian_genetic_programming import CGP
from genepax.supervised_learning.constants_optimization import (
    optimize_constants_with_cmaes,
    optimize_constants_with_lbfgs,
    optimize_constants_with_sgd,
)
from genepax.supervised_learning.regularization import (
    snap_to_pm_target,
    sticky_pm_target_regularizer,
)
from genepax.supervised_learning.scoring_functions import compute_model_predictions


def test_cmaes_output_shape_and_type():
    """Ensure the constants optimization with CMAES have the correct shape."""
    n_genomes, n_features, n_samples = 3, 5, 20
    X = jnp.ones((n_samples, n_features))
    y = jnp.ones((n_samples,))

    cgp = CGP(
        n_inputs=n_features,
        n_outputs=1,
        n_nodes=5,
        weighted_functions=True,
    )
    key = jax.random.key(42)

    # init genomes
    init_key, key = jax.random.split(key)
    init_keys = jax.random.split(init_key, n_genomes)
    genotypes = jax.vmap(jax.jit(cgp.init))(init_keys)

    graph_weights = cgp.get_weights(genotypes)
    prediction_fn = jax.jit(partial(compute_model_predictions, graph_structure=cgp))
    for minibatch_size in [2, 10, 20, 50]:
        optimized_weights = optimize_constants_with_cmaes(
            graph_weights,
            genotypes,
            key,
            X,
            y,
            prediction_fn,
            mini_batch_size=minibatch_size,
        )
        # same keys
        assert set(optimized_weights.keys()) == set(graph_weights.keys())
        # check array shapes
        for k in optimized_weights:
            assert optimized_weights[k].shape == graph_weights[k].shape
            assert isinstance(optimized_weights[k], jnp.ndarray)


def test_lbfgs_output_shape_and_type():
    """Ensure the constants optimization with LBFGS have the correct shape."""
    n_genomes, n_features, n_samples = 3, 5, 20
    X = jnp.ones((n_samples, n_features))
    y = jnp.ones((n_samples,))

    cgp = CGP(
        n_inputs=n_features,
        n_outputs=1,
        n_nodes=5,
        weighted_functions=True,
    )
    key = jax.random.key(42)

    # init genomes
    init_key, key = jax.random.split(key)
    init_keys = jax.random.split(init_key, n_genomes)
    genotypes = jax.vmap(jax.jit(cgp.init))(init_keys)

    graph_weights = cgp.get_weights(genotypes)
    prediction_fn = jax.jit(partial(compute_model_predictions, graph_structure=cgp))
    optimized_weights = optimize_constants_with_lbfgs(
        graph_weights, genotypes, key, X, y, prediction_fn
    )
    # same keys
    assert set(optimized_weights.keys()) == set(graph_weights.keys())
    # check array shapes
    for k in optimized_weights:
        assert optimized_weights[k].shape == graph_weights[k].shape
        assert isinstance(optimized_weights[k], jnp.ndarray)


def test_sgd_output_shape_and_type():
    """Ensure the constants optimization with sgd have the correct shape."""
    n_genomes, n_features, n_samples = 3, 5, 20
    X = jnp.ones((n_samples, n_features))
    y = jnp.ones((n_samples,))

    cgp = CGP(
        n_inputs=n_features,
        n_outputs=1,
        n_nodes=5,
        weighted_functions=True,
    )
    key = jax.random.key(42)

    # init genomes
    init_key, key = jax.random.split(key)
    init_keys = jax.random.split(init_key, n_genomes)
    genotypes = jax.vmap(jax.jit(cgp.init))(init_keys)

    graph_weights = cgp.get_weights(genotypes)
    prediction_fn = jax.jit(partial(compute_model_predictions, graph_structure=cgp))
    for optimizer in [optax.adam(1e-3), optax.rmsprop(1e-3, momentum=0.9)]:
        optimized_weights = optimize_constants_with_sgd(
            graph_weights,
            genotypes,
            key,
            X,
            y,
            prediction_fn,
            optimizer=optimizer,
            batch_size=n_samples,
        )

        # same keys
        assert set(optimized_weights.keys()) == set(graph_weights.keys())
        # check array shapes
        for k in optimized_weights:
            assert optimized_weights[k].shape == graph_weights[k].shape
            assert isinstance(optimized_weights[k], jnp.ndarray)


def test_multiregression_constants_optimization():
    """Ensure the constants optimization works also for multiple outputs."""
    n_genomes, n_features, n_samples, n_outputs = 3, 5, 20, 3
    X = jnp.ones((n_samples, n_features))
    y = jnp.ones((n_samples, n_outputs))

    cgp = CGP(
        n_inputs=n_features,
        n_outputs=n_outputs,
        n_nodes=10,
        weighted_functions=True,
    )
    key = jax.random.key(42)

    # init genomes
    init_key, key = jax.random.split(key)
    init_keys = jax.random.split(init_key, n_genomes)
    genotypes = jax.vmap(jax.jit(cgp.init))(init_keys)

    graph_weights = cgp.get_weights(genotypes)
    prediction_fn = jax.jit(partial(compute_model_predictions, graph_structure=cgp))
    for optimizer in [optax.adam(1e-3), optax.rmsprop(1e-3, momentum=0.9)]:
        optimized_weights = optimize_constants_with_sgd(
            graph_weights,
            genotypes,
            key,
            X,
            y,
            prediction_fn,
            optimizer=optimizer,
            batch_size=n_samples,
        )

        # same keys
        assert set(optimized_weights.keys()) == set(graph_weights.keys())
        # check array shapes
        for k in optimized_weights:
            assert optimized_weights[k].shape == graph_weights[k].shape
            assert isinstance(optimized_weights[k], jnp.ndarray)


def test_sgd_push_to_pm_target():
    """Ensure the constants optimization with sgd have the correct shape."""
    n_genomes, n_features, n_samples = 3, 5, 20
    X = jnp.ones((n_samples, n_features))
    y = jnp.ones((n_samples,))

    for wgt in [True, False]:
        cgp = CGP(
            n_inputs=n_features,
            n_outputs=1,
            n_nodes=5,
            weighted_functions=wgt,
            weighted_inputs=not wgt,
        )
        key = jax.random.key(42)

        # init genomes
        init_key, key = jax.random.split(key)
        init_keys = jax.random.split(init_key, n_genomes)
        genotypes = jax.vmap(jax.jit(cgp.init))(init_keys)

        graph_weights = cgp.get_weights(genotypes)
        prediction_fn = jax.jit(partial(compute_model_predictions, graph_structure=cgp))
        reg_loss_fn = sticky_pm_target_regularizer
        reg_update_fn = partial(snap_to_pm_target, target=1, eps=1e-1)
        optimized_weights = optimize_constants_with_sgd(
            graph_weights,
            genotypes,
            key,
            X,
            y,
            prediction_fn,
            batch_size=n_samples,
            regularization_loss_fn=reg_loss_fn,
            regularization_update_fn=reg_update_fn,
            n_gradient_steps=100,
            optimizer=optax.adam(1e-1),
            regularization_strength=10,
        )
        for w in optimized_weights.values():
            assert jnp.allclose(jnp.abs(w), 1)
