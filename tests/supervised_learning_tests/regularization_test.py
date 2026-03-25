import jax
import jax.numpy as jnp

from genepax.supervised_learning.regularization import (
    no_regularizer,
    no_snap,
    snap_to_pm_target,
    sticky_pm_target_regularizer,
)


def test_sticky_pm_target_regularizer_zero_penalty_at_target():
    weights = {"a": jnp.array([1.0, -1.0, 1.0])}
    penalty = sticky_pm_target_regularizer(weights, target=1)
    assert penalty == 0.0


def test_sticky_pm_target_regularizer_symmetric_penalty():
    weights1 = {"w": jnp.array(1.2)}
    weights2 = {"w": jnp.array(0.8)}

    p1 = sticky_pm_target_regularizer(weights1, target=1)
    p2 = sticky_pm_target_regularizer(weights2, target=1)

    assert jnp.allclose(p1, p2)


def test_sticky_pm_target_regularizer_nearest_target_selected():
    weights = {"w": jnp.array(0.9)}
    penalty = sticky_pm_target_regularizer(weights, target=1)

    expected = (0.9 - 1.0) ** 2
    assert jnp.allclose(penalty, expected)


def test_sticky_pm_target_regularizer_gradient_direction():
    def reg(w):
        return sticky_pm_target_regularizer({"w": w}, target=1)

    grad = jax.grad(reg)(jnp.array(1.3))
    assert grad > 0  # pushes back toward 1.0

    grad = jax.grad(reg)(jnp.array(-1.3))
    assert grad < 0  # pushes back toward -1.0


def test_sticky_pm_target_regularizer_jit_compiles():
    f = jax.jit(lambda w: sticky_pm_target_regularizer({"w": w}, target=1))
    out = f(jnp.array([0.5, 1.5]))
    assert jnp.isfinite(out)


def test_sticky_pm_target_regularizer_multiple_leaves():
    weights = {
        "a": jnp.array([1.0, 2.0]),
        "b": jnp.array([-1.5]),
    }
    penalty = sticky_pm_target_regularizer(weights, target=1)
    assert penalty > 0.0


def test_snap_to_pm_target_basic():
    weights = {"w": jnp.array([0.98, -1.01, 0.5])}
    snapped = snap_to_pm_target(weights, target=1, eps=0.05)
    expected = jnp.array([1.0, -1.0, 0.5])
    assert jnp.allclose(
        snapped["w"], expected
    ), "Weights within eps should snap to ±target"


def test_snap_outside_eps_remains():
    weights = {"w": jnp.array([0.9, -1.1])}
    snapped = snap_to_pm_target(weights, target=1, eps=0.05)
    expected = jnp.array([0.9, -1.1])
    assert jnp.allclose(snapped["w"], expected), "Weights outside eps should not snap"


def test_snap_custom_target():
    weights = {"w": jnp.array([2.04, -2.01, 0.5])}
    snapped = snap_to_pm_target(weights, target=2, eps=0.05)
    expected = jnp.array([2, -2, 0.5])
    assert jnp.allclose(
        snapped["w"], expected
    ), "Snapping should work for arbitrary ±target"


def test_snap_multiple_leaves():
    weights = {"a": jnp.array([0.99, -1.02]), "b": jnp.array([1.01, 0.4])}
    snapped = snap_to_pm_target(weights, target=1, eps=0.05)
    expected_a = jnp.array([1.0, -1.0])
    expected_b = jnp.array([1.0, 0.4])
    assert jnp.allclose(snapped["a"], expected_a)
    assert jnp.allclose(snapped["b"], expected_b)


def test_jit_snap_works():
    jit_snap = jax.jit(lambda w: snap_to_pm_target(w, target=1, eps=0.05))
    weights = {"x": jnp.array([0.98, -1.02])}
    snapped = jit_snap(weights)
    expected = jnp.array([1.0, -1.0])
    assert jnp.allclose(snapped["x"], expected)


def test_no_regularizer_returns_zero():
    weights = {"a": jnp.array([1.0, -1.0]), "b": jnp.array([0.5])}
    result = no_regularizer(weights)
    assert isinstance(result, jnp.ndarray), "Output should be a JAX array"
    assert result == 0.0, "No-op regularizer should return 0.0"


def test_no_regularizer_jit():
    jit_fn = jax.jit(no_regularizer)
    weights = {"a": jnp.array([2.0, -3.0])}
    result = jit_fn(weights)
    assert result == 0.0, "JIT compiled no_regularizer should return 0.0"


def test_no_snap_returns_same_weights():
    weights = {"a": jnp.array([1.0, -1.0]), "b": jnp.array([0.5])}
    result = no_snap(weights)
    for k in weights:
        assert jnp.allclose(
            result[k], weights[k]
        ), f"No-op snap should not change weights for key {k}"


def test_no_snap_jit():
    jit_fn = jax.jit(no_snap)
    weights = {"x": jnp.array([0.1, 0.9])}
    result = jit_fn(weights)
    for k in weights:
        assert jnp.allclose(
            result[k], weights[k]
        ), f"No-op snap JIT should not change weights for key {k}"
