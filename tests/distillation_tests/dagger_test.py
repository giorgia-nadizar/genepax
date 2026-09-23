"""Tests for the plain DAgger implementation."""

import jax
import jax.numpy as jnp
import pytest

from distillation.fit_imitation import apply_linear_scaling, imitation_loss
from distillation.q_dagger import compute_q_dagger_weights
from distillation_experiments.scripts.dagger import (
    select_rows,
)
from distillation_experiments.scripts.dagger_linear import (
    fit_linear_student,
    linear_imitation_loss,
    split_train_test,
)


def test_imitation_loss_is_mean_squared_action_error() -> None:
    expert = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    student = jnp.asarray([[0.0, 2.0], [5.0, 4.0]])

    assert imitation_loss(expert, student) == pytest.approx(1.25)
    assert imitation_loss(
        expert, student, jnp.asarray([1.0, 0.0])
    ) == pytest.approx(0.5)


def test_apply_linear_scaling_uses_stored_slopes_and_intercepts() -> None:
    actions = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    # Two slopes followed by two intercepts.
    weights = jnp.asarray([2.0, -1.0, 0.5, 3.0])

    scaled = apply_linear_scaling(actions, weights)

    assert jnp.allclose(scaled, jnp.asarray([[2.5, 1.0], [6.5, -1.0]]))


def test_apply_linear_scaling_rejects_wrong_custom_weight_count() -> None:
    with pytest.raises(ValueError, match="two custom weights"):
        apply_linear_scaling(jnp.ones((2, 1)), jnp.ones(3))


def test_select_rows_keeps_features_and_labels_aligned() -> None:
    X = jnp.arange(20).reshape(10, 2)
    y = X[:, :1] * 2

    selected_X, selected_y = select_rows(X, y, 4, jax.random.key(0))

    assert len(selected_X) == 4
    assert jnp.array_equal(selected_y, selected_X[:, :1] * 2)


def test_select_rows_rejects_insufficient_data() -> None:
    with pytest.raises(ValueError, match="3 samples"):
        select_rows(
            jnp.zeros((3, 2)),
            jnp.zeros((3, 1)),
            4,
            jax.random.key(0),
        )


def test_q_dagger_weights_are_normalized_q_gaps() -> None:
    observations = jnp.asarray([[1.0], [2.0]])
    expert_actions = jnp.asarray([[1.0], [0.5]])

    def q_value(observation: jax.Array, action: jax.Array) -> jax.Array:
        return observation[:, 0] - jnp.square(action[:, 0] - 1.0)

    weights, raw_weights = compute_q_dagger_weights(
        observations, expert_actions, q_value, action_grid_size=3, batch_size=1
    )

    assert jnp.allclose(raw_weights, jnp.asarray([4.0, 3.75]))
    assert jnp.mean(weights) == pytest.approx(1.0)


def test_q_dagger_weights_support_vector_actions_deterministically() -> None:
    observations = jnp.asarray([[0.0], [1.0]])
    expert_actions = jnp.asarray([[0.0, 0.0], [0.5, -0.5]])

    def q_value(observation: jax.Array, action: jax.Array) -> jax.Array:
        del observation
        return -jnp.sum(jnp.square(action), axis=-1)

    weights, raw_weights = compute_q_dagger_weights(
        observations, expert_actions, q_value, action_grid_size=17, batch_size=1
    )
    repeated_weights, repeated_raw = compute_q_dagger_weights(
        observations, expert_actions, q_value, action_grid_size=17, batch_size=2
    )

    assert weights.shape == (2,)
    assert jnp.mean(weights) == pytest.approx(1.0)
    assert jnp.all(raw_weights > 0)
    assert jnp.allclose(weights, repeated_weights)
    assert jnp.allclose(raw_weights, repeated_raw)


def test_linear_student_honors_sample_weights() -> None:
    X = jnp.asarray([[0.0], [1.0], [2.0]])
    y = jnp.asarray([[0.0], [1.0], [100.0]])

    unweighted_coefficients, _, _ = fit_linear_student(X, y)
    weighted_coefficients, _, _ = fit_linear_student(
        X, y, jnp.asarray([1.0, 1.0, 0.0])
    )

    assert weighted_coefficients[0, 0] == pytest.approx(1.0)
    assert weighted_coefficients[0, 0] < unweighted_coefficients[0, 0]


def test_linear_test_loss_uses_held_out_rows_and_weights() -> None:
    coefficients = jnp.asarray([[2.0]])
    intercept = jnp.asarray([0.0])
    X = jnp.asarray([[1.0], [2.0]])
    y = jnp.asarray([[1.0], [4.0]])

    loss = linear_imitation_loss(
        coefficients, intercept, X, y, jnp.asarray([1.0, 3.0])
    )

    assert loss == pytest.approx(0.25)


def test_linear_train_test_split_is_disjoint_and_aligned() -> None:
    X = jnp.arange(20).reshape(10, 2)
    y = X[:, :1] * 2
    weights = jnp.arange(10, dtype=jnp.float32) + 1

    train_X, train_y, train_weights, test_X, test_y, test_weights = (
        split_train_test(X, y, weights, 0.2, jax.random.key(3))
    )

    assert len(train_X) == 8
    assert len(test_X) == 2
    assert jnp.array_equal(train_y, train_X[:, :1] * 2)
    assert jnp.array_equal(test_y, test_X[:, :1] * 2)
    assert set(map(tuple, train_X.tolist())).isdisjoint(
        set(map(tuple, test_X.tolist()))
    )
    assert len(train_weights) == 8
    assert len(test_weights) == 2
