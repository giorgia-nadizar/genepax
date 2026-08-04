import jax.numpy as jnp

from distillation.evaluation import finite_prefix_mask, finite_transition_mask


def test_finite_transition_mask_checks_observations_and_teacher_actions():
    X = jnp.asarray([[0.0, 1.0], [jnp.nan, 0.0], [0.0, 1.0]])
    y = jnp.asarray([[0.0], [0.0], [jnp.inf]])

    mask = finite_transition_mask(X, y)

    assert jnp.array_equal(mask, jnp.asarray([True, False, False]))


def test_finite_prefix_mask_rejects_everything_after_first_invalid_state():
    X = jnp.asarray([[0.0], [jnp.nan], [1.0]])
    y = jnp.asarray([[0.0], [0.0], [0.0]])

    mask = finite_prefix_mask(X, y)

    assert jnp.array_equal(mask, jnp.asarray([True, False, False]))
