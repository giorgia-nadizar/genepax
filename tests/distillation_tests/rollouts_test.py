import jax.numpy as jnp

from distillation.rollouts import sanitize_action, valid_transition_mask


def test_sanitize_action_replaces_non_finite_values_and_clips_range():
    action = jnp.asarray([jnp.nan, jnp.inf, -jnp.inf, 2.0, -2.0, 0.25])

    sanitized = sanitize_action(action)

    assert jnp.array_equal(
        sanitized,
        jnp.asarray([0.0, 1.0, -1.0, 1.0, -1.0, 0.25]),
    )


def test_valid_transition_mask_keeps_terminal_transition_only():
    dones = jnp.asarray([False, False, True, True, True])

    mask = valid_transition_mask(dones)

    # The action that leads to termination is valid; transitions attempted
    # from the terminal state afterward are not.
    assert jnp.array_equal(mask, jnp.asarray([1, 1, 1, 0, 0]))
