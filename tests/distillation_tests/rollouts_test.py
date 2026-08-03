import jax.numpy as jnp

from distillation.rollouts import valid_transition_mask


def test_valid_transition_mask_keeps_terminal_transition_only():
    dones = jnp.asarray([False, False, True, True, True])

    mask = valid_transition_mask(dones)

    # The action that leads to termination is valid; transitions attempted
    # from the terminal state afterward are not.
    assert jnp.array_equal(mask, jnp.asarray([1, 1, 1, 0, 0]))
