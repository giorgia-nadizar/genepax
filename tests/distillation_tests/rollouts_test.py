import jax
import jax.numpy as jnp
import numpy as np

from distillation.rollouts import masked_return


def test_return_ignores_nonfinite_rewards_after_terminal_transition():
    rewards = jnp.array([1., 2., jnp.nan, jnp.inf])
    dones = jnp.array([0., 1., 0., jnp.nan])
    assert float(jax.jit(masked_return)(rewards, dones)) == 3.


def test_return_preserves_nonfinite_reward_in_valid_episode():
    rewards = jnp.array([1., jnp.nan, 3.])
    dones = jnp.array([0., 1., 0.])
    assert np.isnan(float(jax.jit(masked_return)(rewards, dones)))
