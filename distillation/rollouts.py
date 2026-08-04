"""Shared Brax rollout primitives used by distillation and policy search."""

from collections.abc import Callable

import jax
import jax.numpy as jnp


def sanitize_action(action: jax.Array) -> jax.Array:
    """Maps a policy action to the finite normalized Brax action domain."""
    action = jnp.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
    return jnp.clip(action, -1.0, 1.0)


def valid_transition_mask(dones: jax.Array) -> jax.Array:
    """Returns a mask that excludes transitions after the first terminal one."""
    prior_dones = jnp.concatenate([jnp.zeros_like(dones[:1]), dones[:-1]])
    return (jnp.cumsum(prior_dones) == 0).astype(jnp.float32)


def masked_return(rewards: jax.Array, dones: jax.Array) -> jax.Array:
    return jnp.sum(rewards * valid_transition_mask(dones))


def rollout(
    env,
    key: jax.Array,
    action_fn: Callable[[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
    num_steps: int,
):
    """Rolls out ``action_fn`` and returns observations, labels and rewards.

    ``action_fn(observation, key, step)`` returns ``(environment_action,
    label_action)``.  Keeping both values makes this primitive useful for
    evaluation as well as DAgger dataset collection.
    """
    state = env.reset(key)

    def step_fn(carry, step):
        state, key = carry
        key, action_key = jax.random.split(key)
        action, label = action_fn(state.obs, action_key, step)
        next_state = env.step(state, sanitize_action(action))
        return (next_state, key), (
            state.obs, label, next_state.reward, next_state.done
        )

    _, trace = jax.lax.scan(
        step_fn, (state, key), jnp.arange(num_steps), length=num_steps
    )
    return trace
