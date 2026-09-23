"""State-importance weights used by Q-DAgger imitation objectives."""

from collections.abc import Callable
from typing import Tuple

import jax
import jax.numpy as jnp


def compute_q_dagger_weights(
    observations: jax.Array,
    expert_actions: jax.Array,
    q_value_estimator: Callable[[jax.Array, jax.Array], jax.Array],
    action_grid_size: int,
    batch_size: int,
) -> Tuple[jax.Array, jax.Array]:
    """Approximate expert-to-worst-action Q gaps.

    One-dimensional actions retain the original dense grid.  For vector
    actions, a deterministic candidate set combines axis-aligned extremes
    with uniformly distributed joint actions.  This avoids the exponential
    cost of a Cartesian grid while preserving reproducibility.
    """
    action_dimensions = expert_actions.shape[1]
    if action_dimensions == 1:
        action_grid = jnp.linspace(-1.0, 1.0, action_grid_size)[:, None]
    else:
        identity = jnp.eye(action_dimensions, dtype=expert_actions.dtype)
        anchors = jnp.concatenate(
            [identity, -identity, jnp.ones((1, action_dimensions)),
             -jnp.ones((1, action_dimensions)),
             jnp.zeros((1, action_dimensions))],
            axis=0,
        )
        anchor_count = min(len(anchors), action_grid_size)
        random_count = action_grid_size - anchor_count
        random_actions = jax.random.uniform(
            jax.random.key(0),
            (random_count, action_dimensions),
            minval=-1.0,
            maxval=1.0,
            dtype=expert_actions.dtype,
        )
        action_grid = jnp.concatenate(
            [anchors[:anchor_count], random_actions], axis=0
        )
    weights = []
    for start in range(0, len(observations), batch_size):
        observation_batch = observations[start:start + batch_size]
        action_batch = expert_actions[start:start + batch_size]
        expert_q = q_value_estimator(observation_batch, action_batch)
        repeated_observations = jnp.repeat(
            observation_batch, action_grid_size, axis=0
        )
        tiled_actions = jnp.tile(action_grid, (len(observation_batch), 1))
        grid_q = q_value_estimator(
            repeated_observations, tiled_actions
        ).reshape(len(observation_batch), action_grid_size)
        weights.append(jax.nn.relu(expert_q - jnp.min(grid_q, axis=1)))
    raw_weights = jnp.concatenate(weights)
    positive_mean = jnp.mean(raw_weights)
    if float(positive_mean) <= 0:
        raise ValueError("Q-DAgger produced no positive state weights")
    return raw_weights / positive_mean, raw_weights
