"""Policies comprising one independently evolved feature CGP per action."""

from typing import Any, Dict

import jax
import jax.numpy as jnp

from distillation.fit_feature_imitation import action_mse, feature_policy_action
from genepax.gp.cartesian_genetic_programming import CGP


def independent_feature_action(
    policy: Dict[str, Any], cgp_structure: CGP, observation: jax.Array
) -> jax.Array:
    """Execute each action's own graph and stored scalar linear readout."""
    genotypes = jax.tree.map(lambda *values: jnp.stack(values), *policy['actions'])
    return jax.vmap(feature_policy_action, in_axes=(0, None, None))(
        genotypes, cgp_structure, observation
    ).reshape(-1)


def independent_feature_mse(
    policy: Dict[str, Any], cgp_structure: CGP, X: jax.Array, y: jax.Array,
    sample_weights: jax.Array | None = None,
) -> jax.Array:
    predictions = jax.vmap(independent_feature_action, in_axes=(None, None, 0))(
        policy, cgp_structure, X
    )
    return action_mse(predictions, y, sample_weights)
