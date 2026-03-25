from typing import Any, Dict, Union

import jax
import jax.numpy as jnp


def no_regularizer(weights: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Default regularizer that does nothing.

    Always returns 0.0, so it has no effect on optimization.
    """
    return jnp.array(0.0, dtype=jnp.float32)


def no_snap(weights: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """
    Default snap function that does nothing.

    Returns the weights unchanged.
    """
    return weights


def sticky_pm_target_regularizer(
    weights: Dict[str, jnp.ndarray],
    target: int = 1,
) -> jnp.ndarray:
    """
    Quadratic well around ±target.

    Each weight is penalized by its squared distance to the nearest of {+target, -target}.

    Parameters
    ----------
    weights : Dict[str, jnp.ndarray]
        Weight PyTree.
    target : int
        Magnitude of the target value (default 1).
        Penalty attracts weights toward ±target.

    Returns
    -------
    jnp.ndarray
        Scalar regularization penalty.
    """
    target = float(target)

    def _penalty(w: jnp.ndarray) -> jnp.ndarray:
        dist_sq = jnp.minimum(
            (w - target) ** 2,
            (w + target) ** 2,
        )
        return jnp.sum(dist_sq)

    return jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree.map(_penalty, weights),
    )


def snap_to_pm_target(
    weights: Dict[str, jnp.ndarray],
    target: int = 1,
    eps: float = 1e-2,
) -> Union[Dict[str, jnp.ndarray], Any]:
    """
    Project weights exactly to ±target if they are sufficiently close.

    Parameters
    ----------
    weights : Dict[str, jnp.ndarray]
        Weight PyTree.
    target : int
        The magnitude of the target value to snap to (default 1).
    eps : float
        Distance threshold for snapping.

    Returns
    -------
    Dict[str, jnp.ndarray]
        Snapped weight PyTree.
    """
    target = float(target)

    def _snap(w: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(
            jnp.abs(jnp.abs(w) - target) < eps,
            jnp.sign(w) * target,
            w,
        )

    return jax.tree.map(_snap, weights)
