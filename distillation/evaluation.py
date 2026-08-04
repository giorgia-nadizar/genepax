import jax
import jax.numpy as jnp

from distillation.rollouts import masked_return, rollout, valid_transition_mask


def finite_transition_mask(X, y):
    """Selects transitions with finite observations and teacher actions."""
    return jnp.all(jnp.isfinite(X), axis=-1) & jnp.all(jnp.isfinite(y), axis=-1)


def finite_prefix_mask(X, y):
    """Stops a trajectory at its first non-finite transition."""
    finite = finite_transition_mask(X, y)
    return jnp.cumprod(finite.astype(jnp.int32), axis=-1).astype(bool)


def collect_mixed_policy_dataset(
        genotype,
        cgp_structure,
        actor,
        env,
        expert_weight: float = 0.5,
        num_steps: int = 1000,
        seed: int = 0,
        n_seeds: int = 10
):
    seeds = seed + jnp.arange(n_seeds)

    X, y, masks, returns = jax.vmap(
        lambda s: single_collect_mixed_policy_dataset(
            genotype,
            cgp_structure,
            actor,
            env,
            expert_weight,
            num_steps,
            s,
        )
    )(seeds)
    finite_masks = finite_transition_mask(X, y)
    retained_masks = masks & jax.vmap(finite_prefix_mask)(X, y)
    X_new = jnp.concatenate(
        [X[i][retained_masks[i]] for i in range(n_seeds)],
        axis=0,
    )

    y_new = jnp.concatenate(
        [y[i][retained_masks[i]] for i in range(n_seeds)],
        axis=0,
    )

    avg_return = jnp.mean(returns)
    collection_diagnostics = {
        "candidate_transitions": jnp.sum(masks),
        "invalid_transitions": jnp.sum(masks & ~finite_masks),
        "retained_transitions": jnp.sum(retained_masks),
    }
    return X_new, y_new, avg_return, collection_diagnostics


def single_collect_mixed_policy_dataset(
        genotype,
        cgp_structure,
        actor,
        env,
        expert_weight: float = 0.5,
        num_steps: int = 1000,
        seed: int = 0,
):
    key = jax.random.key(seed)

    def mixed_action(observation, action_key, _step):
        expert_action, _ = actor(observation, action_key)
        symbolic_action = cgp_structure.apply(genotype, observation)
        action = (
            expert_weight * expert_action
            + (1.0 - expert_weight) * symbolic_action
        )
        return action, expert_action

    X, y, rewards, dones = rollout(env, key, mixed_action, num_steps)
    episode_mask = valid_transition_mask(dones).astype(bool)
    retained_mask = episode_mask & finite_prefix_mask(X, y)
    safe_rewards = jnp.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
    valid_return = jnp.sum(jnp.where(retained_mask, safe_rewards, 0.0))
    return X, y, episode_mask, valid_return


def evaluate_symbolic_policy(
        genotype,
        cgp_structure,
        env,
        num_steps: int = 1000,
        seed: int = 0,
        n_seeds: int = 10,
):
    seeds = seed + jnp.arange(n_seeds)

    rewards = jax.vmap(
        lambda s: evaluate_symbolic_policy_single(
            genotype,
            cgp_structure,
            env,
            num_steps,
            s,
        )
    )(seeds)

    return jnp.mean(rewards)


def evaluate_symbolic_policy_single(
        genotype,
        cgp_structure,
        env,
        num_steps: int = 1000,
        seed: int = 0,
):
    """
    Evaluate a symbolic controller.

    Args:
        genotype: CGP genotype
        cgp_structure: CGP object
        env: Brax environment
        num_steps: evaluation horizon
        seed: RNG seed

    Returns:
        cumulative reward
    """

    key = jax.random.key(seed)

    def symbolic_action(observation, _key, _step):
        action = cgp_structure.apply(genotype, observation)
        return action, action

    _, _, rewards, dones = rollout(env, key, symbolic_action, num_steps)
    return masked_return(rewards, dones)
