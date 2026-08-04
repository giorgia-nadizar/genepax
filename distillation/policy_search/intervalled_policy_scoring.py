"""Scoring for policies that periodically defer to an expert."""

from functools import partial

import jax
import jax.numpy as jnp

from distillation.rollouts import masked_return, rollout, sanitize_action
from distillation.policy_search.mixed_policy_scoring import (
    single_mixed_policy_multi_seed_scoring_fn,
)


def single_intervalled_policy_scoring_fn(
    genotype, key, cgp_structure, actor, env, n=2, num_steps=1000
):
    if n <= 0:
        raise ValueError("n must be positive")

    def action_fn(observation, action_key, step):
        expert_action, _ = actor(observation, action_key)
        symbolic_action = sanitize_action(
            cgp_structure.apply(genotype, observation)
        )
        action = jax.lax.select(step % n == 0, expert_action, symbolic_action)
        return action, expert_action

    _, _, rewards, dones = rollout(env, key, action_fn, num_steps)
    return jnp.nan_to_num(masked_return(rewards, dones), nan=-jnp.inf)


def single_intervalled_policy_multi_seed_scoring_fn(
    genotype, key, cgp_structure, actor, env, n=2, num_steps=1000, n_reps=5
):
    keys = jax.random.split(key, n_reps)
    score = partial(
        single_intervalled_policy_scoring_fn, cgp_structure=cgp_structure,
        actor=actor, env=env, n=n, num_steps=num_steps
    )
    return jnp.asarray([jnp.mean(jax.vmap(score, in_axes=(None, 0))(genotype, keys))])


def intervalled_policy_scoring_fn(
    genotypes, key, cgp_structure, actor, env, n=2,
    num_steps=1000, n_reps=5
):
    population_size = jax.tree.leaves(genotypes)[0].shape[0]
    keys = jax.random.split(key, population_size)
    common = dict(cgp_structure=cgp_structure, actor=actor, env=env,
                  num_steps=num_steps, n_reps=n_reps)
    rewards = jax.vmap(
        partial(single_intervalled_policy_multi_seed_scoring_fn, n=n, **common),
        in_axes=(0, 0),
    )(genotypes, keys)
    symbolic_rewards = jax.vmap(
        partial(single_mixed_policy_multi_seed_scoring_fn, beta=1.0, **common),
        in_axes=(0, 0),
    )(genotypes, keys)
    return rewards, {"test_accuracy": symbolic_rewards, "updated_params": genotypes}
