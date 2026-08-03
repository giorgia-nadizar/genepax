"""Population scoring for symbolic and expert/symbolic mixed policies."""

from functools import partial

import jax
import jax.numpy as jnp

from distillation.rollouts import masked_return, rollout


def single_mixed_policy_scoring_fn(
    genotype, key, cgp_structure, actor, env, beta=0.5, num_steps=1000
):
    """Scores one rollout; ``beta`` is the symbolic-policy weight."""
    def action_fn(observation, action_key, _step):
        expert_action, _ = actor(observation, action_key)
        symbolic_action = cgp_structure.apply(genotype, observation)
        action = beta * symbolic_action + (1.0 - beta) * expert_action
        return action, expert_action

    _, _, rewards, dones = rollout(env, key, action_fn, num_steps)
    return jnp.nan_to_num(masked_return(rewards, dones), nan=-jnp.inf)


def single_mixed_policy_multi_seed_scoring_fn(
    genotype, key, cgp_structure, actor, env, beta=0.5,
    num_steps=1000, n_reps=5
):
    keys = jax.random.split(key, n_reps)
    score = partial(
        single_mixed_policy_scoring_fn, cgp_structure=cgp_structure,
        actor=actor, env=env, beta=beta, num_steps=num_steps
    )
    return jnp.asarray([jnp.mean(jax.vmap(score, in_axes=(None, 0))(genotype, keys))])


def _score_population(genotypes, key, scoring_fn):
    population_size = jax.tree.leaves(genotypes)[0].shape[0]
    keys = jax.random.split(key, population_size)
    return jax.vmap(scoring_fn, in_axes=(0, 0))(genotypes, keys)


def mixed_policy_scoring_fn(
    genotypes, key, cgp_structure, actor, env, beta=0.5,
    num_steps=1000, n_reps=5
):
    common = dict(cgp_structure=cgp_structure, actor=actor, env=env,
                  num_steps=num_steps, n_reps=n_reps)
    rewards = _score_population(
        genotypes, key,
        partial(single_mixed_policy_multi_seed_scoring_fn, beta=beta, **common),
    )
    symbolic_rewards = _score_population(
        genotypes, key,
        partial(single_mixed_policy_multi_seed_scoring_fn, beta=1.0, **common),
    )
    return rewards, {"test_accuracy": symbolic_rewards, "updated_params": genotypes}


def symbolic_policy_scoring_fn(
    genotypes, key, cgp_structure, env, num_steps=1000, n_reps=5
):
    def score_one(genotype, genotype_key):
        keys = jax.random.split(genotype_key, n_reps)

        def score_seed(seed_key):
            def action_fn(observation, _action_key, _step):
                action = cgp_structure.apply(genotype, observation)
                return action, action

            _, _, rewards, dones = rollout(env, seed_key, action_fn, num_steps)
            return masked_return(rewards, dones)

        return jnp.asarray([jnp.mean(jax.vmap(score_seed)(keys))])

    rewards = _score_population(genotypes, key, score_one)
    return rewards, {
        "test_accuracy": rewards,
        "updated_params": genotypes,
    }
