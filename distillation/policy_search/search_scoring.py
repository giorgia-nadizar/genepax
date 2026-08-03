"""Scoring adapters shared by the baseline and mixed-policy search scripts."""

from functools import partial

import jax

from distillation.rollouts import masked_return, rollout


def evaluate_genome(
    genome, key, environment, cgp_structure, n_evals=5,
    actor_fn=None, symbolic_weight=1.0, episode_length=1000,
):
    def evaluate_once(seed_key):
        def action_fn(observation, action_key, _step):
            symbolic = cgp_structure.apply(genome, observation)
            if actor_fn is None:
                return symbolic, symbolic
            expert, _ = actor_fn(observation, action_key)
            return symbolic_weight * symbolic + (1.0 - symbolic_weight) * expert, expert

        _, _, rewards, dones = rollout(
            environment, seed_key, action_fn, episode_length
        )
        return masked_return(rewards, dones)

    return jax.vmap(evaluate_once)(jax.random.split(key, n_evals))


def scoring_fn_maker(
    environment, cgp_structure, n_evals=5, actor_fn=None,
    mixed=False, episode_length=1000,
):
    score = partial(
        evaluate_genome, environment=environment, cgp_structure=cgp_structure,
        n_evals=n_evals, actor_fn=actor_fn, episode_length=episode_length,
    )
    if mixed:
        return jax.jit(jax.vmap(score, in_axes=(0, 0, None)))
    return jax.jit(jax.vmap(score, in_axes=(0, 0)))


def mixed_scoring_fn_maker(environment, cgp_structure, actor_fn, n_evals=5,
                           episode_length=1000):
    return scoring_fn_maker(
        environment, cgp_structure, n_evals, actor_fn,
        mixed=True, episode_length=episode_length,
    )
