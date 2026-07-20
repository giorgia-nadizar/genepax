import functools

import jax
import jax.numpy as jnp

from mixedpolicysearch.mixed_policy_scoring import single_mixed_policy_multi_seed_scoring_fn


def intervalled_policy_scoring_fn(
        genotypes,
        key,
        cgp_structure,
        actor,
        actor_params,
        env,
        n: int = 2,
        num_steps: int = 1000,
        n_reps: int = 5
):
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, jax.tree.leaves(genotypes)[0].shape[0])
    scoring_fn = functools.partial(
        single_intervalled_policy_multi_seed_scoring_fn,
        cgp_structure=cgp_structure,
        actor=actor,
        actor_params=actor_params,
        env=env,
        n=n,
        num_steps=num_steps,
        n_reps=n_reps
    )
    rewards = jax.vmap(scoring_fn, in_axes=(0, 0))(genotypes, keys)
    symbolic_scoring_fn = functools.partial(
        single_mixed_policy_multi_seed_scoring_fn,
        cgp_structure=cgp_structure,
        actor=actor,
        actor_params=actor_params,
        env=env,
        beta=0,
        num_steps=num_steps,
        n_reps=n_reps
    )
    symbolic_rewards = jax.vmap(symbolic_scoring_fn, in_axes=(0, 0))(genotypes, keys)

    return rewards, {"test_accuracy": symbolic_rewards}


def single_intervalled_policy_multi_seed_scoring_fn(
        genotype,
        key,
        cgp_structure,
        actor,
        actor_params,
        env,
        n: int = 2,
        num_steps: int = 1000,
        n_reps: int = 5
):
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, n_reps)
    scoring_fn = functools.partial(
        single_intervalled_policy_scoring_fn,
        cgp_structure=cgp_structure,
        actor=actor,
        actor_params=actor_params,
        env=env,
        n=n,
        num_steps=num_steps,
    )
    rewards = jax.vmap(scoring_fn, in_axes=(None, 0))(genotype, keys)
    return jnp.asarray([jnp.mean(rewards)])


def single_intervalled_policy_scoring_fn(
        genotype,
        key,
        cgp_structure,
        actor,
        actor_params,
        env,
        n: int = 2,
        num_steps: int = 1000,
):
    state = env.reset(key)

    def step_fn(carry, _):
        state, key, idx = carry
        key, action_key = jax.random.split(key)
        neural_action = actor.apply(
            actor_params,
            state.obs,
        )
        symbolic_action = cgp_structure.apply(
            genotype,
            state.obs,
        )
        # take the neural action every 1 / n timesteps
        action_to_take = jax.lax.cond(
            idx % n == 0,
            lambda _: neural_action,
            lambda _: symbolic_action,
            operand=None,
        )
        next_state = env.step(
            state,
            action_to_take,
        )
        return (
            next_state,
            key,
            idx + 1
        ), (
            state.obs,
            neural_action,
            next_state.reward,
            next_state.done,
        )

    (_, _, _), data = jax.lax.scan(
        step_fn,
        (state, key, 0),
        None,
        length=num_steps,
    )
    X, y, rewards, dones = data

    alive = 1.0 - jnp.concatenate(
        [jnp.array([0]), jnp.cumsum(dones[:-1])]
    )
    alive = jnp.clip(alive, 0.0, 1.0)

    return jnp.sum(rewards * alive)
