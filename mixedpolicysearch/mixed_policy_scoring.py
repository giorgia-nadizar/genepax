import functools

import jax
import jax.numpy as jnp


def mixed_policy_scoring_fn(
        genotypes,
        key,
        cgp_structure,
        actor,
        actor_params,
        env,
        beta: float = 0.5,
        num_steps: int = 1000,
        n_reps: int = 5
):
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, jax.tree.leaves(genotypes)[0].shape[0])
    scoring_fn = functools.partial(
        single_mixed_policy_multi_seed_scoring_fn,
        cgp_structure=cgp_structure,
        actor=actor,
        actor_params=actor_params,
        env=env,
        beta=beta,
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

    return rewards, {"test_accuracy": symbolic_rewards, "updated_params": genotypes}


def single_mixed_policy_multi_seed_scoring_fn(
        genotype,
        key,
        cgp_structure,
        actor,
        actor_params,
        env,
        beta: float = 0.5,
        num_steps: int = 1000,
        n_reps: int = 5
):
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, n_reps)
    scoring_fn = functools.partial(
        single_mixed_policy_scoring_fn,
        cgp_structure=cgp_structure,
        actor=actor,
        actor_params=actor_params,
        env=env,
        beta=beta,
        num_steps=num_steps,
    )
    rewards = jax.vmap(scoring_fn, in_axes=(None, 0))(genotype, keys)
    return jnp.asarray([jnp.mean(rewards)])


def single_mixed_policy_scoring_fn(
        genotype,
        key,
        cgp_structure,
        actor,
        actor_params,
        env,
        beta: float = 0.5,
        num_steps: int = 1000,
):
    state = env.reset(key)

    def step_fn(carry, _):
        state, key = carry
        key, action_key = jax.random.split(key)
        neural_action = actor.apply(
            actor_params,
            state.obs,
        )
        symbolic_action = cgp_structure.apply(
            genotype,
            state.obs,
        )
        mixed_action = (
                (1.0 - beta) * symbolic_action
                +
                beta * neural_action
        )
        next_state = env.step(
            state,
            mixed_action,
        )
        return (
            next_state,
            key,
        ), (
            state.obs,
            neural_action,
            next_state.reward,
            next_state.done,
        )

    (_, _), data = jax.lax.scan(
        step_fn,
        (state, key),
        None,
        length=num_steps,
    )
    X, y, rewards, dones = data

    alive = 1.0 - jnp.concatenate(
        [jnp.array([0]), jnp.cumsum(dones[:-1])]
    )
    alive = jnp.clip(alive, 0.0, 1.0)
    total_reward = jnp.sum(rewards * alive)

    return jnp.nan_to_num(total_reward, nan=-jnp.inf)
