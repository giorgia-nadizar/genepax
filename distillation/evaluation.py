import jax
import jax.numpy as jnp


def collect_mixed_policy_dataset(
        genotype,
        cgp_structure,
        actor,
        actor_params,
        env,
        beta: float = 0.5,
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
            actor_params,
            env,
            beta,
            num_steps,
            s,
        )
    )(seeds)
    X_new = jnp.concatenate(
        [X[i][masks[i]] for i in range(n_seeds)],
        axis=0,
    )

    y_new = jnp.concatenate(
        [y[i][masks[i]] for i in range(n_seeds)],
        axis=0,
    )

    avg_return = jnp.mean(returns)
    return X_new, y_new, avg_return


def single_collect_mixed_policy_dataset(
        genotype,
        cgp_structure,
        actor,
        actor_params,
        env,
        beta: float = 0.5,
        num_steps: int = 1000,
        seed: int = 0,
):
    key = jax.random.key(seed)

    state = env.reset(key)

    def step_fn(carry, _):
        state, key = carry

        key, action_key = jax.random.split(key)

        # -----------------------------
        # Neural expert action
        # -----------------------------

        neural_action = actor.apply(
            actor_params,
            state.obs,
        )

        # -----------------------------
        # Symbolic action
        # -----------------------------

        symbolic_action = cgp_structure.apply(
            genotype,
            state.obs,
        )

        # -----------------------------
        # Mixture action
        # -----------------------------

        mixed_action = (
                beta * symbolic_action
                +
                (1.0 - beta) * neural_action
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
    valid_mask = alive.astype(bool)

    return X, y, valid_mask, jnp.sum(rewards * alive)


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

    state = env.reset(key)

    def step_fn(carry, _):
        state, key = carry

        # symbolic policy
        action = cgp_structure.apply(
            genotype,
            state.obs,
        )

        # make sure action has shape (action_size,)
        action = jnp.asarray(action)

        next_state = env.step(
            state,
            action,
        )

        return (
            next_state,
            key,
        ), (next_state.reward, next_state.done)

    (_, _), (rewards, dones) = jax.lax.scan(
        step_fn,
        (state, key),
        None,
        length=num_steps,
    )
    alive = 1.0 - jnp.concatenate(
        [jnp.array([0]), jnp.cumsum(dones[:-1])]
    )
    alive = jnp.clip(alive, 0.0, 1.0)

    return jnp.sum(rewards * alive)
