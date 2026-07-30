from brax import envs

import jax
import jax.numpy as jnp

from distillation.networks.sac_utils import load_sac_actor, load_q_value_estimator


def evaluate_policy(
        environment,
        policy_fn,
        num_episodes=10,
        episode_duration=1000,
        seed=0,
):
    """
    Evaluate a deterministic policy over multiple episodes.

    Entire evaluation is JAX-native:
      - vmap: parallelize episodes
      - scan: unroll episode timesteps
      - jit: compile complete evaluation

    Returns:
      episode_rewards: [num_episodes]
      episode_lengths: [num_episodes]
    """

    eval_key = jax.random.PRNGKey(seed)

    # --------------------------------------------------------
    # One episode
    # --------------------------------------------------------

    def run_episode(key):
        reset_key, action_key = jax.random.split(key)

        # Reset environment
        state = environment.reset(reset_key)

        # Track whether the episode is still active.
        active = jnp.asarray(True)

        # Accumulators
        total_reward = jnp.asarray(0.0, dtype=jnp.float32)
        episode_length = jnp.asarray(0, dtype=jnp.int32)

        # ----------------------------------------------------
        # One environment step
        # ----------------------------------------------------

        def step_fn(carry, _):
            state, key, active, total_reward, episode_length = carry

            key, policy_key = jax.random.split(key)

            observation = state.obs

            # ------------------------------------------------
            # Policy action
            #
            # Important:
            # observation should be shape (obs_size,)
            # not (1, obs_size)
            # ------------------------------------------------

            action, _ = policy_fn(
                observation,
                policy_key,
            )

            # ------------------------------------------------
            # Environment step
            # ------------------------------------------------

            next_state = environment.step(
                state,
                action,
            )

            reward = jnp.asarray(
                next_state.reward,
                dtype=jnp.float32,
            )

            done = jnp.asarray(
                next_state.done,
                dtype=jnp.bool_,
            )

            # ------------------------------------------------
            # Only accumulate while episode is active
            # ------------------------------------------------

            reward = jnp.where(
                active,
                reward,
                0.0,
            )

            total_reward = (
                    total_reward + reward
            )

            episode_length = (
                    episode_length + active.astype(jnp.int32)
            )

            # Once done, remain inactive.
            active = active & (~done)

            return (
                next_state,
                key,
                active,
                total_reward,
                episode_length,
            ), None

        # ----------------------------------------------------
        # Scan over episode horizon
        # ----------------------------------------------------

        (
            state,
            key,
            active,
            total_reward,
            episode_length,
        ), _ = jax.lax.scan(
            step_fn,
            (
                state,
                action_key,
                active,
                total_reward,
                episode_length,
            ),
            xs=None,
            length=episode_duration,
        )

        return total_reward, episode_length

    # --------------------------------------------------------
    # Parallelize episodes
    # --------------------------------------------------------

    episode_keys = jax.random.split(
        eval_key,
        num_episodes,
    )

    episode_rewards, episode_lengths = jax.vmap(
        run_episode
    )(episode_keys)

    return episode_rewards, episode_lengths


# ============================================================
# JIT compile evaluation
# ============================================================

evaluate_policy_jit = jax.jit(
    evaluate_policy,
    static_argnames=(
        "environment",
        "policy_fn",
        "num_episodes",
        "episode_duration",
    ),
)

# ================================================================
# Example usage
# ================================================================

if __name__ == "__main__":
    env_name = "inverted_pendulum"
    checkpoint_path = (
        f"checkpoints/{env_name}/final"
    )

    policy_fn, model_config = load_sac_actor(checkpoint_path)

    print("Loaded SAC teacher")
    print(
        "Environment:",
        model_config["env_name"],
    )

    print(
        "Observation size:",
        model_config["observation_size"],
    )

    print(
        "Action size:",
        model_config["action_size"],
    )

    env = envs.create(env_name=env_name, backend=model_config["backend"])

    key = jax.random.PRNGKey(0)

    state = env.reset(key)

    action, policy_extras = policy_fn(state.obs, key)

    print()
    print("Initial observation:")
    print(state.obs)

    print()
    print("Actor action:")
    print(action)

    print()
    print("Policy extras:")
    print(policy_extras)

    # ============================================================
    # Test the Q value estimator
    # ============================================================

    q_value_estimator = load_q_value_estimator(checkpoint_path)

    q_value = q_value_estimator(state.obs, action)

    print()
    print("Q value for actor action:")
    print(q_value)

    # ============================================================
    # Test the actor
    # ============================================================

    print()
    results = evaluate_policy(
        environment=env,
        policy_fn=policy_fn,
        num_episodes=10,
        episode_duration=1000,
        seed=123,
    )

    print(results)
