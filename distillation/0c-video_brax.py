from pathlib import Path

import jax
import jax.numpy as jnp
from brax import envs
from brax.io import html

from distillation.networks.sac_utils import load_sac_actor

# ============================================================
# CONFIG
# ============================================================

ENV_NAME = "inverted_pendulum"
BACKEND = "generalized"

CHECKPOINT_PATH = (
    f"checkpoints/{ENV_NAME}/final"
)

OUTPUT_HTML = (
    f"../../../Documents/videos/{ENV_NAME}_policy.html"
)

EPISODE_DURATION = 1000
SEED = 123


# ============================================================
# RECORD EPISODE
# ============================================================

def record_episode(
        environment,
        policy_fn,
        episode_duration=1000,
        seed=0,
):
    """
    Run one deterministic SAC episode using JAX lax.scan.

    The environment is expected to use the desired backend,
    e.g. backend="generalized".

    Returns:
        trajectory:
            Stacked pipeline states with leading time dimension.

        total_reward:
            Scalar JAX array containing the episode reward.

        episode_length:
            Scalar JAX array containing the number of active steps.
    """

    # --------------------------------------------------------
    # Reset environment
    # --------------------------------------------------------

    key = jax.random.PRNGKey(seed)

    state = environment.reset(key)

    # --------------------------------------------------------
    # Scan step
    # --------------------------------------------------------

    def step_fn(carry, _):
        state, key, active, total_reward, episode_length = carry

        # ----------------------------------------------------
        # Store current pipeline state
        #
        # This is the state BEFORE taking the action.
        # ----------------------------------------------------

        pipeline_state = state.pipeline_state

        # ----------------------------------------------------
        # Split RNG key
        # ----------------------------------------------------

        key, policy_key = jax.random.split(key)

        # ----------------------------------------------------
        # Get deterministic policy action
        # ----------------------------------------------------

        action, _ = policy_fn(
            state.obs,
            policy_key,
        )

        # ----------------------------------------------------
        # Step environment
        # ----------------------------------------------------

        next_state = environment.step(
            state,
            action,
        )

        # ----------------------------------------------------
        # Read reward and done
        # ----------------------------------------------------

        reward = jnp.asarray(
            next_state.reward,
            dtype=jnp.float32,
        )

        done = jnp.asarray(
            next_state.done,
            dtype=jnp.bool_,
        )

        # ----------------------------------------------------
        # Only accumulate while episode is active
        #
        # This prevents rewards after termination from
        # contributing to the total.
        # ----------------------------------------------------

        reward = jnp.where(
            active,
            reward,
            0.0,
        )

        total_reward = (
                total_reward + reward
        )

        episode_length = (
                episode_length
                + active.astype(jnp.int32)
        )

        # ----------------------------------------------------
        # Once done, remain inactive
        # ----------------------------------------------------

        next_active = (
                active & (~done)
        )

        # ----------------------------------------------------
        # Carry
        # ----------------------------------------------------

        next_carry = (
            next_state,
            key,
            next_active,
            total_reward,
            episode_length,
        )

        # ----------------------------------------------------
        # Output trajectory frame
        # ----------------------------------------------------

        return (
            next_carry,
            pipeline_state,
        )

    # --------------------------------------------------------
    # Initial scan carry
    # --------------------------------------------------------

    active = jnp.asarray(
        True,
        dtype=jnp.bool_,
    )

    total_reward = jnp.asarray(
        0.0,
        dtype=jnp.float32,
    )

    episode_length = jnp.asarray(
        0,
        dtype=jnp.int32,
    )

    initial_carry = (
        state,
        key,
        active,
        total_reward,
        episode_length,
    )

    # --------------------------------------------------------
    # Run rollout
    # --------------------------------------------------------

    (
        final_carry,
        trajectory,
    ) = jax.lax.scan(
        step_fn,
        initial_carry,
        xs=None,
        length=episode_duration,
    )

    # --------------------------------------------------------
    # Extract final values
    # --------------------------------------------------------

    (
        final_state,
        final_key,
        final_active,
        total_reward,
        episode_length,
    ) = final_carry

    return (
        trajectory,
        total_reward,
        episode_length,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # Load policy
    # --------------------------------------------------------

    policy_fn, model_config = load_sac_actor(CHECKPOINT_PATH)

    env = envs.create(
        env_name=ENV_NAME,
        backend=BACKEND,
    )

    print()
    print("=" * 60)
    print("ENVIRONMENT CREATED")
    print("=" * 60)

    print(
        "Environment:",
        ENV_NAME,
    )

    print(
        "Backend:",
        BACKEND,
    )

    # --------------------------------------------------------
    # Test initial policy action
    # --------------------------------------------------------

    key = jax.random.PRNGKey(
        SEED
    )

    state = env.reset(
        key
    )

    action, policy_extras = (
        policy_fn(
            state.obs,
            key,
        )
    )

    print()
    print(
        "Initial observation:"
    )

    print(
        state.obs
    )

    print()
    print(
        "Initial actor action:"
    )

    print(
        action
    )

    print()
    print(
        "Policy extras:"
    )

    print(
        policy_extras
    )

    # --------------------------------------------------------
    # Record episode
    # --------------------------------------------------------

    record_episode_jit = jax.jit(
        record_episode,
        static_argnames=(
            "environment",
            "policy_fn",
            "episode_duration",
        ),
    )

    trajectory, total_reward, episode_length = (
        record_episode_jit(
            environment=env,
            policy_fn=policy_fn,
            episode_duration=1000,
            seed=123,
        )
    )

    # --------------------------------------------------------
    # Save HTML visualization
    # --------------------------------------------------------

    trajectory_length = trajectory.x.pos.shape[0]

    trajectory_for_html = [
        jax.tree_util.tree_map(
            lambda x, t=t: x[t],
            trajectory,
        )
        for t in range(
            trajectory_length
        )
    ]

    html_string = html.render(
        env.sys,
        trajectory_for_html,
        height=600,
        # width=1000,
    )

    output_path = Path(
        OUTPUT_HTML
    ).resolve()

    with open(
            output_path,
            "w",
            encoding="utf-8",
    ) as f:
        f.write(
            html_string
        )

    print()
    print(
        "HTML visualization saved to:"
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        "Reward:",
        total_reward,
    )

    print(
        "Length:",
        episode_length,
    )

    print(
        "Visualization:",
        Path(
            OUTPUT_HTML
        ).resolve(),
    )
