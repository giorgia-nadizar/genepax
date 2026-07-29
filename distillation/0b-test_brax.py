import json
from pathlib import Path

import orbax.checkpoint as ocp

from brax import envs
from brax.training.acme import running_statistics
from brax.training.agents.sac import networks as sac_networks

import jax
import jax.numpy as jnp


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


def load_sac_teacher(checkpoint_path):
    """
    Load a saved SAC teacher.

    Expected checkpoint structure:

        checkpoint_path/
            model_config.json
            training_config.json
            final_metrics.json
            training_state/

    Returns:
        policy_fn:
            Callable that maps an observation to an action.

        params:
            Tuple of:
                (
                    normalizer_params,
                    policy_params,
                )

        training_state:
            Complete SAC TrainingState containing:
                - policy_params
                - q_params
                - target_q_params
                - alpha_params
                - optimizer states
                - normalizer_params
                - training counters

        model_config:
            Model configuration dictionary.

        training_config:
            Training configuration dictionary.
    """

    checkpoint_path = Path(checkpoint_path).resolve()

    # ============================================================
    # Load configuration
    # ============================================================

    with open(checkpoint_path / "model_config.json", "r") as f:
        model_config = json.load(f)

    with open(checkpoint_path / "training_config.json", "r") as f:
        training_config = json.load(f)

    # ============================================================
    # Recreate SAC network
    # ============================================================

    network_factory = sac_networks.make_sac_networks

    normalize_fn = lambda x, y: x

    if model_config["normalize_observations"]:
        normalize_fn = running_statistics.normalize

    sac_network = network_factory(
        observation_size=model_config["observation_size"],
        action_size=model_config["action_size"],
        preprocess_observations_fn=normalize_fn,
    )

    # ============================================================
    # Recreate policy inference function
    # ============================================================

    make_policy = sac_networks.make_inference_fn(
        sac_network
    )

    # ============================================================
    # Load Orbax TrainingState
    # ============================================================

    checkpointer = ocp.PyTreeCheckpointer()

    training_state = checkpointer.restore(
        str(checkpoint_path / "training_state")
    )

    # ============================================================
    # Extract policy parameters
    # ============================================================

    normalizer_params = training_state["normalizer_params"]
    policy_params = jax.tree_util.tree_map(
        lambda x: x[0],
        training_state["policy_params"],
    )

    params = (
        normalizer_params,
        policy_params,
    )

    # ============================================================
    # Create deterministic policy
    # ============================================================

    policy_fn = make_policy(
        params,
        deterministic=True,
    )

    return (
        policy_fn,
        params,
        training_state,
        model_config,
        training_config,
    )


# ================================================================
# Example usage
# ================================================================

if __name__ == "__main__":
    env_name = "inverted_double_pendulum"
    checkpoint_path = (
        f"checkpoints/{env_name}/final"
    )

    (
        policy_fn,
        params,
        training_state,
        model_config,
        training_config,
    ) = load_sac_teacher(
        checkpoint_path
    )

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

    print(
        "Environment steps:",
        training_state["env_steps"],
    )

    print(
        "Gradient steps:",
        training_state["gradient_steps"],
    )

    print(
        "Alpha:",
        jnp.exp(training_state["alpha_params"]),
    )

    # ============================================================
    # Test the actor
    # ============================================================

    env = envs.create(
        env_name=model_config["env_name"],
        backend=model_config["backend"],
    )

    key = jax.random.PRNGKey(0)

    state = env.reset(key)

    action, policy_extras = policy_fn(
        state.obs,
        key,
    )

    print()
    print("Initial observation:")
    print(state.obs)

    print()
    print("Actor action:")
    print(action)

    print()
    print("Policy extras:")
    print(policy_extras)

    print()
    results = evaluate_policy(
        environment=env,
        policy_fn=policy_fn,
        num_episodes=10,
        episode_duration=1000,
        seed=123,
    )
    print(results)
