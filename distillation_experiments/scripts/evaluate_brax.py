"""Evaluate a saved Brax SAC teacher and optionally render an HTML episode."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
from brax import envs
from brax.io import html

from distillation.networks.sac_utils import (
    load_q_value_estimator,
    load_sac_actor,
)


def run_episode(environment, policy_fn, key, episode_length=1000):
    """Runs one episode and returns its reward, length, and pipeline states."""
    reset_key, policy_key = jax.random.split(key)
    state = environment.reset(reset_key)

    def step_fn(carry, _):
        state, key, active, total_reward, elapsed_steps = carry
        key, action_key = jax.random.split(key)
        action, _ = policy_fn(state.obs, action_key)
        next_state = environment.step(state, jnp.clip(action, -1.0, 1.0))
        reward = jnp.where(active, next_state.reward, 0.0)
        return (
            next_state,
            key,
            active & jnp.logical_not(next_state.done),
            total_reward + reward,
            elapsed_steps + active.astype(jnp.int32),
        ), state.pipeline_state

    (_, _, _, total_reward, elapsed_steps), trajectory = jax.lax.scan(
        step_fn,
        (
            state,
            policy_key,
            jnp.asarray(True),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        None,
        length=episode_length,
    )
    return total_reward, elapsed_steps, trajectory


def evaluate_policy(
    environment, policy_fn, num_episodes=10, episode_length=1000, seed=0
):
    """Evaluates a deterministic policy over independent episodes."""
    def evaluate_episode(key):
        reset_key, policy_key = jax.random.split(key)
        state = environment.reset(reset_key)

        def step_fn(carry, _):
            state, key, active, reward, length = carry
            key, action_key = jax.random.split(key)
            action, _ = policy_fn(state.obs, action_key)
            next_state = environment.step(state, jnp.clip(action, -1.0, 1.0))
            return (
                next_state,
                key,
                active & jnp.logical_not(next_state.done),
                reward + jnp.where(active, next_state.reward, 0.0),
                length + active.astype(jnp.int32),
            ), None

        _, _, _, reward, length = jax.lax.scan(
            step_fn,
            (state, policy_key, jnp.asarray(True), jnp.float32(0), jnp.int32(0)),
            None,
            length=episode_length,
        )[0]
        return reward, length

    rewards, lengths = jax.vmap(evaluate_episode)(
        jax.random.split(jax.random.key(seed), num_episodes)
    )
    return rewards, lengths


def save_html(environment, trajectory, output_path):
    """Renders a scanned Brax trajectory as a standalone HTML file."""
    frames = [
        jax.tree_util.tree_map(lambda value, index=index: value[index], trajectory)
        for index in range(trajectory.x.pos.shape[0])
    ]
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        html.render(environment.sys, frames, height=600),
        encoding="utf-8",
    )
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="inverted_pendulum")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--video",
        action="store_true",
        help="Render the first evaluation episode to HTML.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="HTML output path (defaults to distillation_experiments/videos/<env>.html).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes <= 0 or args.episode_length <= 0:
        raise ValueError("episodes and episode-length must be positive")

    experiments_dir = Path(__file__).resolve().parents[1]
    checkpoint_path = experiments_dir / "expert_models" / args.env / "final"
    policy_fn, model_config = load_sac_actor(checkpoint_path)
    environment = envs.create(
        env_name=model_config["env_name"],
        backend=model_config["backend"],
    )

    initial_key = jax.random.key(args.seed)
    initial_state = environment.reset(initial_key)
    initial_action, _ = policy_fn(initial_state.obs, initial_key)
    q_value = load_q_value_estimator(checkpoint_path)(
        initial_state.obs, initial_action
    )

    evaluate = jax.jit(
        evaluate_policy,
        static_argnames=(
            "environment", "policy_fn", "num_episodes", "episode_length"
        ),
    )
    rewards, lengths = evaluate(
        environment,
        policy_fn,
        num_episodes=args.episodes,
        episode_length=args.episode_length,
        seed=args.seed,
    )

    print(f"environment: {model_config['env_name']}")
    print(f"initial action: {initial_action}")
    print(f"initial Q value: {q_value}")
    print(f"episode rewards: {rewards}")
    print(f"episode lengths: {lengths}")
    print(f"mean reward: {jnp.mean(rewards):.3f}")

    if args.video:
        record = jax.jit(
            run_episode,
            static_argnames=("environment", "policy_fn", "episode_length"),
        )
        reward, length, trajectory = record(
            environment,
            policy_fn,
            jax.random.key(args.seed),
            episode_length=args.episode_length,
        )
        output_path = (
            args.output or experiments_dir / "videos" / f"{args.env}.html"
        )
        output_path = save_html(environment, trajectory, output_path)
        print(f"video episode: reward={reward:.3f}, length={length}")
        print(f"visualization: {output_path}")


if __name__ == "__main__":
    main()
