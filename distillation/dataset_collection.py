"""Collection of deterministic SAC demonstrations for behavioral cloning."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from brax import envs

from distillation.networks.sac_utils import load_sac_actor


def generate_expert_dataset(
        checkpoint_path,
        dataset_path=None,
        num_envs=10,
        num_rollouts=1,
        episode_length=1000,
        seed=0,
        verbose=False,
):
    """Collect valid ``(observation, deterministic-teacher-action)`` pairs.

    ``num_rollouts`` provides independent resets and therefore better state
    coverage than merely extending one deterministic trajectory.  Transitions
    after a terminal state are excluded because raw Brax environments are not
    auto-reset by this collector.
    """
    if num_envs <= 0 or num_rollouts <= 0 or episode_length <= 0:
        raise ValueError("num_envs, num_rollouts, and episode_length must be positive")

    checkpoint_path = Path(checkpoint_path).resolve()
    dataset_path = Path(dataset_path).resolve() if dataset_path else None
    policy_fn, model_config = load_sac_actor(checkpoint_path)

    env = envs.create(
        env_name=model_config["env_name"],
        backend=model_config["backend"],
        batch_size=num_envs,
    )

    def rollout_dataset(key):
        reset_key, key = jax.random.split(key)
        state = env.reset(reset_key)
        active = jnp.ones((num_envs,), dtype=jnp.bool_)

        def step_fn(carry, _):
            state, key, active = carry
            key, action_key = jax.random.split(key)
            actions, _ = policy_fn(state.obs, action_key)
            next_state = env.step(state, actions)
            return (
                next_state,
                key,
                active & jnp.logical_not(next_state.done),
            ), (state.obs, actions, active)

        _, (observations, actions, valid) = jax.lax.scan(
            step_fn, (state, key, active), None, length=episode_length
        )
        return observations, actions, valid

    rollout_dataset = jax.jit(rollout_dataset)

    if verbose:
        print(
            f"Collecting {num_rollouts} × {num_envs} teacher rollouts "
            f"for {model_config['env_name']}..."
        )

    key = jax.random.key(seed)
    collected_observations, collected_actions = [], []
    for rollout_index in range(num_rollouts):
        observations, actions, valid = rollout_dataset(
            jax.random.fold_in(key, rollout_index)
        )
        valid = np.asarray(valid).reshape(-1)
        observations = np.asarray(observations).reshape(-1, env.observation_size)
        actions = np.asarray(actions).reshape(-1, env.action_size)
        collected_observations.append(observations[valid])
        collected_actions.append(actions[valid])

    X = np.concatenate(collected_observations).astype(np.float32)
    y = np.concatenate(collected_actions).astype(np.float32)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Teacher rollout produced non-finite observations or actions")

    metadata = {
        "format_version": 1,
        "env_name": model_config["env_name"],
        "backend": model_config["backend"],
        "observation_size": int(env.observation_size),
        "action_size": int(env.action_size),
        "num_envs": num_envs,
        "num_rollouts": num_rollouts,
        "episode_length": episode_length,
        "seed": seed,
        "num_transitions": int(X.shape[0]),
        "teacher_checkpoint": str(checkpoint_path),
    }

    if dataset_path:
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dataset_path, X=X, y=y,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )

    if verbose:
        print(f"Collected {X.shape[0]:,} valid transitions")
        if dataset_path:
            print(f"Saved dataset: {dataset_path}")

    return X, y
