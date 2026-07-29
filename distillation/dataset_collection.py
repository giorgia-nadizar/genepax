from pathlib import Path

import jax
import numpy as np

from brax import envs

from distillation.networks.sac_utils import load_sac_teacher


def generate_expert_dataset(
        checkpoint_path,
        dataset_path=None,
        num_envs=10,
        episode_length=1000,
        seed=0,
        verbose=False,
):
    """
    Generate an expert state-action dataset using a saved SAC teacher.

    Args:
        checkpoint_path:
            Path to the saved SAC teacher checkpoint.

        dataset_path:
            Path where the generated .npz dataset will be saved.
            If None, the dataset is not saved.

        num_envs:
            Number of Brax environments simulated in parallel.

        episode_length:
            Number of environment steps collected per environment.

        seed:
            Random seed used to initialize the environment rollouts.

        verbose:
            If True, print information about dataset generation.
            If False, suppress all output from this function.

    Returns:
        X:
            Array of observations with shape
            (num_envs * episode_length, observation_size).

        y:
            Array of actions with shape
            (num_envs * episode_length, action_size).
    """

    checkpoint_path = Path(
        checkpoint_path
    ).resolve()

    if dataset_path is not None:
        dataset_path = Path(
            dataset_path
        ).resolve()

    # ============================================================
    # Load SAC teacher
    # ============================================================

    policy_fn, model_config = (
        load_sac_teacher(
            checkpoint_path
        )
    )

    # ============================================================
    # Create vectorized Brax environment
    # ============================================================

    env = envs.create(
        env_name=model_config[
            "env_name"
        ],
        backend=model_config[
            "backend"
        ],
        batch_size=num_envs,
    )

    # ============================================================
    # Rollout function
    # ============================================================

    def rollout_dataset(key):

        reset_key, key = (
            jax.random.split(
                key
            )
        )

        state = env.reset(
            reset_key
        )

        def step_fn(
                carry,
                _,
        ):
            state, key = carry

            key, action_key = (
                jax.random.split(
                    key
                )
            )

            actions, _ = policy_fn(
                state.obs,
                action_key,
            )

            next_state = env.step(
                state,
                actions,
            )

            return (
                next_state,
                key,
            ), (
                state.obs,
                actions,
            )

        (
            _,
            (
                observations,
                actions,
            ),
        ) = jax.lax.scan(
            step_fn,
            (
                state,
                key,
            ),
            None,
            length=episode_length,
        )

        return (
            observations,
            actions,
        )

    # ============================================================
    # JIT compile rollout
    # ============================================================

    rollout_dataset = jax.jit(
        rollout_dataset
    )

    # ============================================================
    # Collect dataset
    # ============================================================

    if verbose:
        print(
            "Generating expert dataset..."
        )

        print(
            f"Environment:    "
            f"{model_config['env_name']}"
        )

        print(
            f"Num envs:       "
            f"{num_envs}"
        )

        print(
            f"Episode length: "
            f"{episode_length}"
        )

    key = jax.random.key(
        seed
    )

    observations, actions = (
        rollout_dataset(
            key
        )
    )

    # ============================================================
    # Flatten time and environment dimensions
    # ============================================================

    X = np.asarray(
        observations.reshape(
            -1,
            env.observation_size,
        )
    )

    y = np.asarray(
        actions.reshape(
            -1,
            env.action_size,
        )
    )

    # ============================================================
    # Save dataset if requested
    # ============================================================

    if dataset_path is not None:
        dataset_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez(
            dataset_path,
            X=X,
            y=y,
        )

    # ============================================================
    # Print summary
    # ============================================================

    if verbose:
        print()
        print(
            "Expert dataset generated."
        )

        print(
            f"Observations: "
            f"{X.shape}"
        )

        print(
            f"Actions:      "
            f"{y.shape}"
        )

        if dataset_path is not None:
            print(
                f"Saved to:     "
                f"{dataset_path}"
            )
        else:
            print(
                "Dataset was not saved."
            )

    return (
        X,
        y,
    )
