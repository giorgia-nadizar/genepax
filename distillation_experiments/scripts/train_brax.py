import json
import time
from functools import partial
from pathlib import Path

import jax
import orbax.checkpoint as ocp
from brax import envs
from brax.training.agents.sac import networks as sac_networks

from distillation.networks.sac_agent import train as sac_train

COMMON_CONFIG = {
    "environment": {
        "backend": "generalized",
        "episode_length": 1000,
    },
    "network": {
        "hidden_layer_sizes": (256, 256),
        "activation": "swish",
        "distribution_type": "tanh_normal",
        "noise_std_type": "scalar",
        "init_noise_std": 1.0,
        "state_dependent_std": False,
        "policy_network_layer_norm": False,
        "q_network_layer_norm": False,
        "normalize_observations": False,
    },
    "sac": {
        "learning_rate": 1e-4,
        "discounting": 0.99,
        "batch_size": 256,
        "tau": 0.005,
    },
    "training": {
        "num_envs": 128,
        "num_eval_envs": 16,
        "min_replay_size": 10_000,
        "max_replay_size": 1_000_000,
        "deterministic_eval": True,
        "num_evals": 20,
        "seed": 0,
    },
}

# Only values that differ from COMMON_CONFIG belong here.
ENVIRONMENT_OVERRIDES = {
    "halfcheetah": {
        "sac": {"grad_updates_per_step": 16},
        "training": {
            "num_timesteps": 4_000_000,
            "num_envs": 64,
            "num_evals": 40,
        },
    },
    "walker2d": {
        "sac": {"grad_updates_per_step": 64},
        "training": {"num_timesteps": 4_000_000},
    },
    "hopper": {
        "sac": {"grad_updates_per_step": 128},
        "training": {"num_timesteps": 500_000},
    },
    "inverted_double_pendulum": {
        "sac": {"grad_updates_per_step": 128},
        "training": {"num_timesteps": 500_000},
    },
    "inverted_pendulum": {
        "sac": {"grad_updates_per_step": 128},
        "training": {"num_timesteps": 500_000},
    },
}


def make_config(env_name):
    """Builds an independent configuration from defaults and overrides."""
    try:
        overrides = ENVIRONMENT_OVERRIDES[env_name]
    except KeyError as error:
        supported = ", ".join(sorted(ENVIRONMENT_OVERRIDES))
        raise ValueError(
            f"Unsupported environment {env_name!r}; choose one of: {supported}"
        ) from error

    config = {group: values.copy() for group, values in COMMON_CONFIG.items()}
    config["environment"]["env_name"] = env_name
    for group, values in overrides.items():
        config[group].update(values)
    return config


SAC_CONFIGS = {
    env_name: make_config(env_name) for env_name in ENVIRONMENT_OVERRIDES
}

# Backwards-compatible names for notebooks and scripts importing these configs.
HALFCHEETAH_CONFIG = SAC_CONFIGS["halfcheetah"]
WALKER2D_CONFIG = SAC_CONFIGS["walker2d"]
HOPPER_CONFIG = SAC_CONFIGS["hopper"]
INVERTED_DOUBLE_PENDULUM_CONFIG = SAC_CONFIGS["inverted_double_pendulum"]
INVERTED_PENDULUM_CONFIG = SAC_CONFIGS["inverted_pendulum"]

SAC_CONFIG = INVERTED_PENDULUM_CONFIG


def make_network_factory(network_config):
    """Builds the SAC factory whose settings are persisted in model_config."""
    activation_fns = {
        "relu": jax.nn.relu,
        "swish": jax.nn.swish,
        "tanh": jax.nn.tanh,
    }
    try:
        activation = activation_fns[network_config["activation"]]
    except KeyError as error:
        raise ValueError(
            f"Unsupported activation: {network_config['activation']}"
        ) from error

    return partial(
        sac_networks.make_sac_networks,
        hidden_layer_sizes=tuple(network_config["hidden_layer_sizes"]),
        activation=activation,
        policy_network_layer_norm=network_config[
            "policy_network_layer_norm"
        ],
        q_network_layer_norm=network_config["q_network_layer_norm"],
        distribution_type=network_config["distribution_type"],
        noise_std_type=network_config["noise_std_type"],
        init_noise_std=network_config["init_noise_std"],
        state_dependent_std=network_config["state_dependent_std"],
    )


def save_json(path, data):
    """
    Save JSON in a human-readable format.
    """

    path = Path(path)

    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def save_teacher_checkpoint(
        checkpoint_path,
        training_state,
        model_config,
        training_config,
        metrics,
):
    """
    Save the complete SAC teacher.

    This saves:

        - actor parameters
        - critic parameters
        - target critic parameters
        - entropy parameter
        - observation normalizer
        - optimizer states
        - training counters

    The neural-network architecture is saved separately in JSON.
    """

    checkpoint_path = Path(checkpoint_path).resolve()

    checkpoint_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("Saving final SAC teacher")
    print("=" * 70)
    print(f"Path: {checkpoint_path}")

    # --------------------------------------------------------
    # Save architecture
    # --------------------------------------------------------

    model_config_path = checkpoint_path / "model_config.json"

    save_json(
        model_config_path,
        model_config,
    )

    print(f"Saved model config: {model_config_path}")

    # --------------------------------------------------------
    # Save training configuration
    # --------------------------------------------------------

    training_config_path = checkpoint_path / "training_config.json"

    save_json(
        training_config_path,
        training_config,
    )

    print(f"Saved training config: {training_config_path}")

    # --------------------------------------------------------
    # Save final metrics
    #
    # Convert JAX / NumPy values into Python values where possible.
    # --------------------------------------------------------

    serializable_metrics = {}

    for key, value in metrics.items():

        try:
            value = jax.device_get(value)

            if hasattr(value, "item"):
                value = value.item()

        except Exception:
            value = str(value)

        serializable_metrics[key] = value

    metrics_path = checkpoint_path / "final_metrics.json"

    save_json(
        metrics_path,
        serializable_metrics,
    )

    print(f"Saved final metrics: {metrics_path}")

    # --------------------------------------------------------
    # Save complete TrainingState using Orbax
    # --------------------------------------------------------

    state_path = checkpoint_path / "training_state"

    state_path = state_path.resolve()

    checkpointer = ocp.PyTreeCheckpointer()

    # Orbax requires an absolute path.
    checkpointer.save(
        str(state_path),
        training_state,
        force=True,
    )

    print(f"Saved training state: {state_path}")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    # ========================================================
    # Configuration groups
    # ========================================================

    environment_config = SAC_CONFIG["environment"]
    network_config = SAC_CONFIG["network"]
    sac_config = SAC_CONFIG["sac"]
    training_config = SAC_CONFIG["training"]

    # ========================================================
    # Environment
    # ========================================================

    env_name = environment_config["env_name"]
    backend = environment_config["backend"]
    episode_length = environment_config["episode_length"]

    print(
        f"Creating environment: {env_name}"
    )

    environment = envs.create(
        env_name=env_name,
        backend=backend,
    )

    print(
        f"Observation size: {environment.observation_size}"
    )

    print(
        f"Action size:      {environment.action_size}"
    )

    # ========================================================
    # Paths
    # ========================================================

    experiments_dir = Path(__file__).resolve().parents[1]
    checkpoint_path = experiments_dir / "expert_models" / env_name

    checkpoint_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_checkpoint_path = (
            checkpoint_path / "final"
    )

    # ========================================================
    # Model configuration
    # ========================================================

    model_config = {
        "env_name": env_name,
        "backend": backend,

        "observation_size": int(
            environment.observation_size
        ),

        "action_size": int(
            environment.action_size
        ),

        **network_config,
        # Older checkpoints lack this marker and were trained with Brax's
        # default network factory, irrespective of the recorded config.
        "network_factory_configured": True,
    }

    # ========================================================
    # Training configuration
    # ========================================================

    # Copy the configuration so we can save exactly what
    # was used for this training run.
    training_config_to_save = {
        **environment_config,
        **sac_config,
        **training_config,
    }

    # ========================================================
    # Print configuration
    # ========================================================

    print()
    print("=" * 70)
    print("Starting SAC training")
    print("=" * 70)

    print(
        f"Environment:       {env_name}"
    )

    print(
        f"Backend:            {backend}"
    )

    print(
        f"Timesteps:          {training_config['num_timesteps']:,}"
    )

    print(
        f"Seed:               {training_config['seed']}"
    )

    print(
        f"Episode length:     {episode_length}"
    )

    print(
        f"Hidden layers:      "
        f"{network_config['hidden_layer_sizes']}"
    )

    print(
        f"Learning rate:      "
        f"{sac_config['learning_rate']}"
    )

    print(
        f"Discounting:        "
        f"{sac_config['discounting']}"
    )

    print(
        f"Batch size:         "
        f"{sac_config['batch_size']}"
    )

    print(
        f"Tau:                "
        f"{sac_config['tau']}"
    )

    print(
        f"Gradient updates:   "
        f"{sac_config['grad_updates_per_step']}"
    )

    print(
        f"Num envs:            "
        f"{training_config['num_envs']}"
    )

    print(
        f"Num eval envs:       "
        f"{training_config['num_eval_envs']}"
    )

    print(
        f"Checkpoint path:    {checkpoint_path}"
    )

    print(
        f"Final model path:   {final_checkpoint_path}"
    )

    print("=" * 70)
    print()

    # ========================================================
    # Progress callback
    # ========================================================

    start_time = time.time()


    def progress_fn(
            num_steps,
            metrics,
    ):
        elapsed = (
                time.time() - start_time
        )

        print(
            f"steps={num_steps:>10,} "
            f"time={elapsed:>8.1f}s "
            f"metrics={metrics}"
        )


    # ========================================================
    # Train SAC
    # ========================================================

    make_inference_fn, params, metrics, training_state = sac_train(

        environment=environment,

        # ----------------------------------------------------
        # Training budget
        # ----------------------------------------------------

        num_timesteps=training_config["num_timesteps"],

        episode_length=episode_length,

        # ----------------------------------------------------
        # Experience collection
        # ----------------------------------------------------

        num_envs=training_config["num_envs"],

        num_eval_envs=training_config["num_eval_envs"],

        # ----------------------------------------------------
        # SAC hyperparameters
        # ----------------------------------------------------

        learning_rate=sac_config["learning_rate"],

        discounting=sac_config["discounting"],

        batch_size=sac_config["batch_size"],

        tau=sac_config["tau"],

        # ----------------------------------------------------
        # Replay buffer
        # ----------------------------------------------------

        min_replay_size=training_config[
            "min_replay_size"
        ],

        max_replay_size=training_config[
            "max_replay_size"
        ],

        # ----------------------------------------------------
        # Optimization
        # ----------------------------------------------------

        grad_updates_per_step=sac_config[
            "grad_updates_per_step"
        ],

        # ----------------------------------------------------
        # Observations
        # ----------------------------------------------------

        normalize_observations=network_config[
            "normalize_observations"
        ],

        # Keep training and loading tied to the exact architecture saved in
        # model_config.json.
        network_factory=make_network_factory(network_config),

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        deterministic_eval=training_config[
            "deterministic_eval"
        ],

        num_evals=training_config[
            "num_evals"
        ],

        # ----------------------------------------------------
        # Random seed
        # ----------------------------------------------------

        seed=training_config["seed"],

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        progress_fn=progress_fn,
    )

    # ========================================================
    # Training finished
    # ========================================================

    elapsed = (
            time.time() - start_time
    )

    print()
    print("=" * 70)
    print("Training finished")
    print("=" * 70)

    print(
        f"Final metrics: {metrics}"
    )

    print(
        f"Elapsed time:  {elapsed:.1f}s"
    )

    # ========================================================
    # Save final complete teacher
    # ========================================================

    save_teacher_checkpoint(
        checkpoint_path=final_checkpoint_path,

        training_state=training_state,

        model_config=model_config,

        training_config=training_config_to_save,

        metrics=metrics,
    )

    print()
    print("Done.")
