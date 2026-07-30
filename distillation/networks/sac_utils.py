import json
from pathlib import Path

import orbax.checkpoint as ocp

from brax.training.acme import running_statistics
from brax.training.agents.sac import networks as sac_networks

import jax
import jax.numpy as jnp


def _make_sac_network(model_config, preprocess_observations_fn):
    """Recreates the exact network architecture recorded in a checkpoint."""
    # Checkpoints created before ``network_factory_configured`` existed were
    # trained with ``make_sac_networks`` defaults.  Their JSON contains the
    # desired config, not the architecture actually used.
    if not model_config.get("network_factory_configured", False):
        return sac_networks.make_sac_networks(
            observation_size=model_config["observation_size"],
            action_size=model_config["action_size"],
            preprocess_observations_fn=preprocess_observations_fn,
        )

    activation_name = model_config["activation"]
    activation_fns = {
        "relu": jax.nn.relu,
        "swish": jax.nn.swish,
        "tanh": jax.nn.tanh,
    }
    try:
        activation = activation_fns[activation_name]
    except KeyError as error:
        raise ValueError(f"Unsupported checkpoint activation: {activation_name}") from error

    return sac_networks.make_sac_networks(
        observation_size=model_config["observation_size"],
        action_size=model_config["action_size"],
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=tuple(model_config["hidden_layer_sizes"]),
        activation=activation,
        policy_network_layer_norm=model_config["policy_network_layer_norm"],
        q_network_layer_norm=model_config["q_network_layer_norm"],
        distribution_type=model_config["distribution_type"],
        noise_std_type=model_config["noise_std_type"],
        init_noise_std=model_config["init_noise_std"],
        state_dependent_std=model_config["state_dependent_std"],
    )


def _unreplicate(tree):
    """Removes the leading pmap device axis saved in TrainingState."""
    return jax.tree_util.tree_map(lambda x: x[0], tree)


def load_sac_actor(checkpoint_path):
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

        model_config:
            Model configuration dictionary.

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

    normalize_fn = lambda x, y: x

    if model_config["normalize_observations"]:
        normalize_fn = running_statistics.normalize

    sac_network = _make_sac_network(model_config, normalize_fn)

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

    normalizer_params = _unreplicate(training_state["normalizer_params"])
    policy_params = _unreplicate(training_state["policy_params"])

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
        model_config,
    )


def load_q_value_estimator(
        checkpoint_path,
        use_target=False,
):
    """
        Load a saved SAC Q-value estimator.

        Expected checkpoint structure:

            checkpoint_path/
                model_config.json
                training_config.json
                final_metrics.json
                training_state/

        Args:
            checkpoint_path:
                Path to the saved SAC checkpoint.

            use_target:
                If True, load the target Q-network parameters.
                If False, load the current Q-network parameters.

        Returns:
            q_value_estimator:
                JIT-compiled callable that maps observations and actions
                to the minimum Q-value predicted by the two SAC critics.
                Supports both single observations/actions and batches.
        """
    checkpoint_path = Path(
        checkpoint_path
    ).resolve()

    with open(
            checkpoint_path / "model_config.json",
            "r",
    ) as f:
        model_config = json.load(f)

    normalize_fn = lambda x, y: x

    if model_config[
        "normalize_observations"
    ]:
        normalize_fn = (
            running_statistics.normalize
        )

    sac_network = _make_sac_network(model_config, normalize_fn)

    checkpointer = (
        ocp.PyTreeCheckpointer()
    )

    training_state = (
        checkpointer.restore(
            str(
                checkpoint_path
                / "training_state"
            )
        )
    )

    normalizer_params = _unreplicate(training_state["normalizer_params"])

    if use_target:
        q_params = (
            training_state["target_q_params"]
        )
    else:
        q_params = (
            training_state["q_params"]
        )

    q_params = _unreplicate(q_params)

    q_network = (
        sac_network.q_network
    )

    @jax.jit
    def q_value_estimator(
            observations,
            actions,
    ):
        observations = jnp.asarray(
            observations
        )

        actions = jnp.asarray(
            actions
        )

        if observations.ndim == 1:
            observations = observations[None, :]

        if actions.ndim == 1:
            actions = actions[None, :]

        q_values = q_network.apply(
            normalizer_params,
            q_params,
            observations,
            actions,
        )

        q_value = jnp.min(
            q_values,
            axis=-1,
        )

        return q_value

    return q_value_estimator
