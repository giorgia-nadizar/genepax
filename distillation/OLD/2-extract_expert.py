import pickle

import jax.numpy as jnp

from brax import envs
from stable_baselines3 import SAC

# ============================================================
# Paths
# ============================================================

ENV_NAME = "inverted_double_pendulum"

MODEL_PATH = f"./sac_brax_{ENV_NAME}.zip"
OUTPUT_PATH = f"./sac_jax_{ENV_NAME}.pkl"

# ============================================================
# Load trained SB3 SAC model
# ============================================================

sb3_model = SAC.load(MODEL_PATH)

torch_actor = sb3_model.policy.actor

torch_critic = sb3_model.policy.critic

state_dict = torch_critic.state_dict()

print("Loaded SB3 SAC model")

# ============================================================
# Brax environment
# ============================================================

env = envs.create(
    env_name=ENV_NAME,
    backend="generalized",
)

print("Observation size:", env.observation_size)
print("Action size:", env.action_size)


# ============================================================
# Infer architecture from PyTorch parameters
# ============================================================

def infer_critic_architecture(torch_critic):
    state_dict = torch_critic.state_dict()

    hidden_sizes = []

    layer = 0

    while True:

        key = f"qf0.{layer}.weight"

        if key not in state_dict:
            break

        out_dim = state_dict[key].shape[0]

        # ignore final q layer
        if out_dim != 1:
            hidden_sizes.append(out_dim)

        layer += 2

    input_dim = (
        state_dict["qf0.0.weight"]
        .shape[1]
    )

    return {

        "hidden_sizes": hidden_sizes,

        "input_dim": input_dim,
    }


def convert_single_q(
        state_dict,
        prefix,
        hidden_sizes
):
    params = {}

    for i, _ in enumerate(hidden_sizes):
        torch_idx = i * 2

        params[f"fc{i + 1}"] = {

            "kernel":
                jnp.array(
                    state_dict[
                        f"{prefix}.{torch_idx}.weight"
                    ]
                    .cpu()
                    .numpy()
                    .T
                ),

            "bias":
                jnp.array(
                    state_dict[
                        f"{prefix}.{torch_idx}.bias"
                    ]
                    .cpu()
                    .numpy()
                ),
        }

    final_idx = len(hidden_sizes) * 2

    params["q"] = {

        "kernel":
            jnp.array(
                state_dict[
                    f"{prefix}.{final_idx}.weight"
                ]
                .cpu()
                .numpy()
                .T
            ),

        "bias":
            jnp.array(
                state_dict[
                    f"{prefix}.{final_idx}.bias"
                ]
                .cpu()
                .numpy()
            ),
    }

    return params


def infer_actor_architecture(torch_actor):
    state_dict = torch_actor.state_dict()

    print("\nActor parameters:")
    for k, v in state_dict.items():
        print(k, v.shape)

    hidden_sizes = []

    layer_idx = 0

    while True:

        weight_name = (
            f"latent_pi.{layer_idx}.weight"
        )

        if weight_name not in state_dict:
            break

        weight = state_dict[weight_name]

        # PyTorch Linear:
        # [out_features, in_features]
        hidden_sizes.append(
            weight.shape[0]
        )

        layer_idx += 2

    obs_dim = (
        state_dict["latent_pi.0.weight"]
        .shape[1]
    )

    action_dim = (
        state_dict["mu.weight"]
        .shape[0]
    )

    architecture = {

        "hidden_sizes":
            hidden_sizes,

        "obs_dim":
            obs_dim,

        "action_dim":
            action_dim,
    }

    return architecture


# ============================================================
# Convert PyTorch actor -> Flax parameters
# ============================================================

def convert_actor(torch_actor):
    architecture = infer_actor_architecture(
        torch_actor
    )

    state_dict = torch_actor.state_dict()

    params = {
        "params": {}
    }

    # Hidden layers
    for i, _ in enumerate(
            architecture["hidden_sizes"]
    ):
        torch_idx = i * 2

        weight_name = (
            f"latent_pi.{torch_idx}.weight"
        )

        bias_name = (
            f"latent_pi.{torch_idx}.bias"
        )

        params["params"][
            f"fc{i + 1}"
        ] = {

            "kernel":
                jnp.array(
                    state_dict[weight_name]
                    .cpu()
                    .numpy()
                    .T
                ),

            "bias":
                jnp.array(
                    state_dict[bias_name]
                    .cpu()
                    .numpy()
                ),
        }

    # Output mean layer
    params["params"]["mu"] = {

        "kernel":
            jnp.array(
                state_dict["mu.weight"]
                .cpu()
                .numpy()
                .T
            ),

        "bias":
            jnp.array(
                state_dict["mu.bias"]
                .cpu()
                .numpy()
            ),
    }

    return params, architecture


def convert_critic(torch_critic):
    state_dict = torch_critic.state_dict()

    architecture = (
        infer_critic_architecture(
            torch_critic
        )
    )

    params = {

        "q1":
            {
                "params":
                    convert_single_q(
                        state_dict,
                        "qf0",
                        architecture["hidden_sizes"]
                    )
            },

        "q2":
            {
                "params":
                    convert_single_q(
                        state_dict,
                        "qf1",
                        architecture["hidden_sizes"]
                    )
            },

    }

    return architecture, params


# ============================================================
# Convert actor and critic
# ============================================================

actor_params, actor_architecture = convert_actor(
    torch_actor
)

critic_architecture, critic_params = convert_critic(
    torch_critic
)


print("\nActor converted to JAX")
print("Critic converted to JAX")


# ============================================================
# Save full SAC policy
# ============================================================

sac_data = {

    "actor": {

        "architecture":
            actor_architecture,

        "params":
            actor_params,
    },


    "critic": {

        "architecture":
            critic_architecture,

        "params":
            critic_params,
    },
}


with open(
        OUTPUT_PATH,
        "wb"
) as f:

    pickle.dump(
        sac_data,
        f
    )


print("\nSaved JAX SAC policy:")
print(OUTPUT_PATH)
