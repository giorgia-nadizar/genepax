import pickle

import jax
import numpy as np

from brax import envs

from distillation.networks.jax_actor import JaxActor

# ============================================================
# Configuration
# ============================================================

ENV_NAME = "inverted_double_pendulum"

CHECKPOINT = "td3_expert_fast.pkl"

NUM_ENVS = 10
EPISODE_LENGTH = 1000

DATASET_PATH = f"expert_dataset_{ENV_NAME}.npz"

# ============================================================
# Load trained SB3 model
# ============================================================

MODEL_PATH = f"./sac_jax_{ENV_NAME}.pkl"

# Load parameters

with open(MODEL_PATH, "rb") as f:
    sac_data = pickle.load(f)

actor_data = sac_data["actor"]
architecture = actor_data["architecture"]
actor_params = actor_data["params"]

# Recreate network

actor = JaxActor(
    hidden_sizes=tuple(
        architecture["hidden_sizes"]
    ),
    action_dim=architecture["action_dim"],
)

# ============================================================
# Vectorized Brax environment
# ============================================================

env = envs.create(
    env_name=ENV_NAME,
    batch_size=NUM_ENVS,
)


# ============================================================
# Rollout function
# ============================================================

def rollout_dataset(key):
    reset_key, key = jax.random.split(key)

    state = env.reset(reset_key)

    def step_fn(carry, _):
        state, key = carry

        key, action_key = jax.random.split(key)

        actions = actor.apply(
            actor_params,
            state.obs
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

    (_, _), (observations, actions) = jax.lax.scan(
        step_fn,
        (state, key),
        None,
        length=EPISODE_LENGTH,
    )

    return observations, actions


# JIT compilation
rollout_dataset = jax.jit(
    rollout_dataset
)

# ============================================================
# Collect data
# ============================================================

key = jax.random.key(0)

X, y = rollout_dataset(
    key
)

# scan output:
#
# X:
# (time, env, obs_size)
#
# y:
# (time, env, action_size)


X = np.asarray(
    X.reshape(
        -1,
        env.observation_size,
    )
)

y = np.asarray(
    y.reshape(
        -1,
        env.action_size,
    )
)

print("Dataset:")
print("X:", X.shape)
print("y:", y.shape)

# ============================================================
# Save
# ============================================================

np.savez(
    DATASET_PATH,
    X=X,
    y=y,
)

print(
    "saved dataset:",
    DATASET_PATH,
)
