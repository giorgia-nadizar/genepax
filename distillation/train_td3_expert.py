import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from brax import envs

from qdax.baselines.td3 import TD3, TD3Config
from qdax.core.neuroevolution.buffers.buffer import (
    ReplayBuffer,
    Transition,
)

# ============================================================
# Configuration
# ============================================================

ENV_NAME = "hopper"

SEED = 0

# Enough for a decent Hopper policy
TOTAL_STEPS = 1_500_000

# Let the replay buffer fill with diverse actions
WARMUP_STEPS = 20_000

REPLAY_BUFFER_SIZE = 1_000_000

CHECKPOINT_PATH = "hopper_td3_expert.pkl"

# ============================================================
# Environment
# ============================================================

env = envs.get_environment(
    env_name=ENV_NAME,
)

obs_size = env.observation_size
action_size = env.action_size

print("Observation size:", obs_size)
print("Action size:", action_size)

# ============================================================
# TD3 agent
# ============================================================

td3_config = TD3Config(
    episode_length=1000,

    # Larger batches stabilize Hopper
    batch_size=256,

    # Standard TD3 values
    critic_learning_rate=3e-4,
    policy_learning_rate=1e-4,

    # Hopper benefits from slightly stronger discount
    discount=0.995,

    # More exploration early
    expl_noise=0.15,
)

td3 = TD3(
    config=td3_config,
    action_size=action_size,
)

key = jax.random.key(SEED)

training_state = td3.init(
    key,
    action_size,
    obs_size,
)

# ============================================================
# Replay buffer
# ============================================================

dummy_transition = Transition(
    obs=jnp.zeros((obs_size,)),
    next_obs=jnp.zeros((obs_size,)),
    rewards=jnp.zeros(()),
    dones=jnp.zeros(()),
    truncations=jnp.zeros(()),
    actions=jnp.zeros((action_size,)),
)

replay_buffer = ReplayBuffer.init(
    buffer_size=REPLAY_BUFFER_SIZE,
    transition=dummy_transition,
)

# ============================================================
# Training loop
# ============================================================

state = env.reset(key)

episode_reward = 0.0
episode_length = 0

for step in range(TOTAL_STEPS):

    key, action_key = jax.random.split(key)

    # --------------------------------------------------------
    # Action selection
    # --------------------------------------------------------

    if step < WARMUP_STEPS:

        # random exploration
        action = jax.random.uniform(
            action_key,
            shape=(action_size,),
            minval=-1.0,
            maxval=1.0,
        )

    else:

        action = td3.select_action(
            obs=state.obs,
            policy_params=training_state.policy_params,
            key=action_key,
            expl_noise=max(
                0.05,
                td3_config.expl_noise * (1 - step / TOTAL_STEPS)
            ),
            deterministic=False,
        )

    # --------------------------------------------------------
    # Environment step
    # --------------------------------------------------------

    next_state = env.step(
        state,
        action,
    )

    transition = Transition(
        obs=state.obs,
        next_obs=next_state.obs,
        rewards=next_state.reward,
        dones=next_state.done,
        truncations=next_state.done,
        actions=action,
    )

    replay_buffer = replay_buffer.insert(
        transition
    )

    episode_reward += float(next_state.reward)
    episode_length += 1

    # --------------------------------------------------------
    # TD3 update
    # --------------------------------------------------------

    if step >= WARMUP_STEPS:
        training_state, replay_buffer, metrics = td3.update(
            training_state,
            replay_buffer,
        )

    # --------------------------------------------------------
    # Episode reset
    # --------------------------------------------------------

    if next_state.done:

        if step % 5_000 == 0:
            print(
                f"step={step}, "
                f"episode reward={episode_reward:.2f}, "
                f"length={episode_length}"
            )

        key, reset_key = jax.random.split(key)

        state = env.reset(
            reset_key
        )

        episode_reward = 0.0
        episode_length = 0

    else:
        state = next_state

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    # if step % 10_000 == 0:
    # print(
    #     "training step:",
    #     step
    # )

# ============================================================
# Save expert
# ============================================================

checkpoint = {
    "training_state": training_state,
    "config": td3_config,
    "obs_size": obs_size,
    "action_size": action_size,
}

with open(CHECKPOINT_PATH, "wb") as f:
    pickle.dump(
        checkpoint,
        f,
    )

print(
    "Saved checkpoint:",
    CHECKPOINT_PATH,
)


# ============================================================
# Deterministic expert wrapper
# ============================================================

def expert_action(obs, key):
    return td3.select_action(
        obs=obs,
        policy_params=training_state.policy_params,
        key=key,
        expl_noise=0.0,
        deterministic=True,
    )


def expert_q(obs, action):
    q1, q2 = td3._critic.apply(
        training_state.critic_params,
        obs,
        action,
    )

    return jnp.minimum(q1, q2)


# Quick sanity check

key, test_key = jax.random.split(key)

obs = env.reset(test_key).obs

action = expert_action(
    obs,
    test_key,
)

q = expert_q(
    obs,
    action,
)

print("Example expert action:", action)
print("Example Q value:", q)
