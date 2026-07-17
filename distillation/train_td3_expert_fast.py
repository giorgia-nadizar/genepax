import pickle

import jax
import jax.numpy as jnp

from brax import envs

from qdax.baselines.td3 import (
    TD3,
    TD3Config,
    generate_unroll,
)
from qdax.core.neuroevolution.buffers.buffer import (
    ReplayBuffer,
    Transition,
)

ENV_NAME = "inverted_pendulum"
eval_env = envs.get_environment(
    env_name=ENV_NAME,
)


### EVALUATION FUNCTION ####

# def evaluate(training_state, num_steps=1000):
#     key = jax.random.key(123)
#
#     state = eval_env.reset(key)
#
#     total_reward = 0.0
#     print(total_reward)
#
#     for st in range(num_steps):
#         key, action_key = jax.random.split(key)
#
#         action = td3.select_action(
#             obs=state.obs,
#             policy_params=training_state.policy_params,
#             key=action_key,
#             expl_noise=0.0,
#             deterministic=True,
#         )
#
#         state = eval_env.step(
#             state,
#             action,
#         )
#         print(st, state.reward)
#
#         total_reward += float(state.reward)
#
#     return total_reward

def evaluate(training_state, num_steps=1000):
    eval_env = envs.get_environment(
        env_name=ENV_NAME,
    )

    key = jax.random.key(123)

    state = eval_env.reset(key)

    def step_fn(carry, _):
        state, key, alive = carry

        key, action_key = jax.random.split(key)

        action = td3.select_action(
            obs=state.obs,
            policy_params=training_state.policy_params,
            key=action_key,
            expl_noise=0.0,
            deterministic=True,
        )

        next_state = eval_env.step(
            state,
            action,
        )

        reward = next_state.reward * alive

        alive = alive * (1.0 - next_state.done)

        return (
            next_state,
            key,
            alive,
        ), reward

    (_, _, _), rewards = jax.lax.scan(
        step_fn,
        (
            state,
            key,
            jnp.array(1.0),
        ),
        None,
        length=num_steps,
    )

    return jnp.sum(rewards)


# ============================================================
# Config
# ============================================================


SEED = 0
NUM_ENVS = 256
EPISODE_LENGTH = 1000
UNROLL_LENGTH = 20
# NUM_ITERATIONS = 500
NUM_ITERATIONS = 10000
WARMUP_STEPS = 50
REPLAY_BUFFER_SIZE = 1_000_000
GRADIENT_STEPS = 4

# ============================================================
# Environment
# ============================================================

env = envs.create(
    env_name=ENV_NAME,
    batch_size=NUM_ENVS,
)

obs_size = env.observation_size
action_size = env.action_size

print("obs:", obs_size)
print("actions:", action_size)

# ============================================================
# TD3
# ============================================================

config = TD3Config(
    episode_length=EPISODE_LENGTH,
    batch_size=256,
    critic_learning_rate=3e-4,
    policy_learning_rate=3e-4,
    discount=0.99,
    # expl_noise=0.1,
    expl_noise=0.2
)

td3 = TD3(
    config=config,
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
    obs=jnp.zeros((NUM_ENVS, obs_size)),
    next_obs=jnp.zeros((NUM_ENVS, obs_size)),
    rewards=jnp.zeros((NUM_ENVS,)),
    dones=jnp.zeros((NUM_ENVS,)),
    truncations=jnp.zeros((NUM_ENVS,)),
    actions=jnp.zeros((NUM_ENVS, action_size)),
)

replay_buffer = ReplayBuffer.init(
    REPLAY_BUFFER_SIZE,
    dummy_transition,
)

# ============================================================
# Vectorized environment reset
# ============================================================

key, reset_key = jax.random.split(key)

env_state = env.reset(
    reset_key
)

# ============================================================
# Jitted rollout
# ============================================================

play_step_fn = lambda env_state, training_state: td3.play_step_fn(
    env_state,
    training_state,
    env,
    deterministic=False,
)


@jax.jit
def rollout(
        env_state,
        training_state,
):
    return generate_unroll(
        env_state,
        training_state,
        UNROLL_LENGTH,
        play_step_fn,
    )


# warm up
for i in range(WARMUP_STEPS):
    env_state, _, transitions = rollout(
        env_state,
        training_state,
    )

    replay_buffer = replay_buffer.insert(
        transitions
    )

print("warmup done")

# ============================================================
# Training
# ============================================================

for i in range(NUM_ITERATIONS):

    (
        env_state,
        training_state,
        transitions,
    ) = rollout(
        env_state,
        training_state,
    )

    replay_buffer = replay_buffer.insert(
        transitions
    )

    training_state, replay_buffer, metrics = td3.update(
        training_state,
        replay_buffer,
    )

    for _ in range(GRADIENT_STEPS):
        training_state, replay_buffer, metrics = td3.update(
            training_state,
            replay_buffer,
        )

    if i % 100 == 0:
        print(
            i,
            metrics,
        )

        evaluation_reward = evaluate(training_state)
        print(evaluation_reward)
        print(replay_buffer.current_size)

# ============================================================
# Save
# ============================================================

checkpoint = {
    "training_state": training_state,
    "obs_size": obs_size,
    "action_size": action_size,
}

with open(
        "td3_expert_fast_2.pkl",
        "wb",
) as f:
    pickle.dump(
        checkpoint,
        f,
    )

print("saved")
