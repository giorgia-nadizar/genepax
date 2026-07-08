import gymnasium as gym
import jax
import jax.numpy as jnp
import minari
import numpy as np

from flax import serialization

from policy import MLPPolicy

# -------------------------
# Load normalization
# -------------------------
stats = np.load("normalization.npz")

obs_mean = stats["obs_mean"]
obs_std = stats["obs_std"]

# -------------------------
# Build model
# -------------------------
dataset = minari.load_dataset("mujoco/invertedpendulum/expert-v0")
# env = dataset.recover_environment(render_mode="human")
env = dataset.recover_environment()

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

model = MLPPolicy(action_dim=act_dim)

params = model.init(
    jax.random.PRNGKey(0),
    jnp.ones((1, obs_dim))
)["params"]

with open("bc_params.msgpack", "rb") as f:
    params = serialization.from_bytes(params, f.read())


def act(observation):
    observation = (observation - obs_mean) / (obs_std + 1e-8)

    observation = jnp.asarray(observation)[None, :]

    action = model.apply(
        {"params": params},
        observation,
    )

    return np.asarray(action[0])


num_episodes = 10

returns = []

for ep in range(num_episodes):

    obs, _ = env.reset()

    done = False
    total_reward = 0.0

    while not done:
        action = act(obs)

        obs, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        total_reward += reward

    returns.append(total_reward)

    print(
        f"Episode {ep:02d}: {total_reward:.1f}"
    )

print()

print("Average return:", np.mean(returns))
print("Std return:", np.std(returns))
