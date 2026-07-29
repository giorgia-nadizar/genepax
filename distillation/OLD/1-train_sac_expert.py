import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

from brax import envs
from brax.envs.wrappers import gym as brax_gym


@dataclass
class SACConfig:
    # Environment
    env_name: str = "inverted_pendulum"

    # Training
    total_timesteps: int = 180_000

    # SAC parameters
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    batch_size: int = 256

    gamma: float = 0.99
    tau: float = 0.005

    learning_starts: int = 100

    train_freq: int = 1
    gradient_steps: int = 1

    ent_coef: str = "auto"

    # Network
    net_arch: tuple = (256, 256)

    # Hardware
    device: str = "cuda"

    # Logging
    tensorboard_log: str = "./tensorboard/"


PENDULUM = SACConfig(
    env_name="inverted_pendulum",
    total_timesteps=180_000,
)
DOUBLE_PENDULUM = SACConfig(
    env_name="inverted_double_pendulum",
    total_timesteps=1_000_000,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=512,
    gamma=0.99,
    tau=0.005,
    learning_starts=10_000,
    train_freq=1,
    gradient_steps=4,
    ent_coef="auto",
    net_arch=(256, 256, 256),
)
SWIMMER = SACConfig(
    env_name="swimmer",
    total_timesteps=2_000_000,

    learning_rate=3e-4,

    buffer_size=1_000_000,
    batch_size=256,

    gamma=0.99,
    tau=0.005,

    learning_starts=10_000,

    train_freq=1,
    gradient_steps=1,

    ent_coef="auto",

    net_arch=(256, 256),
)

config = SWIMMER


# --------------------------------------------------
# Create Brax Gymnasium environment
# --------------------------------------------------

def make_env():
    brax_env = envs.create(
        env_name=config.env_name,
        backend="generalized"
    )

    env = brax_gym.GymWrapper(
        brax_env
    )

    return env


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(model, env, episodes=10):
    rewards = []

    for ep in range(episodes):

        obs, info = env.reset()

        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = env.step(
                action
            )

            total_reward += reward

            done = terminated or truncated

        rewards.append(total_reward)

    print()
    print("Evaluation")
    print("----------------")
    print(
        f"Mean reward: {np.mean(rewards):.2f}"
    )
    print(
        f"Std reward : {np.std(rewards):.2f}"
    )


# --------------------------------------------------
# Training
# --------------------------------------------------

def main():
    print("Creating Brax environment")

    env = make_env()

    print(
        "Observation space:",
        env.observation_space
    )

    print(
        "Action space:",
        env.action_space
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,

        learning_rate=config.learning_rate,

        buffer_size=config.buffer_size,
        batch_size=config.batch_size,

        gamma=config.gamma,
        tau=config.tau,

        learning_starts=config.learning_starts,

        train_freq=config.train_freq,
        gradient_steps=config.gradient_steps,

        ent_coef=config.ent_coef,

        policy_kwargs=dict(
            net_arch=list(config.net_arch)
        ),

        verbose=1,
        device=config.device,

        tensorboard_log=config.tensorboard_log
    )

    print("\nStarting SAC training")

    model.learn(
        total_timesteps=config.total_timesteps
    )

    print("Training finished")

    model_path = f"sac_brax_{config.env_name}.zip"
    model.save(
        model_path
    )

    print(
        f"Saved model to {model_path}"
    )


if __name__ == "__main__":
    main()
