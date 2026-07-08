import pickle
import jax.numpy as jnp
import jax
import minari
import numpy as np

from genepax.gp.cartesian_genetic_programming import CGP

env_name = "invertedpendulum/expert-v0"
dataset = minari.load_dataset(f"mujoco/{env_name}", download=True)

try:
    file = open(f"../results/CGP_featsls_IL_0.pickle", 'rb')
except FileNotFoundError:
    exit(1)
repertoire = pickle.load(file)
best_idx = jnp.argmax(repertoire.fitnesses, axis=0)
best_genotype = jax.tree.map(lambda x: x[best_idx][0], repertoire.genotypes)
best_weights = best_genotype["weights"]["custom_weights"]

n_features = 6
cgp_structure = CGP(
    n_inputs=4,
    n_outputs=n_features,
    n_nodes=50,
    outputs_wrapper=lambda x: x,
    n_custom_weights=n_features + 1
)


def policy_play_fn(observation):
    action_features_space = cgp_structure.apply(best_genotype, observation)
    action_features_space = jnp.concatenate([action_features_space, jnp.asarray([1.])])
    action = action_features_space @ best_weights
    return jnp.asarray([action])


def evaluate_policy(env, n_episodes=1):
    returns = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        for step in range(1000):
            # policy should output action given obs
            action = policy_play_fn(obs)  # <- your model here

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            print(step)
            if done:
                break

            total_reward += reward

        returns.append(total_reward)
        print(total_reward)

    return np.mean(returns), np.std(returns)


env = dataset.recover_environment()
mean, std = evaluate_policy(env)
print("Return:", mean, "+/-", std)
