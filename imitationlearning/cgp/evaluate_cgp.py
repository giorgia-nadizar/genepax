import csv
import pickle

import jax
import minari
import numpy as np
import jax.numpy as jnp

from cgp_policy import CGPPolicy
from genepax.gp.cartesian_genetic_programming import CGP
from imitationlearning.dataset import load_d4rl_dataset

csv_path = f"evaluation_pearson.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "seed",
        "best_fitness",
        "active_size",
        "mean_return",
        "std_return",
        "best_return",
        "worst_return",
    ])

for seed in range(2, 10):
    try:
        file = open(f"repertoire_pearson_{seed}.pickle", 'rb')
    except FileNotFoundError:
        exit(1)
    repertoire = pickle.load(file)
    best_idx = jnp.argmax(repertoire.fitnesses, axis=0)
    best_genome = jax.tree.map(lambda x: x[best_idx][0], repertoire.genotypes)
    print(seed, repertoire.fitnesses[best_idx])

    config = {
        "n_pop": 100,
        "tournament_size": 3,
        "n_gens": 100
    }

    dataset_id = "mujoco/invertedpendulum/expert-v0"
    obs, actions = load_d4rl_dataset(dataset_id)
    dataset = minari.load_dataset("mujoco/invertedpendulum/expert-v0")

    obs = jnp.array(obs)
    actions = jnp.array(actions)

    obs_mean = jnp.mean(obs, axis=0)
    obs_std = jnp.std(obs, axis=0) + 1e-8

    cgp_structure = CGP(
        n_inputs=4,
        n_outputs=1,
        n_nodes=50,
        outputs_wrapper=lambda x: jnp.tanh(x),
    )
    readable_representation = cgp_structure.get_readable_expression(best_genome)
    print(readable_representation)

    active_mask = cgp_structure.compute_active_mask(best_genome)
    active_size = jnp.sum(active_mask) / len(active_mask)

    policy = CGPPolicy(
        genome=best_genome,
        cgp_structure=cgp_structure,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )

    env = dataset.recover_environment()

    num_episodes = 10
    returns = []

    for ep in range(num_episodes):

        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = policy.act(obs)

            obs, reward, terminated, truncated, _ = env.step(
                np.asarray(action)
            )

            done = terminated or truncated
            total_reward += reward

        returns.append(total_reward)

        print(f"Episode {ep:02d}: {total_reward:.1f}")

    print("Mean return:", np.mean(returns))
    print("Std return:", np.std(returns))
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            seed,
            repertoire.fitnesses[best_idx][0],
            active_size,
            np.mean(returns),
            np.std(returns),
            np.max(returns),
            np.min(returns),
        ])
    print()
