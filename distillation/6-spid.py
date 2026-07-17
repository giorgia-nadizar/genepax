import csv
import pickle

import jax

from distillation.evaluation import collect_mixed_policy_dataset, \
    evaluate_symbolic_policy
from distillation.fit_dataset import fit_dataset
import jax.numpy as jnp
from brax import envs

from distillation.networks.jax_actor import JaxActor
from distillation.networks.jax_critic import JaxCritic
from distillation.networks.jax_q_value import JaxQValueEstimator
from genepax.gp.cartesian_genetic_programming import CGP

if __name__ == '__main__':
    # ENV_NAME = "inverted_pendulum"
    # n_iterations = 200
    # sr_generations = 20
    # max_dataset_size = 15_000
    # target_reward = 999

    ENV_NAME = "inverted_double_pendulum"
    n_iterations = 500
    sr_generations = 50
    max_dataset_size = 20_000
    target_reward = 9000

    for SEED in range(10):
        MODEL_PATH = f"./sac_jax_{ENV_NAME}.pkl"
        eval_env = envs.get_environment(
            env_name=ENV_NAME,
        )
        REPERTOIRE_PATH = f"repertoires/symbolic_{ENV_NAME}_{SEED}"
        CSV_PATH = f"{REPERTOIRE_PATH}.csv"

        # we have already collected the initial dataset at expert_dataset.npz
        dataset_path = f"expert_dataset_{ENV_NAME}.npz"
        data = jnp.load(dataset_path)

        expert_X = data["X"]
        expert_y = data["y"]

        # Init the CGP policy graph with default values
        cgp_structure = CGP(
            n_inputs=expert_X.shape[1],
            n_outputs=expert_y.shape[1],
        )
        repertoire = None
        X = expert_X
        y = expert_y

        # Load SAC actor and critic
        with open(MODEL_PATH, "rb") as f:
            sac_data = pickle.load(f)

        actor_data = sac_data["actor"]
        actor_architecture = actor_data["architecture"]
        actor_params = actor_data["params"]
        critic_params = sac_data["critic"]
        critic_architecture = critic_params["architecture"]

        actor = JaxActor(
            hidden_sizes=tuple(
                actor_architecture["hidden_sizes"]
            ),
            action_dim=actor_architecture["action_dim"],
        )

        q1 = JaxCritic(
            hidden_sizes=tuple(
                critic_architecture["hidden_sizes"]
            )
        )

        q2 = JaxCritic(
            hidden_sizes=tuple(
                critic_architecture["hidden_sizes"]
            )
        )

        q_value_estimator = JaxQValueEstimator(q1, q2, critic_params["params"]["q1"], critic_params["params"]["q2"])

        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "loss", "reward", "mixed_reward"])

        for iteration in range(n_iterations):

            repertoire, loss = fit_dataset(X, y, cgp_structure,
                                           # bootstrap_repertoire=repertoire,
                                           q_value_estimator=q_value_estimator,
                                           n_gens=sr_generations, seed=iteration + SEED)
            print(f"{iteration} loss {loss}")

            best_idx = jnp.argmax(repertoire.fitnesses, axis=0)
            best_genotype = jax.tree.map(lambda x: x[best_idx][0], repertoire.genotypes)

            evaluation_result = evaluate_symbolic_policy(
                best_genotype,
                cgp_structure,
                eval_env,
            )
            print(f"{iteration} evaluation {evaluation_result}")

            mixed_evaluation_result = collect_mixed_policy_dataset(
                best_genotype,
                cgp_structure,
                actor,
                actor_params,
                eval_env,
                n_seeds=20
            )
            X_new, y_new, mixed_evaluation_reward = mixed_evaluation_result

            X = jnp.vstack([X, X_new])
            y = jnp.vstack([y, y_new])
            X_unique, indices = jnp.unique(
                X,
                axis=0,
                return_index=True,
            )
            y_unique = y[indices]
            if len(X) > max_dataset_size:
                X = X[-max_dataset_size:]
                y = y[-max_dataset_size:]
            print(f"{iteration} mixed-policy reward {mixed_evaluation_reward}")
            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([iteration, loss, evaluation_result, mixed_evaluation_reward])
            if evaluation_result >= target_reward:
                print("ENV SOLVED")
                break

        path = f"{REPERTOIRE_PATH}_{iteration}.pickle"
        with open(path, "wb") as file:
            pickle.dump(repertoire, file)
