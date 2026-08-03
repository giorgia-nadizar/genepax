import csv
import pickle
from pathlib import Path

import jax

from distillation.evaluation import collect_mixed_policy_dataset, \
    evaluate_symbolic_policy
from distillation.fit_dataset import fit_dataset
import jax.numpy as jnp
from brax import envs

from distillation.networks.sac_utils import load_sac_actor, load_q_value_estimator
from genepax.gp.cartesian_genetic_programming import CGP

if __name__ == '__main__':
    experiments_dir = Path(__file__).resolve().parents[1]
    ENV_NAME = "inverted_pendulum"
    checkpoint_path = experiments_dir / "expert_models" / ENV_NAME / "final"
    n_iterations = 200
    sr_generations = 50
    sr_pop_size = 50
    max_dataset_size = 100_000
    dataset_batch_size = 4_096
    target_reward = 999

    # ENV_NAME = "inverted_double_pendulum"
    # n_iterations = 500
    # sr_generations = 50
    # max_dataset_size = 20_000
    # target_reward = 9000

    for SEED in range(10):
        eval_env = envs.get_environment(
            env_name=ENV_NAME,
            backend="generalized"
        )
        REPERTOIRE_PATH = (
            experiments_dir / "repertoires" / f"brax_symbolic_{ENV_NAME}_{SEED}"
        )
        CSV_PATH = REPERTOIRE_PATH.with_suffix(".csv")
        Path(REPERTOIRE_PATH).parent.mkdir(parents=True, exist_ok=True)

        # we have already collected the initial dataset at expert_dataset.npz
        dataset_path = (
            experiments_dir / "expert_datasets" / f"expert_{ENV_NAME}.npz"
        )
        data = jnp.load(dataset_path)

        expert_X = jnp.asarray(data["X"], dtype=jnp.float32)
        expert_y = jnp.asarray(data["y"], dtype=jnp.float32)

        # Init the CGP policy graph with default values
        cgp_structure = CGP(
            n_inputs=expert_X.shape[1],
            n_outputs=data["y"].shape[1],
        )
        repertoire = None
        # SAC is the sole teacher.  The CGP is trained against SAC actions on
        # the initial demonstrations and on every subsequent DAgger rollout.
        sac_actor, _ = load_sac_actor(checkpoint_path)

        X, indices = jnp.unique(
            expert_X,
            axis=0,
            return_index=True,
        )
        y = expert_y[indices]
        if len(X) > max_dataset_size:
            indices = jax.random.permutation(
                jax.random.key(SEED), len(X)
            )[:max_dataset_size]
            X, y = X[indices], y[indices]

        # The critic gives an additional value-regression penalty; action
        # labels themselves always come from SAC.
        q_value_estimator = load_q_value_estimator(checkpoint_path)

        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "iteration", "bc_q_loss", "symbolic_reward",
                "mixed_reward", "symbolic_weight", "dataset_size",
            ])

        for iteration in range(n_iterations):
            repertoire, loss = fit_dataset(X, y, cgp_structure,
                                           bootstrap_repertoire=repertoire,
                                           n_pop=sr_pop_size,
                                           q_value_estimator=q_value_estimator,
                                           n_gens=sr_generations,
                                           dataset_batch_size=dataset_batch_size,
                                           seed=iteration + SEED)
            print(f"{iteration} bc_q_loss {loss}")

            best_idx = jnp.argmax(repertoire.fitnesses)
            best_genotype = jax.tree.map(
                lambda x: x[best_idx], repertoire.genotypes
            )

            evaluation_result = evaluate_symbolic_policy(
                best_genotype,
                cgp_structure,
                eval_env,
            )
            print(f"{iteration} evaluation {evaluation_result}")

            mixed_evaluation_result = collect_mixed_policy_dataset(
                best_genotype,
                cgp_structure,
                sac_actor,
                eval_env,
                beta=min(1.0, 0.5 + 0.5 * iteration / max(n_iterations - 1, 1)),
                n_seeds=20,
                seed=SEED * 10_000 + iteration * 100,
            )
            X_new, y_new, mixed_evaluation_reward = mixed_evaluation_result
            X_new, indices = jnp.unique(
                X_new,
                axis=0,
                return_index=True,
            )
            y_new = y_new[indices]

            # Aggregate, rather than replacing the previous DAgger data.  A
            # capped random subset avoids the historical bug where a larger
            # initial dataset made the remaining capacity negative.
            X = jnp.vstack([X, X_new])
            y = jnp.vstack([y, y_new])
            X, indices = jnp.unique(
                X,
                axis=0,
                return_index=True,
            )
            y = y[indices]
            if len(X) > max_dataset_size:
                indices = jax.random.permutation(
                    jax.random.key(SEED * 100_000 + iteration), len(X)
                )[:max_dataset_size]
                X, y = X[indices], y[indices]

            print(f"{iteration} mixed-policy reward {mixed_evaluation_reward}")
            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    iteration, loss, evaluation_result,
                    mixed_evaluation_reward,
                    min(1.0, 0.5 + 0.5 * iteration / max(n_iterations - 1, 1)),
                    len(X),
                ])
            if evaluation_result >= target_reward:
                print("ENV SOLVED")
                break

        path = f"{REPERTOIRE_PATH}_{iteration}.pickle"
        with open(path, "wb") as file:
            pickle.dump(repertoire, file)
