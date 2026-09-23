"""Plain DAgger with a linear-regression student.

This is the continuous-action linear baseline from Algorithm 1 of Kohler et
al. (2025): fit on expert demonstrations, collect states with the current
student, label them with the ANN expert, aggregate, and refit.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from brax import envs
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.linear_model import LinearRegression

from distillation.evaluation import finite_prefix_mask
from distillation.networks.sac_utils import load_q_value_estimator, load_sac_actor
from distillation.q_dagger import compute_q_dagger_weights
from distillation.rollouts import (
    masked_return,
    rollout,
    sanitize_action,
    valid_transition_mask,
)
from distillation_experiments.scripts.experiments.dagger import positive_int, select_rows


def create_run_directory(root, environment, run_name):
    if run_name is None:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = root / f"dagger_linear_{environment}" / run_name
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def fit_linear_student(X, y, sample_weights=None):
    """Fit weighted least squares and return JAX-compatible parameters."""
    weights = None if sample_weights is None else np.asarray(sample_weights)
    model = LinearRegression().fit(
        np.asarray(X), np.asarray(y), sample_weight=weights
    )
    coefficients = jnp.asarray(model.coef_, dtype=jnp.float32)
    intercept = jnp.asarray(model.intercept_, dtype=jnp.float32)
    predictions = jnp.asarray(model.predict(np.asarray(X)), dtype=jnp.float32)
    per_sample_loss = jnp.mean(jnp.square(predictions - y), axis=-1)
    if sample_weights is None:
        loss = jnp.mean(per_sample_loss)
    else:
        loss = jnp.sum(per_sample_loss * sample_weights) / jnp.sum(sample_weights)
    return coefficients, intercept, float(loss)


def linear_imitation_loss(
    coefficients, intercept, X, y, sample_weights=None
):
    """Evaluate weighted action MSE without fitting on the evaluated rows."""
    predictions = X @ coefficients.T + intercept
    per_sample_loss = jnp.mean(jnp.square(predictions - y), axis=-1)
    if sample_weights is None:
        return float(jnp.mean(per_sample_loss))
    return float(
        jnp.sum(per_sample_loss * sample_weights) / jnp.sum(sample_weights)
    )


def split_train_test(X, y, weights, test_fraction, key):
    """Create a deterministic held-out split that remains fixed during DAgger."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("validation-fraction must be strictly between 0 and 1")
    test_size = max(1, round(len(X) * test_fraction))
    indices = jax.random.permutation(key, len(X))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return (
        X[train_indices], y[train_indices], weights[train_indices],
        X[test_indices], y[test_indices], weights[test_indices],
    )


def linear_action(coefficients, intercept, observation):
    return sanitize_action(coefficients @ observation + intercept)


def evaluate_linear_student(
    coefficients,
    intercept,
    environment,
    *,
    num_steps,
    seed,
    trajectories,
):
    def evaluate_one(trajectory_seed):
        key = jax.random.key(trajectory_seed)

        def action_fn(observation, _key, _step):
            action = linear_action(coefficients, intercept, observation)
            return action, action

        _, _, rewards, dones = rollout(environment, key, action_fn, num_steps)
        return masked_return(rewards, dones)

    seeds = seed + jnp.arange(trajectories)
    return float(jnp.mean(jax.vmap(evaluate_one)(seeds)))


def collect_linear_student_trajectories(
    coefficients,
    intercept,
    actor,
    environment,
    *,
    num_steps,
    seed,
    trajectories,
):
    """Collect a fixed number of student trajectories with expert labels."""
    def collect_one(trajectory_seed):
        key = jax.random.key(trajectory_seed)

        def action_fn(observation, action_key, _step):
            expert_action, _ = actor(observation, action_key)
            action = linear_action(coefficients, intercept, observation)
            return action, expert_action

        X, y, rewards, dones = rollout(environment, key, action_fn, num_steps)
        episode_mask = valid_transition_mask(dones).astype(bool)
        retained_mask = episode_mask & finite_prefix_mask(X, y)
        safe_rewards = jnp.nan_to_num(
            rewards, nan=0.0, posinf=0.0, neginf=0.0
        )
        episode_return = jnp.sum(
            jnp.where(retained_mask, safe_rewards, 0.0)
        )
        invalid = jnp.sum(episode_mask & ~finite_prefix_mask(X, y))
        return X, y, retained_mask, episode_return, invalid

    seeds = seed + jnp.arange(trajectories)
    X, y, masks, returns, invalid = jax.vmap(collect_one)(seeds)
    X_new = jnp.concatenate(
        [X[index][masks[index]] for index in range(trajectories)]
    )
    y_new = jnp.concatenate(
        [y[index][masks[index]] for index in range(trajectories)]
    )
    if len(X_new) == 0:
        raise RuntimeError("Linear student produced no valid transitions")
    return X_new, y_new, {
        "behavior_reward": float(jnp.mean(returns)),
        "collection_trajectories": trajectories,
        "invalid_transitions": int(jnp.sum(invalid)),
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run plain DAgger with a linear-regression student"
    )
    parser.add_argument("--env", default="inverted_pendulum")
    parser.add_argument("--backend", default="generalized")
    parser.add_argument("--run-name")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=positive_int, default=5)
    parser.add_argument("--iterations", type=positive_int, default=10)
    parser.add_argument(
        "--expert-bootstrap-samples", type=positive_int, default=10_000
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--trajectories-per-iteration", type=positive_int, default=20
    )
    parser.add_argument("--rollout-steps", type=positive_int, default=1_000)
    parser.add_argument("--evaluation-trajectories", type=positive_int, default=10)
    parser.add_argument("--target-reward", type=float, default=999.0)
    parser.add_argument(
        "--stop-when-solved", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--state-weighting", choices=("uniform", "q_dagger"), default="uniform"
    )
    parser.add_argument("--q-action-grid-size", type=positive_int, default=101)
    parser.add_argument("--q-batch-size", type=positive_int, default=1_000)
    return parser.parse_args()


def main():
    args = parse_arguments()
    experiments_directory = Path(__file__).resolve().parents[2]
    checkpoint_path = experiments_directory / "artifacts" / "expert_models" / args.env / "final"
    dataset_path = args.dataset_path or (
        experiments_directory / "artifacts" / "expert_datasets" / f"expert_{args.env}.npz"
    )
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Expert model not found: {checkpoint_path}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Expert dataset not found: {dataset_path}")

    run_directory = create_run_directory(
        experiments_directory / "artifacts" / "repertoires", args.env, args.run_name
    )
    configuration = vars(args) | {
        "algorithm": "DAgger",
        "student": "LinearRegression",
        "student_objective": "mean_squared_expert_action_error",
        "script": "distillation_experiments.scripts.experiments.dagger_linear",
    }
    configuration["dataset_path"] = str(dataset_path)
    with (run_directory / "config.json").open("w") as file:
        json.dump(configuration, file, indent=2, sort_keys=True)

    expert_data = np.load(dataset_path)
    all_expert_X = jnp.asarray(expert_data["X"], dtype=jnp.float32)
    all_expert_y = jnp.asarray(expert_data["y"], dtype=jnp.float32)
    actor = None
    if args.iterations > 1:
        actor, _ = load_sac_actor(checkpoint_path)
    q_value_estimator = (
        load_q_value_estimator(checkpoint_path)
        if args.state_weighting == "q_dagger"
        else None
    )
    summaries = []
    print(f"Writing linear DAgger results to {run_directory}")

    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        environment = envs.get_environment(args.env, backend=args.backend)
        seed_directory = run_directory / f"seed_{seed}"
        seed_directory.mkdir()
        metrics_path = seed_directory / "metrics.csv"
        with metrics_path.open("w", newline="") as file:
            csv.writer(file).writerow([
                "iteration", "imitation_loss", "test_imitation_loss",
                "student_reward",
                "behavior_reward", "dataset_size", "new_samples",
                "collection_trajectories", "invalid_transitions",
            ])

        bootstrap_X, bootstrap_y = select_rows(
            all_expert_X,
            all_expert_y,
            args.expert_bootstrap_samples,
            jax.random.key(seed),
        )
        if q_value_estimator is None:
            bootstrap_weights = jnp.ones(
                len(bootstrap_X), dtype=jnp.float32
            )
        else:
            bootstrap_weights, _ = compute_q_dagger_weights(
                bootstrap_X, bootstrap_y, q_value_estimator,
                args.q_action_grid_size, args.q_batch_size,
            )
        X, y, sample_weights, test_X, test_y, test_weights = split_train_test(
            bootstrap_X,
            bootstrap_y,
            bootstrap_weights,
            args.validation_fraction,
            jax.random.key(seed + 1_000_000),
        )
        best_reward = -float("inf")
        best_iteration = 0
        ever_solved = False
        pending_collection = None

        for iteration in range(args.iterations):
            coefficients, intercept, loss = fit_linear_student(
                X, y, sample_weights
            )
            test_loss = linear_imitation_loss(
                coefficients, intercept, test_X, test_y, test_weights
            )
            reward = evaluate_linear_student(
                coefficients,
                intercept,
                environment,
                num_steps=args.rollout_steps,
                seed=seed * 100_000 + 90_000,
                trajectories=args.evaluation_trajectories,
            )
            if reward > best_reward:
                best_reward = reward
                best_iteration = iteration
            solved = reward >= args.target_reward
            ever_solved = ever_solved or solved
            new_samples = (
                args.expert_bootstrap_samples
                if pending_collection is None
                else pending_collection["new_samples"]
            )
            with metrics_path.open("a", newline="") as file:
                csv.writer(file).writerow([
                    iteration,
                    loss,
                    test_loss,
                    reward,
                    "" if pending_collection is None else pending_collection[
                        "behavior_reward"
                    ],
                    len(X),
                    new_samples,
                    "" if pending_collection is None else pending_collection[
                        "collection_trajectories"
                    ],
                    "" if pending_collection is None else pending_collection[
                        "invalid_transitions"
                    ],
                ])
            print(
                f"seed={seed} iteration={iteration} loss={loss:.6g} "
                f"test_loss={test_loss:.6g} "
                f"reward={reward:.3f} samples={len(X)}"
            )
            if args.stop_when_solved and solved:
                break
            if iteration + 1 == args.iterations:
                break

            X_new, y_new, pending_collection = (
                collect_linear_student_trajectories(
                    coefficients,
                    intercept,
                    actor,
                    environment,
                    num_steps=args.rollout_steps,
                    seed=seed * 100_000 + iteration * 10_000,
                    trajectories=args.trajectories_per_iteration,
                )
            )
            pending_collection["new_samples"] = len(X_new)
            X = jnp.concatenate([X, X_new])
            y = jnp.concatenate([y, y_new])
            if q_value_estimator is None:
                new_weights = jnp.ones(len(X_new), dtype=jnp.float32)
            else:
                new_weights, _ = compute_q_dagger_weights(
                    X_new, y_new, q_value_estimator,
                    args.q_action_grid_size, args.q_batch_size,
                )
            sample_weights = jnp.concatenate([sample_weights, new_weights])

        np.savez_compressed(
            seed_directory / "final_model.npz",
            coefficients=np.asarray(coefficients),
            intercept=np.asarray(intercept),
        )
        np.savez_compressed(
            seed_directory / "final_dataset.npz",
            X=np.asarray(X),
            y=np.asarray(y),
            sample_weights=np.asarray(sample_weights),
            test_X=np.asarray(test_X),
            test_y=np.asarray(test_y),
            test_weights=np.asarray(test_weights),
        )
        summary = {
            "seed": seed,
            "last_iteration": iteration,
            "final_reward": reward,
            "final_imitation_loss": loss,
            "final_test_imitation_loss": test_loss,
            "best_observed_reward": best_reward,
            "best_observed_iteration": best_iteration,
            "solved": ever_solved,
            "dataset_size": len(X),
            "test_dataset_size": len(test_X),
            "state_weighting": args.state_weighting,
            "weight_effective_sample_size": float(
                jnp.square(jnp.sum(sample_weights))
                / jnp.sum(jnp.square(sample_weights))
            ),
        }
        with (seed_directory / "summary.json").open("w") as file:
            json.dump(summary, file, indent=2)
        summaries.append(summary)
        with (run_directory / "aggregate_summary.json").open("w") as file:
            json.dump(
                {
                    "completed_seeds": len(summaries),
                    "requested_seeds": args.num_seeds,
                    "solved_seeds": sum(item["solved"] for item in summaries),
                    "seeds": summaries,
                },
                file,
                indent=2,
            )


if __name__ == "__main__":
    main()
