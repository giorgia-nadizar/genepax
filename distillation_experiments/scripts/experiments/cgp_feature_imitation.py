"""Behavioral cloning with k CGP features and a fitted linear action readout."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import time

from brax import envs
import jax
import jax.numpy as jnp
import numpy as np

from distillation.fit_feature_imitation import (
    feature_policy_action, feature_policy_mse, fit_feature_imitation, make_feature_cgp,
)
from distillation.rollouts import masked_return, rollout
from distillation.networks.sac_utils import load_q_value_estimator
from distillation.q_dagger import compute_q_dagger_weights
from distillation_experiments.scripts.experiments.dagger import positive_int, select_rows
from distillation_experiments.scripts.experiments.feature_reference_data import load_reference_split


def evaluate_feature_policy(genotype, cgp_structure, environment, *, seed, steps, trajectories):
    def evaluate_one(trajectory_seed):
        def action_fn(observation, _key, _step):
            action = feature_policy_action(genotype, cgp_structure, observation)
            return action, action

        _, _, rewards, dones = rollout(
            environment, jax.random.key(trajectory_seed), action_fn, steps
        )
        return masked_return(rewards, dones)

    return jax.vmap(evaluate_one)(seed + jnp.arange(trajectories))


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="inverted_pendulum")
    parser.add_argument("--backend", default="generalized")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--reference-run", type=Path, help="Reuse exact per-seed data and state weights from this run")
    parser.add_argument("--k", type=positive_int, default=8)
    parser.add_argument("--n-nodes", type=positive_int, default=50)
    parser.add_argument("--generations", type=positive_int, default=100)
    parser.add_argument("--population-size", type=positive_int, default=100)
    parser.add_argument("--expert-samples", type=positive_int, default=10_000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=positive_int, default=1)
    parser.add_argument("--rollout-steps", type=positive_int, default=1_000)
    parser.add_argument("--evaluation-trajectories", type=positive_int, default=10)
    parser.add_argument("--state-weighting", choices=("uniform", "q_dagger"), default="uniform")
    parser.add_argument("--q-action-grid-size", type=positive_int, default=101)
    parser.add_argument("--q-batch-size", type=positive_int, default=1000)
    return parser.parse_args()


def main():
    args = parse_arguments()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be strictly between zero and one")
    val_count = max(1, round(args.expert_samples * args.validation_fraction))
    if args.expert_samples - val_count < 2:
        raise ValueError("The split must leave at least two training samples")
    experiments = Path(__file__).resolve().parents[2]
    dataset_path = args.dataset_path or experiments / "artifacts" / "expert_datasets" / f"expert_{args.env}.npz"
    with np.load(dataset_path) as data:
        X, y = jnp.asarray(data["X"]), jnp.asarray(data["y"])
    environment = envs.get_environment(args.env, backend=args.backend)
    if X.ndim != 2 or y.ndim != 2 or X.shape != (len(y), environment.observation_size) or y.shape[1] != environment.action_size:
        raise ValueError("Dataset dimensions do not match the environment")
    if len(X) < args.expert_samples or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Dataset must be finite and contain expert-samples rows")
    cgp = make_feature_cgp(X.shape[1], y.shape[1], args.k, args.n_nodes)
    critic = (load_q_value_estimator(experiments / "artifacts" / "expert_models" / args.env / "final")
              if args.state_weighting == "q_dagger" and args.reference_run is None else None)
    root = args.output_root or experiments / "artifacts" / "repertoires"
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = root / f"cgp_feature_{args.env}" / run_name
    run_directory.mkdir(parents=True, exist_ok=False)
    config = vars(args) | {
        "dataset_path": str(dataset_path.resolve()), "output_root": str(root.resolve()),
        "reference_run": str(args.reference_run.resolve()) if args.reference_run else None,
        "n_inputs": X.shape[1], "n_actions": y.shape[1],
        "n_custom_weights": cgp.n_custom_weights,
        "readout_layout": "row-major (k+1, n_actions); final row is intercept",
        "feature_transform": "nan_to_num then clip to [-1000, 1000]",
        "selection": "minimum training clipped-action MSE using the requested state weighting",
        "regression": "least squares with intercept; independent action columns over shared k features",
        "devices": [str(device) for device in jax.devices()],
    }
    (run_directory / "config.json").write_text(json.dumps(config, indent=2))
    summaries = []
    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        sampled_X, sampled_y = select_rows(X, y, args.expert_samples, jax.random.key(seed))
        train_X, train_y = sampled_X[val_count:], sampled_y[val_count:]
        val_X, val_y = sampled_X[:val_count], sampled_y[:val_count]
        train_weights = val_weights = None
        if critic is not None:
            weights, _ = compute_q_dagger_weights(
                sampled_X, sampled_y, critic, args.q_action_grid_size, args.q_batch_size,
            )
            train_weights, val_weights = weights[val_count:], weights[:val_count]
        if args.reference_run is not None:
            train_X, train_y, val_X, val_y, train_weights, val_weights = load_reference_split(
                args.reference_run, seed, env=args.env, dataset_path=dataset_path,
                weighting=args.state_weighting, train_count=args.expert_samples - val_count,
                validation_count=val_count,
            )
        fit_started = time.perf_counter()
        result = fit_feature_imitation(
            train_X, train_y, cgp, seed=seed,
            n_gens=args.generations, n_pop=args.population_size,
            sample_weights=train_weights,
        )
        fit_seconds = time.perf_counter() - fit_started
        genotype = result["genotype"]
        directory = run_directory / f"seed_{seed}"
        directory.mkdir()
        for filename, value in (("final_individual.pickle", genotype), ("final_population.pickle", result["repertoire"])):
            with (directory / filename).open("wb") as file:
                pickle.dump(value, file)
        np.savez_compressed(directory / "dataset.npz", X=train_X, y=train_y, validation_X=val_X, validation_y=val_y,
                            train_weights=np.ones(len(train_X)) if train_weights is None else train_weights,
                            validation_weights=np.ones(len(val_X)) if val_weights is None else val_weights)
        with (directory / "search_history.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=result["history"][0].keys())
            writer.writeheader()
            writer.writerows(result["history"])
        evaluation_started = time.perf_counter()
        returns = evaluate_feature_policy(
            genotype, cgp, environment, seed=seed * 100_000 + 90_000,
            steps=args.rollout_steps, trajectories=args.evaluation_trajectories,
        )
        jax.block_until_ready(returns)
        evaluation_seconds = time.perf_counter() - evaluation_started
        summary = {
            "seed": seed, "k": args.k, "state_weighting": args.state_weighting,
            "train_mse": float(feature_policy_mse(genotype, cgp, train_X, train_y)),
            "train_weighted_mse": result["loss"],
            "validation_weighted_mse": float(feature_policy_mse(genotype, cgp, val_X, val_y, val_weights)),
            "validation_mse": float(feature_policy_mse(genotype, cgp, val_X, val_y)),
            "reward": float(jnp.mean(returns)), "trajectory_returns": np.asarray(returns).tolist(),
            "readout": np.asarray(genotype["weights"]["custom_weights"]).reshape(args.k + 1, -1).tolist(),
            "fit_seconds": fit_seconds,
            "evaluation_seconds": evaluation_seconds,
        }
        (directory / "summary.json").write_text(json.dumps(summary, indent=2))
        summaries.append(summary)
        (run_directory / "aggregate_summary.json").write_text(json.dumps({"variants": summaries}, indent=2))
        print(f"seed={seed} reward={summary['reward']:.3f} validation_mse={summary['validation_mse']:.6g}")


if __name__ == "__main__":
    main()
