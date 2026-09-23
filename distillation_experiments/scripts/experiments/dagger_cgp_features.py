"""Q-DAgger with CGP features, weighted linear readouts, and full replay."""

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

from distillation.evaluation import finite_prefix_mask
from distillation.fit_feature_imitation import (
    feature_policy_action, feature_policy_mse, fit_feature_imitation, make_feature_cgp,
)
from distillation.networks.sac_utils import load_q_value_estimator, load_sac_actor
from distillation.q_dagger import compute_q_dagger_weights
from distillation.rollouts import rollout, valid_transition_mask
from distillation_experiments.scripts.experiments.cgp_feature_imitation import evaluate_feature_policy
from distillation_experiments.scripts.experiments.dagger import positive_int, select_rows


def collect_feature_trajectories(genotype, cgp, actor, environment, *, seed, steps, trajectories,
                                 policy_action=feature_policy_action):
    """Execute the student; label visited states with the deterministic expert."""
    def collect_one(trajectory_seed):
        def action_fn(observation, key, _step):
            label, _ = actor(observation, key)
            return policy_action(genotype, cgp, observation), label

        X, y, rewards, dones = rollout(environment, jax.random.key(trajectory_seed), action_fn, steps)
        episode_mask = valid_transition_mask(dones).astype(bool)
        finite = finite_prefix_mask(X, y)
        retained = episode_mask & finite
        reward = jnp.sum(jnp.where(retained, jnp.nan_to_num(rewards, nan=0., posinf=0., neginf=0.), 0.))
        return X, y, retained, reward, jnp.sum(episode_mask & ~finite)

    X, y, masks, rewards, invalid = jax.vmap(collect_one)(seed + jnp.arange(trajectories))
    X = jnp.concatenate([X[i][masks[i]] for i in range(trajectories)])
    y = jnp.concatenate([y[i][masks[i]] for i in range(trajectories)])
    if not len(X):
        raise RuntimeError("Student collected no valid transitions")
    return X, y, {"behavior_reward": float(jnp.mean(rewards)),
                  "invalid_transitions": int(jnp.sum(invalid)), "new_samples": len(X)}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="inverted_double_pendulum")
    parser.add_argument("--backend", default="generalized")
    parser.add_argument("--k", type=positive_int, default=8)
    parser.add_argument("--n-nodes", type=positive_int, default=50)
    parser.add_argument("--num-seeds", type=positive_int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--iterations", type=positive_int, default=10)
    parser.add_argument("--generations", type=positive_int, default=100)
    parser.add_argument("--population-size", type=positive_int, default=100)
    parser.add_argument("--expert-samples", type=positive_int, default=10000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--trajectories-per-iteration", type=positive_int, default=20)
    parser.add_argument("--evaluation-trajectories", type=positive_int, default=10)
    parser.add_argument("--rollout-steps", type=positive_int, default=1000)
    parser.add_argument("--q-action-grid-size", type=positive_int, default=101)
    parser.add_argument("--q-batch-size", type=positive_int, default=1000)
    parser.add_argument("--target-reward", type=float, default=9359.)
    parser.add_argument("--stop-when-solved", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-name")
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main():
    args = parse_arguments()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be between zero and one")
    test_size = max(1, round(args.expert_samples * args.validation_fraction))
    if args.expert_samples - test_size < 2:
        raise ValueError("At least two training samples must remain")
    experiments = Path(__file__).resolve().parents[2]
    dataset = experiments / "artifacts" / "expert_datasets" / f"expert_{args.env}.npz"
    checkpoint = experiments / "artifacts" / "expert_models" / args.env / "final"
    with np.load(dataset) as data:
        all_X, all_y = jnp.asarray(data["X"], dtype=jnp.float32), jnp.asarray(data["y"], dtype=jnp.float32)
    environment = envs.get_environment(args.env, backend=args.backend)
    if all_X.shape != (len(all_y), environment.observation_size) or all_y.shape[1] != environment.action_size:
        raise ValueError("Dataset dimensions do not match environment")
    if len(all_X) < args.expert_samples or not np.isfinite(all_X).all() or not np.isfinite(all_y).all():
        raise ValueError("Expert dataset must be finite with sufficient samples")
    actor, _ = load_sac_actor(checkpoint)
    critic = load_q_value_estimator(checkpoint)
    cgp = make_feature_cgp(all_X.shape[1], all_y.shape[1], args.k, args.n_nodes)
    root = args.output_root or experiments / "artifacts" / "repertoires"
    name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = root / f"dagger_cgp_feature_{args.env}" / name
    run.mkdir(parents=True, exist_ok=False)
    config = vars(args) | {
        "output_root": str(root.resolve()), "dataset_path": str(dataset.resolve()),
        "algorithm": "Q-DAgger", "state_weighting": "q_dagger",
        "refit_strategy": "from_scratch_each_iteration", "fit_sampling": "all_replay_rows",
        "weight_definition": "max(Q(s, expert_action) - min_grid Q(s, action), 0)",
        "weight_normalization": "global training-replay mean; raw gaps retained across batches",
        "readout_layout": "row-major (k+1, n_actions); final row is intercept",
        "n_inputs": all_X.shape[1], "n_actions": all_y.shape[1],
        "n_custom_weights": cgp.n_custom_weights,
        "selection": "minimum Q-weighted clipped-action training MSE",
        "split": "same per-seed bootstrap permutation as cgp_feature_imitation; first rows held out",
        "device": str(jax.devices()),
    }
    (run / "config.json").write_text(json.dumps(config, indent=2))
    summaries = []
    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        directory = run / f"seed_{seed}"
        directory.mkdir()
        X, y = select_rows(all_X, all_y, args.expert_samples, jax.random.key(seed))
        _, raw_gaps = compute_q_dagger_weights(X, y, critic, args.q_action_grid_size, args.q_batch_size)
        test_X, test_y, test_gaps = X[:test_size], y[:test_size], raw_gaps[:test_size]
        X, y, gaps = X[test_size:], y[test_size:], raw_gaps[test_size:]
        np.savez_compressed(directory / "initial_dataset.npz", X=X, y=y, raw_q_gaps=gaps,
                            test_X=test_X, test_y=test_y, test_raw_q_gaps=test_gaps)
        metrics = []
        collection = {"behavior_reward": None, "new_samples": len(X), "invalid_transitions": 0}
        best_reward = -float("inf")
        for iteration in range(args.iterations):
            # Normalize the whole replay together, not each collection batch.
            weights = gaps / jnp.mean(gaps)
            started = time.perf_counter()
            fit = fit_feature_imitation(
                X, y, cgp, seed=seed + iteration * 10000, n_gens=args.generations,
                n_pop=args.population_size, sample_weights=weights,
            )
            fit_seconds = time.perf_counter() - started
            genotype = fit["genotype"]
            checkpoint_dir = directory / f"iteration_{iteration}"
            checkpoint_dir.mkdir()
            for filename, value in (("individual.pickle", genotype), ("population.pickle", fit["repertoire"])):
                with (checkpoint_dir / filename).open("wb") as f:
                    pickle.dump(value, f)
            with (checkpoint_dir / "search_history.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fit["history"][0].keys())
                writer.writeheader()
                writer.writerows(fit["history"])
            returns = evaluate_feature_policy(genotype, cgp, environment, seed=seed * 100000 + 90000,
                                              steps=args.rollout_steps, trajectories=args.evaluation_trajectories)
            reward = float(jnp.mean(returns))
            metric = {
                "iteration": iteration, "student_reward": reward,
                "train_weighted_mse": fit["loss"],
                "train_mse": float(feature_policy_mse(genotype, cgp, X, y)),
                "validation_weighted_mse": float(feature_policy_mse(genotype, cgp, test_X, test_y, test_gaps)),
                "validation_mse": float(feature_policy_mse(genotype, cgp, test_X, test_y)),
                "dataset_size": len(X), "fit_seconds": fit_seconds,
                "weight_effective_sample_size": float(jnp.sum(weights)**2 / jnp.sum(weights**2)),
                **collection,
            }
            metrics.append(metric)
            with (directory / "metrics.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=metric.keys())
                writer.writeheader()
                writer.writerows(metrics)
            (checkpoint_dir / "summary.json").write_text(json.dumps(metric | {
                "trajectory_returns": np.asarray(returns).tolist(),
                "readout": np.asarray(genotype["weights"]["custom_weights"]).reshape(args.k + 1, -1).tolist(),
            }, indent=2))
            if reward > best_reward:
                best_reward, best_iteration = reward, iteration
            print(f"seed={seed} iteration={iteration} reward={reward:.3f} weighted_mse={fit['loss']:.6g} replay={len(X)}", flush=True)
            if (args.stop_when_solved and reward >= args.target_reward) or iteration + 1 == args.iterations:
                break
            new_X, new_y, collection = collect_feature_trajectories(
                genotype, cgp, actor, environment, seed=2000000 + seed * 100000 + iteration * 1000,
                steps=args.rollout_steps, trajectories=args.trajectories_per_iteration,
            )
            _, new_gaps = compute_q_dagger_weights(new_X, new_y, critic, args.q_action_grid_size, args.q_batch_size)
            np.savez_compressed(checkpoint_dir / "collected_dataset.npz", X=new_X, y=new_y, raw_q_gaps=new_gaps)
            X, y, gaps = jnp.concatenate([X, new_X]), jnp.concatenate([y, new_y]), jnp.concatenate([gaps, new_gaps])
        np.savez_compressed(directory / "final_dataset.npz", X=X, y=y, raw_q_gaps=gaps,
                            sample_weights=weights, test_X=test_X, test_y=test_y, test_raw_q_gaps=test_gaps)
        with (directory / "final_individual.pickle").open("wb") as f:
            pickle.dump(genotype, f)
        summary = {"seed": seed, "k": args.k, "mode": "q_dagger", "last_iteration": iteration,
                   "final_reward": reward, "best_observed_reward": best_reward,
                   "best_observed_iteration": best_iteration, "solved": best_reward >= args.target_reward,
                   "final_validation_mse": metric["validation_mse"],
                   "final_validation_weighted_mse": metric["validation_weighted_mse"], "dataset_size": len(X)}
        (directory / "summary.json").write_text(json.dumps(summary, indent=2))
        summaries.append(summary)
        (run / "aggregate_summary.json").write_text(json.dumps({"variants": summaries}, indent=2))


if __name__ == "__main__":
    main()
