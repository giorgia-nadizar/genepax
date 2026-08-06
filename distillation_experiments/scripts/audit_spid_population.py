"""Audit SPID fitness against Brax reward on fixed saved populations."""

import argparse
import json
import pickle
from pathlib import Path

from brax import envs
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from distillation.evaluation import evaluate_symbolic_policy
from distillation.fit_dataset import feature_construction_scoring_fn
from distillation.networks.sac_utils import load_q_value_estimator
from genepax.gp.cartesian_genetic_programming import CGP


def load_iteration_dataset(iteration_dir):
    """Reloads and combines the exact partitions used by a saved iteration."""
    data = np.load(iteration_dir / "dataset.npz")
    X_parts = [data[name] for name in ("expert_X", "dagger_X", "recovery_X")]
    y_parts = [data[name] for name in ("expert_y", "dagger_y", "recovery_y")]
    return (
        jnp.asarray(np.concatenate(X_parts), dtype=jnp.float32),
        jnp.asarray(np.concatenate(y_parts), dtype=jnp.float32),
    )


def rank_descending(values):
    """Returns one-based ranks, using the minimum rank for ties."""
    return pd.Series(values).rank(method="min", ascending=False).astype(int).to_numpy()


def correlation(values_a, values_b, method):
    """Computes a finite correlation or returns None for a constant vector."""
    value = pd.Series(values_a).corr(pd.Series(values_b), method=method)
    return None if pd.isna(value) else float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--evaluation-trajectories", type=int, default=20)
    parser.add_argument("--evaluation-seed", type=int, default=400_000)
    parser.add_argument("--candidate-batch-size", type=int, default=10)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text())
    experiments_dir = Path(__file__).resolve().parents[1]
    expert_data = np.load(
        experiments_dir / "expert_datasets" / f"expert_{config['env']}.npz"
    )
    cgp = CGP(
        n_inputs=expert_data["X"].shape[1],
        n_outputs=expert_data["y"].shape[1],
    )
    environment = envs.get_environment(
        env_name=config["env"], backend=config["backend"]
    )
    q_estimator = load_q_value_estimator(
        experiments_dir / "expert_models" / config["env"] / "final"
    )
    audit_root = run_dir / "population_audit"
    audit_root.mkdir(exist_ok=True)
    aggregate_path = audit_root / "aggregate_summary.json"
    if aggregate_path.exists():
        existing_summaries = json.loads(aggregate_path.read_text())
    else:
        existing_summaries = []
    audit_summaries = {
        summary["seed"]: summary for summary in existing_summaries
    }

    for seed in args.seeds:
        seed_dir = run_dir / f"seed_{seed}"
        seed_summary = json.loads((seed_dir / "summary.json").read_text())
        iteration = seed_summary["best_iteration"]
        iteration_dir = seed_dir / "iterations" / f"iteration_{iteration:03d}"
        X, y = load_iteration_dataset(iteration_dir)
        with (seed_dir / "best_repertoire.pickle").open("rb") as file:
            repertoire = pickle.load(file)

        fitness, components = feature_construction_scoring_fn(
            repertoire.genotypes,
            jax.random.key(0),
            X,
            y,
            cgp,
            q_estimator,
            config["dataset_batch_size"],
            config["gm_alpha"],
            config["gm_epsilon"],
        )
        losses = -np.asarray(fitness).reshape(-1)
        performance_gaps = np.asarray(components["performance_gap"]).reshape(-1)
        fidelity_gaps = np.asarray(components["fidelity_gap"]).reshape(-1)

        population_size = len(losses)
        reward_batches = []
        for start in range(0, population_size, args.candidate_batch_size):
            stop = min(start + args.candidate_batch_size, population_size)
            genotypes = jax.tree.map(
                lambda value: value[start:stop], repertoire.genotypes
            )
            rewards = jax.vmap(
                lambda genotype: evaluate_symbolic_policy(
                    genotype,
                    cgp,
                    environment,
                    num_steps=config["rollout_steps"],
                    seed=args.evaluation_seed + seed * 10_000,
                    n_seeds=args.evaluation_trajectories,
                )
            )(genotypes)
            reward_batches.append(np.asarray(rewards))
        rewards = np.concatenate(reward_batches)

        frame = pd.DataFrame({
            "candidate": np.arange(population_size),
            "spid_loss": losses,
            "spid_fitness": -losses,
            "performance_gap": performance_gaps,
            "fidelity_gap": fidelity_gaps,
            "reward": rewards,
        })
        frame["loss_rank"] = rank_descending(-losses)
        frame["reward_rank"] = rank_descending(rewards)
        output_dir = audit_root / f"seed_{seed}_iteration_{iteration:03d}"
        output_dir.mkdir(exist_ok=False)
        frame.to_csv(output_dir / "population.csv", index=False)

        best_reward_row = frame.loc[frame["reward"].idxmax()]
        best_loss_row = frame.loc[frame["spid_loss"].idxmin()]
        audit_summary = {
            "seed": seed,
            "iteration": iteration,
            "dataset_size": len(X),
            "population_size": population_size,
            "evaluation_trajectories": args.evaluation_trajectories,
            "evaluation_seed": args.evaluation_seed + seed * 10_000,
            "pearson_loss_reward": correlation(losses, rewards, "pearson"),
            "spearman_loss_reward": correlation(losses, rewards, "spearman"),
            "pearson_performance_gap_reward": correlation(
                performance_gaps, rewards, "pearson"
            ),
            "spearman_performance_gap_reward": correlation(
                performance_gaps, rewards, "spearman"
            ),
            "best_reward": float(best_reward_row["reward"]),
            "best_reward_loss_rank": int(best_reward_row["loss_rank"]),
            "best_loss_reward": float(best_loss_row["reward"]),
            "best_loss": float(best_loss_row["spid_loss"]),
            "reward_min": float(rewards.min()),
            "reward_median": float(np.median(rewards)),
            "reward_max": float(rewards.max()),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(audit_summary, indent=2)
        )
        audit_summaries[seed] = audit_summary
        print(json.dumps(audit_summary, indent=2))

    aggregate_path.write_text(
        json.dumps(
            [audit_summaries[seed] for seed in sorted(audit_summaries)],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
