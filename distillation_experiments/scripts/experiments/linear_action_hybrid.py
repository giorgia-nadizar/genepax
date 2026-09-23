"""Evaluate subsets of fitted linear actions inside an ANN expert policy."""

import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path

from brax import envs
import jax
import jax.numpy as jnp
import numpy as np

from distillation.networks.sac_utils import load_sac_actor
from distillation.rollouts import masked_return, rollout, sanitize_action
from distillation_experiments.scripts.experiments.dagger import positive_int


SOURCE_RUNS = {
    "uniform": "linear_initial_uniform_5seeds_test_split",
    "q_dagger": "linear_initial_qdagger_5seeds_test_split",
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="hopper")
    parser.add_argument("--backend", default="generalized")
    parser.add_argument("--run-name")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=positive_int, default=5)
    parser.add_argument(
        "--state-weighting", choices=("uniform", "q_dagger", "both"),
        default="both",
    )
    parser.add_argument(
        "--linear-action-count", choices=("one", "two", "both"), default="both"
    )
    parser.add_argument("--rollout-steps", type=positive_int, default=1_000)
    parser.add_argument("--evaluation-trajectories", type=positive_int, default=10)
    parser.add_argument("--target-reward", type=float, default=3_250.0)
    return parser.parse_args()


def evaluate_linear_hybrid(
    coefficients, intercept, actor, environment, action_indices,
    *, steps, seed, trajectories,
):
    """Replace selected ANN outputs with their fitted linear action heads."""
    indices = jnp.asarray(action_indices)

    def evaluate_one(trajectory_seed):
        def hybrid_action(observation, action_key, _step):
            expert_action, _ = actor(observation, action_key)
            linear_actions = coefficients[indices] @ observation + intercept[indices]
            action = expert_action.at[indices].set(linear_actions)
            action = sanitize_action(action)
            return action, expert_action

        _, _, rewards, dones = rollout(
            environment, jax.random.key(trajectory_seed), hybrid_action, steps
        )
        return masked_return(rewards, dones)

    seeds = seed + jnp.arange(trajectories)
    return float(jnp.mean(jax.vmap(evaluate_one)(seeds)))


def per_action_validation_mse(coefficients, intercept, dataset):
    test_X = jnp.asarray(dataset["test_X"], dtype=jnp.float32)
    test_y = jnp.asarray(dataset["test_y"], dtype=jnp.float32)
    weights = jnp.asarray(dataset["test_weights"], dtype=jnp.float32)
    squared_errors = jnp.square(test_X @ coefficients.T + intercept - test_y)
    return np.asarray(
        jnp.sum(squared_errors * weights[:, None], axis=0) / jnp.sum(weights)
    ).tolist()


def main():
    args = parse_arguments()
    experiments = Path(__file__).resolve().parents[2]
    modes = (
        ("uniform", "q_dagger")
        if args.state_weighting == "both" else (args.state_weighting,)
    )
    action_counts = {
        "one": (1,), "two": (2,), "both": (1, 2),
    }[args.linear_action_count]
    run_name = args.run_name or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_directory = (
        experiments / "artifacts" / "repertoires" / f"linear_action_hybrid_{args.env}" / run_name
    )
    run_directory.mkdir(parents=True, exist_ok=False)
    actor, _ = load_sac_actor(experiments / "artifacts" / "expert_models" / args.env / "final")
    environment = envs.get_environment(args.env, backend=args.backend)
    action_count = int(environment.action_size)
    action_subsets = [
        subset for count in action_counts
        for subset in combinations(range(action_count), count)
    ]
    requested_variants = len(action_subsets) * len(modes) * args.num_seeds
    config = vars(args) | {
        "action_count": action_count,
        "linear_action_counts": list(action_counts),
        "source_runs": SOURCE_RUNS,
        "experiment": "selected_linear_actions_with_remaining_ann_expert_actions",
        "fit_strategy": "reuse_action_heads_from_initial_linear_runs",
        "requested_variants": requested_variants,
        "script": "distillation_experiments.scripts.experiments.linear_action_hybrid",
    }
    (run_directory / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True)
    )
    summaries = []
    print(f"Writing linear-action hybrid results to {run_directory}")

    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        for mode_index, mode in enumerate(modes):
            source_directory = (
                experiments / "artifacts" / "repertoires" / f"dagger_linear_{args.env}"
                / SOURCE_RUNS[mode] / f"seed_{seed}"
            )
            model = np.load(source_directory / "final_model.npz")
            coefficients = jnp.asarray(model["coefficients"], dtype=jnp.float32)
            intercept = jnp.asarray(model["intercept"], dtype=jnp.float32)
            dataset = np.load(source_directory / "final_dataset.npz")
            validation_mse = per_action_validation_mse(
                coefficients, intercept, dataset
            )
            for linear_action_indices in action_subsets:
                ann_action_indices = [
                    index for index in range(action_count)
                    if index not in linear_action_indices
                ]
                evaluation_seed = seed * 100_000 + mode_index * 10_000 + 90_000
                reward = evaluate_linear_hybrid(
                    coefficients, intercept, actor, environment,
                    list(linear_action_indices),
                    steps=args.rollout_steps, seed=evaluation_seed,
                    trajectories=args.evaluation_trajectories,
                )
                summary = {
                    "seed": seed,
                    "mode": mode,
                    "linear_action_indices": list(linear_action_indices),
                    "ann_action_indices": ann_action_indices,
                    "coefficients": np.asarray(
                        coefficients[jnp.asarray(linear_action_indices)]
                    ).tolist(),
                    "intercept": np.asarray(
                        intercept[jnp.asarray(linear_action_indices)]
                    ).tolist(),
                    "source_validation_mse": [
                        validation_mse[index] for index in linear_action_indices
                    ],
                    "evaluation_seed": evaluation_seed,
                    "reward": reward,
                    "solved": bool(reward >= args.target_reward),
                }
                subset_name = "_".join(map(str, linear_action_indices))
                result_directory = (
                    run_directory / f"linear_action_count_{len(linear_action_indices)}"
                    / f"linear_actions_{subset_name}" / f"seed_{seed}" / mode
                )
                result_directory.mkdir(parents=True)
                (result_directory / "summary.json").write_text(
                    json.dumps(summary, indent=2)
                )
                summaries.append(summary)
                (run_directory / "aggregate_summary.json").write_text(json.dumps({
                    "completed_variants": len(summaries),
                    "requested_variants": requested_variants,
                    "solved_variants": sum(item["solved"] for item in summaries),
                    "variants": summaries,
                }, indent=2))
                print(
                    f"linear_actions={list(linear_action_indices)} "
                    f"ann_actions={ann_action_indices} seed={seed} mode={mode} "
                    f"reward={reward:.3f}"
                )


if __name__ == "__main__":
    main()
