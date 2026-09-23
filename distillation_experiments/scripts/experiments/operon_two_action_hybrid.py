"""Evaluate two fitted Operon actions at a time inside an ANN expert policy."""

import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path

from brax import envs

from distillation.networks.sac_utils import load_sac_actor
from distillation_experiments.scripts.experiments.dagger import positive_int
from distillation_experiments.scripts.experiments.operon_imitation import expression_to_jax
from distillation_experiments.scripts.experiments.operon_single_action_hybrid import (
    evaluate_hybrid_expressions,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="hopper")
    parser.add_argument("--backend", default="generalized")
    parser.add_argument("--source-run", default="operon_initial_both_5seeds")
    parser.add_argument("--run-name")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=positive_int, default=5)
    parser.add_argument(
        "--state-weighting", choices=("uniform", "q_dagger", "both"),
        default="both",
    )
    parser.add_argument("--rollout-steps", type=positive_int, default=1_000)
    parser.add_argument("--evaluation-trajectories", type=positive_int, default=10)
    parser.add_argument("--target-reward", type=float, default=3_250.0)
    return parser.parse_args()


def main():
    args = parse_arguments()
    experiments = Path(__file__).resolve().parents[2]
    source_directory = (
        experiments / "artifacts" / "repertoires" / f"operon_{args.env}" / args.source_run
    )
    if not source_directory.is_dir():
        raise FileNotFoundError(f"Operon source run does not exist: {source_directory}")
    run_name = args.run_name or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_directory = (
        experiments / "artifacts" / "repertoires" / f"operon_two_action_hybrid_{args.env}"
        / run_name
    )
    run_directory.mkdir(parents=True, exist_ok=False)
    checkpoint = experiments / "artifacts" / "expert_models" / args.env / "final"
    actor, _ = load_sac_actor(checkpoint)
    environment = envs.get_environment(args.env, backend=args.backend)
    action_count = int(environment.action_size)
    if action_count < 3:
        raise ValueError("Two-action hybrid evaluation requires at least 3 actions")
    operon_action_pairs = list(combinations(range(action_count), 2))
    modes = (
        ("uniform", "q_dagger")
        if args.state_weighting == "both"
        else (args.state_weighting,)
    )
    requested_variants = len(operon_action_pairs) * len(modes) * args.num_seeds
    config = vars(args) | {
        "action_count": action_count,
        "operon_action_count": 2,
        "operon_action_pairs": [list(pair) for pair in operon_action_pairs],
        "experiment": "two_operon_actions_with_remaining_ann_expert_actions",
        "fit_strategy": "reuse_independently_fitted_actions_from_source_run",
        "source_directory": str(source_directory),
        "requested_variants": requested_variants,
        "script": "distillation_experiments.scripts.experiments.operon_two_action_hybrid",
    }
    (run_directory / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True)
    )
    summaries = []
    print(f"Writing two-action hybrid results to {run_directory}")

    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        for mode_index, mode in enumerate(modes):
            source_summary_path = (
                source_directory / f"seed_{seed}" / mode / "summary.json"
            )
            source_summary = json.loads(source_summary_path.read_text())
            expressions = source_summary["expressions"]
            if len(expressions) != action_count:
                raise ValueError(
                    f"Expected {action_count} expressions in {source_summary_path}, "
                    f"found {len(expressions)}"
                )
            for operon_action_indices in operon_action_pairs:
                ann_action_indices = [
                    index for index in range(action_count)
                    if index not in operon_action_indices
                ]
                selected_expressions = [
                    expressions[index] for index in operon_action_indices
                ]
                callables = [
                    expression_to_jax(expression, int(environment.observation_size))
                    for expression in selected_expressions
                ]
                evaluation_seed = seed * 100_000 + mode_index * 10_000 + 90_000
                reward = evaluate_hybrid_expressions(
                    callables, actor, environment, list(operon_action_indices),
                    steps=args.rollout_steps,
                    seed=evaluation_seed,
                    trajectories=args.evaluation_trajectories,
                )
                summary = {
                    "seed": seed,
                    "mode": mode,
                    "operon_action_indices": list(operon_action_indices),
                    "ann_action_indices": ann_action_indices,
                    "expressions": selected_expressions,
                    "source_validation_mse": [
                        source_summary["per_action_validation_mse"][index]
                        for index in operon_action_indices
                    ],
                    "evaluation_seed": evaluation_seed,
                    "reward": reward,
                    "solved": bool(reward >= args.target_reward),
                }
                pair_name = "_".join(map(str, operon_action_indices))
                result_directory = (
                    run_directory / f"operon_actions_{pair_name}"
                    / f"seed_{seed}" / mode
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
                    f"operon_actions={list(operon_action_indices)} "
                    f"ann_actions={ann_action_indices} seed={seed} mode={mode} "
                    f"reward={reward:.3f}"
                )


if __name__ == "__main__":
    main()
