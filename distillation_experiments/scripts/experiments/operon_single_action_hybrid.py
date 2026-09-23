"""Evaluate one fitted Operon action at a time inside an ANN expert policy."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from brax import envs
import jax
import jax.numpy as jnp

from distillation.networks.sac_utils import load_sac_actor
from distillation.rollouts import masked_return, rollout, sanitize_action
from distillation_experiments.scripts.experiments.dagger import positive_int
from distillation_experiments.scripts.experiments.operon_imitation import expression_to_jax


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


def evaluate_hybrid_expression(
    callable_, actor, environment, action_index, *, steps, seed, trajectories
):
    """Evaluate a policy with one ANN action replaced by an Operon expression."""

    return evaluate_hybrid_expressions(
        [callable_], actor, environment, [action_index],
        steps=steps, seed=seed, trajectories=trajectories,
    )


def evaluate_hybrid_expressions(
    callables, actor, environment, action_indices, *, steps, seed, trajectories
):
    """Evaluate a policy with selected ANN actions replaced by Operon outputs."""
    if len(callables) != len(action_indices):
        raise ValueError("callables and action_indices must have equal length")

    def evaluate_one(trajectory_seed):
        def hybrid_action(observation, action_key, _step):
            expert_action, _ = actor(observation, action_key)
            symbolic_actions = jnp.stack([
                jnp.asarray(callable_(*observation)).reshape(())
                for callable_ in callables
            ])
            action = expert_action.at[jnp.asarray(action_indices)].set(
                symbolic_actions
            )
            action = sanitize_action(action)
            return action, expert_action

        _, _, rewards, dones = rollout(
            environment, jax.random.key(trajectory_seed), hybrid_action, steps
        )
        return masked_return(rewards, dones)

    seeds = seed + jnp.arange(trajectories)
    return float(jnp.mean(jax.vmap(evaluate_one)(seeds)))


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
        experiments / "artifacts" / "repertoires" / f"operon_single_action_hybrid_{args.env}"
        / run_name
    )
    run_directory.mkdir(parents=True, exist_ok=False)
    checkpoint = experiments / "artifacts" / "expert_models" / args.env / "final"
    actor, _ = load_sac_actor(checkpoint)
    environment = envs.get_environment(args.env, backend=args.backend)
    action_count = int(environment.action_size)
    modes = (
        ("uniform", "q_dagger")
        if args.state_weighting == "both"
        else (args.state_weighting,)
    )
    config = vars(args) | {
        "action_count": action_count,
        "experiment": "single_operon_action_with_remaining_ann_expert_actions",
        "fit_strategy": "reuse_independently_fitted_action_from_source_run",
        "source_directory": str(source_directory),
        "requested_variants": action_count * len(modes) * args.num_seeds,
        "script": "distillation_experiments.scripts.experiments.operon_single_action_hybrid",
    }
    (run_directory / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True)
    )
    summaries = []
    print(f"Writing single-action hybrid results to {run_directory}")

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
            for action_index, expression in enumerate(expressions):
                callable_ = expression_to_jax(
                    expression, int(environment.observation_size)
                )
                evaluation_seed = (
                    seed * 100_000 + mode_index * 10_000
                    + 90_000
                )
                reward = evaluate_hybrid_expression(
                    callable_, actor, environment, action_index,
                    steps=args.rollout_steps,
                    seed=evaluation_seed,
                    trajectories=args.evaluation_trajectories,
                )
                summary = {
                    "seed": seed,
                    "mode": mode,
                    "operon_action_index": action_index,
                    "ann_action_indices": [
                        index for index in range(action_count)
                        if index != action_index
                    ],
                    "expression": expression,
                    "source_validation_mse": source_summary[
                        "per_action_validation_mse"
                    ][action_index],
                    "evaluation_seed": evaluation_seed,
                    "reward": reward,
                    "solved": bool(reward >= args.target_reward),
                }
                result_directory = (
                    run_directory / f"action_{action_index}" / f"seed_{seed}" / mode
                )
                result_directory.mkdir(parents=True)
                (result_directory / "summary.json").write_text(
                    json.dumps(summary, indent=2)
                )
                summaries.append(summary)
                (run_directory / "aggregate_summary.json").write_text(json.dumps({
                    "completed_variants": len(summaries),
                    "requested_variants": action_count * len(modes) * args.num_seeds,
                    "solved_variants": sum(item["solved"] for item in summaries),
                    "variants": summaries,
                }, indent=2))
                print(
                    f"action={action_index} seed={seed} mode={mode} "
                    f"reward={reward:.3f}"
                )


if __name__ == "__main__":
    main()
