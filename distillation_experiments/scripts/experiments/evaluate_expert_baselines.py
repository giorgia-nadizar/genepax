"""Evaluate saved ANN experts and persist seed-matched reward baselines."""

import argparse
import json
from pathlib import Path

from brax import envs
import jax
import jax.numpy as jnp

from distillation.networks.sac_utils import load_sac_actor
from distillation.rollouts import masked_return, rollout


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument("--backend", default="generalized")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--rollout-steps", type=int, default=1_000)
    parser.add_argument("--evaluation-trajectories", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def evaluate_actor(actor, environment, seeds, rollout_steps):
    def evaluate_one(trajectory_seed):
        def action_fn(observation, action_key, _step):
            action, _ = actor(observation, action_key)
            return action, action

        _, _, rewards, dones = rollout(
            environment,
            jax.random.key(trajectory_seed),
            action_fn,
            rollout_steps,
        )
        return masked_return(rewards, dones)

    return jax.vmap(evaluate_one)(jnp.asarray(seeds))


def main():
    args = parse_arguments()
    experiments = Path(__file__).resolve().parents[2]
    results = {}
    for environment_name in args.envs:
        environment = envs.get_environment(
            environment_name, backend=args.backend
        )
        actor, _ = load_sac_actor(
            experiments / "artifacts" / "expert_models" / environment_name / "final"
        )
        seed_results = {}
        all_returns = []
        for dataset_seed in range(
            args.seed_start, args.seed_start + args.num_seeds
        ):
            evaluation_seeds = (
                dataset_seed * 100_000
                + 90_000
                + jnp.arange(args.evaluation_trajectories)
            )
            returns = evaluate_actor(
                actor, environment, evaluation_seeds, args.rollout_steps
            )
            all_returns.append(returns)
            seed_results[str(dataset_seed)] = {
                "mean_reward": float(jnp.mean(returns)),
                "min_reward": float(jnp.min(returns)),
                "max_reward": float(jnp.max(returns)),
                "trajectories": args.evaluation_trajectories,
            }
        all_returns = jnp.concatenate(all_returns)
        results[environment_name] = {
            "seeds": seed_results,
            "overall_mean_reward": float(jnp.mean(all_returns)),
            "overall_min_reward": float(jnp.min(all_returns)),
            "overall_max_reward": float(jnp.max(all_returns)),
            "rollout_steps": args.rollout_steps,
            "evaluation_trajectories_per_seed": args.evaluation_trajectories,
        }
        print(
            f"{environment_name}: mean_reward="
            f"{results[environment_name]['overall_mean_reward']:.3f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"Saved expert baselines to {args.output.resolve()}")


if __name__ == "__main__":
    main()
