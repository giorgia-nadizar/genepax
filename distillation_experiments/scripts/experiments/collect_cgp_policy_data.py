"""Collect uniformly weighted demonstrations from a saved CGP policy."""

import argparse
import json
import pickle
from pathlib import Path

from brax import envs
import jax
import jax.numpy as jnp
import numpy as np

from distillation.rollouts import masked_return, rollout, sanitize_action, valid_transition_mask
from distillation_experiments.scripts.experiments.dagger import positive_int
from genepax.gp.cartesian_genetic_programming import CGP


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--backend", default="generalized")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2_000_000)
    parser.add_argument("--trajectories", type=positive_int, default=100)
    parser.add_argument("--rollout-steps", type=positive_int, default=1_000)
    return parser.parse_args()


def main():
    args = parse_arguments()
    environment = envs.get_environment(args.env, backend=args.backend)
    cgp = CGP(
        n_inputs=environment.observation_size,
        n_outputs=environment.action_size,
    )
    with args.policy.open("rb") as file:
        genotype = pickle.load(file)

    def collect_one(trajectory_seed):
        def action_fn(observation, _key, _step):
            action = sanitize_action(cgp.apply(genotype, observation))
            return action, action

        X, y, rewards, dones = rollout(
            environment, jax.random.key(trajectory_seed), action_fn,
            args.rollout_steps,
        )
        mask = valid_transition_mask(dones).astype(bool)
        finite = jnp.all(jnp.isfinite(X), axis=-1) & jnp.all(
            jnp.isfinite(y), axis=-1
        )
        retained = mask & jnp.cumprod(finite.astype(jnp.int32)).astype(bool)
        return X, y, retained, masked_return(rewards, dones)

    seeds = args.seed + jnp.arange(args.trajectories)
    X, y, masks, returns = jax.vmap(collect_one)(seeds)
    retained_X = jnp.concatenate([
        X[index][masks[index]] for index in range(args.trajectories)
    ])
    retained_y = jnp.concatenate([
        y[index][masks[index]] for index in range(args.trajectories)
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=np.asarray(retained_X),
        y=np.asarray(retained_y),
        trajectory_returns=np.asarray(returns),
        trajectory_seeds=np.asarray(seeds),
    )
    metadata = {
        "environment": args.env,
        "backend": args.backend,
        "teacher": "direct-policy-search CGP",
        "policy_path": str(args.policy.resolve()),
        "collection_seed": args.seed,
        "trajectories": args.trajectories,
        "rollout_steps": args.rollout_steps,
        "samples": len(retained_X),
        "mean_teacher_return": float(jnp.mean(returns)),
        "median_teacher_return": float(jnp.median(returns)),
        "min_teacher_return": float(jnp.min(returns)),
        "max_teacher_return": float(jnp.max(returns)),
        "state_weighting": "uniform",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
