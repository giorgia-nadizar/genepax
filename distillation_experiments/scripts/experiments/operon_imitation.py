"""Fit Operon symbolic policies to ANN expert actions and validate in Brax."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from brax import envs
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pyoperon.sklearn import SymbolicRegressor
import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    parse_expr,
    standard_transformations,
)

from distillation.networks.sac_utils import load_q_value_estimator
from distillation.q_dagger import compute_q_dagger_weights
from distillation.rollouts import masked_return, rollout, sanitize_action
from distillation_experiments.scripts.experiments.dagger import positive_int, select_rows
from distillation_experiments.scripts.experiments.operon_primitives import (
    OPERON_PRIMITIVE_PROFILES,
    operon_allowed_symbols,
)


def split_dataset(X, y, weights, fraction, key):
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation-fraction must be strictly between 0 and 1")
    indices = jax.random.permutation(key, len(X))
    validation_size = max(1, round(len(X) * fraction))
    validation = indices[:validation_size]
    training = indices[validation_size:]
    return (
        X[training], y[training], weights[training],
        X[validation], y[validation], weights[validation],
    )


def weighted_mse(predictions, targets, weights):
    per_sample = jnp.mean(jnp.square(predictions - targets), axis=-1)
    return float(jnp.sum(per_sample * weights) / jnp.sum(weights))


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="inverted_pendulum")
    parser.add_argument("--backend", default="generalized")
    parser.add_argument("--run-name")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=positive_int, default=1)
    parser.add_argument(
        "--state-weighting", choices=("uniform", "q_dagger", "both"),
        default="both",
    )
    parser.add_argument("--expert-samples", type=positive_int, default=10_000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--generations", type=positive_int, default=100)
    parser.add_argument("--population-size", type=positive_int, default=500)
    parser.add_argument("--max-evaluations", type=positive_int, default=100_000)
    parser.add_argument("--max-length", type=positive_int, default=25)
    parser.add_argument(
        "--primitive-set",
        choices=tuple(OPERON_PRIMITIVE_PROFILES),
        default="compact",
        help="Operon primitive profile; 'cgp' mirrors the numeric CGP functions.",
    )
    parser.add_argument("--threads", type=positive_int, default=8)
    parser.add_argument("--rollout-steps", type=positive_int, default=1_000)
    parser.add_argument("--evaluation-trajectories", type=positive_int, default=10)
    parser.add_argument("--target-reward", type=float, default=999.0)
    parser.add_argument("--q-action-grid-size", type=positive_int, default=101)
    parser.add_argument("--q-batch-size", type=positive_int, default=1_000)
    return parser.parse_args()


def create_run_directory(root, environment, run_name):
    if run_name is None:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"operon_{environment}" / run_name
    path.mkdir(parents=True, exist_ok=False)
    return path


def expression_to_jax(expression, input_count):
    symbols = sympy.symbols(f"X1:{input_count + 1}")
    local_dict = {str(symbol): symbol for symbol in symbols}
    parsed = parse_expr(
        expression,
        local_dict=local_dict,
        transformations=standard_transformations + (convert_xor,),
    )
    return sympy.lambdify(symbols, parsed, modules="jax")


def evaluate_expressions(
    callables, environment, *, steps, seed, trajectories
):
    def evaluate_one(trajectory_seed):
        def action_fn(observation, _key, _step):
            raw_action = jnp.stack([
                jnp.asarray(callable_(*observation)).reshape(())
                for callable_ in callables
            ])
            action = sanitize_action(raw_action)
            return action, action

        _, _, rewards, dones = rollout(
            environment, jax.random.key(trajectory_seed), action_fn, steps
        )
        return masked_return(rewards, dones)

    seeds = seed + jnp.arange(trajectories)
    return float(jnp.mean(jax.vmap(evaluate_one)(seeds)))


def evaluate_expression(
    callable_, environment, *, steps, seed, trajectories
):
    """Backward-compatible evaluator for single-action environments."""
    return evaluate_expressions(
        [callable_], environment, steps=steps, seed=seed,
        trajectories=trajectories,
    )


def main():
    args = parse_arguments()
    experiments = Path(__file__).resolve().parents[2]
    dataset_path = args.dataset_path or (
        experiments / "artifacts" / "expert_datasets" / f"expert_{args.env}.npz"
    )
    checkpoint_path = experiments / "artifacts" / "expert_models" / args.env / "final"
    run_directory = create_run_directory(
        experiments / "artifacts" / "repertoires", args.env, args.run_name
    )
    with (run_directory / "config.json").open("w") as file:
        config = vars(args).copy()
        config["dataset_path"] = str(dataset_path)
        json.dump(config, file, indent=2, sort_keys=True)
    data = np.load(dataset_path)
    all_X = jnp.asarray(data["X"], dtype=jnp.float32)
    all_y = jnp.asarray(data["y"], dtype=jnp.float32)
    modes = (
        ("uniform", "q_dagger")
        if args.state_weighting == "both"
        else (args.state_weighting,)
    )
    q_estimator = (
        load_q_value_estimator(checkpoint_path)
        if "q_dagger" in modes
        else None
    )
    summaries = []
    print(f"Writing Operon results to {run_directory}")

    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        sampled_X, sampled_y = select_rows(
            all_X, all_y, args.expert_samples, jax.random.key(seed)
        )
        q_weights = None
        if q_estimator is not None:
            q_weights, _ = compute_q_dagger_weights(
                sampled_X, sampled_y, q_estimator,
                args.q_action_grid_size, args.q_batch_size,
            )
        environment = envs.get_environment(args.env, backend=args.backend)

        for mode_index, mode in enumerate(modes):
            weights = (
                jnp.ones(len(sampled_X), dtype=jnp.float32)
                if mode == "uniform"
                else q_weights
            )
            train_X, train_y, train_weights, val_X, val_y, val_weights = (
                split_dataset(
                    sampled_X, sampled_y, weights, args.validation_fraction,
                    jax.random.key(seed + 50_000),
                )
            )
            # Operon's sklearn binding has no sample_weight argument. Drawing a
            # fixed-size weighted bootstrap makes its empirical MSE approximate
            # the same Q-weighted objective used by the other regressors.
            rng = np.random.default_rng(seed * 10 + mode_index)
            probabilities = np.asarray(train_weights, dtype=np.float64)
            probabilities /= probabilities.sum()
            training_indices = rng.choice(
                len(train_X), size=len(train_X), replace=True,
                p=probabilities,
            )
            fit_X = np.asarray(train_X)[training_indices]
            search_history = []
            candidates = []
            selected_expressions = []
            selected_positions = []
            selected_validation_losses = []
            for action_index in range(all_y.shape[1]):
                fit_y = np.asarray(train_y[:, action_index])[training_indices]
                model = SymbolicRegressor(
                    allowed_symbols=operon_allowed_symbols(args.primitive_set),
                    generations=1,
                    population_size=args.population_size,
                    pool_size=args.population_size,
                    max_evaluations=args.max_evaluations,
                    max_length=args.max_length,
                    max_depth=10,
                    initialization_max_length=10,
                    objectives=["mse", "length"],
                    model_selection_criterion="minimum_description_length",
                    optimizer="lm",
                    optimizer_iterations=1,
                    add_model_scale_term=True,
                    add_model_intercept_term=True,
                    n_threads=args.threads,
                    random_state=seed * 100 + mode_index * 10 + action_index,
                    warm_start=True,
                )
                for generation in range(args.generations):
                    model.fit(fit_X, fit_y)
                    current_losses = np.asarray([
                        solution["mean_squared_error"]
                        for solution in model.pareto_front_
                    ], dtype=float)
                    search_history.append({
                        "action_index": action_index,
                        "generation": generation,
                        "best_loss": float(np.min(current_losses)),
                        "median_loss": float(np.median(current_losses)),
                        "candidate_count": len(current_losses),
                    })
                original_model = model.model_
                action_candidates = []
                for position, solution in enumerate(model.pareto_front_):
                    model.model_ = solution["tree"]
                    predictions = jnp.asarray(
                        model.predict(np.asarray(val_X)), dtype=jnp.float32
                    ).reshape(-1, 1)
                    loss = weighted_mse(
                        predictions,
                        val_y[:, action_index:action_index + 1],
                        val_weights,
                    )
                    action_candidates.append({
                        "action_index": action_index,
                        "position": position,
                        "expression": solution["model"],
                        "length": int(solution["length"]),
                        "operon_mse": float(solution["mean_squared_error"]),
                        "validation_mse": loss,
                    })
                model.model_ = original_model
                candidates.extend(action_candidates)
                action_frame = pd.DataFrame(action_candidates)
                best = action_frame.loc[action_frame.validation_mse.idxmin()]
                selected_expressions.append(best.expression)
                selected_positions.append(int(best.position))
                selected_validation_losses.append(float(best.validation_mse))
            candidate_frame = pd.DataFrame(candidates)
            result_directory = run_directory / f"seed_{seed}" / mode
            result_directory.mkdir(parents=True)
            pd.DataFrame(search_history).to_csv(
                result_directory / "search_history.csv", index=False
            )
            candidate_frame.to_csv(result_directory / "candidates.csv", index=False)
            try:
                selected_callables = [
                    expression_to_jax(expression, val_X.shape[1])
                    for expression in selected_expressions
                ]
                reward = evaluate_expressions(
                    selected_callables,
                    environment,
                    steps=args.rollout_steps,
                    seed=seed * 100_000 + mode_index * 10_000 + 90_000,
                    trajectories=args.evaluation_trajectories,
                )
            except (FloatingPointError, TypeError, ValueError):
                reward = 0.0
            validation_mse = float(np.mean(selected_validation_losses))
            summary = {
                "seed": seed,
                "mode": mode,
                "candidate_count": len(candidate_frame),
                "action_count": int(all_y.shape[1]),
                "selected_positions": selected_positions,
                "expressions": selected_expressions,
                "per_action_validation_mse": selected_validation_losses,
                "validation_mse": validation_mse,
                "reward": reward,
                "solved": bool(reward >= args.target_reward),
                "weight_effective_sample_size": float(
                    jnp.square(jnp.sum(weights)) / jnp.sum(jnp.square(weights))
                ),
                "q_weight_implementation": (
                    "not_applicable" if mode == "uniform"
                    else "fixed_size_weighted_bootstrap"
                ),
            }
            with (result_directory / "summary.json").open("w") as file:
                json.dump(summary, file, indent=2)
            summaries.append(summary)
            with (run_directory / "aggregate_summary.json").open("w") as file:
                json.dump({"variants": summaries}, file, indent=2)
            print(
                f"seed={seed} mode={mode} reward={reward:.1f} "
                f"validation_mse={validation_mse:.6g} "
                f"expressions={selected_expressions}"
            )


if __name__ == "__main__":
    main()
