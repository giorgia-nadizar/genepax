"""Reproduce the Hopper small-ANN DAgger protocol with CGP feature readouts.

Run without arguments to execute all 20 variants in isolated GPU processes.
Completed variants are skipped on restart; partial variants restart from bootstrap.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'distillation_experiments/artifacts/repertoires'
REFERENCE = BASE / 'dagger_neural_hopper/small_ann_32x32_dagger_both_5seeds_target3250'
RUN = BASE / 'dagger_cgp_feature_hopper/k16_ann_protocol_both_5seeds_target3250'


def load_bootstrap(reference, seed, mode):
    """The ANN replay is append-only: its first 8,000 rows are the bootstrap."""
    import numpy as np
    with np.load(reference / f'seed_{seed}' / mode / 'final_dataset.npz') as data:
        return {key: data[key][:8000].copy() if key in ('X', 'y', 'sample_weights')
                else data[key].copy()
                for key in ('X', 'y', 'sample_weights', 'test_X', 'test_y', 'test_weights')}


def audit_replay(directory, last_iteration):
    """Verify every saved replay row and ensure the held-out split never changes."""
    import numpy as np
    with np.load(directory / 'initial_dataset.npz') as initial, np.load(directory / 'final_dataset.npz') as final:
        for key in ('test_X', 'test_y', 'test_weights'):
            np.testing.assert_array_equal(initial[key], final[key])
        for key in ('X', 'y', 'sample_weights'):
            chunks = [initial[key]]
            for iteration in range(last_iteration):
                with np.load(directory / f'iteration_{iteration}/collected_dataset.npz') as batch:
                    chunks.append(batch[key])
            np.testing.assert_array_equal(np.concatenate(chunks), final[key])
    (directory / 'audit.json').write_text(json.dumps({'replay_exact': True, 'held_out_unchanged': True}))


def run_variant(args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from brax import envs
    from distillation.fit_feature_imitation import (
        feature_policy_action, feature_policy_mse, fit_feature_imitation, make_feature_cgp,
    )
    from distillation.independent_feature_policy import independent_feature_action, independent_feature_mse
    from distillation.networks.sac_utils import load_sac_actor, load_q_value_estimator
    from distillation.q_dagger import compute_q_dagger_weights
    from distillation_experiments.scripts.experiments.cgp_feature_imitation import evaluate_feature_policy
    from distillation_experiments.scripts.experiments.cgp_action_feature_imitation import evaluate_feature_policy as evaluate_independent
    from distillation_experiments.scripts.experiments.dagger_cgp_features import collect_feature_trajectories

    if jax.default_backend() != 'gpu':
        raise RuntimeError('This experiment requires the GPU; refusing silent CPU fallback')
    config = json.loads((REFERENCE / 'config.json').read_text())
    directory = args.output / args.architecture / f'seed_{args.seed}' / args.mode
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'summary.json').exists():
        return
    data = load_bootstrap(REFERENCE, args.seed, args.mode)
    assert data['X'].shape == (8000, 11) and data['test_X'].shape == (2000, 11)
    np.savez_compressed(directory / 'initial_dataset.npz', **data)
    X, y, weights, test_X, test_y, test_weights = (
        jnp.asarray(data[k]) for k in ('X', 'y', 'sample_weights', 'test_X', 'test_y', 'test_weights'))
    independent = args.architecture == 'independent'
    cgp = make_feature_cgp(X.shape[1], 1 if independent else y.shape[1], 16, 50)
    action = independent_feature_action if independent else feature_policy_action
    mse = independent_feature_mse if independent else feature_policy_mse
    evaluate = evaluate_independent if independent else evaluate_feature_policy
    checkpoint = ROOT / 'distillation_experiments/artifacts/expert_models/hopper/final'
    actor, _ = load_sac_actor(checkpoint)
    critic = load_q_value_estimator(checkpoint) if args.mode == 'q_dagger' else None
    environment = envs.get_environment('hopper', backend=config['backend'])
    metrics = []
    pending = dict(behavior_reward=None, new_samples=len(X), invalid_transitions=0)
    for iteration in range(args.iterations):
        started = time.perf_counter()
        destination = directory / f'iteration_{iteration}'
        destination.mkdir(exist_ok=True)
        actions = []
        for index in range(y.shape[1] if independent else 1):
            fit = fit_feature_imitation(
                X, y[:, index:index+1] if independent else y, cgp,
                seed=args.seed * 1000000 + (args.mode == 'q_dagger') * 100000 + iteration * 1000 + index,
                n_gens=args.generations, n_pop=args.population_size, sample_weights=weights)
            actions.append(fit['genotype'])
            with (destination / f'action_{index}_fit.pickle').open('wb') as f:
                pickle.dump(fit, f)
        policy = {'actions': tuple(actions)} if independent else actions[0]
        fit_seconds = time.perf_counter() - started
        with (destination / 'individual.pickle').open('wb') as f:
            pickle.dump(policy, f)
        returns = evaluate(policy, cgp, environment, seed=args.seed * 100000 + 90000,
                           steps=args.rollout_steps, trajectories=args.evaluation_trajectories)
        reward = float(jnp.mean(returns))
        metric = dict(iteration=iteration, student_reward=reward,
            imitation_loss=float(mse(policy, cgp, X, y, weights)),
            test_imitation_loss=float(mse(policy, cgp, test_X, test_y, test_weights)),
            train_mse=float(mse(policy, cgp, X, y)),
            validation_mse=float(mse(policy, cgp, test_X, test_y)),
            dataset_size=len(X), fit_seconds=fit_seconds, **pending)
        metrics.append(metric)
        with (directory / 'metrics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metric.keys())
            writer.writeheader()
            writer.writerows(metrics)
        (destination / 'summary.json').write_text(json.dumps(metric | {'trajectory_returns': np.asarray(returns).tolist()}, indent=2))
        print(f'{args.architecture} seed={args.seed} {args.mode} iteration={iteration} reward={reward:.3f} rows={len(X)}', flush=True)
        if reward >= config['target_reward'] or iteration + 1 == args.iterations:
            break
        new_X, new_y, pending = collect_feature_trajectories(
            policy, cgp, actor, environment, seed=args.seed * 100000 + iteration * 10000,
            steps=args.rollout_steps, trajectories=args.trajectories, policy_action=action)
        if critic is None:
            new_weights = jnp.ones(len(new_X))
        else:
            new_weights, _ = compute_q_dagger_weights(new_X, new_y, critic,
                config['q_action_grid_size'], config['q_batch_size'])
        np.savez_compressed(destination / 'collected_dataset.npz', X=new_X, y=new_y, sample_weights=new_weights)
        X, y, weights = jnp.concatenate([X, new_X]), jnp.concatenate([y, new_y]), jnp.concatenate([weights, new_weights])
        jax.clear_caches()
    np.savez_compressed(directory / 'final_dataset.npz', X=X, y=y, sample_weights=weights,
                        test_X=test_X, test_y=test_y, test_weights=test_weights)
    with (directory / 'final_individual.pickle').open('wb') as f:
        pickle.dump(policy, f)
    audit_replay(directory, iteration)
    summary = dict(seed=args.seed, architecture=args.architecture, mode=args.mode, k=16,
        last_iteration=iteration, final_reward=reward, solved=reward >= config['target_reward'],
        dataset_size=len(X), best_observed_reward=max(m['student_reward'] for m in metrics))
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture', choices=['shared', 'independent'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', choices=['uniform', 'q_dagger'], default='uniform')
    parser.add_argument('--output', type=Path, default=RUN)
    parser.add_argument('--generations', type=int, default=100)
    parser.add_argument('--population-size', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--rollout-steps', type=int, default=1000)
    parser.add_argument('--evaluation-trajectories', type=int, default=10)
    parser.add_argument('--trajectories', type=int, default=20)
    args = parser.parse_args()
    if min(args.generations, args.population_size, args.iterations, args.rollout_steps,
           args.evaluation_trajectories, args.trajectories) < 1 or not 0 <= args.seed < 5:
        parser.error('Counts must be positive and seed must be between 0 and 4')
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((REFERENCE / 'config.json').read_text()) | dict(
        student='CGP features plus least-squares readout with intercept', k=16, n_nodes=50,
        generations=args.generations, population_size=args.population_size,
        iterations=args.iterations, rollout_steps=args.rollout_steps,
        evaluation_trajectories=args.evaluation_trajectories, trajectories_per_iteration=args.trajectories,
        run_name=args.output.name, script=__name__, reference_run=str(REFERENCE),
        fit_sampling='all replay rows', q_weight_normalization='per collection batch, matching ANN reference',
        architectures=['shared', 'independent'], independent_budget='three separately evolved graphs, 16 features and 50 nodes each',
        bootstrap='exact initial rows and weights from saved ANN replay; fixed held-out set',
        model_selection='minimum weighted clipped-action training MSE; held-out set reporting only')
    for key in ('hidden_sizes', 'epochs', 'batch_size', 'learning_rate', 'patience'):
        protocol.pop(key, None)
    protocol_path = args.output / 'config.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Output directory contains a different experiment configuration')
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2))
    if args.architecture:
        run_variant(args)
        return
    from distillation_experiments.scripts.analysis.hopper_feature_dagger_report import refresh
    refresh(args.output)
    status_path = args.output / 'status.json'
    for seed in range(5):
        for architecture in ('shared', 'independent'):
            for mode in ('uniform', 'q_dagger'):
                cmd = [sys.executable, '-u', '-m', 'distillation_experiments.scripts.experiments.hopper_feature_dagger',
                       '--architecture', architecture, '--seed', str(seed), '--mode', mode, '--output', str(args.output)]
                for name in ('generations', 'population_size', 'iterations', 'rollout_steps', 'evaluation_trajectories', 'trajectories'):
                    cmd += ['--' + name.replace('_', '-'), str(getattr(args, name))]
                current = dict(architecture=architecture, seed=seed, mode=mode)
                status_path.write_text(json.dumps(dict(status='running', current=current), indent=2))
                try:
                    subprocess.run(cmd, cwd=ROOT, env=os.environ.copy(), check=True)
                except subprocess.CalledProcessError as error:
                    status_path.write_text(json.dumps(dict(status='failed', current=current, exit_code=error.returncode), indent=2))
                    raise
                refresh(args.output)
    status_path.write_text(json.dumps(dict(status='complete', completed_variants=20), indent=2))
    print('All 20 CGP DAgger variants completed.', flush=True)


if __name__ == '__main__':
    main()
