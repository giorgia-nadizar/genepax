"""Audit completed ANN sweep artifacts against their frozen reference data."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--verify-mse', action='store_true',
                        help='Recompute train/held-out losses for seed 0 at k=1 and k=32 in every group')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    sweep = root / 'distillation_experiments/artifacts/repertoires/ann_feature_sensitivity_5seeds'
    manifest = json.loads((sweep / 'manifest.json').read_text())
    for path, expected in manifest['reference_sha256'].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Reference data changed: {path}')
    cache = {}
    evaluators = {}
    records = []
    for job in manifest['jobs']:
        directory = root / job['run_directory'] / f"seed_{job['seed']}"
        if not (directory / 'summary.json').exists():
            if args.allow_partial:
                continue
            raise FileNotFoundError(directory / 'summary.json')
        summary = json.loads((directory / 'summary.json').read_text())
        reference_path = root / job['reference_run'] / f"seed_{job['seed']}" / 'dataset.npz'
        if reference_path not in cache:
            with np.load(reference_path) as data:
                cache[reference_path] = {key: data[key] for key in data.files}
        reference = cache[reference_path]
        with np.load(directory / 'dataset.npz') as data:
            for key in ('X', 'y', 'validation_X', 'validation_y'):
                np.testing.assert_array_equal(data[key], reference[key])
            if job['weighting'] == 'q_dagger':
                for key in ('train_weights', 'validation_weights'):
                    np.testing.assert_array_equal(data[key], reference[key])
            actions = data['y'].shape[1]
        with (directory / 'final_individual.pickle').open('rb') as file:
            genotype = pickle.load(file)
        graphs = genotype['actions'] if job['architecture'] == 'independent' else (genotype,)
        assert len(graphs) == (actions if job['architecture'] == 'independent' else 1)
        for graph in graphs:
            assert graph['genes']['outputs'].shape == (job['k'],)
            assert graph['genes']['functions'].shape == (50,)
            expected_weights = (job['k'] + 1) * (1 if job['architecture'] == 'independent' else actions)
            assert graph['weights']['custom_weights'].shape == (expected_weights,)
            assert np.isfinite(np.asarray(graph['weights']['custom_weights'])).all()
        readout = ([np.asarray(graph['weights']['custom_weights']).tolist() for graph in graphs]
                   if job['architecture'] == 'independent' else
                   np.asarray(genotype['weights']['custom_weights']).reshape(job['k'] + 1, actions))
        np.testing.assert_array_equal(readout, summary['readout'])
        assert np.isfinite(summary['train_mse']) and np.isfinite(summary['validation_mse'])
        reloaded_metrics = {}
        if args.verify_mse and job['seed'] == 0 and job['k'] in (1, 32):
            import jax
            import jax.numpy as jnp
            from distillation.fit_feature_imitation import make_feature_cgp, feature_policy_mse
            from distillation.independent_feature_policy import independent_feature_mse
            evaluator_key = (job['environment'], job['architecture'], job['k'])
            with np.load(directory / 'dataset.npz') as data:
                if evaluator_key not in evaluators:
                    independent = job['architecture'] == 'independent'
                    cgp = make_feature_cgp(data['X'].shape[1], 1 if independent else actions, job['k'], 50)
                    metric = independent_feature_mse if independent else feature_policy_mse
                    # Match the experiment runner's evaluation path exactly;
                    # an extra outer jit can change float32 cancellation.
                    evaluators[evaluator_key] = (
                        lambda policy, X, y, w, cgp=cgp, metric=metric: metric(policy, cgp, X, y, w))
                evaluate = evaluators[evaluator_key]
                for prefix, X, y, weights in [('train', 'X', 'y', 'train_weights'),
                                             ('validation', 'validation_X', 'validation_y', 'validation_weights')]:
                    for weighted in ([False, True] if job['weighting'] == 'q_dagger' else [False]):
                        name = prefix + ('_weighted_mse' if weighted else '_mse')
                        value = float(evaluate(genotype, jnp.asarray(data[X]), jnp.asarray(data[y]),
                                               jnp.asarray(data[weights]) if weighted else None))
                        # CPU and CUDA float32 graph evaluation differ slightly,
                        # especially for small losses with cancelling readouts.
                        np.testing.assert_allclose(value, summary[name], rtol=1e-4, atol=1e-6,
                                                   err_msg=f'{directory}: {name}')
                        reloaded_metrics[name] = dict(value=value, reported=summary[name],
                                                      absolute_difference=abs(value - summary[name]),
                                                      backend=jax.default_backend())
        records.append(job | dict(passed=True, graphs=len(graphs), raw_reward_finite=bool(np.isfinite(summary['reward'])),
                                  reloaded_metrics=reloaded_metrics))
    if not args.allow_partial:
        assert len(records) == 360
    (sweep / ('partial_audit.json' if args.allow_partial else 'audit.json')).write_text(json.dumps(records, indent=2) + '\n')
    print(f'Audited {len(records)} policy bundles / {sum(r["graphs"] for r in records)} graphs; exact reference splits and Q weights verified.')
    if args.verify_mse:
        print(f'Reloaded MSE verified for {sum(bool(r["reloaded_metrics"]) for r in records)} representative policies.')


if __name__ == '__main__':
    main()
