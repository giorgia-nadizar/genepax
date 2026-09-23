"""Resumable ANN-teacher feature sweep across four environments on CUDA.

Each fresh policy is a separate process/checkpoint. Completed historical fits
are reused. Per-architecture k=8 reference splits and weights are fixed across k.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

FEATURE_COUNTS = (1, 2, 4, 8, 16, 32)
ENVIRONMENTS = ('inverted_pendulum', 'inverted_double_pendulum', 'hopper', 'walker2d')


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def build_manifest(experiments):
    root = experiments.parent
    repertoires = experiments / 'artifacts' / 'repertoires'
    sweep = repertoires / 'ann_feature_sensitivity_5seeds'
    jobs = []
    fingerprints = {}
    # Finish complete k blocks, yielding useful comparisons during a long run.
    for k in (8, 1, 2, 4, 16, 32):
        for env in ENVIRONMENTS:
            for architecture in (('shared', 'independent') if env in ('hopper', 'walker2d') else ('shared',)):
                family = 'cgp_feature' if architecture == 'shared' else 'cgp_action_feature'
                for mode in ('uniform', 'q_dagger'):
                    name = (f'ann_independent_{mode}_k8_5seeds' if architecture == 'independent'
                            else 'ann_teacher_bc_k8_5seeds_gpu' if env == 'inverted_double_pendulum' and mode == 'uniform'
                            else f'ann_initial_{mode}_k8_5seeds')
                    reference = repertoires / f'{family}_{env}' / name
                    config = json.loads((reference / 'config.json').read_text())
                    for key, expected in [('env', env), ('k', 8), ('n_nodes', 50), ('generations', 100),
                                          ('population_size', 100), ('expert_samples', 10000),
                                          ('validation_fraction', .2), ('rollout_steps', 1000), ('evaluation_trajectories', 10)]:
                        if config[key] != expected:
                            raise ValueError(f'Reference {reference}: {key} differs from sweep protocol')
                    if config.get('state_weighting', 'uniform') != mode:
                        raise ValueError('Reference weighting differs')
                    dataset = experiments / 'artifacts' / 'expert_datasets' / f'expert_{env}.npz'
                    if Path(config['dataset_path']).resolve() != dataset.resolve():
                        raise ValueError('Reference is not the requested ANN teacher')
                    for seed in range(5):
                        reused = k == 8 or (env == 'inverted_double_pendulum' and mode == 'uniform')
                        if k == 8:
                            run = reference
                        elif reused:
                            run = repertoires / f'{family}_{env}' / f'ann_teacher_bc_k{k}_sensitivity_5seeds_gpu'
                        else:
                            run = sweep / 'runs' / f'{family}_{env}' / f'{architecture}_{mode}_k{k}_seed{seed}'
                        if reused:
                            old_config = json.loads((run / 'config.json').read_text())
                            for key in ('env', 'n_nodes', 'generations', 'population_size', 'expert_samples',
                                        'validation_fraction', 'rollout_steps', 'evaluation_trajectories'):
                                if old_config[key] != config[key]:
                                    raise ValueError(f'Historical {key} differs: {run}')
                            summary = json.loads((run / f'seed_{seed}' / 'summary.json').read_text())
                            if summary['seed'] != seed or summary['k'] != k:
                                raise ValueError('Historical summary does not match setting')
                        reference_data = reference / f'seed_{seed}' / 'dataset.npz'
                        relative_data = str(reference_data.relative_to(root))
                        if relative_data not in fingerprints:
                            fingerprints[relative_data] = hashlib.sha256(reference_data.read_bytes()).hexdigest()
                        jobs.append(dict(environment=env, architecture=architecture, weighting=mode, k=k, seed=seed,
                                         reused=reused, run_directory=str(run.relative_to(root)),
                                         reference_run=str(reference.relative_to(root)),
                                         dataset=str(dataset.relative_to(root))))
    return dict(feature_counts=list(FEATURE_COUNTS), seeds=list(range(5)), teacher='ANN',
                n_nodes_per_cgp=50, population_size_per_cgp=100, generations_per_cgp=100,
                training_rows=8000, held_out_rows=2000, evaluation_episodes=10, rollout_steps=1000,
                reference_sha256=fingerprints, jobs=jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    experiments = Path(__file__).resolve().parents[2]
    root = experiments.parent
    sweep = experiments / 'artifacts' / 'repertoires/ann_feature_sensitivity_5seeds'
    sweep.mkdir(exist_ok=True)
    manifest = build_manifest(experiments)
    manifest_path = sweep / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Existing manifest/reference fingerprints differ; inspect before resuming')
    else:
        write_json(manifest_path, manifest)
    print(f"Policies={len(manifest['jobs'])}, reused={sum(j['reused'] for j in manifest['jobs'])}", flush=True)
    if args.plan_only:
        return
    environment = os.environ | {'JAX_PLATFORMS': 'cuda', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'}
    status_path = sweep / 'status.json'
    statuses = json.loads(status_path.read_text()) if status_path.exists() else {}
    for job in manifest['jobs']:
        identity = f"{job['environment']}_{job['architecture']}_{job['weighting']}_k{job['k']}_seed{job['seed']}"
        run = root / job['run_directory']
        summary_path = run / f"seed_{job['seed']}" / 'summary.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary['seed'] != job['seed'] or summary['k'] != job['k']:
                raise ValueError(f'Unexpected completed summary: {summary_path}')
            statuses.setdefault(identity, dict(state='reused' if job['reused'] else 'completed', job=job))
            statuses[identity]['state'] = 'reused' if job['reused'] else 'completed'
            write_json(status_path, statuses)
            continue
        if job['reused']:
            raise FileNotFoundError(summary_path)
        # Preserve an interrupted seed attempt; only that policy is restarted.
        if run.exists():
            archived = run.with_name(run.name + '.incomplete_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
            run.rename(archived)
        command = [sys.executable, '-u', '-m', 'distillation_experiments.scripts.experiments.' +
                   ('cgp_feature_imitation' if job['architecture'] == 'shared' else 'cgp_action_feature_imitation'),
                   '--env', job['environment'], '--state-weighting', job['weighting'], '--k', str(job['k']),
                   '--seed-start', str(job['seed']), '--num-seeds', '1', '--n-nodes', '50',
                   '--generations', '100', '--population-size', '100', '--expert-samples', '10000',
                   '--validation-fraction', '.2', '--rollout-steps', '1000', '--evaluation-trajectories', '10',
                   '--dataset-path', str(root / job['dataset']), '--reference-run', str(root / job['reference_run']),
                   '--output-root', str(sweep / 'runs'), '--run-name', run.name]
        statuses[identity] = dict(state='running', job=job, started_utc=datetime.now(timezone.utc).isoformat())
        write_json(status_path, statuses)
        print('START', identity, flush=True)
        started = time.perf_counter()
        with (sweep / f'{identity}.log').open('w') as log:
            process = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
        statuses[identity].update(state='completed' if process.returncode == 0 else 'failed',
                                  process_seconds=time.perf_counter() - started, returncode=process.returncode)
        write_json(status_path, statuses)
        if process.returncode:
            raise RuntimeError(f'{identity} failed; inspect its log. Completed policies will be reused on restart.')
        print('DONE', identity, flush=True)
    print('SWEEP COMPLETE', flush=True)


if __name__ == '__main__':
    main()
