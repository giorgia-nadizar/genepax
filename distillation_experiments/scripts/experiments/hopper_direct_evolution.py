"""Five GPU direct-CGP Hopper runs on the Generalized backend, with saved policies."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = ROOT / 'distillation_experiments/artifacts/policy_search/baselines/hopper_generalized_5seeds'


def evaluate_saved(directory, seed, steps, trajectories):
    import pickle
    import jax
    import jax.numpy as jnp
    import numpy as np
    from brax import envs
    from genepax.gp.cartesian_genetic_programming import CGP
    from distillation.rollouts import rollout, masked_return, sanitize_action
    environment = envs.get_environment('hopper', backend='generalized')
    cgp = CGP(n_inputs=environment.observation_size, n_outputs=environment.action_size)
    def evaluate_one(policy, episode_seed):
        def act(obs, key, step):
            action = sanitize_action(cgp.apply(policy, obs))
            return action, action
        _, _, rewards, dones = rollout(environment, jax.random.key(episode_seed), act, steps)
        return masked_return(rewards, dones)
    evaluate = jax.jit(jax.vmap(evaluate_one, in_axes=(None, 0)))
    episode_seeds = seed * 100000 + 90000 + jnp.arange(trajectories)
    records = {}
    for name in ('best_individual', 'final_individual'):
        with (directory / (name + '.pickle')).open('rb') as f:
            policy = pickle.load(f)
        returns = np.asarray(evaluate(policy, episode_seeds))
        if not np.isfinite(returns).all():
            raise ValueError(f'Nonfinite saved-policy evaluation: {name}')
        records[name] = dict(mean_reward=float(returns.mean()), episode_returns=returns.tolist(),
                             reached_3250=bool(returns.mean() >= 3250))
    result = dict(seed=seed, backend='generalized', episode_length=steps,
                  evaluation_seeds=np.asarray(episode_seeds).tolist(), policies=records,
                  selection='best observed training fitness; comparison episodes never used for selection')
    (directory / 'comparison_evaluation.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=range(5))
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--generations', type=int, default=1500)
    parser.add_argument('--population-size', type=int, default=100)
    parser.add_argument('--elite-size', type=int, default=10)
    parser.add_argument('--episode-length', type=int, default=1000)
    parser.add_argument('--evaluations', type=int, default=5)
    parser.add_argument('--comparison-trajectories', type=int, default=10)
    args = parser.parse_args()
    if min(args.generations, args.population_size, args.elite_size, args.episode_length,
           args.evaluations, args.comparison_trajectories) < 1 or args.elite_size >= args.population_size:
        parser.error('Counts must be positive and elite size smaller than population size')
    run = args.run_root
    run.mkdir(parents=True, exist_ok=True)
    config = dict(backend='generalized', seeds=list(range(5)), n_nodes=50,
                  generations=args.generations, population_size=args.population_size,
                  elite_size=args.elite_size, episode_length=args.episode_length,
                  evaluations=args.evaluations, comparison_trajectories=args.comparison_trajectories,
                  fixed_budget=True, environment='brax.envs.get_environment',
                  comparison_seed_rule=f'seed * 100000 + 90000 + arange({args.comparison_trajectories})',
                  selection='best observed search fitness', policy='CGP directly outputs three actions; no regression')
    config_path = run / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Existing run configuration differs')
    config_path.write_text(json.dumps(config, indent=2))
    if args.seed is not None:
        directory = run / 'hopper' / f'seed_{args.seed}'
        if not (directory / 'summary.json').exists():
            if directory.exists():
                archive = directory.with_name(directory.name + '_interrupted_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
                directory.rename(archive)
            from distillation_experiments.scripts.experiments.policy_search_gp import main as search
            sys.argv = ['policy_search_gp', '--env', 'hopper', '--backend', 'generalized',
                        '--seed', str(args.seed), '--run-name', f'seed_{args.seed}', '--output-root', str(run),
                        '--fixed-budget', '--unwrapped-env', '--require-gpu', '--target-reward', '3250']
            for name in ('generations','population_size','elite_size','episode_length','evaluations'):
                sys.argv += ['--' + name.replace('_','-'), str(getattr(args,name))]
            search()
        evaluate_saved(directory, args.seed, args.episode_length, args.comparison_trajectories)
        return
    summaries = []
    for seed in range(5):
        directory = run / 'hopper' / f'seed_{seed}'
        status = dict(status='running', seed=seed, completed_seeds=len(summaries))
        (run / 'status.json').write_text(json.dumps(status, indent=2))
        if not (directory / 'comparison_evaluation.json').exists():
            command = [sys.executable, '-u', '-m', 'distillation_experiments.scripts.experiments.hopper_direct_evolution',
                       '--seed', str(seed), '--run-root', str(run)]
            for name in ('generations','population_size','elite_size','episode_length','evaluations','comparison_trajectories'):
                command += ['--' + name.replace('_','-'), str(getattr(args,name))]
            with (run / f'seed_{seed}.log').open('a') as log:
                result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                (run / 'status.json').write_text(json.dumps(status | dict(status='failed',exit_code=result.returncode),indent=2))
                raise RuntimeError(f'Seed {seed} failed; see seed_{seed}.log')
        summaries.append(json.loads((directory / 'comparison_evaluation.json').read_text()))
        (run / 'aggregate_summary.json').write_text(json.dumps(dict(completed_seeds=len(summaries), seeds=summaries),indent=2))
        print(f'Seed {seed} complete',flush=True)
    (run / 'status.json').write_text(json.dumps(dict(status='complete',completed_seeds=5),indent=2))


if __name__ == '__main__':
    main()
