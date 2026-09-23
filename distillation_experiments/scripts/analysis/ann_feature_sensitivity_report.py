"""Artifact-only analysis and notebook reporting for the multi-environment ANN sweep."""
import argparse
import base64
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ENV_LABELS = {'inverted_pendulum': 'Inverted pendulum', 'inverted_double_pendulum': 'Inverted double pendulum',
              'hopper': 'Hopper', 'walker2d': 'Walker2d'}
ARCH_LABELS = {'shared': 'Single CGP', 'independent': 'CGP per action'}
COLORS = {'uniform': '#4C72B0', 'q_dagger': '#DD8452'}
WEIGHT_LABELS = {'uniform': 'Uniform', 'q_dagger': 'Q-DAgger'}
ARCH_STYLES = {'shared': ('-', 'o'), 'independent': ('--', 's')}
COUNTS = [1, 2, 4, 8, 16, 32]
CELL_IDS = ('ann-multienv-feature-protocol', 'ann-multienv-feature-results', 'ann-multienv-feature-findings')


def collect_results(root):
    root = Path(root)
    base = root / 'distillation_experiments/artifacts/repertoires'
    sweep = base / 'ann_feature_sensitivity_5seeds'
    manifest = json.loads((sweep / 'manifest.json').read_text())
    statuses = json.loads((sweep / 'status.json').read_text()) if (sweep / 'status.json').exists() else {}
    corrections = json.loads((base / 'initial_feature_comparison_evaluation/corrections.json').read_text())
    rows = []
    for job in manifest['jobs']:
        path = root / job['run_directory'] / f"seed_{job['seed']}" / 'summary.json'
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        reward = record['reward']
        corrected = False
        if job['k'] == 8 and job['architecture'] == 'shared':
            for correction in corrections:
                if (correction['model'].startswith('CGP') and correction['environment_key'] == job['environment']
                    and correction['seed'] == job['seed'] and correction['weighting'] ==
                    ('Uniform' if job['weighting'] == 'uniform' else 'Q-DAgger')):
                    reward = correction['reward']
                    corrected = True
        identity = f"{job['environment']}_{job['architecture']}_{job['weighting']}_k{job['k']}_seed{job['seed']}"
        fit_seconds = record.get('fit_seconds', np.nan)
        evaluation_seconds = record.get('evaluation_seconds', np.nan)
        rows.append(job | dict(reward=reward, reward_corrected=corrected,
            train_mse=record['train_mse'], held_out_mse=record['validation_mse'],
            weighted_train_mse=record.get('train_weighted_mse', record['train_mse']),
            weighted_held_out_mse=record.get('validation_weighted_mse', record['validation_mse']),
            fit_seconds=fit_seconds, evaluation_seconds=evaluation_seconds,
            fit_and_evaluation_seconds=fit_seconds + evaluation_seconds,
            process_seconds=statuses.get(identity, {}).get('process_seconds', np.nan)))
    return pd.DataFrame(rows), manifest


def summarize(frame):
    metrics = ('reward', 'train_mse', 'held_out_mse', 'weighted_train_mse', 'weighted_held_out_mse',
               'fit_seconds', 'evaluation_seconds', 'fit_and_evaluation_seconds', 'process_seconds')
    rows = []
    for keys, values in frame.groupby(['environment', 'architecture', 'weighting', 'k'], sort=False):
        row = dict(zip(('environment', 'architecture', 'weighting', 'k'), keys))
        row['seeds'] = len(values)
        for metric in metrics:
            finite = values.loc[np.isfinite(values[metric]), metric]
            row[f'{metric}_n'] = len(finite)
            row[f'{metric}_mean'] = finite.mean()
            row[f'{metric}_std'] = finite.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['environment', 'architecture', 'weighting', 'k'])


def plot_results(frame):
    """Overlay weighting modes by color and architectures by line/marker style."""
    from matplotlib.lines import Line2D

    plt.style.use('seaborn-v0_8-whitegrid')
    metrics = [('reward', 'Evaluation reward', False), ('train_mse', 'Training action MSE', True),
               ('held_out_mse', 'Held-out action MSE', True), ('fit_seconds', 'Fit time (seconds)', True)]
    fig, axes = plt.subplots(4, 4, figsize=(22, 17))
    for column, env in enumerate(ENV_LABELS):
        subset = frame[frame.environment.eq(env)]
        architectures = ('shared', 'independent') if env in ('hopper', 'walker2d') else ('shared',)
        for row_index, (metric, label, logarithmic) in enumerate(metrics):
            ax = axes[row_index, column]
            for weighting in WEIGHT_LABELS:
                for architecture in architectures:
                    values = subset[subset.weighting.eq(weighting) & subset.architecture.eq(architecture)]
                    color = COLORS[weighting]
                    linestyle, marker = ARCH_STYLES[architecture]
                    means, low, high = [], [], []
                    for position, k in enumerate(COUNTS):
                        points = values[values.k.eq(k)][metric].to_numpy()
                        points = points[np.isfinite(points)]
                        if logarithmic:
                            points = points[points > 0]
                        ax.scatter(np.full(len(points), position) + np.linspace(-.09, .09, len(points)), points,
                                   s=15, alpha=.25, color=color, marker=marker, linewidths=0)
                        complete = len(values[values.k.eq(k)]) == 5
                        means.append(np.mean(points) if complete and len(points) else np.nan)
                        low.append(np.quantile(points, .25) if complete and len(points) else np.nan)
                        high.append(np.quantile(points, .75) if complete and len(points) else np.nan)
                    ax.plot(range(len(COUNTS)), means, marker=marker, linestyle=linestyle,
                            color=color, linewidth=1.8, markersize=5,
                            label=f'{WEIGHT_LABELS[weighting]} — {ARCH_LABELS[architecture]}')
                    ax.fill_between(range(len(COUNTS)), low, high, color=color, alpha=.07)
            ax.set(xticks=range(len(COUNTS)), xticklabels=COUNTS, xlabel='Features k per CGP',
                   ylabel=label, title=f'{ENV_LABELS[env]}: {label}')
            if logarithmic:
                ax.set_yscale('log')
            if metric == 'fit_seconds' and env == 'inverted_double_pendulum':
                ax.text(.02, .98, 'Uniform: historical k=8 timing unavailable',
                        transform=ax.transAxes, va='top', fontsize=8)
    handles = [Line2D([], [], color=COLORS[weighting], linestyle=ARCH_STYLES[architecture][0],
                      marker=ARCH_STYLES[architecture][1], linewidth=1.8,
                      label=f'{WEIGHT_LABELS[weighting]} — {ARCH_LABELS[architecture]}')
               for weighting in WEIGHT_LABELS for architecture in ARCH_LABELS]
    fig.suptitle(f'ANN teacher feature sensitivity — Uniform and Q-DAgger ({len(frame)}/360 policies)\n'
                 'Points: seeds; lines: five-seed means; bands: interquartile range. MSE is unweighted.', fontsize=16)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .953), ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .92))
    return [('combined', fig)]


NOTEBOOK_CODE = '''from pathlib import Path
import sys
import matplotlib.pyplot as plt
from IPython.display import display
sweep_root = next(p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents) if (p / 'distillation_experiments').is_dir())
if str(sweep_root) not in sys.path:
    sys.path.insert(0, str(sweep_root))
from distillation_experiments.scripts.analysis.ann_feature_sensitivity_report import collect_results, summarize, plot_results
ann_sweep_results, ann_sweep_manifest = collect_results(sweep_root)
ann_sweep_summary = summarize(ann_sweep_results)
print(f"Completed policies: {len(ann_sweep_results)}/{len(ann_sweep_manifest['jobs'])}")
for _, figure in plot_results(ann_sweep_results):
    display(figure)
    plt.close(figure)
display(ann_sweep_summary)
'''


def render_report(root):
    root = Path(root)
    base = root / 'distillation_experiments/artifacts/repertoires'
    sweep = base / 'ann_feature_sensitivity_5seeds'
    frame, manifest = collect_results(root)
    summary = summarize(frame)
    frame.to_csv(sweep / 'per_seed_results.csv', index=False)
    summary.to_csv(sweep / 'summary.csv', index=False)
    outputs = [dict(output_type='stream', name='stdout', text=f'Completed policies: {len(frame)}/360\n')]
    for weighting, fig in plot_results(frame):
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
        (sweep / f'{weighting}_sensitivity.png').write_bytes(buffer.getvalue())
        outputs.append(dict(output_type='display_data', metadata={}, data={'image/png': base64.b64encode(buffer.getvalue()).decode()}))
        plt.close(fig)
    outputs.append(dict(output_type='display_data', metadata={}, data={'text/plain': summary.to_string(index=False), 'text/html': summary.to_html(index=False)}))
    status = 'COMPLETE' if len(frame) == 360 else 'IN PROGRESS'
    protocol = f'''## ANN teacher: feature-count sensitivity across four environments

**{status}: {len(frame)}/360 policies available.** Last refreshed {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.

Feature counts **k={{1,2,4,8,16,32}}**, five seeds, Uniform and Q-DAgger weighting. Both pendulums use one CGP; Hopper and Walker2d compare one shared CGP against one independently evolved CGP per action. Every readout includes a constant 1/intercept, stored with its individual. The independent architecture shares no graph nodes or regression coefficients between actions.

Each CGP has 50 nodes, population 100, and 100 generations. Independent models receive 3×/6× the total search budget and node capacity in Hopper/Walker2d. The counts specify features **per CGP**, so independent policies have 3k/6k total outputs. Training uses 8,000 fixed ANN-expert rows, with 2,000 held-out rows; rewards average ten 1,000-step episodes. No dataset expansion is used.

Each new fit loads its architecture's exact k=8 split and weights. This fixes data and weights across k within each architecture; historical single-CGP and independent-CGP Q weights differ slightly as documented above. Matching completed runs are reused (85 policies); 275 policies are newly fitted. Shared Walker2d k=8 reward corrections use the existing replay overlay. Reference data hashes and run provenance are saved in the sweep manifest.

The combined figure overlays **Uniform (blue)** and **Q-DAgger (orange)**. Solid lines/circles indicate a single shared CGP; dashed lines/squares indicate an independent CGP per action (Hopper and Walker2d only). It shows **ordinary training and held-out action MSE** for direct comparison across fitting weights. The table additionally includes weighted MSE, seed counts, finite-value counts, fit and evaluation times, and new-process wall times. Means/std exclude nonfinite values and counts remain explicit. Lines/bands require all five seeds to be available; individual points may show partial progress. Timing includes compilation and is hardware-specific; independent fit time also includes checkpoint writes. Historical IDP Uniform k=8 timing was never recorded and remains missing. Runtime is measured, not inferred from the old run's start/end timestamps.
'''
    if len(frame) == 360:
        winners = []
        minimum_mse_at_largest_k = 0
        for keys, group in summary.groupby(['environment', 'architecture', 'weighting']):
            minimum_mse_at_largest_k += int(group.loc[group.held_out_mse_mean.idxmin(), 'k'] == 32)
            eligible = group[group.reward_n.eq(5)]
            if len(eligible):
                best = eligible.loc[eligible.reward_mean.idxmax()]
                winners.append(f"- {ENV_LABELS[keys[0]]}, {ARCH_LABELS[keys[1]]}, {keys[2]}: highest observed mean reward at k={int(best.k)} ({best.reward_mean:.1f}).\n")
        paired = summary[summary.architecture.eq('independent')].merge(
            summary[summary.architecture.eq('shared')], on=['environment', 'weighting', 'k'],
            suffixes=('_independent', '_shared'))
        mse_improvements = int((paired.held_out_mse_mean_independent < paired.held_out_mse_mean_shared).sum())
        reward_improvements = int((paired.reward_mean_independent > paired.reward_mean_shared).sum())
        findings = f'''### Sweep findings

All **360 policies / 72 five-seed groups** are complete. **k=32 gives the lowest mean held-out MSE in {minimum_mse_at_largest_k}/12 environment/architecture/weighting groups**, but the highest mean reward occurs at different feature counts. Lower cloning error does not consistently translate into better rollout reward.

Independent CGPs achieve lower mean held-out MSE in **{mse_improvements}/{len(paired)}** matched Hopper/Walker2d comparisons and higher mean reward in **{reward_improvements}/{len(paired)}**. These comparisons include their larger total graph capacity and search budget; they do not isolate architecture at equal compute.

Highest observed mean reward by group:

''' + ''.join(winners)
        findings += '\nThese are descriptive five-seed results; selecting k by these evaluation rewards requires a separate evaluation to assess generalization.\n'
        findings += '\n**Measured fit time (range of five-seed means):**\n\n'
        for label, mask in [('Single CGP', summary.architecture.eq('shared')),
                            ('Independent CGPs, Hopper', summary.architecture.eq('independent') & summary.environment.eq('hopper')),
                            ('Independent CGPs, Walker2d', summary.architecture.eq('independent') & summary.environment.eq('walker2d'))]:
            times = summary.loc[mask, 'fit_seconds_mean'].dropna()
            findings += f'- {label}: {times.min():.1f}–{times.max():.1f} seconds per policy.\n'
        processes = frame.loc[~frame.reused, 'process_seconds'].dropna()
        findings += f'\nThe {len(processes)} newly timed policy processes took {processes.sum()/3600:.2f} hours in total, including initialization and evaluation. Historical missing timings remain excluded.\n'
        findings += '\nFull metrics: [per-seed CSV](ann_feature_sensitivity_5seeds/per_seed_results.csv), [group means and standard deviations](ann_feature_sensitivity_5seeds/summary.csv).\n'
        audit_path = sweep / 'audit.json'
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            if len(audit) == 360 and all(record['passed'] for record in audit):
                reloaded = sum(bool(record.get('reloaded_metrics')) for record in audit)
                backends = sorted({metric['backend'] for record in audit for metric in record.get('reloaded_metrics', {}).values()})
                differences = [metric['absolute_difference'] for record in audit for metric in record.get('reloaded_metrics', {}).values()]
                findings += f'\nValidation: all 360 policy bundles / {sum(record["graphs"] for record in audit)} graphs passed the artifact audit; {reloaded} endpoint policies reproduced training and held-out MSE after reload using the original evaluation path on {", ".join(backends)} (maximum absolute difference {max(differences, default=0):.3g}; tolerance: absolute 1e-6 plus relative 1e-4).\n'
    else:
        findings = '### Sweep status\n\nThe sweep is running. Missing groups and historical missing timings are not filled with estimates. Final comparisons will be written after all 360 policies are available.\n'
    cells = [dict(cell_type='markdown', id=CELL_IDS[0], metadata={}, source=protocol.splitlines(keepends=True)),
             dict(cell_type='code', id=CELL_IDS[1], metadata={}, source=NOTEBOOK_CODE.splitlines(keepends=True), execution_count=1, outputs=outputs),
             dict(cell_type='markdown', id=CELL_IDS[2], metadata={}, source=findings.splitlines(keepends=True))]
    notebook_path = base / 'distillation_comparison.ipynb'
    notebook = json.loads(notebook_path.read_text())
    notebook['cells'] = [cell for cell in notebook['cells'] if cell.get('id') not in CELL_IDS]
    notebook['cells'].extend(cells)
    temp = notebook_path.with_suffix('.tmp')
    temp.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')
    temp.replace(notebook_path)
    print(f'Report refreshed: {len(frame)}/360 policies', flush=True)
    return len(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    last_count = -1
    while True:
        frame, _ = collect_results(root)
        count = len(frame)
        if count != last_count and (not args.watch or count == 360 or count // 5 != last_count // 5):
            render_report(root)
            last_count = count
        if not args.watch or count == 360:
            return
        time.sleep(45)


if __name__ == '__main__':
    main()
