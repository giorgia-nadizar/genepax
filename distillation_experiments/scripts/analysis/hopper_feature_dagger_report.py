"""Notebook progress report for the Hopper CGP feature DAgger reproduction."""
import base64
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from distillation_experiments.scripts.experiments.hopper_feature_dagger import BASE, RUN


def direct_evolution_results():
    """Load separate evaluation of each completed run's best-search policy."""
    directory = BASE.parent / 'policy_search/baselines/hopper_generalized_5seeds/hopper'
    rows = []
    for seed in range(5):
        path = directory / f'seed_{seed}/comparison_evaluation.json'
        if path.exists():
            record = json.loads(path.read_text())
            rows.append(dict(seed=seed, reward=record['policies']['best_individual']['mean_reward']))
    return pd.DataFrame(rows, columns=['seed', 'reward'])


def collect(run=RUN):
    frames, summaries = [], []
    for architecture in ('shared', 'independent'):
        for seed in range(5):
            for mode in ('uniform', 'q_dagger'):
                directory = Path(run) / architecture / f'seed_{seed}' / mode
                if (directory / 'metrics.csv').exists():
                    frames.append(pd.read_csv(directory / 'metrics.csv').assign(architecture=architecture, seed=seed, mode=mode))
                if (directory / 'summary.json').exists():
                    summaries.append(json.loads((directory / 'summary.json').read_text()))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), pd.DataFrame(summaries)


def plot(history):
    fig, axes = plt.subplots(4, 2, figsize=(15, 17))
    specs = [('student_reward', 'Policy reward'), ('imitation_loss', 'Training action MSE'),
             ('test_imitation_loss', 'Fixed expert test MSE'), ('dataset_size', 'Training-set size')]
    for column, architecture in enumerate(('shared', 'independent')):
        for row, (metric, label) in enumerate(specs):
            ax = axes[row, column]
            if not history.empty:
                for mode, color in [('uniform', 'tab:blue'), ('q_dagger', 'tab:orange')]:
                    subset = history[(history.architecture == architecture) & (history['mode'] == mode)]
                    for seed, values in subset.groupby('seed'):
                        ax.plot(values.iteration, values[metric], 'o-', color=color, alpha=.3, linewidth=1)
                    if not subset.empty:
                        median = subset.groupby('iteration')[metric].median()
                        ax.plot(median.index, median.values, color=color, linewidth=2.5, label=mode + ' available-seed median')
            if metric == 'student_reward':
                ax.axhline(3250, color='black', linestyle='--', label='Target 3250')
            ax.set(title=f'{architecture}: {label}', xlabel='DAgger iteration', ylabel=label)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
            ax.grid(alpha=.2)
    fig.suptitle('Hopper CGP features + linear regression DAgger — k=16, five seeds')
    fig.tight_layout(rect=(0, 0, 1, .97))
    return fig


def best_per_run(history):
    """Select each run's maximum evaluation reward, keeping its first peak iteration."""
    ordered = history.sort_values('iteration').reset_index(drop=True)
    indices = ordered.groupby(['architecture', 'mode', 'seed']).student_reward.idxmax()
    return ordered.loc[indices, ['architecture', 'mode', 'seed', 'iteration', 'student_reward']].rename(
        columns={'iteration': 'best_iteration', 'student_reward': 'best_reward'}).reset_index(drop=True)


def plot_best_per_run(history):
    from distillation_experiments.scripts.analysis.hopper_best_iteration_report import collect_best_iterations, MODELS
    all_best = collect_best_iterations()
    cgp_best = best_per_run(history)
    cgp_best['model'] = cgp_best.architecture.map({'shared': 'Single CGP (k=16)', 'independent': 'CGP per action (k=16)'})
    best = pd.concat([all_best[~all_best.model.isin(MODELS[-2:])], cgp_best], ignore_index=True)
    direct = direct_evolution_results()
    fig, grid = plt.subplots(2, 3, figsize=(14, 9), sharey=True)
    axes = grid.ravel()
    for ax, title in zip(axes, MODELS):
        for position, (mode, color) in enumerate((('uniform', 'tab:blue'), ('q_dagger', 'tab:orange'))):
            values = best[(best.model == title) & (best['mode'] == mode)].sort_values('seed')
            if values.empty:
                continue
            ax.boxplot([values.best_reward.to_numpy()], positions=[position], widths=.45,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=color, edgecolor=color, alpha=.2),
                       medianprops=dict(color=color, linewidth=2),
                       whiskerprops=dict(color=color), capprops=dict(color=color))
            for i, row in enumerate(values.itertuples()):
                x = position + (i - (len(values) - 1) / 2) * .065
                ax.scatter(x, row.best_reward, color=color, edgecolors='white', s=65, zorder=3)
                ax.annotate(f's{row.seed}', (x, row.best_reward), xytext=(5, 5),
                            textcoords='offset points', fontsize=8, color=color)
        ax.axhline(3250, color='black', linestyle='--', linewidth=1.2, label='Target: 3,250')
        ax.set(title=title, xticks=[0, 1], xticklabels=['Uniform', 'Q-DAgger'], xlim=(-.5, 1.65),
               ylim=(0, max(3500, float(best.best_reward.max()) * 1.1)))
        ax.grid(axis='y', alpha=.2)
    ax = axes[5]
    color = 'tab:purple'
    if not direct.empty:
        ax.boxplot([direct.reward.to_numpy()], positions=[0], widths=.45, patch_artist=True,
                   showfliers=False, boxprops=dict(facecolor=color, edgecolor=color, alpha=.2),
                   medianprops=dict(color=color, linewidth=2), whiskerprops=dict(color=color),
                   capprops=dict(color=color))
        for i, row in enumerate(direct.itertuples()):
            x = (i - (len(direct) - 1) / 2) * .065
            ax.scatter(x, row.reward, color=color, edgecolors='white', s=65, zorder=3)
            ax.annotate(f's{row.seed}', (x, row.reward), xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.axhline(3250, color='black', linestyle='--', linewidth=1.2)
    ax.set(title=f'Direct CGP evolution\n{len(direct)}/5 seeds completed', xticks=[0],
           xticklabels=['Best-search policy'], xlim=(-.6, .8))
    ax.grid(axis='y', alpha=.2)
    axes[0].set_ylim(0, max(3500, float(best.best_reward.max()) * 1.1,
                           0 if direct.empty else float(direct.reward.max()) * 1.1))
    for ax in axes[::3]:
        ax.set_ylabel('Mean episode reward')
    fig.suptitle('Hopper: DAgger best per run and direct CGP evolution — Generalized backend', fontsize=13)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .93),
               ncol=2, frameon=False)
    fig.text(.5, .025, 'Dots = seeds; boxes = IQR and median. DAgger: best evaluation iteration. Direct: evaluation of best-search policy.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .90))
    return fig


def refresh(run=RUN):
    history, summaries = collect(run)
    status = f'{len(summaries)}/20 variants completed'
    protocol = f'''## Hopper CGP-feature DAgger — k=16, both architectures

**{status}.** Matches the small-ANN experiment's five seeds, Uniform and Q-DAgger modes, exact 8,000-row initial training set and weights, fixed 2,000-row held-out set, collection seeds, and evaluation seeds. Up to ten fits, 20 pure-student collection trajectories per aggregation, ten evaluation trajectories, 1,000 steps per episode, stop at mean reward 3,250. Every collected valid transition is retained. Q weights are normalized separately for each collection batch, matching the ANN reference.

Each refit starts from scratch: population 100, 100 generations, 50 nodes per CGP. Shared: one graph outputs 16 features for three linear action readouts. Independent: three separate graphs each output 16 features for its own scalar readout (48 total features and three times the search budget). All readouts include an intercept and use least-squares regression; evolution selects by weighted clipped-action training MSE. The fixed held-out set is only reported, whereas the ANN used it for early stopping. Search budgets and student classes therefore differ.

Thin lines show actual seed histories; thick lines show the median of seeds available at each iteration. Early-stopped histories are not extrapolated; partial results are not final five-seed comparisons. Training and test losses use each mode's weights, matching the ANN panel; ordinary MSE is also saved in metrics.csv.

The distribution plot below includes the completed Generalized-backend direct-evolution runs; the historical Spring reference line has been removed.

Artifacts: `dagger_cgp_feature_hopper/k16_ann_protocol_both_5seeds_target3250/`. Checkpoints, fitted readouts, populations, search histories, episode returns, and collected datasets are retained for every iteration.
'''
    source = '''from pathlib import Path
import sys
import matplotlib.pyplot as plt
from IPython.display import display
cgp_root = next(p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents) if (p / 'distillation_experiments').is_dir())
if str(cgp_root) not in sys.path:
    sys.path.insert(0, str(cgp_root))
from distillation_experiments.scripts.analysis.hopper_feature_dagger_report import collect, plot
cgp_dagger_history, cgp_dagger_summaries = collect()
print(f"{len(cgp_dagger_summaries)}/20 variants completed")
if not cgp_dagger_history.empty:
    display(plot(cgp_dagger_history))
    plt.close()
display(cgp_dagger_summaries)
'''
    outputs = [dict(output_type='stream', name='stdout', text=status + '\n')]
    if not history.empty:
        fig = plot(history)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=110, bbox_inches='tight')
        plt.close(fig)
        (Path(run) / 'progression.png').write_bytes(buffer.getvalue())
        outputs.append(dict(output_type='display_data', metadata={}, data={'image/png': base64.b64encode(buffer.getvalue()).decode()}))
        history.to_csv(Path(run) / 'all_metrics.csv', index=False)
    if not summaries.empty:
        outputs.append(dict(output_type='display_data', metadata={}, data={'text/plain': summaries.to_string(index=False), 'text/html': summaries.to_html(index=False)}))
    (Path(run) / 'aggregate_summary.json').write_text(json.dumps({'completed_variants': len(summaries), 'requested_variants': 20, 'variants': summaries.to_dict('records')}, indent=2))
    path = BASE / 'distillation_comparison.ipynb'
    notebook = json.loads(path.read_text())
    prefix = 'hopper-k16-feature-dagger-'
    notebook['cells'] = [c for c in notebook['cells'] if not c.get('id', '').startswith(prefix)]
    notebook['cells'] += [dict(cell_type='markdown', id=prefix+'protocol', metadata={}, source=protocol.splitlines(keepends=True)),
        dict(cell_type='code', id=prefix+'results', metadata={}, execution_count=1, source=source.splitlines(keepends=True), outputs=outputs)]
    if len(summaries) == 20:
        groups = summaries.groupby(['architecture', 'mode']).agg(
            seeds=('seed', 'nunique'), mean_final_reward=('final_reward', 'mean'),
            std_final_reward=('final_reward', 'std'), median_final_reward=('final_reward', 'median'),
            solved=('solved', 'sum')).reset_index()
        groups.to_csv(Path(run) / 'final_group_summary.csv', index=False)
        best = summaries.loc[summaries.final_reward.idxmax()]
        findings = (
            f'### Completed Hopper CGP-feature DAgger results\n\n'
            f'All **20/20 runs** finished, with **{len(history)} fitting iterations** recorded. '
            f'**{int(summaries.solved.sum())}/20** final policies reached the 3,250 target.\n\n'
            '| Architecture | Weighting | Seeds | Final reward mean ± SD | Final reward median | Solved |\n'
            '|---|---|---:|---:|---:|---:|\n')
        for row in groups.itertuples():
            architecture = 'Single CGP' if row.architecture == 'shared' else 'CGP per action'
            mode = 'Uniform' if row.mode == 'uniform' else 'Q-DAgger'
            findings += (f'| {architecture} | {mode} | {row.seeds} | '
                         f'{row.mean_final_reward:,.1f} ± {row.std_final_reward:,.1f} | '
                         f'{row.median_final_reward:,.1f} | {row.solved}/5 |\n')
        findings += (f'\nThe strongest final policy was **{best.final_reward:,.1f}**, '
                     f'{best.architecture} CGP, {best["mode"]}, seed {best.seed}. '
                     'These are final-iteration rewards, not retrospectively selected peaks. '
                     'The per-action architecture has three times the graph/search budget. '
                     'Direct-evolution results on the same backend appear in the distribution below.\n')
        audits = [json.loads(p.read_text()) for p in Path(run).glob('*/seed_*/*/audit.json')]
        if len(audits) == 20 and all(a.get('replay_exact') and a.get('held_out_unchanged') for a in audits):
            findings += '\nAll 20 saved replay audits passed: exact reconstruction from collected batches and unchanged held-out datasets.\n'
        notebook['cells'].append(dict(cell_type='markdown', id=prefix+'findings', metadata={},
                                      source=findings.splitlines(keepends=True)))
    if not history.empty:
        best = best_per_run(history)
        best.to_csv(Path(run) / 'best_per_run.csv', index=False)
        fig = plot_best_per_run(history)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=140, bbox_inches='tight')
        plt.close(fig)
        (Path(run) / 'best_per_run.png').write_bytes(buffer.getvalue())
        text = ('### Best reward achieved in each run\n\n'
                'The five distillation panels cover linear regression, small ANN, Operon, single CGP (k=16), and CGP per action (k=16), each under Uniform and Q-DAgger. Each dot is the maximum mean evaluation reward over all available fitting iterations for one seed, '
                'regardless of when it occurred. Boxes show the interquartile range and median across seeds. '
                'Seed labels identify individual runs; `best_iteration_by_method.csv` records all 50 peak rewards and their corresponding iterations '
                '(first occurrence in a tie). These are retrospective peaks on the evaluation seeds, '
                'not independent evaluations of selected policies. The sixth panel shows separate evaluation of '
                'each completed direct-evolution run’s best-search policy on the same ten episode seeds '
                'as its DAgger counterpart. Both use Generalized. Direct evolution selects by training fitness, '
                'whereas the DAgger boxes use retrospective evaluation peaks. Missing direct-evolution seeds '
                'are excluded. Fitting budgets and model-selection protocols differ across distillation methods.\n')
        direct = direct_evolution_results()
        text += f'\n**Direct evolution: {len(direct)}/5 completed seeds.** All 50 distillation runs are included.\n'
        direct_root = BASE.parent / 'policy_search/baselines/hopper_generalized_5seeds/hopper'
        for seed in range(5):
            if seed not in direct.seed.to_list():
                metrics_path = direct_root / f'seed_{seed}/metrics.csv'
                count = len(pd.read_csv(metrics_path)) if metrics_path.exists() else 0
                text += (f'\nDirect-evolution seed {seed}: {count}/1,500 generations recorded at refresh; '
                         'its completed-policy evaluation is not yet available and is excluded from the boxplot.\n')
        code = ('from distillation_experiments.scripts.analysis.hopper_feature_dagger_report import collect, plot_best_per_run\n'
                'from IPython.display import display\nimport matplotlib.pyplot as plt\n'
                'best_history, _ = collect()\nbest_figure = plot_best_per_run(best_history)\n'
                'display(best_figure)\nplt.close(best_figure)\n')
        notebook['cells'] += [dict(cell_type='markdown', id=prefix+'best-protocol', metadata={}, source=text.splitlines(keepends=True)),
            dict(cell_type='code', id=prefix+'best-distribution', metadata={}, execution_count=1,
                 source=code.splitlines(keepends=True), outputs=[dict(output_type='display_data', metadata={},
                     data={'image/png': base64.b64encode(buffer.getvalue()).decode()})])]
    from distillation_experiments.scripts.analysis.hopper_best_iteration_report import collect_best_iterations, plot_best_iterations
    iterations = collect_best_iterations()
    iterations.to_csv(Path(run) / 'best_iteration_by_method.csv', index=False)
    fig = plot_best_iterations(iterations)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    (Path(run) / 'best_iteration_by_method.png').write_bytes(buffer.getvalue())
    explanation = ('### Iteration of the best Hopper distillation policy\n\n'
                   'For each of five seeds, take the iteration with the highest mean evaluation reward, '
                   'using its first occurrence in a tie. Iteration 0 is the initial fit; iterations 1–9 '
                   'follow dataset aggregation. Each scatter point shows that iteration on the x-axis and its mean '
                   'evaluation reward on the y-axis. Coordinates are exact, with no jitter. Separate panels '
                   'show Uniform and Q-DAgger, with colors and marker shapes identifying linear regression, small ANN, '
                   'Operon, shared CGP and per-action CGP. Early-stopped runs use only observed iterations. '
                   'This describes retrospective peak timing; an iteration is not an equal compute budget '
                   'across models, and a later peak does not necessarily mean a higher reward.\n')
    code = ('from distillation_experiments.scripts.analysis.hopper_best_iteration_report import collect_best_iterations, plot_best_iterations\n'
            'from IPython.display import display\nimport matplotlib.pyplot as plt\n'
            'hopper_best_iterations = collect_best_iterations()\n'
            'iteration_figure = plot_best_iterations(hopper_best_iterations)\n'
            'display(iteration_figure)\nplt.close(iteration_figure)\n')
    notebook['cells'] += [dict(cell_type='markdown', id=prefix+'iteration-protocol', metadata={}, source=explanation.splitlines(keepends=True)),
        dict(cell_type='code', id=prefix+'iteration-distribution', metadata={}, execution_count=1,
             source=code.splitlines(keepends=True), outputs=[dict(output_type='display_data', metadata={},
                 data={'image/png': base64.b64encode(buffer.getvalue()).decode()})])]
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')
    temp.replace(path)


if __name__ == '__main__':
    refresh()
