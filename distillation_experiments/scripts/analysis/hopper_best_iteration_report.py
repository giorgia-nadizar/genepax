"""Compare when Hopper distillation runs reach their best evaluation reward."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parents[2] / 'artifacts' / 'repertoires'
MODELS = ['Linear regression', 'Small ANN', 'Operon', 'Single CGP (k=16)', 'CGP per action (k=16)']


def collect_best_iterations():
    rows = []
    for model in MODELS:
        for mode in ('uniform', 'q_dagger'):
            for seed in range(5):
                if model == 'Linear regression':
                    tag = 'uniform' if mode == 'uniform' else 'qdagger'
                    path = BASE / f'dagger_linear_hopper/linear_dagger_{tag}_5seeds_target3250/seed_{seed}/metrics.csv'
                elif model == 'Small ANN':
                    path = BASE / f'dagger_neural_hopper/small_ann_32x32_dagger_both_5seeds_target3250/seed_{seed}/{mode}/metrics.csv'
                elif model == 'Operon':
                    path = BASE / f'dagger_operon_hopper/operon_dagger_both_5seeds_target3250/seed_{seed}/{mode}/metrics.csv'
                else:
                    architecture = 'shared' if model == 'Single CGP (k=16)' else 'independent'
                    path = BASE / f'dagger_cgp_feature_hopper/k16_ann_protocol_both_5seeds_target3250/{architecture}/seed_{seed}/{mode}/metrics.csv'
                history = pd.read_csv(path).sort_values('iteration')
                best = history.loc[history.student_reward.idxmax()]
                rows.append(dict(model=model, mode=mode, seed=seed, best_iteration=int(best.iteration),
                                 best_reward=float(best.student_reward), last_iteration=int(history.iteration.max()),
                                 source=str(path.relative_to(BASE))))
    return pd.DataFrame(rows)


def plot_best_iterations(best):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange']
    markers = ['o', 's', '^', 'D', 'P']
    for ax, mode, title in zip(axes, ('uniform', 'q_dagger'), ('Uniform', 'Q-DAgger')):
        for model, color, marker in zip(MODELS, colors, markers):
            values = best[(best.model == model) & (best['mode'] == mode)].sort_values('seed')
            ax.scatter(values.best_iteration, values.best_reward, color=color, marker=marker,
                       s=75, alpha=.8, edgecolors='white', linewidths=.7, zorder=3)
        ax.set(title=title, xticks=range(10), xlim=(-.4, 9.4),
               ylim=(0, max(3500, float(best.best_reward.max()) * 1.06)),
               xlabel='DAgger iteration of best policy')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Reward at best iteration (mean episode return)')
    fig.suptitle('Hopper distillation: best iteration and reward per run', fontsize=14)
    fig.legend(handles=[Line2D([], [], color=color, marker=marker, linestyle='none', label=model)
                        for model, color, marker in zip(MODELS, colors, markers)],
               loc='upper center', bbox_to_anchor=(.5, .935), ncol=5, frameon=False, fontsize=9)
    fig.text(.5, .025, 'Each point is one run (five seeds per method). Exact iteration positions; overlapping points are possible.\n'
             'Iteration 0 = initial fit. Ties use the first occurrence.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .87))
    return fig
