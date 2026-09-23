# Distillation experiments

This directory contains runnable workflows and their generated artifacts. The
reusable implementation remains in the `distillation` package.


All saved models, datasets, run histories, figures, and analysis notebooks live
under `artifacts/`. Scripts are split by purpose:

- `scripts/experiments/`: training, search, demonstration collection, rollout evaluation, and their shared helpers.
- `scripts/analysis/`: aggregation, figures, saved-run audits, and replay checks. Analysis tools do not train new policies; audits and replay checks may evaluate saved policies.

```text
distillation_experiments/
├── scripts/
│   ├── experiments/
│   └── analysis/
├── artifacts/
│   ├── expert_models/      # SAC teacher checkpoints (policy and critics)
│   ├── expert_datasets/    # Saved observations and teacher actions
│   ├── bc_ann_models/      # Original ANN behavioral-cloning checkpoints
│   ├── policy_search/      # Direct/mixed search policies and results
│   └── repertoires/        # Distillation runs, audits, plots, and notebooks
├── README.md
└── EXPERIMENT_LOG.md
```

The main comparison notebook is
[distillation_comparison.ipynb](artifacts/repertoires/distillation_comparison.ipynb).
The pendulum diagnostics are in
[pendulum_distillation.ipynb](artifacts/repertoires/cgp_expression_adam_inverted_pendulum/expression_adam_5seeds/pendulum_distillation.ipynb).
Script defaults and saved configuration/manifest paths use this layout. Historical
log text retains the paths printed when those experiments ran. Explicit output
path options still allow writing a new run elsewhere.

Run modules from the repository root. For example, train/search with
`python -m distillation_experiments.scripts.experiments.policy_search_gp`, or
generate a saved-run report with
`python -m distillation_experiments.scripts.analysis.ann_feature_sensitivity_report`.

Other experiment commands:

```bash
python -m distillation_experiments.scripts.experiments.train_brax
python -m distillation_experiments.scripts.experiments.evaluate_brax --env inverted_pendulum
python -m distillation_experiments.scripts.experiments.evaluate_brax --env inverted_pendulum --video
python -m distillation_experiments.scripts.experiments.collect_expert_data
python -m distillation_experiments.scripts.experiments.neural_imitation
python -m distillation_experiments.scripts.experiments.spid
python -m distillation_experiments.scripts.experiments.spid_validated
python -m distillation_experiments.scripts.experiments.dagger
python -m distillation_experiments.scripts.experiments.dagger_linear
python -m distillation_experiments.scripts.experiments.cgp_expression_then_adam
python -m distillation_experiments.scripts.experiments.policy_search_gp --seed 0
python -m distillation_experiments.scripts.experiments.policy_search_mixed --seed 0
```

`spid.py` is the original baseline implementation. `spid_validated.py` is the
experimental variant with recovery-state reservoirs, top-k Brax validation,
guarded best-policy aggregation, validation checkpoints during CGP search,
reward-guided repertoire bootstrapping, and complete iteration persistence.

For a small inverted-pendulum baseline SPID smoke test:

```bash
python -m distillation_experiments.scripts.experiments.spid \
  --env inverted_pendulum \
  --run-name smoke \
  --num-seeds 1 \
  --iterations 3 \
  --sr-generations 3 \
  --population-size 10 \
  --rollout-steps 200 \
  --evaluation-trajectories 2 \
  --max-dataset-size 5000 \
  --trajectories-per-iteration 2
```

For the validated workflow, use the same core options with the validated
module and its validation controls:

```bash
python -m distillation_experiments.scripts.experiments.spid_validated \
  --env inverted_pendulum \
  --run-name validated-smoke \
  --num-seeds 1 \
  --iterations 3 \
  --sr-generations 30 \
  --population-size 10 \
  --validation-top-k 10 \
  --validation-checkpoints 10,30 \
  --aggregation-policy best \
  --rollout-steps 200 \
  --evaluation-trajectories 2 \
  --max-dataset-size 5000 \
  --trajectories-per-iteration 2
```

Both SPID scripts write runs below `artifacts/repertoires/spid_<environment>/<run-name>/`.
Every seed has its own metrics, summary, best-validation genotype, and the
population repertoire from which it was selected. Load `best_genotype.pickle`
with `pickle.load` when evaluating a saved controller. An existing run name is
never overwritten. Omit `--run-name` to use a UTC timestamp.

`dagger.py` is the plain DAgger baseline adapted from Algorithm 1 of Kohler et
al. (2025). It fits CGP only on mean squared expert-action error: the first
sample batch comes from stored expert demonstrations and later batches come
from pure student rollouts labelled by the ANN expert. Reward is used only for
reporting. For example:

```bash
python -m distillation_experiments.scripts.experiments.dagger \
  --env inverted_pendulum \
  --run-name plain-dagger \
  --iterations 10 \
  --expert-bootstrap-samples 10000 \
  --trajectories-per-iteration 20 \
  --sr-generations 50 \
  --population-size 100
```

DAgger runs are written below
`artifacts/repertoires/dagger_<environment>/<run-name>/`. Each seed stores its metrics,
summary, final CGP population, final individual, and exact final aggregated
dataset. Intermediate checkpoints are not stored. Pass `--warm-start` to make
each CGP fit continue from the preceding population; by default every refit is
independent.

Pass `--linear-scaling` to fit a per-output affine calibration for every CGP
individual. Slopes and intercepts are stored in the genotype's
`weights["custom_weights"]` and are applied before action clipping during both
Brax evaluation and DAgger collection.

`dagger_linear.py` runs the same protocol with ordinary least squares,
matching the paper's continuous-action linear-policy baseline. It writes
reloadable coefficients, datasets, metrics, and summaries below
`artifacts/repertoires/dagger_linear_<environment>/<run-name>/`.

`cgp_expression_then_adam.py` separates discrete structure search from numeric
optimization. It evolves unweighted expressions (raw, linearly scaled, or
both), freezes the top structures, and then uses Adam to optimize both weights
of every CGP input connection. The scaled mode refits affine output scaling
after Adam and both stages are evaluated on held-out expert samples and in
Brax.

Pass `--state-weighting q_dagger` to weight every supervised state by the SAC
expert's estimated disadvantage of its worst action. For continuous
one-dimensional actions, the worst action is approximated on a configurable
grid over `[-1, 1]`. These weights are used by expression fitness, linear
scaling, held-out error, and Adam connection optimization.

Artifact directories are resolved relative to this directory, not the current
working directory.

### CGP features with a linear action readout

```bash
python -m distillation_experiments.scripts.experiments.cgp_feature_imitation \
  --env inverted_pendulum --k 8 --num-seeds 5 --run-name features8_bc
```

This fixed-dataset behavioral-cloning experiment evolves a CGP graph with `k`
feature outputs (default 8). Every candidate receives a least-squares
readout from its features to all expert action dimensions, including an
intercept. All actions share the same `k` features, with an independent
regression and intercept for each action (not `k` features per action).
Evolution selects by training MSE after action clipping; held-out MSE
and Brax reward are reporting-only. The readout is fitted on training rows only.
Feature outputs are sanitized and clipped to `[-1000, 1000]` consistently during
fitting and acting. Redundant features are handled with an SVD least-squares
solve. This workflow defaults to uniform sample weights and has no DAgger
aggregation. Pass `--state-weighting q_dagger` to weight both the least-squares
fit and evolutionary fitness using the saved ANN expert's critic. Summaries
save both ordinary and weighted train/validation MSE; `train_mse` and
`validation_mse` always denote ordinary MSE. The notebook's initial-fit plot
uses weighted MSE in its Q-DAgger group, matching the existing baselines.

Results go under `artifacts/repertoires/cgp_feature_<environment>/<run-name>/`. Each seed
saves the fitted population, `final_individual.pickle`, exact train/validation
data, search history, readout, and evaluation returns. Custom weights contain
a flattened `(k + 1, n_actions)` matrix in row-major order: the first `k` rows
are coefficients and the final row contains intercepts.

To execute a saved individual, reconstruct the structure from `config.json`
with `make_feature_cgp(config["n_inputs"], config["n_actions"], config["k"],
config["n_nodes"])` from `distillation.fit_feature_imitation`, load the genotype
with `pickle.load`, and call `feature_policy_action(genotype, cgp, observation)`.
This helper applies the stored readout and clips actions to `[-1, 1]`.

### Q-DAgger with CGP features

```bash
python -m distillation_experiments.scripts.experiments.dagger_cgp_features \
  --env inverted_double_pendulum --k 8 --num-seeds 5 \
  --run-name ann_qdagger_k8_5seeds
```

This uses the ANN actor and critic, with ten fitting iterations and 20 pure
student collection trajectories between fits. Every valid collected state is
labelled by the ANN, assigned its expert-to-worst-grid-action Q gap, and appended
to replay. The complete replay is used for weighted least-squares readouts and
Q-weighted CGP fitness (100 generations, population 100 per refit by default).
The initial 2,000-row held-out set stays fixed. Raw Q gaps are retained across
collection batches and normalized over the complete training replay each time;
batch-local normalization does not change their relative importance.

Each fit starts from a fresh population. Both ordinary and Q-weighted MSE are
reported, alongside rollout reward and replay size. Seeds stop at mean reward
9,359 by default; `--no-stop-when-solved` runs all requested iterations. Results
under `artifacts/repertoires/dagger_cgp_feature_<environment>/<run-name>/` include every
iteration's fitted population, individual, history and collected batch, plus
the exact initial/final datasets and raw Q gaps. Policies reload with the same
`make_feature_cgp` and `feature_policy_action` helpers as fixed-dataset cloning.

SAC expert models produced by `train_brax.py` are stored in `artifacts/expert_models/`.
Behavioral-cloning artificial neural networks produced by
`neural_imitation.py` are stored in `artifacts/bc_ann_models/`.

### Independent feature CGP for each action

```bash
python -m distillation_experiments.scripts.experiments.cgp_action_feature_imitation \
  --env hopper --k 8 --num-seeds 5 --state-weighting uniform \
  --run-name independent_features8_uniform
```

Use `--state-weighting q_dagger` for the Q-weighted initial fit. Each action
has its own independently evolved 50-node CGP producing eight features and
its own least-squares regression plus intercept. No graph nodes, features,
coefficients, or populations are shared across actions. The 100-generation,
100-individual budget applies to each action separately. Search seeds are
`seed + 10000 * action_index`; all actions use the same training rows and
state weights, paired with their own target action column.

Runs are saved under `artifacts/repertoires/cgp_action_feature_<env>/<run-name>/`.
The combined `final_individual.pickle` stores a tuple under `actions`; each
entry is a full CGP genotype with nine fitted `custom_weights`. Per-action
populations, individuals and histories are saved in `action_<index>/`.
Reload using `make_feature_cgp(n_inputs, 1, k, n_nodes)` and
`independent_feature_action(policy, cgp, observation)` from
`distillation.independent_feature_policy`. The seed-level convergence curve
averages the independent per-action losses; it is not a joint population.

### ANN-teacher feature-count sweep

```bash
python -m distillation_experiments.scripts.experiments.ann_feature_sensitivity
python -m distillation_experiments.scripts.analysis.ann_feature_sensitivity_report
JAX_PLATFORMS=cuda python -m distillation_experiments.scripts.analysis.ann_feature_sensitivity_audit --verify-mse
```

The CUDA sweep tests k=1,2,4,8,16,32 with seeds 0–4, Uniform and Q-DAgger
weighting, across both pendulums, Hopper and Walker2d. Hopper and Walker2d
include shared and independent per-action CGPs. All readouts include an
intercept. This is fixed-dataset behavioral cloning, with 8,000 training and
2,000 held-out ANN-expert rows and ten evaluation episodes per policy.

There are 360 policies: 85 matching historical fits are reused and 275 are
new fits. Each architecture's exact k=8 data split and Q weights are reused
across feature counts. The manifest records reference hashes and provenance.
Rerunning the driver resumes completed policies and archives interrupted seed
attempts. Use `--plan-only` to validate the manifest without starting fits.

Results are saved under `artifacts/repertoires/ann_feature_sensitivity_5seeds/`.
The report writes per-seed and aggregate CSVs, one combined four-by-four figure, and
a section in `distillation_comparison.ipynb`. Add `--watch` to
refresh it during the sweep. Figures show reward, ordinary training/held-out
MSE and fit time; CSVs also contain weighted MSE and evaluation/process times.
Missing historical timings remain missing. The audit checks reference hashes,
exact data/weights, independent graph counts, and stored readout shapes/values;
use `--allow-partial` while the sweep is running. `--verify-mse` also reloads
seed 0 at both endpoint feature counts in every environment/architecture/
weighting group and recomputes ordinary and weighted train/held-out MSE.
Use the original CUDA backend for numerical replay; CPU and extra outer JIT
compilation can change float32 cancellation in fitted readouts. Structural
auditing without `--verify-mse` can run with `JAX_PLATFORMS=cpu`.

### Hopper DAgger with 16 CGP features

Reproduce the small-ANN Hopper dataset-expansion protocol with both shared
and per-action CGPs, five seeds and Uniform/Q-DAgger weighting:

```bash
python -u -m distillation_experiments.scripts.experiments.hopper_feature_dagger
```

Requires the GPU. Each variant runs in an isolated process; completed variants
are skipped on restart and unfinished variants restart from bootstrap. Uses the
ANN's saved exact initial training/held-out data and weights, ten fitting
iterations, 20 collection trajectories, ten evaluation trajectories, and target
3250. Fits use all replay rows, population 100, 100 generations, and 50 nodes per
graph. Independent Hopper policies have three 16-feature graphs and three times
the search budget. Q weighting preserves the ANN's per-batch normalization.
The held-out split is reported only; unlike ANN early stopping, CGP selection
uses training error. Thus model classes and fitting budgets differ.

Results and replay audits are saved in
`artifacts/repertoires/dagger_cgp_feature_hopper/k16_ann_protocol_both_5seeds_target3250/`.
The driver refreshes the notebook after each completed variant. Refresh partial
progress manually with `python -m distillation_experiments.scripts.analysis.hopper_feature_dagger_report`.

### Hopper direct evolution on Generalized (five seeds)

```bash
python -u -m distillation_experiments.scripts.experiments.hopper_direct_evolution
```

Runs seeds 0–4 sequentially in isolated GPU processes. Each uses a 50-node
CGP directly producing three actions, population 100 (10 elites), tournament
size 3, and 1,500 generations without early stopping. Fitness averages five
fresh 1,000-step episodes per candidate per generation. The raw Brax
`hopper` environment uses `generalized`, matching the DAgger experiments.

Artifacts are in `artifacts/policy_search/baselines/hopper_generalized_5seeds/`:
`config.json`, `status.json`, per-seed logs, and `aggregate_summary.json`.
Each `hopper/seed_N/` saves `best_training_individual.pickle` on every search
record, `best_individual.pickle` (the best search policy),
`final_individual.pickle` (final-generation winner), and
`final_population.pickle`, alongside search metrics and configuration.
After training, both saved policies are reloaded and evaluated on the same
ten 1,000-step episode seeds used by the corresponding DAgger seed.
`comparison_evaluation.json` retains all returns; those episodes are never
used for search selection. The best search score is a noisy maximum, so use
these separate evaluation results for the DAgger comparison.
Restarting skips completed seeds and preserves interrupted attempts in a
renamed directory before restarting that seed.
