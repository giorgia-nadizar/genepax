# Distillation experiment log

This is a living record of the policy-distillation experiments, the reasons
for running them, and the conclusions drawn from them. Add new entries with
the date, exact configuration or result directory, outcome, and next decision.

## Current question

Can a symbolic CGP controller reliably imitate a trained ANN expert and solve
Brax `inverted_pendulum`? If not, is the limiting factor data aggregation, the
SPID objective, the CGP representation, or CGP optimization?

The task is considered solved at an average evaluation reward of at least
999. The maximum observed return is 1000.

## Work completed before the current ablations

- Reorganized the distillation package and moved runnable workflows and their
  outputs under `distillation_experiments/`.
- Kept `spid.py` as the original baseline and `spid_validated.py` as the
  guarded experimental implementation.
- Fixed transition masking so invalid transitions and all transitions after
  the first invalid one are excluded.
- Added action and loss sanitization, guarded CGP functions, configurable beta
  scheduling, bounded aggregation reservoirs, recovery-state sampling,
  rollout validation, and population auditing.
- Confirmed that direct CGP policy search can solve inverted pendulum. The
  retained policy-search baseline solved it at generation 58 with reward
  1000. This establishes that the environment is representable and solvable
  by the current CGP policy class.

Direct policy-search result:

`artifacts/policy_search/baselines/inverted_pendulum/seed_0_protected_actions/`

## Validated SPID run

Retained result:

`artifacts/repertoires/spid_inverted_pendulum/ann_reward_bootstrap_ten_seeds/`

Configuration and methodology:

- Ten seeds.
- ANN/SAC expert labels.
- Reward-aware geometric-mean SPID loss.
- Student-state aggregation with expert and recovery reservoirs.
- Top-k rollout validation during CGP optimization.
- Maximum dataset size of 20,000 in the completed runs.

Outcome:

- 1 of 10 seeds solved the task.
- Seed 6 reached reward 1000 at iteration 9.
- The other best validation rewards ranged from 236.2 to 577.9.
- Population audits showed that imitation-loss rank is not a consistently
  reliable proxy for rollout reward. In some populations the highest-reward
  individual was far down the loss ranking.

Interpretation:

- SPID can produce a successful controller, but reliability is poor.
- Numerical sanitization removed the obvious non-finite failure mode but did
  not resolve the optimization and model-selection problem.
- Reward validation is necessary for CGP because small supervised-loss
  differences can correspond to large closed-loop reward differences.

The detailed plots and analysis are in:

`artifacts/repertoires/spid_inverted_pendulum/ann_reward_bootstrap_ten_seeds/analysis.ipynb`

## Plain DAgger with CGP

Motivation:

Kohler et al. (2025), *Evaluating Interpretable Reinforcement Learning by
Distilling Policies into Programs*, use a simple DAgger loop without SPID's
geometric-mean objective. We implemented a plain CGP version to determine
whether the SPID loss itself was the main difficulty.

Implementation:

- `scripts/dagger.py`
- `distillation/fit_imitation.py`
- Expert-only bootstrap followed by pure student rollouts.
- Expert actions label every retained student state.
- CGP fitness is mean squared expert-action error only.
- Reward is evaluation-only and does not select the next behavior policy.
- Final population, individual, dataset, metrics, configuration, and summary
  are reloadable; intermediate checkpoints are not stored.

### Exact-transition-budget attempt

Initial configuration:

- Five requested seeds.
- 10 iterations.
- Nominal total of 100,000 samples.
- Population 100, 50 generations per refit.
- Fresh CGP population at every iteration.

Observed seed-0 behavior included:

- Iteration 0: imitation loss 0.00282, reward 141.2.
- Iteration 1: imitation loss 0.05479, reward 3.0.

The implementation had to launch many additional trajectories to obtain an
exact transition count after weak students began terminating early. This made
runtime variable and expensive. The attempt was stopped.

### Fixed-trajectory GPU attempt

The collector was changed to execute exactly 20 student trajectories per
aggregation iteration for every seed. Dataset growth was therefore allowed to
depend on episode length.

The first attempt ran on CPU because an active Jupyter kernel occupied about
5.9 GB of the 8.15 GB GPU. The kernel was stopped, GPU availability was
verified, and the experiment was restarted on the GPU.

Important observations before termination:

- Seed 0 expert-only fit: reward 233.8, loss 0.00269.
- Seed 0 final iteration: reward 76.2; its best reward remained the
  expert-only iteration.
- Seed 1 expert-only fit: reward 174.5.
- Seed 1 fell to reward 7.0 after its first DAgger refit and later recovered
  only partially.
- No invalid transitions were collected.
- Weak students produced only roughly 80--500 retained transitions from 20
  trajectories, so the 10,000 expert samples dominated the aggregated data.

Decision:

The fixed-trajectory run was interrupted and its result directory deleted.
The repeated collapse after independently refitting CGP showed that running
the remaining seeds would not answer a new question.

Interpretation:

- Removing the geometric mean did not make CGP imitation reliable.
- The first, expert-only CGP fit was already far below the solution threshold.
- Fresh CGP refitting introduced substantial instability.
- Plain DAgger cannot compensate for a student learner that cannot reliably
  fit the initial supervised problem.

## Linear-regression control experiment

Implementation:

- `scripts/dagger_linear.py`
- Ordinary least-squares continuous-action student, matching the relevant
  policy class in Kohler et al.
- Five seeds, each with 10,000 randomly selected expert samples.
- Ten requested DAgger iterations, 20 collection trajectories per iteration,
  and early stopping at reward 999.

Retained result:

`artifacts/repertoires/dagger_linear_inverted_pendulum/linear_5seeds_fixed20/`

Outcome:

| Seed | Training MSE | Reward |
|---:|---:|---:|
| 0 | 0.00225083 | 1000 |
| 1 | 0.00258375 | 1000 |
| 2 | 0.00215478 | 1000 |
| 3 | 0.00232795 | 1000 |
| 4 | 0.00218290 | 1000 |

All five seeds solved at iteration 0, so early stopping occurred before any
DAgger aggregation.

Conclusions:

- The ANN expert dataset is sufficient and correctly labelled.
- The Brax evaluation and action interface are correct.
- The expert policy is well approximated by a compact affine controller.
- DAgger is unnecessary for this environment when the supervised learner can
  fit the initial expert distribution.
- The remaining bottleneck is specifically CGP supervised optimization or
  action calibration, rather than the expert, environment, masking, or data
  aggregation pipeline.

## CGP linear-scaling experiment (2026-08-06)

Hypothesis:

CGP may discover a useful expression whose magnitude and offset are poorly
calibrated. Per-individual analytic linear scaling can remove those two search
dimensions.

Implementation:

- For each CGP output, fit `scaled = slope * raw + intercept` analytically by
  least squares over the complete training dataset.
- Store all slopes followed by all intercepts in
  `genotype["weights"]["custom_weights"]`. For inverted pendulum this remains
  the established two-value layout `[slope, intercept]`.
- Use a two-pass chunked scorer: first compute global regression statistics,
  then evaluate clipped scaled-action MSE.
- Store the fitted weights in every evaluated genome through Lamarckian
  repertoire updates.
- Apply the stored scaling before action clipping during Brax validation and
  DAgger collection.
- Log raw and scaled imitation losses separately.

Current fixed-dataset test:

- Result directory:
  `artifacts/repertoires/dagger_inverted_pendulum/cgp_linear_scaling_bc_5seeds/`
- Five seeds.
- One expert-only fitting iteration; no DAgger aggregation.
- 10,000 expert samples per seed.
- Population 100 and 50 generations.
- Ten Brax evaluation trajectories.
- GPU execution.

Outcome:

| Seed | Raw MSE | Scaled MSE | Reward | Solved |
|---:|---:|---:|---:|:---:|
| 0 | 0.034843 | 0.002331 | 832.8 | No |
| 1 | 0.596405 | 0.003216 | 46.0 | No |
| 2 | 0.006242 | 0.002559 | 624.4 | No |
| 3 | 0.514655 | 0.003076 | 89.9 | No |
| 4 | 0.645673 | 0.002599 | 58.5 | No |

Linear scaling substantially reduced supervised error for every seed and
produced two much stronger controllers (seeds 0 and 2), but none reached the
999 solution threshold. Similar scaled MSEs still led to drastically different
closed-loop rewards. This reinforces that average expert-action MSE alone does
not characterize controller stability and motivates post-search optimization
of graph connection weights followed by Brax validation.

Success criteria:

- Scaled loss should improve materially over raw loss.
- Held policy rewards should improve over the unscaled CGP expert-only range
  of roughly 140--234.
- Ideally at least one seed should reach reward 999 before DAgger is
  reintroduced.

## Current decision tree

1. Finish the fixed-dataset linear-scaling experiment.
2. If scaling solves or substantially improves CGP, repeat over more seeds and
   evaluate on a larger held-out rollout set.
3. Only then reintroduce DAgger, initially with population warm-starting.
4. If scaling does not improve reward, compare CGP and linear regression on
   the exact same train/held-out samples and inspect expression capacity,
   observation normalization, and per-state error around failure states.
5. Consider balanced weighting of expert and newest student batches only after
   expert-only CGP fitting is reliable.

## Two-stage expression search and Adam connection tuning (2026-08-06)

Implementation:

- `scripts/cgp_expression_then_adam.py`
- Independently supports raw and linearly scaled unweighted expression search,
  or runs both for a paired comparison.
- Selects the top loss-ranked final expressions.
- Freezes graph genes and enables `weighted_inputs=True` only for the Adam
  stage, exposing both `inputs1` and `inputs2` connection-weight arrays.
- Uses the existing batched `optimize_constants_with_sgd` implementation with
  Optax Adam, mini-batching, gradient clipping, finite-gradient guards, and
  configurable steps and learning rate.
- Recomputes analytic output scaling after Adam for the scaled branch.
- Records train and held-out MSE plus Brax reward before and after Adam for
  every candidate.
- Saves the search population, candidates before and after Adam, selected final
  individual, exact train/validation split, candidate tables, and summaries.

Verification:

- Batched synthetic Adam optimization completed with the expected connection
  and custom-weight genotype shapes.
- A minimal GPU Brax smoke test completed both raw and scaled branches and
  produced reloadable artifacts; temporary smoke-test outputs were removed.

Full five-seed result:

`artifacts/repertoires/cgp_expression_adam_inverted_pendulum/expression_adam_5seeds/`

- All ten seed/mode variants completed; none reached reward 999.
- Raw search had mean best reward 101.1 before Adam and 100.6 after Adam.
- Scaled search had mean best reward 302.7 before Adam and 326.4 after Adam,
  but the median fell from 66.9 to 64.4 because the mean improvement was driven
  by seed 3 rising from 423.3 to 558.2.
- The best policy was scaled seed 2 before Adam at reward 936.5. Adam reduced
  its held-out MSE but also reduced reward to 927.2.
- Adam reduced the best held-out MSE in every seed/mode variant, yet reward
  decreased in four of ten variants. Connection-weight optimization therefore
  works numerically but average action error remains misaligned with the
  closed-loop objective.
- Candidate-level held-out MSE was more correlated with reward in the scaled
  branch than the raw branch, but similar errors still admitted large reward
  differences.

Comparison notebook:

`artifacts/repertoires/cgp_expression_adam_inverted_pendulum/expression_adam_5seeds/pendulum_distillation.ipynb`

The notebook combines the linear-regression, analytic linear-scaling, and
expression-plus-Adam results. It includes success and reward summaries,
per-seed comparisons, Adam before/after deltas, candidate-level held-out-MSE
versus reward plots, scaling gains, candidate tables, and conclusions.

## Q-DAgger-weighted expression search and Adam (2026-08-06)

Question:

Does emphasizing states where a wrong action has high expert-Q cost improve
CGP structure search and connection-weight optimization relative to uniform
action MSE?

Implementation:

- Uses `Q(s, a_expert) - min_a Q(s, a)` as the non-negative state weight.
- Approximates the minimum for inverted pendulum with 101 evenly spaced
  actions over `[-1, 1]` because the saved SAC critic has a continuous,
  one-dimensional action input.
- Normalizes weights to mean one without changing relative importance.
- Applies weights consistently to CGP expression fitness, analytic linear
  scaling, train/held-out error, and mini-batch Adam connection updates.
- Stores normalized train/validation weights, raw Q gaps, effective sample
  size, and weight-distribution statistics with every result.
- Retains the same five seeds, raw/scaled branches, dataset split, population,
  generations, top-k candidates, Adam configuration, and rollout validation as
  the uniform-loss experiment.

Result directory:

`artifacts/repertoires/cgp_expression_adam_inverted_pendulum/expression_adam_qdagger_5seeds/`

Status: running at the time this entry was written.

## Template for future entries

```text
## YYYY-MM-DD: short experiment name

Question:
Configuration:
Result directory:
Outcome:
Interpretation:
Decision / next experiment:
```

## 2026-09-14: CGP feature-count sensitivity (both teachers)

Completed k={1,2,4,8,16,32}, five seeds per setting, for saved CGP and SAC ANN teachers on inverted double pendulum. Reused ten k=8 results and ran 50 new experiments sequentially on GPU. Fixed 50 nodes, population 100, 100 generations, 8,000 training/2,000 held-out rows and ten 1,000-step evaluation episodes per seed. Exact saved splits match across k. Every readout includes the additional constant input 1.

Results and audit: `artifacts/repertoires/cgp_feature_inverted_double_pendulum/k_sensitivity_5seeds/`. Rendered analysis is in `artifacts/repertoires/distillation_comparison.ipynb`.

| k | CGP mean reward | CGP solved / 5 | ANN mean reward | ANN solved / 5 |
|---:|---:|---:|---:|---:|
| 1 | 5742.7 | 0 | 1077.6 | 0 |
| 2 | 7298.5 | 0 | 1612.3 | 0 |
| 4 | 7032.8 | 0 | 1222.5 | 0 |
| 8 | 8103.2 | 0 | 5742.2 | 2 |
| 16 | 8854.2 | 0 | 3097.6 | 1 |
| 32 | 9228.7 | 2 | 2266.6 | 0 |

Success threshold: mean episode reward >=9,350. The notebook also reports >=9,359. k=32 is strongest for the CGP teacher; k=8 remains strongest for the ANN teacher. Increasing k improves imitation error but does not consistently improve control. No setting solves more than 2/5 seeds. Fit timing is roughly 31–33 seconds for newly timed k<=16 settings and 40–41 seconds for k=32 (including compilation); historical k=8 timing is unavailable.

## 2026-09-15: Q-DAgger with k=8 CGP features and full replay

Implemented `scripts/dagger_cgp_features.py` with exact Q-weighted least-squares readouts and CGP fitness. Raw expert-to-worst-grid-action Q gaps are preserved across collection batches and normalized over the full replay.

Five seeds, k=8 plus intercept, 50 nodes, population 100, 100 generations per independent refit, up to ten fits, 20 ANN-labelled student trajectories between fits, and ten evaluation episodes per fit. Initial data: 8,000 training and 2,000 fixed held-out rows, identical to the prior uniform k=8 split. Stop threshold: 9,359.

Results: `artifacts/repertoires/dagger_cgp_feature_inverted_double_pendulum/ann_qdagger_k8_5seeds_full_replay/`.

| Seed | Last iteration | Final reward | Best observed reward | Final replay rows |
|---:|---:|---:|---:|---:|
| 0 | 9 | 4098.9 | 8056.0 | 75735 |
| 1 | 4 | 9359.8 | 9359.8 | 18551 |
| 2 | 9 | 4417.9 | 7921.0 | 75768 |
| 3 | 0 | 9359.6 | 9359.6 | 8000 |
| 4 | 9 | 3409.5 | 8987.2 | 59790 |

Outcome: 2/5 solved at 9,359, mean final reward 6129.1. Seed 3 solved at bootstrap; seed 1 solved after four expansions. Other seeds lost stronger intermediate policies during refitting. Compared with uniform BC, success at 9,350 remains 2/5; this is not an equal-budget ablation of Q weighting.

Validation: six tests and a two-iteration GPU smoke passed; all final replays reconstruct from saved batches, held-out rows remain fixed, global Q normalization is verified, and all collection invalid-transition counts are zero. Notebook updated with full metrics, baseline comparisons, and a six-panel progression figure.

## 2026-09-15: Initial BC comparison with eight shared CGP features

Extended the initial behavioral-cloning plot to Inverted Pendulum, Inverted Double Pendulum, Hopper and Walker2d, with Linear, ANN32×32, Operon, and CGP features + LR under Uniform and Q-DAgger weighting. Five seeds per group. Ran 35 new CGP fits plus ten missing ANN fits on Inverted Pendulum; reused five uniform IDP CGP fits and completed historical baselines. All new fits used CUDA.

CGP: k=8 shared outputs, 50 nodes, population 100, 100 generations, 8,000 training / 2,000 held-out expert samples, ten 1,000-step evaluation episodes. Each action has an independent least-squares readout including an intercept: 9×1 for the pendulums, 9×3 for Hopper, 9×6 for Walker2d. Q weights apply to regression and evolutionary fitness; initial-fit weights are normalized over the sampled dataset before splitting (the full-replay DAgger runner normalizes over its training replay).

| Environment | Uniform mean reward | Q-DAgger mean reward | Uniform held-out MSE | Q-weighted held-out MSE |
|---|---:|---:|---:|---:|
| Inverted pendulum | 532.40 | 97.72 | 0.000937894 | 0.0155739 |
| Inverted double pendulum | 5742.24 | 2681.76 | 0.000402914 | 0.00103296 |
| Hopper | 300.13 | 162.64 | 0.0912438 | 0.0877796 |
| Walker2d | 64.80 | 25.89 | 0.155889 | 0.144332 |

Inverted Pendulum success (mean reward >=999): Linear 5/5 Uniform and 4/5 Q-weighted; ANN 3/5 and 0/5; Operon 3/5 and 5/5; CGP 2/5 and 0/5. Linear values come from iteration-zero metrics, excluding the Q-weighted seed-4 improvement after expansion.

Fixed `distillation.rollouts.masked_return`: select valid rewards with `where` rather than multiplying by a zero mask, so post-terminal NaNs cannot contaminate returns. Four initially nonfinite Walker2d summaries (CGP Uniform seed 0, CGP Q seed 1, Operon Uniform seed 3, Operon Q seed 4) were re-evaluated using their original policies and evaluation seeds. The uniform Operon replay directly reproduces NaNs only after termination. The other three original NaNs do not recur in instrumented replay; their original causes remain unconfirmed. No invalid rewards occurred before termination in any replay. Original summaries remain intact; the notebook applies the explicit correction overlay in `artifacts/repertoires/initial_feature_comparison_evaluation/corrections.json`, alongside saved traces.

Artifacts: `artifacts/repertoires/initial_feature_comparison_{per_seed,summary}.csv`, `initial_feature_comparison.png`, `initial_feature_comparison_audit.json`, and the updated `distillation_comparison.ipynb`. New CGP directories use `cgp_feature_<env>/ann_initial_<uniform|q_dagger>_k8_5seeds`; Inverted Pendulum ANN uses `neural_initial_inverted_pendulum/small_ann_32x32_both_5seeds`.

Validation: eight tests passed; all 40 CGP saved policies reproduce their reported held-out MSE, store eight graph outputs with correctly shaped action readouts, and have matching Uniform/Q-weighted data splits. Notebook cells execute, the notebook validates, and the rendered comparison includes a Walker2d reward zoom.

## 2026-09-15: Independent k=8 feature CGP and regression per action

Completed five seeds for Hopper and Walker2d under Uniform and Q-DAgger weighting. Each action has its own independently evolved CGP with eight features, a 50-node graph, population 100, 100 generations and a fitted scalar linear regression plus intercept. No shared graphs or populations. Hopper has three graphs per policy; Walker2d six. Total search budget and node capacity are 3× / 6× the shared-feature baseline. Regression coefficient count remains nine per action.

Initial BC only: 8,000 training / 2,000 held-out expert rows and ten 1,000-step evaluation episodes. All 20 policies / 90 action graphs completed on CUDA. Search seed = dataset seed + 10000 × action index. Q weights use the same critic/grid protocol, applied to each scalar fit and objective. Exact data splits match the prior shared-feature runs. Recomputed Q weights are not bitwise identical: max absolute differences 1.16e-5 (Hopper) and 0.00340 (Walker2d); Walker2d mean absolute discrepancies are approximately 5e-6. These differences are recorded rather than treated as exact equality.

| Environment | Weighting | Shared reward | Independent reward | Shared ordinary test MSE | Independent ordinary test MSE |
|---|---|---:|---:|---:|---:|
| Hopper | Uniform | 300.13 | 684.40 | 0.0912438 | 0.0803599 |
| Hopper | Q-DAgger | 162.64 | 308.63 | 0.0936197 | 0.0846578 |
| Walker2d | Uniform | 64.80 | 76.17 | 0.155889 | 0.114837 |
| Walker2d | Q-DAgger | 25.89 | 123.32 | 0.170791 | 0.126262 |

Independent graphs improve mean reward and ordinary held-out MSE in all four groups. Gains are not an equal-compute ablation of feature sharing, and absolute rewards remain below the ANN experts. Shared Walker2d comparisons use the documented terminal-mask replay overlay from the initial comparison.

Runner: `scripts/cgp_action_feature_imitation.py`. Execution helper: `distillation.independent_feature_policy.independent_feature_action`. Artifacts: `artifacts/repertoires/cgp_action_feature_<env>/ann_independent_<uniform|q_dagger>_k8_5seeds/`; every action stores its individual, population, history and summary. The full policy stores the tuple of individuals in `final_individual.pickle`.

Report: new independent-feature block in `artifacts/repertoires/distillation_comparison.ipynb`, with two rows (environments) and convergence / ordinary MSE / reward columns, plus weighted MSE and timing in the table. Exported `independent_feature_comparison.png`, per-seed and summary CSVs, and `independent_feature_audit.json`. Ten tests passed; all 20 bundles / 90 action graphs passed saved-readout, split and MSE reproduction audits; all evaluation rewards finite. Notebook validated and plot inspected.

## 2026-09-17: ANN feature-count sensitivity across four environments

Completed k=1,2,4,8,16,32, seeds 0–4, Uniform and Q-DAgger initial behavioral cloning from ANN teachers. Both pendulums use shared CGP features; Hopper and Walker2d include shared CGP and one independently evolved CGP per action. Every readout includes an intercept. Each CGP has 50 nodes, population 100, 100 generations; 8,000 training and 2,000 held-out expert samples; ten 1,000-step evaluation episodes. No dataset expansion. Independent models receive 3×/6× total search budget and graph capacity in Hopper/Walker2d.

All 360 policies / 72 groups completed: 85 compatible historical fits reused, 275 new fits on CUDA. New fits load each architecture's exact k=8 data split and weights; reference hashes are frozen in the manifest. Cross-architecture historical Q-weight differences remain documented in the preceding experiment. Existing shared Walker2d k=8 reward corrections are applied explicitly. The 275 new processes took 11.39 hours combined, including initialization and evaluation.

k=32 minimizes mean ordinary held-out MSE in all 12 environment/architecture/weighting groups. Best reward is not monotonic in feature count. Independent CGPs lower held-out MSE in 24/24 matched locomotion comparisons and improve mean reward in 20/24, with a larger compute budget. Inverted Pendulum Uniform k=4 achieves reward 1,000 in all five seeds.

Highest observed mean reward per group (MSE below is ordinary, not Q-weighted):

| Environment | Architecture | Weighting | k | Reward mean ± SD | Train MSE | Held-out MSE | Mean fit seconds |
|---|---|---|---:|---:|---:|---:|---:|
| hopper | independent | q_dagger | 32 | 416.02 ± 427.70 | 0.0637597 | 0.0648976 | 112.4 |
| hopper | independent | uniform | 8 | 684.40 ± 373.10 | 0.07973 | 0.0803599 | 110.3 |
| hopper | shared | q_dagger | 16 | 968.02 ± 777.57 | 0.0828684 | 0.0828874 | 37.9 |
| hopper | shared | uniform | 32 | 325.45 ± 133.66 | 0.0665919 | 0.0671815 | 42.5 |
| inverted_double_pendulum | shared | q_dagger | 16 | 3760.61 ± 4299.22 | 0.000282407 | 0.000442251 | 35.8 |
| inverted_double_pendulum | shared | uniform | 8 | 5742.24 ± 4632.76 | 0.000311592 | 0.000402914 | unrecorded |
| inverted_pendulum | shared | q_dagger | 32 | 521.90 ± 466.10 | 0.00127273 | 0.00143217 | 39.9 |
| inverted_pendulum | shared | uniform | 4 | 1000.00 ± 0.00 | 0.0015722 | 0.00164572 | 32.4 |
| walker2d | independent | q_dagger | 16 | 220.59 ± 112.03 | 0.106324 | 0.107296 | 199.4 |
| walker2d | independent | uniform | 4 | 102.30 ± 75.60 | 0.134717 | 0.134773 | 215.9 |
| walker2d | shared | q_dagger | 16 | 126.43 ± 74.03 | 0.13423 | 0.134494 | 36.4 |
| walker2d | shared | uniform | 32 | 141.47 ± 120.47 | 0.102104 | 0.102511 | 42.3 |

These are descriptive five-seed comparisons; tuning k on these rewards needs a separate evaluation for generalization claims. No default feature-count setting was changed.

Fit-time ranges of group means: shared CGP 29.7–42.7 seconds, independent Hopper 95.3–114.3 seconds, independent Walker2d 171.6–226.2 seconds. Historical IDP Uniform k=8 timings are unavailable and were not reconstructed. Per-seed CSVs also retain evaluation times, total process times for new runs, weighted MSE, and provenance; aggregate CSVs include means, sample standard deviations and finite counts.

Artifacts: `artifacts/repertoires/ann_feature_sensitivity_5seeds/{manifest.json,status.json,per_seed_results.csv,summary.csv,uniform_sensitivity.png,q_dagger_sensitivity.png,audit.json}` plus per-seed runs and logs. The final section of `distillation_comparison.ipynb` contains the two four-by-four sensitivity figures and all group metrics. Existing comparison plots are preserved.

Validation: 12 tests passed; all 360 bundles / 780 graphs pass structural/readout checks and exact reference-data/Q-weight comparisons. All 72 groups have five finite rewards and ordinary/weighted training and held-out MSE values. Replaying seed 0 at k=1 and k=32 in every group verifies 72 losses across 24 policies on the original CUDA evaluation path (maximum absolute difference 2.06e-7). An extra outer JIT on CPU produced small float32 differences, so the authoritative replay uses the runner's original evaluation path and backend.

## 2026-09-17: Hopper k=16 CGP-feature DAgger reproduction — launched

Launched 20 variants: seeds 0–4 × Uniform/Q-DAgger × shared/per-action CGP.
Matches the small-ANN DAgger reference's exact saved bootstrap data and weights,
fixed held-out split, collection/evaluation seeds, ten fitting iterations,
20 collection trajectories, ten evaluation trajectories, 1,000-step episodes,
and 3,250 stopping target. Preserves the ANN's per-batch Q-weight normalization.
All replay rows participate in every from-scratch CGP refit; population 100,
100 generations, 50 nodes, k=16 with linear readout and intercept. Independent
Hopper policies use three separate graphs (48 features total, 3× search budget).
CGP uses training-only selection; the reference ANN used held-out early stopping.

Both architectures passed two-iteration GPU smoke tests (reduced population,
generations and rollout length), with exact replay reconstruction and unchanged
held-out datasets. Ten focused tests passed. All five saved ANN bootstrap splits
match exactly between weighting modes. Every production variant audits its saved
replay before marking completion. Production outcomes are pending.

Runner: `scripts/hopper_feature_dagger.py`; report: `scripts/hopper_feature_dagger_report.py`.
Artifacts, driver log and status: `artifacts/repertoires/dagger_cgp_feature_hopper/k16_ann_protocol_both_5seeds_target3250/`.
The notebook has a new progress section, automatically refreshed after each
completed variant. Restarting the runner skips completed variants and restarts
unfinished ones from bootstrap.

## 2026-09-18: Hopper direct CGP evolution on Generalized — launched

Five seeds (0–4), 1,500 generations each, population 100, ten elites, tournament
size three, 50-node CGP directly producing three actions. Fitness averages five
fresh 1,000-step episodes per candidate per generation. Fixed search budget;
no target-based stopping or validation-based selection. Uses raw Brax Hopper
with the Generalized backend, matching the CGP DAgger experiments.

GPU-only runner: `scripts/hopper_direct_evolution.py`. Sequential isolated seed
processes save best-search policy checkpoints on every improvement, the final
best-search policy, final-generation winner, and final population. Both saved
policies are reloaded and evaluated on the corresponding DAgger seed's ten
1,000-step episodes (seed * 100000 + 90000 + arange(10)); those episodes never
enter search selection. Results are pending.

Artifacts: `artifacts/policy_search/baselines/hopper_generalized_5seeds/` (status, config,
per-seed logs, aggregate summary, and `hopper/seed_N/` policies and metrics).
A two-generation reduced GPU smoke test passed, including policy persistence
and reloaded-policy evaluation with the expected episode seeds. Four rollout
and evaluation tests passed. The full five-seed driver has been launched.
