# Distillation analyses — editable inventory

Prepared 2026-09-21 from the current notebook cells. This is a selection list for a future summary notebook; no new notebook or experiments have been created.

## Scope and editing

- **Notebook A:** [Inverted-pendulum distillation comparison](cgp_expression_adam_inverted_pendulum/expression_adam_5seeds/pendulum_distillation.ipynb) — 39 cells.
- **Notebook B:** [Policy distillation across Brax environments](distillation_comparison.ipynb) — 63 cells.
- These are the two substantive comparison notebooks assumed to be intended. Other related notebooks are listed at the end.
- Check `[x]` for items to include. Leave unchecked, delete, reorder, or annotate items as desired. IDs remain stable when reordered.
- Cell numbers below are **1-based**, counting both Markdown and code cells as of this inventory.
- Unless stated otherwise, “five seeds” means five per model/weighting/architecture setting.

## Summary-notebook preferences

- Working title:
- Intended audience:
- Main question/message:
- Preferred section order (item IDs):
- Items to merge:
- Items to omit:
- Additional figures or analyses wanted:

## Notebook A — Inverted Pendulum: fitting and controller diagnosis

- [ ] **A01 — Overall inverted-pendulum student comparison**
  - Source: notebook A, cells 1–3, 5.
  - Scope: Uniform and Q-weighted linear regression; CGP analytic scaling; raw/scaled expression search before and after Adam; per-generation Adam variants. Five seeds per setting; reward target 999.
  - Available: Per-seed comparison tables, reward/loss comparisons, solved counts and conclusions.
  - Interpretation/detail to preserve: Distinguish post-search Adam from optimizing every individual each generation.
  - Your edits:

- [ ] **A02 — Analytic output scaling for CGP**
  - Source: notebook A, cells 4.
  - Scope: Raw versus analytically scaled action outputs, paired by seed.
  - Available: Raw/scaled training-MSE table, percentage reduction and paired log-scale plot.
  - Interpretation/detail to preserve: Lower action error need not produce a successful controller.
  - Your edits:

- [ ] **A03 — Effect of Q-DAgger state weighting**
  - Source: notebook A, cells 6–7.
  - Scope: Uniform versus critic-gap weighting for linear regression and CGP/Adam.
  - Available: Effective sample-size table, solved counts and reward interpretation.
  - Interpretation/detail to preserve: Q-weighted MSE and ordinary MSE are different objectives; check initial versus final linear results before combining notebooks.
  - Your edits:

- [ ] **A04 — PySR: loss-selected versus reward-best expression**
  - Source: notebook A, cells 8–9.
  - Scope: Five seeds under Uniform and Q-DAgger; 40 tracked search iterations.
  - Available: Expressions, held-out loss, selected-policy reward, best hall-of-fame reward, candidate counts and solved counts.
  - Interpretation/detail to preserve: Reward-best hall-of-fame selection is retrospective and differs from validation-loss selection.
  - Your edits:

- [ ] **A05 — Operon trial and comparison with PySR**
  - Source: notebook A, cells 10–11.
  - Scope: Five seeds under both weightings; 100 tracked generations, population 500, maximum expression length 25.
  - Available: Expression/metric tables and selected versus reward-best bar comparison across the two engines.
  - Interpretation/detail to preserve: Operon approximates training weights through resampling; validation uses exact Q weights.
  - Your edits:

- [ ] **A06 — Symbolic-search convergence across engines**
  - Source: notebook A, cells 12–14.
  - Scope: PySR, Operon, CGP raw/scaled, and CGP with per-generation Adam; Uniform and Q-weighted objectives.
  - Available: Median best-training-loss curves with seed minimum–maximum bands and linear reference.
  - Interpretation/detail to preserve: Engine-native search steps and compute budgets are not equivalent.
  - Your edits:

- [ ] **A07 — Stacked convergence and final-loss distributions**
  - Source: notebook A, cells 15–16.
  - Scope: Rearrangement of the preceding multi-engine comparison, including linear regression.
  - Available: Stacked convergence panels and final tracked loss distributions; accompanying reward data are assembled here.
  - Interpretation/detail to preserve: Alternative presentation of A06, not a separate experiment.
  - Your edits:

- [ ] **A08 — Training loss, held-out loss and reward distributions**
  - Source: notebook A, cells 17–18.
  - Scope: Five-seed distributions for the symbolic methods and linear reference, split by weighting.
  - Available: Two-by-three boxplot/point layout: final training loss, held-out loss and rollout reward.
  - Interpretation/detail to preserve: Keep model-specific selection and loss conventions visible.
  - Your edits:

- [ ] **A09 — Local policy sensitivity and feedback strength**
  - Source: notebook A, cells 19–22.
  - Scope: Uniform linear, PySR, Operon and scaled CGP with generation Adam; finite differences on expert states and near their median.
  - Available: Sensitivity distributions, feedback-gain norm distributions and per-coordinate gain table.
  - Interpretation/detail to preserve: Sharp feedback is not necessarily mathematical discontinuity or instability.
  - Your edits:

- [ ] **A10 — One-coordinate action response curves**
  - Source: notebook A, cells 23–24.
  - Scope: Vary cart position, pole angle, cart velocity and pole angular velocity separately.
  - Available: Four response panels with seed medians and minimum–maximum envelopes.
  - Interpretation/detail to preserve: Exposes omitted, weak or inconsistent corrective feedback despite low imitation MSE.
  - Your edits:

- [ ] **A11 — Cross-method pole recovery maps**
  - Source: notebook A, cells 25–26.
  - Scope: Representative seed 2 for the four methods; pole angle versus angular velocity, other state coordinates fixed.
  - Available: Executed-action heatmaps on a common state plane.
  - Interpretation/detail to preserve: Representative visualization, not a five-seed aggregate.
  - Your edits:

- [ ] **A12 — Trajectory roughness and survival**
  - Source: notebook A, cells 27–28.
  - Scope: 300-step Generalized-backend rollouts of the same policies, using valid pre-terminal transitions.
  - Available: Lifetime, action total variation, jerk, normalized variation, and comparison plots/tables.
  - Interpretation/detail to preserve: Separates smoothness on expert states from action behavior after state distribution shifts.
  - Your edits:

- [ ] **A13 — Interpretation of smoothness and inductive bias**
  - Source: notebook A, cells 29–30.
  - Scope: Synthesis of A09–A12.
  - Available: Narrative: PySR can be smooth yet under-responsive; globally consistent feedback helps the linear controller.
  - Interpretation/detail to preserve: The notebook does not perform a formal closed-loop stability/eigenvalue analysis.
  - Your edits:

- [ ] **A14 — CGP effective feature-gain heatmap**
  - Source: notebook A, cells 31–34.
  - Scope: 20 per-generation-Adam CGP policies: five seeds × raw/scaled × Uniform/Q-DAgger.
  - Available: Normalized gain heatmap by physical coordinate and policy, marking solved variants.
  - Interpretation/detail to preserve: An input appearing in a graph is not evidence that it has an effective corrective gain.
  - Your edits:

- [ ] **A15 — CGP properties associated with reward**
  - Source: notebook A, cells 35–36.
  - Scope: Gain alignment and magnitude relative to same-seed linear controllers, recovery-map error, saturation and neighboring action jumps.
  - Available: Reward scatter plots and solved-versus-failed diagnostic summaries.
  - Interpretation/detail to preserve: Associations from 20 policies; not causal evidence.
  - Your edits:

- [ ] **A16 — Solved-versus-failed CGP recovery geometry**
  - Source: notebook A, cells 37–39.
  - Scope: Representative solved and failed scaled policies for both weightings.
  - Available: Four recovery maps and CGP-specific interpretation.
  - Interpretation/detail to preserve: Feature-path availability and adequate feedback strength matter; saturation can accompany successful control.
  - Your edits:

## Notebook B — Multiple environments, teacher comparisons and DAgger

- [ ] **B01 — Initial behavioral cloning across four environments**
  - Source: notebook B, cells 1–5.
  - Scope: Inverted Pendulum, Inverted Double Pendulum, Hopper and Walker2d; linear, small ANN, Operon, shared k=8 CGP features, plus per-action k=8 CGPs for locomotion; five seeds per available setting and both weightings.
  - Available: Reward and held-out-MSE boxplots, individual seeds, coverage/summary tables, Walker2d zoom and Inverted Pendulum solved counts.
  - Interpretation/detail to preserve: Fixed teacher data, no aggregation. Initial Q-DAgger panels retain Q-weighted MSE. Walker2d corrected-return overlays are applied.
  - Your edits:

- [ ] **B02 — Shared versus independent k=8 CGP features**
  - Source: notebook B, cells 6–8.
  - Scope: Hopper and Walker2d; one shared graph versus one graph per action, under both weightings.
  - Available: Evolutionary convergence, ordinary held-out MSE, reward, timing/summary tables and saved-policy audits.
  - Interpretation/detail to preserve: Independent graphs use 3×/6× the nodes and search budget; historical Q weights have documented small discrepancies.
  - Your edits:

- [ ] **B03 — Hopper: replace one ANN action with Operon**
  - Source: notebook B, cells 9–11.
  - Scope: Three possible replaced actions × two weightings × five seeds; initial fitting only.
  - Available: Paired-seed reward plots, median-reward and success heatmaps, held-out-MSE versus reward scatter.
  - Interpretation/detail to preserve: Tests which action substitutions are tolerable and reveals catastrophic outliers.
  - Your edits:

- [ ] **B04 — Hopper: replace two ANN actions with Operon**
  - Source: notebook B, cells 12–14.
  - Scope: Three Operon action pairs, retaining one ANN action; two weightings and five seeds.
  - Available: Paired reward plots, summaries/heatmaps and MSE–reward diagnostics.
  - Interpretation/detail to preserve: Complementary action-ablation experiment to B03.
  - Your edits:

- [ ] **B05 — Hopper linear/ANN action hybrids**
  - Source: notebook B, cells 15–16.
  - Scope: Replace one or two ANN actions with linear regression; both weightings and five seeds.
  - Available: Median-reward heatmaps across action choices.
  - Interpretation/detail to preserve: Initial hybrid policies, not DAgger dataset expansion.
  - Your edits:

- [ ] **B06 — Operon versus linear hybrid summary**
  - Source: notebook B, cells 17–18.
  - Scope: Reuse B03–B05.
  - Available: Four heatmaps with a common color scale: model class × one/two replaced actions.
  - Interpretation/detail to preserve: Combined presentation of existing ablations.
  - Your edits:

- [ ] **B07 — Hopper linear-regression DAgger progression**
  - Source: notebook B, cells 19–21.
  - Scope: Five seeds for Uniform and Q-DAgger; 20 student collection trajectories between fits.
  - Available: Reward, training loss, fixed expert-test loss and dataset size over iterations; seed curves, medians and ranges.
  - Interpretation/detail to preserve: Initial held-out expert split stays fixed.
  - Your edits:

- [ ] **B08 — Hopper small-ANN DAgger progression**
  - Source: notebook B, cells 22–24.
  - Scope: 32×32 ANN; five seeds per weighting; refit from scratch after aggregation.
  - Available: Same four progression metrics as B07, with expert and target reward references.
  - Interpretation/detail to preserve: Uniform seed 4 reaches 3,250 at iteration 9; the notebook reports no Q-DAgger success.
  - Your edits:

- [ ] **B09 — Hopper Operon DAgger progression**
  - Source: notebook B, cells 25–27.
  - Scope: Five seeds per weighting; ten fits, 20 collection trajectories, ten Operon generations per refit.
  - Available: Reward, sampled training MSE, fixed test MSE and replay size over iterations.
  - Interpretation/detail to preserve: Each refit samples 10,000 replay rows; training loss is not full-replay MSE as in ANN/linear fits.
  - Your edits:

- [ ] **B10 — Inverted Double Pendulum: Operon primitive-set ablation**
  - Source: notebook B, cells 28–30.
  - Scope: Compact versus expanded CGP-like primitive sets; both weightings, five seeds, ten DAgger fits, target 9,359.
  - Available: Reward, fixed expert-test MSE and replay-size curves.
  - Interpretation/detail to preserve: Resumed compact-run artifacts are deduplicated by seed and weighting.
  - Your edits:

- [ ] **B11 — Combined Hopper DAgger progression**
  - Source: notebook B, cells 31–32.
  - Scope: Linear regression, small ANN and Operon from B07–B09.
  - Available: Overlaid model/weighting median curves for reward, training/test loss and replay size.
  - Interpretation/detail to preserve: Presentation alternative to separate progression panels; method-specific training losses differ.
  - Your edits:

- [ ] **B12 — Behavioral cloning of direct-search CGP teachers**
  - Source: notebook B, cells 33–36.
  - Scope: Inverted Pendulum and Inverted Double Pendulum; linear, small ANN and Operon; Uniform, five seeds.
  - Available: Reward boxplots against teacher reward, summary statistics and interpretation.
  - Interpretation/detail to preserve: Fixed teacher data only; low held-out error can coexist with rollout failures.
  - Your edits:

- [ ] **B13 — Inverted Double Pendulum: CGP teacher to k=8 CGP features**
  - Source: notebook B, cells 37–38.
  - Scope: Five-seed fixed-dataset cloning with a learned linear readout and intercept.
  - Available: Per-seed training/test loss, reward and episode ranges; comparison table against other students.
  - Interpretation/detail to preserve: 0/5 reach 9,350; mean reward about 8,103.2. Readout already includes the constant 1/intercept.
  - Your edits:

- [ ] **B14 — Inverted Double Pendulum: ANN teacher to k=8 CGP features**
  - Source: notebook B, cells 39–40, 43.
  - Scope: Same feature-student architecture, ANN teacher, Uniform, five seeds.
  - Available: Per-seed metrics, baseline comparison and teacher-to-teacher table.
  - Interpretation/detail to preserve: 2/5 reach 9,350; mean reward about 5,742.2. Different teacher state distributions limit causal comparisons.
  - Your edits:

- [ ] **B15 — Combined feature-student view for both teachers**
  - Source: notebook B, cells 41–42.
  - Scope: Reuse B13–B14.
  - Available: Two teacher rows × convergence, train/held-out MSE and reward columns, including individual episode returns.
  - Interpretation/detail to preserve: Shared axes; no additional experiment or separate intercept ablation.
  - Your edits:

- [ ] **B16 — Inverted Double Pendulum distillation recap**
  - Source: notebook B, cells 44–45.
  - Scope: ANN versus CGP teachers; linear, ANN, Operon and k=8 CGP-feature students, Uniform, five seeds each.
  - Available: Held-out-MSE and reward boxplots arranged by teacher.
  - Interpretation/detail to preserve: Teacher distributions, train/test splits and budgets differ across methods.
  - Your edits:

- [ ] **B17 — Feature-count sensitivity with ANN and CGP teachers**
  - Source: notebook B, cells 46–49.
  - Scope: Inverted Double Pendulum; k=1,2,4,8,16,32, five seeds, Uniform; 60 policies including reused k=8 fits.
  - Available: Reward, success at 9,350/9,359, train/test MSE, runtime and search-convergence comparisons.
  - Interpretation/detail to preserve: Exact within-teacher/seed data reuse; historical k=8 timing is missing. CGP-teacher reward favors k=32, ANN-teacher reward favors k=8 in this sweep.
  - Your edits:

- [ ] **B18 — Inverted Double Pendulum Q-DAgger with k=8 CGP features**
  - Source: notebook B, cells 50–53.
  - Scope: ANN teacher, five seeds, full replay, 100 CGP generations per fit, up to ten fits.
  - Available: Reward, weighted/ordinary train/test MSE, replay size, effective weight concentration and fitting time; comparison to initial cloning.
  - Interpretation/detail to preserve: 2/5 final policies reach 9,359. Global replay Q-weight normalization differs from the Hopper ANN-matched protocol.
  - Your edits:

- [ ] **B19 — ANN-teacher feature-count sweep across four environments**
  - Source: notebook B, cells 54–56.
  - Scope: k=1,2,4,8,16,32; both weightings, shared/per-action architectures where available; 360 policies in 72 five-seed groups.
  - Available: Four-by-four figure: metric rows and environment columns, with weighting/architecture overlays; per-seed/group tables, timings and audits.
  - Interpretation/detail to preserve: k=32 minimizes mean ordinary held-out MSE in all 12 model/environment/weighting groups, but reward-optimal k varies. Independent search budgets are larger.
  - Your edits:

- [ ] **B20 — Hopper k=16 CGP-feature DAgger progression and final outcomes**
  - Source: notebook B, cells 57–59.
  - Scope: Shared and per-action architectures × Uniform/Q-DAgger × five seeds; 20 runs and 200 fitting iterations.
  - Available: Reward, training/test MSE, replay-size progression and final-reward mean/SD/median/solved table.
  - Interpretation/detail to preserve: Exact ANN bootstrap reuse; per-batch Q normalization; all 20 runs completed without reaching 3,250. Final rewards are distinct from best-ever rewards.
  - Your edits:

- [ ] **B21 — Hopper best-ever reward distributions across distillation methods**
  - Source: notebook B, cells 60–61.
  - Scope: Linear, small ANN, Operon, shared k=16 CGP and per-action k=16 CGP; both weightings, five seeds each (50 runs).
  - Available: Five panels of boxplots plus seed dots using each run’s maximum reward across available DAgger iterations.
  - Interpretation/detail to preserve: Retrospective evaluation maxima; not final-policy rewards or independently confirmed selected policies.
  - Your edits:

- [ ] **B22 — Hopper direct-evolution baseline on the same backend**
  - Source: notebook B, cells 60–61, sixth panel.
  - Scope: Five completed Generalized-backend direct-CGP runs; 1,500 generations, population 100, 50 nodes; policies saved.
  - Available: Separate boxplot of evaluation rewards for each saved best-search policy, on the corresponding DAgger evaluation seeds.
  - Interpretation/detail to preserve: Direct selection uses training search fitness and separate evaluation, unlike B21’s retrospective evaluation peaks. The old Spring horizontal reference was removed.
  - Your edits:

- [ ] **B23 — Hopper best iteration versus achieved reward**
  - Source: notebook B, cells 62–63.
  - Scope: The same 50 distillation runs as B21, with the earliest occurrence used for ties.
  - Available: Scatter plot: x = best DAgger iteration, y = reward at that iteration; Uniform/Q-DAgger panels; method-specific colors and markers.
  - Interpretation/detail to preserve: Iteration 0 is the initial fit; all points use exact coordinates. This replaces the earlier best-iteration boxplot.
  - Your edits:

## Overlap and details to resolve when assembling the summary

- A06–A08 offer multiple presentations of largely the same convergence/loss/reward comparisons; select or merge them.
- B01 includes the per-action CGP results examined in detail in B02. B06 summarizes B03–B05; B11 summarizes B07–B09.
- B15 combines B13–B14; B16 is the wider teacher/student recap. Avoid presenting reused k=8 policies in B17/B19 as new independent repetitions.
- B21 and B23 use the same per-run maxima, respectively emphasizing reward distributions and the iteration at which peaks occur. B20 instead summarizes final policies.
- Keep Uniform MSE, exact Q-weighted MSE, and resampled fitting objectives distinct. The notebooks do not use one universally identical loss convention.
- Verify the linear Inverted Pendulum Q-DAgger narrative before reusing it: Notebook A says all five were solved at iteration 0, while Notebook B explicitly reports 4/5 initial fits and notes later improvement for seed 4. Resolve against the actual metrics rather than copying both claims.
- Feature independence changes both architecture and total search budget. Native generation/iteration counts also differ among CGP, Operon, PySR and ANN.
- Notebook A’s diagnostic cells load policies and perform new evaluations, including Brax rollouts. Notebook B mainly reads saved artifacts. Preserve or precompute diagnostics if the summary should run without GPU/Brax execution.
- Direct evolution in B22 is a comparison baseline, not distillation. Its separately evaluated best-search policies should remain distinguishable from retrospectively selected DAgger reward peaks.

## Other related notebooks (outside the assumed two-notebook scope)

- [ ] **C01 — Legacy symbolic/mixed-policy learning curves:** [artifacts/repertoires/analysis.ipynb](analysis.ipynb), four cells. Inverted Pendulum, ten seeds; reward, mixed reward and training loss versus iteration.
- [ ] **C02 — SPID reward-bootstrap study:** [SPID analysis](spid_inverted_pendulum/ann_reward_bootstrap_ten_seeds/analysis.ipynb), 17 cells. Aggregate outcomes, learning/expert-withdrawal curves, loss–reward alignment, search validation checkpoints, held-out confirmation, best symbolic expressions and fixed-dataset population audit.
- [ ] **C03 — Historical mixed-policy search:** [mixed_policy.ipynb](../policy_search/results/mixed_policy.ipynb), six cells. Separate historical notebook; not included in the main inventory.

## Reusable data and plotting helpers

- Multi-environment feature sweep: `../scripts/ann_feature_sensitivity_report.py`.
- Hopper CGP progression, final outcomes and best-ever reward figures: `../scripts/hopper_feature_dagger_report.py`.
- Hopper per-run peak iteration/reward collection and scatter: `../scripts/hopper_best_iteration_report.py`.
- Hopper peak data for all five distillation methods: `dagger_cgp_feature_hopper/k16_ann_protocol_both_5seeds_target3250/best_iteration_by_method.csv`.
- Hopper direct-evolution saved policies and evaluation returns: `../policy_search/baselines/hopper_generalized_5seeds/hopper/seed_*/`.

## Your notes

<!-- Add decisions or a proposed narrative here. -->
