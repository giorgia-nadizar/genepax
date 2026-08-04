# Distillation experiments

This directory contains runnable workflows and their generated artifacts. The
reusable implementation remains in the `distillation` package.

Run scripts from the repository root, for example:

```bash
python -m distillation_experiments.scripts.train_brax
python -m distillation_experiments.scripts.evaluate_brax --env inverted_pendulum
python -m distillation_experiments.scripts.evaluate_brax --env inverted_pendulum --video
python -m distillation_experiments.scripts.collect_data
python -m distillation_experiments.scripts.neural_imitation
python -m distillation_experiments.scripts.spid
python -m distillation_experiments.scripts.policy_search_gp --seed 0
python -m distillation_experiments.scripts.policy_search_mixed --seed 0
```

For a small inverted-pendulum SPID smoke test:

```bash
python -m distillation_experiments.scripts.spid \
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

SPID writes each run below `repertoires/spid_<environment>/<run-name>/`.
Every seed has its own metrics, summary, best-validation genotype, and the
population repertoire from which it was selected. Load `best_genotype.pickle`
with `pickle.load` when evaluating a saved controller. An existing run name is
never overwritten. Omit `--run-name` to use a UTC timestamp.

Artifact directories are resolved relative to this directory, not the current
working directory.

SAC expert models produced by `train_brax.py` are stored in `expert_models/`.
Behavioral-cloning artificial neural networks produced by
`neural_imitation.py` are stored in `bc_ann_models/`.
