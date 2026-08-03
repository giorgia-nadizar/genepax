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

Artifact directories are resolved relative to this directory, not the current
working directory.

SAC expert models produced by `train_brax.py` are stored in `expert_models/`.
Behavioral-cloning artificial neural networks produced by
`neural_imitation.py` are stored in `bc_ann_models/`.
