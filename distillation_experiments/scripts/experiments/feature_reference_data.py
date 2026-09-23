"""Load immutable per-seed reference splits for controlled feature sweeps."""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np


def load_reference_split(run, seed, *, env, dataset_path, weighting, train_count, validation_count):
    run = Path(run)
    config = json.loads((run / 'config.json').read_text())
    if config['env'] != env or config.get('state_weighting', 'uniform') != weighting:
        raise ValueError('Reference environment/weighting does not match this fit')
    if Path(config['dataset_path']).resolve() != Path(dataset_path).resolve():
        raise ValueError('Reference teacher dataset does not match this fit')
    with np.load(run / f'seed_{seed}' / 'dataset.npz') as data:
        arrays = [data[key] for key in ('X', 'y', 'validation_X', 'validation_y')]
        if len(arrays[0]) != train_count or len(arrays[2]) != validation_count:
            raise ValueError('Reference split sizes do not match this fit')
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError('Nonfinite reference data')
        weights = [None, None]
        if weighting == 'q_dagger':
            weights = [data[key] for key in ('train_weights', 'validation_weights')]
            for w, count in zip(weights, (train_count, validation_count)):
                if w.shape != (count,) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
                    raise ValueError('Invalid reference state weights')
            weights = [jnp.asarray(w) for w in weights]
    return (*[jnp.asarray(array) for array in arrays], *weights)
