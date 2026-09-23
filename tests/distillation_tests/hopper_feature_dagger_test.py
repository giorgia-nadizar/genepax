"""Protect the exact ANN bootstrap reuse against accidentally training on replay."""

from pathlib import Path
import numpy as np

from distillation_experiments.scripts.hopper_feature_dagger import load_bootstrap


def test_bootstrap_excludes_aggregated_states_and_preserves_reference_weights(tmp_path: Path) -> None:
    directory = tmp_path / 'seed_2/q_dagger'
    directory.mkdir(parents=True)
    X = np.arange(8010 * 2).reshape(8010, 2)
    y = X[:, :1] / 100
    weights = np.linspace(.1, 2, len(X))
    test_X = -np.ones((2000, 2))
    test_y = np.zeros((2000, 1))
    test_weights = np.linspace(.2, 1, 2000)
    np.savez(directory / 'final_dataset.npz', X=X, y=y, sample_weights=weights,
             test_X=test_X, test_y=test_y, test_weights=test_weights)
    loaded = load_bootstrap(tmp_path, 2, 'q_dagger')
    for key, expected in [('X', X[:8000]), ('y', y[:8000]), ('sample_weights', weights[:8000]),
                          ('test_X', test_X), ('test_y', test_y), ('test_weights', test_weights)]:
        np.testing.assert_array_equal(loaded[key], expected)


def test_replay_audit_detects_changed_batch_weights(tmp_path: Path) -> None:
    import pytest
    from distillation_experiments.scripts.hopper_feature_dagger import audit_replay
    initial = dict(X=np.ones((2, 2)), y=np.ones((2, 1)), sample_weights=np.array([.2, .8]),
                   test_X=np.zeros((1, 2)), test_y=np.zeros((1, 1)), test_weights=np.ones(1))
    np.savez(tmp_path / 'initial_dataset.npz', **initial)
    batch = dict(X=np.full((1, 2), 2.), y=np.full((1, 1), 2.), sample_weights=np.ones(1))
    (tmp_path / 'iteration_0').mkdir()
    np.savez(tmp_path / 'iteration_0/collected_dataset.npz', **batch)
    final = initial | {key: np.concatenate([initial[key], batch[key]]) for key in batch}
    np.savez(tmp_path / 'final_dataset.npz', **final)
    audit_replay(tmp_path, 1)
    final['sample_weights'][-1] = 2
    np.savez(tmp_path / 'final_dataset.npz', **final)
    with pytest.raises(AssertionError):
        audit_replay(tmp_path, 1)
