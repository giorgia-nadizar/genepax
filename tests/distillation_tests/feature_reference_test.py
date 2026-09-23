from pathlib import Path

import json

import numpy as np
import pytest

from distillation_experiments.scripts.experiments.feature_reference_data import load_reference_split


def test_reference_preserves_weighted_split_exactly(tmp_path: Path) -> None:
    (tmp_path/'seed_0').mkdir()
    dataset=tmp_path/'teacher.npz'
    (tmp_path/'config.json').write_text(json.dumps(dict(env='hopper',state_weighting='q_dagger',dataset_path=str(dataset))))
    arrays=dict(X=np.array([[1.,2.],[3.,4.]],np.float32),y=np.array([[.1],[.2]],np.float32),
                validation_X=np.array([[5.,6.]],np.float32),validation_y=np.array([[.3]],np.float32),
                train_weights=np.array([.2,1.7],np.float32),validation_weights=np.array([1.1],np.float32))
    np.savez(tmp_path/'seed_0/dataset.npz',**arrays)
    loaded=load_reference_split(tmp_path,0,env='hopper',dataset_path=dataset,weighting='q_dagger',train_count=2,validation_count=1)
    for actual,expected in zip(loaded,arrays.values()):np.testing.assert_array_equal(actual,expected)
    with pytest.raises(ValueError,match='environment/weighting'):
        load_reference_split(tmp_path,0,env='hopper',dataset_path=dataset,weighting='uniform',train_count=2,validation_count=1)
    with pytest.raises(ValueError,match='teacher dataset'):
        load_reference_split(tmp_path,0,env='hopper',dataset_path=tmp_path/'other.npz',weighting='q_dagger',train_count=2,validation_count=1)


def test_legacy_uniform_reference_keeps_unweighted_solver_path(tmp_path: Path) -> None:
    (tmp_path/'seed_0').mkdir()
    dataset=tmp_path/'teacher.npz'
    (tmp_path/'config.json').write_text(json.dumps(dict(env='hopper',dataset_path=str(dataset))))
    np.savez(tmp_path/'seed_0/dataset.npz',X=np.zeros((2,2)),y=np.zeros((2,1)),validation_X=np.zeros((1,2)),validation_y=np.zeros((1,1)))
    loaded=load_reference_split(tmp_path,0,env='hopper',dataset_path=dataset,weighting='uniform',train_count=2,validation_count=1)
    assert loaded[-2:]==(None,None)
