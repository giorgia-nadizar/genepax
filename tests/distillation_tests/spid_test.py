import argparse
import json
import pickle
from types import SimpleNamespace

import pytest

import jax
import jax.numpy as jnp

import distillation_experiments.scripts.spid_validated as spid
from distillation_experiments.scripts.spid_validated import (
    combine_dataset_parts,
    create_run_directory,
    fit_and_validate_blocks,
    mixture_weight,
    positive_int,
    positive_int_list,
    recovery_state_mask,
    save_iteration,
    scheduled_expert_weight,
    select_aggregation_policy,
    select_validated_candidate,
    update_reservoir,
)


def test_fixed_expert_weight():
    assert mixture_weight(50, 100, 0.5, 0.5) == pytest.approx(0.5)


def test_expert_weight_linear_schedule_includes_endpoints():
    assert mixture_weight(0, 5, 0.8, 0.2) == pytest.approx(0.8)
    assert mixture_weight(2, 5, 0.8, 0.2) == pytest.approx(0.5)
    assert mixture_weight(4, 5, 0.8, 0.2) == pytest.approx(0.2)


def test_expert_schedule_holds_end_weight_after_annealing():
    assert scheduled_expert_weight(1, 9, 0.8, 0.0) == pytest.approx(0.8)
    assert scheduled_expert_weight(5, 9, 0.8, 0.0) == pytest.approx(0.4)
    assert scheduled_expert_weight(9, 9, 0.8, 0.0) == pytest.approx(0.0)
    assert scheduled_expert_weight(29, 9, 0.8, 0.0) == pytest.approx(0.0)


def test_best_aggregation_policy_guards_against_latest_regression():
    selected = select_aggregation_policy(
        "best", "latest", 7, 20.0, "best", 3, 200.0
    )

    assert selected == ("best", 3, 200.0)


def test_current_aggregation_policy_preserves_original_behavior():
    selected = select_aggregation_policy(
        "current", "latest", 7, 20.0, "best", 3, 200.0
    )

    assert selected == ("latest", 7, 20.0)


@pytest.mark.parametrize("start,end", [(-0.1, 0.5), (0.5, 1.1)])
def test_expert_weight_rejects_invalid_values(start, end):
    with pytest.raises(ValueError):
        mixture_weight(0, 10, start, end)


def test_dagger_reservoir_is_bounded_and_keeps_labels_aligned():
    X = jnp.empty((0, 1))
    y = jnp.empty((0, 1))
    priorities = jnp.empty((0,))

    X, y, priorities = update_reservoir(
        X,
        y,
        priorities,
        X_new=jnp.arange(10, dtype=jnp.float32)[:, None],
        y_new=(10 * jnp.arange(10, dtype=jnp.float32))[:, None],
        max_size=4,
        key=jax.random.key(0),
    )

    assert len(X) == len(y) == len(priorities) == 4
    assert jnp.array_equal(y[:, 0], 10 * X[:, 0])


def test_dagger_reservoir_deduplicates_states():
    X, y, priorities = update_reservoir(
        jnp.asarray([[1.0]]),
        jnp.asarray([[10.0]]),
        jnp.asarray([0.5]),
        X_new=jnp.asarray([[1.0], [2.0]]),
        y_new=jnp.asarray([[10.0], [20.0]]),
        max_size=10,
        key=jax.random.key(0),
    )

    assert jnp.array_equal(X[:, 0], jnp.asarray([1.0, 2.0]))
    assert jnp.array_equal(y[:, 0], jnp.asarray([10.0, 20.0]))
    assert len(priorities) == 2


def test_recovery_states_are_partitioned_by_observation_threshold():
    X = jnp.asarray([[0.0, 0.05], [0.0, -0.2], [0.0, 0.1]])

    assert jnp.array_equal(
        recovery_state_mask(X, observation_index=1, threshold=0.1),
        jnp.asarray([False, True, True]),
    )


def test_dataset_parts_skip_empty_partitions():
    X, y = combine_dataset_parts(
        (jnp.asarray([[1.0]]), jnp.asarray([[10.0]])),
        (jnp.empty((0, 1)), jnp.empty((0, 1))),
        (jnp.asarray([[2.0]]), jnp.asarray([[20.0]])),
    )

    assert jnp.array_equal(X[:, 0], jnp.asarray([1.0, 2.0]))
    assert jnp.array_equal(y[:, 0], jnp.asarray([10.0, 20.0]))


def test_top_k_validation_can_select_second_best_loss(monkeypatch):
    repertoire = SimpleNamespace(
        fitnesses=jnp.asarray([[0.0], [3.0], [2.0], [1.0]]),
        genotypes={"score": jnp.asarray([0.0, 5.0, 10.0, 1.0])},
    )
    monkeypatch.setattr(
        spid,
        "evaluate_symbolic_policy",
        lambda genotype, *_args, **_kwargs: genotype["score"],
    )

    selected = select_validated_candidate(
        repertoire, 2, None, None, 1000, seed=0, n_seeds=1
    )

    assert selected["repertoire_index"] == 2
    assert selected["fitness_rank"] == 2
    assert selected["reward"] == 10.0


def test_blockwise_validation_retains_best_rollout_candidate(monkeypatch):
    fit_generation_counts = []

    def fake_fit(*_args, n_gens, bootstrap_repertoire, **_kwargs):
        fit_generation_counts.append(n_gens)
        block = len(fit_generation_counts)
        repertoire = SimpleNamespace(
            fitnesses=jnp.asarray([[float(block)]]),
            genotypes={"score": jnp.asarray([float(block)])},
        )
        return repertoire, float(block), {"block": block}

    rewards = iter([10.0, 30.0, 20.0])
    monkeypatch.setattr(spid, "fit_dataset", fake_fit)
    monkeypatch.setattr(
        spid,
        "select_validated_candidate",
        lambda repertoire, *_args, **_kwargs: {
            "genotype": {"score": repertoire.genotypes["score"][0]},
            "fitness_rank": 1,
            "reward": next(rewards),
            "candidate_indices": [0],
            "candidate_rewards": [0.0],
        },
    )

    result = fit_and_validate_blocks(
        jnp.ones((1, 1)), jnp.ones((1, 1)), None, None, None,
        bootstrap_repertoire=None, n_pop=1, n_gens=25,
        validation_interval=10, validation_top_k=1,
        validation_checkpoints=None,
        dataset_batch_size=1, alpha=0.1, epsilon=0.1,
        search_seed=0, evaluation_seed=0, evaluation_trajectories=1,
        rollout_steps=1,
    )

    assert fit_generation_counts == [10, 11, 6]
    assert result["validated"]["reward"] == 30.0
    assert result["best_generation"] == 20
    assert [record["generation"] for record in result["block_validations"]] == [
        10, 20, 25
    ]


def test_explicit_validation_checkpoints_and_reward_bootstrap(monkeypatch):
    bootstraps = []

    def fake_fit(*_args, n_gens, bootstrap_repertoire, **_kwargs):
        bootstraps.append(bootstrap_repertoire)
        block = len(bootstraps)
        repertoire = SimpleNamespace(
            fitnesses=jnp.asarray([[float(block)]]),
            genotypes={"score": jnp.asarray([float(block)])},
        )
        return repertoire, float(block), {"block": block}

    rewards = iter([30.0, 10.0, 20.0])
    monkeypatch.setattr(spid, "fit_dataset", fake_fit)
    monkeypatch.setattr(
        spid,
        "select_validated_candidate",
        lambda repertoire, *_args, **_kwargs: {
            "genotype": {"score": repertoire.genotypes["score"][0]},
            "fitness_rank": 1,
            "reward": next(rewards),
            "candidate_indices": [0],
            "candidate_rewards": [0.0],
        },
    )

    result = fit_and_validate_blocks(
        jnp.ones((1, 1)), jnp.ones((1, 1)), None, None, None,
        bootstrap_repertoire=None, n_pop=1, n_gens=50,
        validation_interval=10, validation_checkpoints=(10, 30, 50),
        validation_top_k=1, dataset_batch_size=1, alpha=0.1, epsilon=0.1,
        search_seed=0, evaluation_seed=0, evaluation_trajectories=1,
        rollout_steps=1,
    )

    assert [record["generation"] for record in result["block_validations"]] == [
        10, 30, 50
    ]
    assert bootstraps[1] is result["validated_repertoire"]
    assert bootstraps[2] is result["validated_repertoire"]
    assert result["validated"]["reward"] == 30.0


def test_iteration_artifacts_are_reloadable(tmp_path):
    genotype = {"value": jnp.asarray([1.0])}
    array = jnp.asarray([[1.0]])
    empty = jnp.empty((0, 1))
    save_iteration(
        tmp_path, 0, genotype,
        array, array, empty, empty, empty, empty,
        {"iteration": 0},
    )
    iteration_dir = tmp_path / "iterations" / "iteration_000"

    with (iteration_dir / "selected_genotype.pickle").open("rb") as file:
        restored = pickle.load(file)
    assert jnp.array_equal(restored["value"], genotype["value"])
    assert json.loads((iteration_dir / "metadata.json").read_text()) == {
        "iteration": 0
    }
    data = jnp.load(iteration_dir / "dataset.npz")
    assert jnp.array_equal(data["expert_X"], array)


def test_run_directory_is_isolated_and_not_overwritten(tmp_path):
    run_dir = create_run_directory(
        tmp_path, "inverted_pendulum", "smoke"
    )

    assert run_dir == tmp_path / "spid_inverted_pendulum" / "smoke"
    with pytest.raises(FileExistsError):
        create_run_directory(tmp_path, "inverted_pendulum", "smoke")


@pytest.mark.parametrize("value", ["0", "-1"])
def test_positive_int_rejects_non_positive_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(value)


def test_positive_int_list_parses_increasing_checkpoints():
    assert positive_int_list("10,30,50") == (10, 30, 50)


@pytest.mark.parametrize("value", ["", "0,10", "10,10", "20,10", "x"])
def test_positive_int_list_rejects_invalid_checkpoints(value):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int_list(value)
