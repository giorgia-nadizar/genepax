import argparse

import pytest

import jax
import jax.numpy as jnp

from distillation_experiments.scripts.spid import (
    create_run_directory,
    mixture_weight,
    positive_int,
    update_reservoir,
)


def test_fixed_expert_weight():
    assert mixture_weight(50, 100, 0.5, 0.5) == pytest.approx(0.5)


def test_expert_weight_linear_schedule_includes_endpoints():
    assert mixture_weight(0, 5, 0.8, 0.2) == pytest.approx(0.8)
    assert mixture_weight(2, 5, 0.8, 0.2) == pytest.approx(0.5)
    assert mixture_weight(4, 5, 0.8, 0.2) == pytest.approx(0.2)


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
