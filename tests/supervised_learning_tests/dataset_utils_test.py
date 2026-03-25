import jax
import jax.numpy as jnp
import pytest

from genepax.supervised_learning.dataset_utils import downsample_dataset


@pytest.fixture
def sample_data():
    X = jnp.arange(20).reshape(10, 2)  # 10 samples, 2 features
    y = jnp.arange(10)  # 10 labels
    return X, y


def test_downsample_dataset_full_dataset_returned(sample_data):
    X, y = sample_data
    key = jax.random.PRNGKey(0)
    X_sub, y_sub = downsample_dataset(X, y, key)
    assert X_sub.shape == X.shape
    assert y_sub.shape == y.shape


def test_downsample_dataset_by_ratio(sample_data):
    X, y = sample_data
    key = jax.random.PRNGKey(1)
    X_sub, y_sub = downsample_dataset(X, y, key, ratio=0.5)
    expected_size = int(X.shape[0] * 0.5)
    assert X_sub.shape[0] == expected_size
    assert y_sub.shape[0] == expected_size


def test_downsample_dataset_by_size(sample_data):
    X, y = sample_data
    key = jax.random.PRNGKey(2)
    X_sub, y_sub = downsample_dataset(X, y, key, size=3)
    assert X_sub.shape[0] == 3
    assert y_sub.shape[0] == 3


def test_downsample_dataset_size_larger_than_dataset(sample_data):
    X, y = sample_data
    key = jax.random.PRNGKey(3)
    X_sub, y_sub = downsample_dataset(X, y, key, size=20)
    assert X_sub.shape[0] == X.shape[0]
    assert y_sub.shape[0] == y.shape[0]


def test_downsample_dataset_deterministic_sampling(sample_data):
    X, y = sample_data
    key = jax.random.PRNGKey(4)
    X_sub1, y_sub1 = downsample_dataset(X, y, key, size=5)
    X_sub2, y_sub2 = downsample_dataset(X, y, key, size=5)
    assert jnp.array_equal(X_sub1, X_sub2)
    assert jnp.array_equal(y_sub1, y_sub2)


def test_downsample_dataset_alignment_of_X_and_y():
    X = jnp.arange(12).reshape(6, 2)
    y = jnp.arange(6)
    key = jax.random.PRNGKey(5)
    X_sub, y_sub = downsample_dataset(X, y, key, size=4)
    # Check that y_sub matches the corresponding rows in X_sub
    for xi, yi in zip(X_sub, y_sub):
        # yi should exist as row index in original X
        row_idx = int(yi)
        assert jnp.array_equal(xi, X[row_idx])


def test_downsample_dataset_ratio_one_returns_full_dataset(sample_data):
    X, y = sample_data
    key = jax.random.PRNGKey(6)
    X_sub, y_sub = downsample_dataset(X, y, key, ratio=1.0)
    assert X_sub.shape == X.shape
    assert y_sub.shape == y.shape


def test_downsample_dataset_empty_dataset():
    X = jnp.empty((0, 2))
    y = jnp.empty((0,))
    key = jax.random.PRNGKey(7)
    X_sub, y_sub = downsample_dataset(X, y, key, ratio=0.5)
    assert X_sub.shape[0] == 0
    assert y_sub.shape[0] == 0


def test_downsample_dataset_size_zero(sample_data):
    X, y = sample_data
    key = jax.random.PRNGKey(8)
    X_sub, y_sub = downsample_dataset(X, y, key, size=0)
    assert X_sub.shape[0] == 0
    assert y_sub.shape[0] == 0
