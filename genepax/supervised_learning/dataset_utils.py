from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy
from qdax.custom_types import RNGKey
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

from datasets.regression import dcgp


def downsample_dataset(
    X: jnp.ndarray,
    y: jnp.ndarray,
    random_key: RNGKey,
    ratio: float | None = None,
    size: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Randomly downsample a dataset to a given size or ratio.

    You may specify either:
    - `ratio`: a float in (0, 1], representing the fraction of data to keep, or
    - `size`: the exact number of samples to keep.

    If neither is given, the full dataset is returned unchanged.

    Parameters
    ----------
    X : jnp.ndarray
        Input feature matrix of shape (N, ...).
    y : jnp.ndarray
        Target values of shape (N, ...).
    random_key : RNGKey
        JAX PRNG key used for sampling.
    ratio : float, optional
        Fraction of the dataset to retain. Ignored if `size` is provided.
    size : int, optional
        Exact number of samples to retain.

    Returns
    -------
    (X_sub, y_sub) : tuple of jnp.ndarray
        Downsampled feature matrix and target array.
    """
    if size is None:
        size = int(X.shape[0] * ratio) if ratio is not None else X.shape[0]

    size = min(size, X.shape[0])  # safety

    # Randomly choose indices without replacement
    indices = jax.random.choice(random_key, X.shape[0], shape=(size,), replace=False)

    X_batch = jnp.take(X, indices, axis=0)
    y_batch = jnp.take(y, indices, axis=0)

    return X_batch, y_batch


def load_dataset(
    dataset_name: str,
    scale_x: bool = False,
    scale_y: bool = False,
    test_split: float = 0.25,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a dataset, split into train/test sets, and optionally scale features and targets.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset ('diabetes', Feynman TSV, or custom CSV).
    scale_x : bool, default=False
        If True, standardize input features.
    scale_y : bool
        If True, standardize target values, default=False.
    test_split : float, default=0.25
        Fraction of data for testing (ignored for pre-split CSVs).
    random_state : int, default=0
        Seed for reproducible train/test split.

    Returns
    -------
    X_train, X_test,  y_train, y_test : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Training and testing features, and training and testing targets, reshaped to (-1, 1).
    """
    uci_classification_datasets = {"breast_cancer": 17, "glass": 42}
    local_classification_datasets = ["diabetes_classification"]

    if dataset_name in uci_classification_datasets.keys():
        dataset = fetch_ucirepo(id=uci_classification_datasets[dataset_name])
        X = dataset.data.features.to_numpy()
        y_raw = dataset.data.targets.squeeze()
        encoder = OneHotEncoder(sparse_output=False)
        y = encoder.fit_transform(y_raw.to_frame())
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=random_state
        )
    elif dataset_name in local_classification_datasets:
        df = pd.read_csv(f"../datasets/classification/{dataset_name}.csv")
        X = df.iloc[:, :-1].to_numpy()
        y_raw = df.iloc[:, -1].to_numpy().reshape(-1, 1)
        encoder = OneHotEncoder(sparse_output=False)
        y = encoder.fit_transform(y_raw)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=random_state
        )
    elif "diabetes" in dataset_name:
        X, y = load_diabetes(return_X_y=True)
        y = y.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=random_state
        )
    elif "feynman" in dataset_name:
        df = pd.read_csv(f"../datasets/regression/{dataset_name}.tsv", sep="\t")
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        y = y.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=random_state
        )
    elif "mtr" in dataset_name:
        statistics = pd.read_csv("../datasets/mtr/statistics.csv")
        n_targets = statistics.loc[
            statistics["name"] == dataset_name.replace("mtr/", ""), "targets"
        ].iloc[0]
        raw_data = scipy.io.arff.loadarff(f"../datasets/{dataset_name}.arff")
        df = pd.DataFrame(raw_data[0])
        X = df.iloc[:, :-n_targets].to_numpy()
        y = df.iloc[:, -n_targets:].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=random_state
        )
    elif "dcgp" in dataset_name:
        X_train, X_test, y_train, y_test = getattr(dcgp, dataset_name)(
            seed=random_state
        )
    else:
        df_train = pd.read_csv(f"../datasets/regression/{dataset_name}_train.csv")
        df_test = pd.read_csv(f"../datasets/regression/{dataset_name}_test.csv")
        X_train = df_train.drop(columns=["target"], inplace=False).to_numpy()
        X_test = df_test.drop(columns=["target"], inplace=False).to_numpy()
        y_train = df_train["target"].to_numpy().reshape(-1, 1)
        y_test = df_test["target"].to_numpy().reshape(-1, 1)

    # Create scalers
    if scale_x:
        X_scaler = StandardScaler()
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)
    if scale_y:
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train)
        y_test = y_scaler.transform(y_test)

    return X_train, X_test, y_train, y_test
