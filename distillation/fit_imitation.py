"""CGP fitting utilities for action-imitation baselines."""

from __future__ import annotations

import functools
from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.custom_types import Genotype, RNGKey

from distillation.rollouts import sanitize_action
from genepax.evolution.custom_emitters import CustomMixingEmitter
from genepax.evolution.evolution_metrics import custom_ga_metrics
from genepax.evolution.genetic_algorithm_extra_scores import (
    GeneticAlgorithmWithExtraScores,
)
from genepax.evolution.tournament_selector import TournamentSelector
from genepax.gp.cartesian_genetic_programming import CGP


def imitation_loss(
    expert_actions: jax.Array,
    student_actions: jax.Array,
    valid_mask: jax.Array | None = None,
) -> jax.Array:
    """Return mean per-action squared error, optionally ignoring padding."""
    per_sample = jnp.mean(jnp.square(student_actions - expert_actions), axis=-1)
    if valid_mask is None:
        return jnp.mean(per_sample)
    valid_mask = valid_mask.astype(per_sample.dtype)
    return jnp.sum(jnp.where(valid_mask > 0, per_sample, 0.0)) / jnp.maximum(
        jnp.sum(valid_mask), 1.0
    )


def apply_linear_scaling(
    actions: jax.Array, custom_weights: jax.Array
) -> jax.Array:
    """Apply per-output slopes and intercepts stored in custom weights."""
    n_outputs = actions.shape[-1]
    if custom_weights.shape[-1] != 2 * n_outputs:
        raise ValueError("Linear scaling requires two custom weights per output")
    slopes = custom_weights[..., :n_outputs]
    intercepts = custom_weights[..., n_outputs:]
    return slopes * actions + intercepts


def scaled_cgp_action(
    genotype: Genotype, cgp_structure: CGP, observation: jax.Array
) -> jax.Array:
    """Evaluate CGP and apply its stored affine output calibration."""
    raw_action = cgp_structure.apply(genotype, observation)
    scaled_action = apply_linear_scaling(
        raw_action, genotype["weights"]["custom_weights"]
    )
    return sanitize_action(scaled_action)


def _score_genome(
    genotype: Genotype,
    X: jax.Array,
    y: jax.Array,
    cgp_structure: CGP,
    valid_mask: jax.Array | None,
) -> Tuple[jax.Array, Dict[str, Any]]:
    predictions = jax.vmap(cgp_structure.apply, in_axes=(None, 0))(genotype, X)
    predictions = sanitize_action(predictions)
    loss = imitation_loss(y, predictions, valid_mask)
    fitness = jnp.nan_to_num(-loss, nan=-jnp.inf, posinf=-jnp.inf)
    return jnp.asarray([fitness]), {
        "test_accuracy": fitness,
        "updated_params": genotype,
        "imitation_loss": loss,
    }


def _score_population(
    genotypes: Genotype,
    key: RNGKey,
    X: jax.Array,
    y: jax.Array,
    cgp_structure: CGP,
    dataset_batch_size: int,
    sample_weights: jax.Array,
) -> tuple[jax.Array, dict[str, Any]]:
    """Score a population in fixed-size chunks to limit device memory."""
    n_samples = X.shape[0]
    n_chunks = (n_samples + dataset_batch_size - 1) // dataset_batch_size
    padded_size = n_chunks * dataset_batch_size
    padding = padded_size - n_samples
    X = jnp.pad(X, ((0, padding), (0, 0))).reshape(
        n_chunks, dataset_batch_size, X.shape[-1]
    )
    y = jnp.pad(y, ((0, padding), (0, 0))).reshape(
        n_chunks, dataset_batch_size, y.shape[-1]
    )
    valid = jnp.pad(sample_weights, (0, padding)).reshape(
        n_chunks, dataset_batch_size
    )

    def score_chunk(
        total: jax.Array, chunk: Tuple[jax.Array, jax.Array, jax.Array]
    ) -> Tuple[jax.Array, None]:
        X_chunk, y_chunk, valid_chunk = chunk
        score_fn = partial(
            _score_genome,
            X=X_chunk,
            y=y_chunk,
            cgp_structure=cgp_structure,
            valid_mask=valid_chunk,
        )
        fitness, _ = jax.vmap(score_fn)(genotypes)
        return total + fitness * jnp.sum(valid_chunk), None

    population_size = genotypes["genes"]["inputs1"].shape[0]
    fitness, _ = jax.lax.scan(
        score_chunk,
        jnp.zeros((population_size, 1)),
        (X, y, valid),
    )
    normalizer = jnp.sum(sample_weights)
    fitness = fitness / normalizer
    return fitness, {
        "test_accuracy": fitness[:, 0],
        "updated_params": genotypes,
        "imitation_loss": -fitness[:, 0],
    }


def _score_scaled_population(
    genotypes: Genotype,
    key: RNGKey,
    X: jax.Array,
    y: jax.Array,
    cgp_structure: CGP,
    dataset_batch_size: int,
    sample_weights: jax.Array,
) -> Tuple[jax.Array, Dict[str, Any]]:
    """Fit affine output scaling globally, then score scaled clipped actions."""
    del key
    n_samples = X.shape[0]
    n_outputs = y.shape[-1]
    n_chunks = (n_samples + dataset_batch_size - 1) // dataset_batch_size
    padded_size = n_chunks * dataset_batch_size
    padding = padded_size - n_samples
    X = jnp.pad(X, ((0, padding), (0, 0))).reshape(
        n_chunks, dataset_batch_size, X.shape[-1]
    )
    y = jnp.pad(y, ((0, padding), (0, 0))).reshape(
        n_chunks, dataset_batch_size, n_outputs
    )
    valid = jnp.pad(sample_weights, (0, padding)).reshape(
        n_chunks, dataset_batch_size
    )
    population_size = genotypes["genes"]["inputs1"].shape[0]

    def predict_population(X_chunk: jax.Array) -> jax.Array:
        def predict_genome(genotype: Genotype) -> jax.Array:
            predictions = jax.vmap(
                cgp_structure.apply, in_axes=(None, 0)
            )(genotype, X_chunk)
            return jnp.nan_to_num(
                predictions, nan=0.0, posinf=1.0, neginf=-1.0
            )

        return jax.vmap(predict_genome)(genotypes)

    def collect_statistics(
        carry: Tuple[jax.Array, jax.Array, jax.Array],
        chunk: Tuple[jax.Array, jax.Array, jax.Array],
    ) -> Tuple[Tuple[jax.Array, jax.Array, jax.Array], None]:
        X_chunk, y_chunk, valid_chunk = chunk
        predictions = predict_population(X_chunk)
        mask = valid_chunk[None, :, None]
        sum_prediction, sum_prediction_squared, sum_cross = carry
        return (
            sum_prediction + jnp.sum(mask * predictions, axis=1),
            sum_prediction_squared
            + jnp.sum(mask * jnp.square(predictions), axis=1),
            sum_cross
            + jnp.sum(mask * predictions * y_chunk[None, :, :], axis=1),
        ), None

    statistic_shape = (population_size, n_outputs)
    (sum_prediction, sum_prediction_squared, sum_cross), _ = jax.lax.scan(
        collect_statistics,
        (
            jnp.zeros(statistic_shape),
            jnp.zeros(statistic_shape),
            jnp.zeros(statistic_shape),
        ),
        (X, y, valid),
    )
    weight_total = jnp.sum(sample_weights)
    sum_target = jnp.sum(valid[:, :, None] * y, axis=(0, 1))
    mean_prediction = sum_prediction / weight_total
    mean_target = sum_target / weight_total
    covariance = sum_cross - sum_prediction * mean_target
    variance = sum_prediction_squared - sum_prediction * mean_prediction
    slopes = jnp.where(variance > 1e-12, covariance / variance, 0.0)
    intercepts = mean_target - slopes * mean_prediction
    custom_weights = jnp.concatenate([slopes, intercepts], axis=-1)
    updated_genotypes = jax.vmap(
        lambda genotype, weights: cgp_structure.update_weights(
            genotype, {"custom_weights": weights}
        )
    )(genotypes, custom_weights)

    def score_scaled(
        carry: Tuple[jax.Array, jax.Array],
        chunk: Tuple[jax.Array, jax.Array, jax.Array],
    ) -> Tuple[Tuple[jax.Array, jax.Array], None]:
        X_chunk, y_chunk, valid_chunk = chunk
        raw_predictions = predict_population(X_chunk)
        scaled_predictions = slopes[:, None, :] * raw_predictions + intercepts[:, None, :]
        scaled_predictions = sanitize_action(scaled_predictions)
        mask = valid_chunk[None, :, None]
        raw_squared_error = jnp.where(
            mask > 0,
            mask * jnp.square(raw_predictions - y_chunk[None, :, :]),
            0.0,
        )
        scaled_squared_error = jnp.where(
            mask > 0,
            mask * jnp.square(scaled_predictions - y_chunk[None, :, :]),
            0.0,
        )
        raw_total, scaled_total = carry
        return (
            raw_total + jnp.sum(raw_squared_error, axis=(1, 2)),
            scaled_total + jnp.sum(scaled_squared_error, axis=(1, 2)),
        ), None

    (raw_total, scaled_total), _ = jax.lax.scan(
        score_scaled,
        (jnp.zeros(population_size), jnp.zeros(population_size)),
        (X, y, valid),
    )
    normalizer = weight_total * n_outputs
    raw_loss = raw_total / normalizer
    scaled_loss = scaled_total / normalizer
    fitness = -scaled_loss[:, None]
    return fitness, {
        "test_accuracy": fitness[:, 0],
        "updated_params": updated_genotypes,
        "imitation_loss": scaled_loss,
        "raw_imitation_loss": raw_loss,
    }


def _score_adam_population(
    genotypes: Genotype,
    key: RNGKey,
    X: jax.Array,
    y: jax.Array,
    cgp_structure: CGP,
    dataset_batch_size: int,
    sample_weights: jax.Array,
    linear_scaling: bool,
    adam_steps: int,
    adam_learning_rate: float,
    adam_batch_size: int,
) -> Tuple[jax.Array, Dict[str, Any]]:
    """Optimize every individual's connection weights before scoring it."""
    if linear_scaling:
        _, scaling_scores = _score_scaled_population(
            genotypes, key, X, y, cgp_structure,
            dataset_batch_size, sample_weights,
        )
        genotypes = scaling_scores["updated_params"]
    graph_weights = cgp_structure.get_weights(genotypes)
    if not graph_weights:
        raise ValueError("Per-generation Adam requires trainable CGP weights")
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(adam_learning_rate)
    )
    optimizer_states = jax.vmap(optimizer.init)(graph_weights)

    def single_step(
        weights: Dict[str, jax.Array],
        genotype: Genotype,
        optimizer_state: optax.OptState,
        X_batch: jax.Array,
        y_batch: jax.Array,
        w_batch: jax.Array,
    ) -> Tuple[Dict[str, jax.Array], optax.OptState, jax.Array]:
        def loss_fn(current_weights: Dict[str, jax.Array]) -> jax.Array:
            predictions = jax.vmap(
                cgp_structure.apply, in_axes=(None, 0, None)
            )(genotype, X_batch, current_weights)
            if linear_scaling:
                predictions = apply_linear_scaling(
                    predictions, genotype["weights"]["custom_weights"]
                )
            predictions = sanitize_action(predictions)
            per_sample = jnp.mean(
                jnp.square(predictions - y_batch), axis=-1
            )
            return jnp.sum(per_sample * w_batch) / jnp.maximum(
                jnp.sum(w_batch), 1e-12
            )

        loss, gradients = jax.value_and_grad(loss_fn)(weights)
        gradients = jax.tree.map(
            lambda gradient: jnp.where(
                jnp.isfinite(gradient), gradient, 0.0
            ),
            gradients,
        )
        updates, optimizer_state = optimizer.update(
            gradients, optimizer_state
        )
        weights = optax.apply_updates(weights, updates)
        weights = jax.tree.map(
            lambda value: jnp.clip(value, -1e4, 1e4), weights
        )
        return weights, optimizer_state, loss

    batched_step = jax.vmap(
        single_step, in_axes=(0, 0, 0, None, None, None)
    )
    batch_size = min(adam_batch_size, len(X))
    for _ in range(adam_steps):
        key, batch_key = jax.random.split(key)
        indices = jax.random.randint(batch_key, (batch_size,), 0, len(X))
        graph_weights, optimizer_states, _ = batched_step(
            graph_weights,
            genotypes,
            optimizer_states,
            X[indices],
            y[indices],
            sample_weights[indices],
        )
    updated_genotypes = jax.vmap(
        cgp_structure.update_weights, in_axes=(0, 0)
    )(genotypes, graph_weights)
    scoring_fn = (
        _score_scaled_population if linear_scaling else _score_population
    )
    fitness, extra_scores = scoring_fn(
        updated_genotypes,
        key,
        X,
        y,
        cgp_structure,
        dataset_batch_size,
        sample_weights,
    )
    return fitness, extra_scores | {
        "updated_params": extra_scores.get(
            "updated_params", updated_genotypes
        )
    }


def fit_imitation_dataset(
    X: jax.Array | np.ndarray,
    y: jax.Array | np.ndarray,
    cgp_structure: CGP,
    *,
    seed: int = 0,
    n_gens: int = 100,
    n_pop: int = 100,
    dataset_batch_size: int = 4096,
    bootstrap_repertoire: GARepertoire | None = None,
    linear_scaling: bool = False,
    sample_weights: jax.Array | np.ndarray | None = None,
    generation_adam_steps: int = 0,
    generation_adam_learning_rate: float = 1e-3,
    generation_adam_batch_size: int = 256,
) -> dict[str, Any]:
    """Fit CGP solely by expert-action MSE and return its final population."""
    if len(X) == 0:
        raise ValueError("Imitation fitting requires a non-empty dataset")
    finite_rows = jnp.all(jnp.isfinite(X), axis=-1) & jnp.all(
        jnp.isfinite(y), axis=-1
    )
    invalid_rows = int(jnp.sum(~finite_rows))
    if invalid_rows:
        raise ValueError(
            f"Imitation dataset contains {invalid_rows} non-finite samples"
        )
    if sample_weights is None:
        sample_weights = jnp.ones(len(X), dtype=jnp.float32)
    sample_weights = jnp.asarray(sample_weights, dtype=jnp.float32)
    if sample_weights.shape != (len(X),):
        raise ValueError("Sample weights must have one value per dataset row")
    if not bool(jnp.all(jnp.isfinite(sample_weights))):
        raise ValueError("Sample weights must be finite")
    if bool(jnp.any(sample_weights < 0)) or float(jnp.sum(sample_weights)) <= 0:
        raise ValueError("Sample weights must be non-negative with positive sum")
    if bootstrap_repertoire is not None and (
        len(bootstrap_repertoire.fitnesses) != n_pop
    ):
        raise ValueError("Bootstrap repertoire size must match n_pop")
    expected_custom_weights = 2 * y.shape[-1] if linear_scaling else 0
    if cgp_structure.n_custom_weights != expected_custom_weights:
        raise ValueError(
            "CGP custom-weight count does not match the scaling configuration"
        )

    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    if bootstrap_repertoire is None:
        init_keys = jax.random.split(init_key, n_pop)
        initial_population = jax.jit(jax.vmap(cgp_structure.init))(init_keys)
    else:
        initial_population = bootstrap_repertoire.genotypes

    metrics_function = functools.partial(
        custom_ga_metrics,
        extra_scores_metrics={"test_accuracy": jnp.ravel},
    )
    mutation_fn = jax.jit(jax.vmap(cgp_structure.mutate, in_axes=(0, 0)))
    emitter = CustomMixingEmitter(
        mutation_fn=mutation_fn,
        variation_fn=None,
        variation_percentage=0,
        batch_size=n_pop,
        selector=TournamentSelector(),
    )
    if generation_adam_steps < 0:
        raise ValueError("generation_adam_steps cannot be negative")
    if generation_adam_steps:
        scoring_fn = partial(
            _score_adam_population,
            X=X,
            y=y,
            cgp_structure=cgp_structure,
            dataset_batch_size=dataset_batch_size,
            sample_weights=sample_weights,
            linear_scaling=linear_scaling,
            adam_steps=generation_adam_steps,
            adam_learning_rate=generation_adam_learning_rate,
            adam_batch_size=generation_adam_batch_size,
        )
    else:
        scoring_fn = partial(
            _score_scaled_population if linear_scaling else _score_population,
            X=X,
            y=y,
            cgp_structure=cgp_structure,
            dataset_batch_size=dataset_batch_size,
            sample_weights=sample_weights,
        )
    ga = GeneticAlgorithmWithExtraScores(
        scoring_function=scoring_fn,
        emitter=emitter,
        metrics_function=metrics_function,
        lamarckian=linear_scaling or generation_adam_steps > 0,
    )
    key, ga_key = jax.random.split(key)
    repertoire, emitter_state, metrics = ga.init(
        genotypes=initial_population,
        population_size=n_pop,
        key=ga_key,
    )
    history = [{
        "generation": 0,
        "best_loss": float(-jnp.max(repertoire.fitnesses[:, 0])),
        "median_loss": float(-jnp.median(repertoire.fitnesses[:, 0])),
        "finite_fitness_fraction": float(
            jnp.mean(jnp.isfinite(repertoire.fitnesses[:, 0]))
        ),
    }]
    for generation in range(1, n_gens):
        key, update_key = jax.random.split(key)
        repertoire, emitter_state, metrics = ga.update(
            repertoire=repertoire,
            emitter_state=emitter_state,
            key=update_key,
        )
        jax.block_until_ready(metrics["max_fitness"])
        history.append({
            "generation": generation,
            "best_loss": float(-jnp.max(repertoire.fitnesses[:, 0])),
            "median_loss": float(-jnp.median(repertoire.fitnesses[:, 0])),
            "finite_fitness_fraction": float(
                jnp.mean(jnp.isfinite(repertoire.fitnesses[:, 0]))
            ),
        })

    stored_repertoire = GARepertoire.init(
        genotypes=repertoire.genotypes,
        fitnesses=repertoire.fitnesses,
        population_size=len(repertoire.fitnesses),
    )
    finite_fitness = jnp.isfinite(repertoire.fitnesses)
    if not bool(jnp.all(finite_fitness)):
        raise FloatingPointError(
            "Non-finite imitation fitness detected; "
            f"finite fraction={float(jnp.mean(finite_fitness)):.3f}"
        )
    best_index = int(jnp.argmax(repertoire.fitnesses[:, 0]))
    best_genotype = jax.tree.map(
        lambda value: value[best_index], repertoire.genotypes
    )
    result = {
        "repertoire": stored_repertoire,
        "genotype": best_genotype,
        "loss": float(-repertoire.fitnesses[best_index, 0]),
        "finite_fitness_fraction": float(jnp.mean(finite_fitness)),
        "history": history,
    }
    if linear_scaling:
        result["raw_loss"] = float(
            repertoire.extra_scores["raw_imitation_loss"][best_index]
        )
        result["scaling_weights"] = np.asarray(
            best_genotype["weights"]["custom_weights"]
        ).tolist()
    return result
