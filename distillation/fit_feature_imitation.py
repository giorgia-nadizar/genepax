"""Evolve shared CGP features with independently fitted linear action readouts."""

from functools import partial
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.custom_types import Genotype, RNGKey

from distillation.rollouts import sanitize_action
from genepax.evolution.custom_emitters import CustomMixingEmitter
from genepax.evolution.evolution_metrics import custom_ga_metrics
from genepax.evolution.genetic_algorithm_extra_scores import GeneticAlgorithmWithExtraScores
from genepax.evolution.tournament_selector import TournamentSelector
from genepax.gp.cartesian_genetic_programming import CGP


def sanitize_features(features: jax.Array) -> jax.Array:
    """Use the same finite, bounded features during fitting and execution."""
    return jnp.clip(
        jnp.nan_to_num(features, nan=0.0, posinf=1e3, neginf=-1e3), -1e3, 1e3
    )


def make_feature_cgp(n_inputs: int, n_actions: int, k: int = 8, n_nodes: int = 50) -> CGP:
    if min(n_inputs, n_actions, k, n_nodes) < 1:
        raise ValueError("Input, action, feature and node counts must be positive")
    return CGP(
        n_inputs=n_inputs, n_outputs=k, n_nodes=n_nodes,
        outputs_wrapper=sanitize_features,
        n_custom_weights=(k + 1) * n_actions,
        weights_mutation=False,
    )


def fit_readout(
    features: jax.Array, actions: jax.Array, sample_weights: jax.Array | None = None
) -> jax.Array:
    """Return (k + 1, n_actions) least-squares weights; final row is intercept.

    Standardize columns for numerical conditioning, solve with SVD (including
    redundant/constant features), then fold standardization into the weights.
    No validation samples or environment rewards enter this fit.
    Optional state weights apply to feature standardization and the solve.
    All actions share the same k features. Solving multiple right-hand sides
    fits each action column independently, including its own intercept.
    """
    mean = (jnp.mean(features, axis=0) if sample_weights is None else
            jnp.sum(features * sample_weights[:, None], axis=0) / jnp.sum(sample_weights))
    centered = features - mean
    variance = (jnp.mean(jnp.square(centered), axis=0) if sample_weights is None else
                jnp.sum(jnp.square(centered) * sample_weights[:, None], axis=0) / jnp.sum(sample_weights))
    scale = jnp.sqrt(variance)
    scale = jnp.where(scale > 0, scale, 1.0)
    design = jnp.concatenate(
        [centered / scale, jnp.ones((len(features), 1))], axis=1
    )
    if sample_weights is not None:
        root_weights = jnp.sqrt(sample_weights)[:, None]
        design, actions = design * root_weights, actions * root_weights
    solution = jnp.linalg.lstsq(design, actions, rcond=None)[0]
    coefficients = solution[:-1] / scale[:, None]
    intercept = solution[-1] - mean @ coefficients
    return jnp.concatenate([coefficients, intercept[None]], axis=0)


def feature_policy_action(
    genotype: Genotype, cgp_structure: CGP, observation: jax.Array
) -> jax.Array:
    """Execute only the graph and its stored readout; never refit at rollout."""
    features = cgp_structure.apply(genotype, observation)
    readout = genotype["weights"]["custom_weights"].reshape(
        cgp_structure.n_outputs + 1, -1
    )
    return sanitize_action(features @ readout[:-1] + readout[-1])


def action_mse(
    predictions: jax.Array, targets: jax.Array, sample_weights: jax.Array | None = None
) -> jax.Array:
    per_sample = jnp.mean(jnp.square(predictions - targets), axis=-1)
    if sample_weights is None:
        return jnp.mean(per_sample)
    return jnp.sum(per_sample * sample_weights) / jnp.sum(sample_weights)


def feature_policy_mse(
    genotype: Genotype, cgp_structure: CGP, X: jax.Array, y: jax.Array,
    sample_weights: jax.Array | None = None,
) -> jax.Array:
    predictions = jax.vmap(feature_policy_action, in_axes=(None, None, 0))(
        genotype, cgp_structure, X
    )
    return action_mse(predictions, y, sample_weights)


def _score_individual(
    genotype: Genotype, *, X: jax.Array, y: jax.Array, cgp_structure: CGP,
    sample_weights: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, Any]]:
    features = jax.vmap(cgp_structure.apply, in_axes=(None, 0))(genotype, X)
    readout = fit_readout(features, y, sample_weights)
    updated = cgp_structure.update_weights(
        genotype, {"custom_weights": readout.ravel()}
    )
    predictions = sanitize_action(features @ readout[:-1] + readout[-1])
    loss = action_mse(predictions, y, sample_weights)
    valid = jnp.all(jnp.isfinite(readout)) & jnp.isfinite(loss)
    fitness = jnp.where(valid, -loss, -jnp.inf)
    return jnp.asarray([fitness]), {
        "updated_params": updated, "test_accuracy": fitness,
    }


def fit_feature_imitation(
    X: jax.Array | np.ndarray,
    y: jax.Array | np.ndarray,
    cgp_structure: CGP,
    *,
    seed: int = 0,
    n_gens: int = 100,
    n_pop: int = 100,
    sample_weights: jax.Array | np.ndarray | None = None,
) -> Dict[str, Any]:
    """Fit each candidate's readout on training data and evolve by action MSE."""
    X, y = jnp.asarray(X, dtype=jnp.float32), jnp.asarray(y, dtype=jnp.float32)
    if X.ndim != 2 or y.ndim != 2 or len(X) != len(y) or len(X) < 2:
        raise ValueError("X and y must be aligned 2D arrays with at least two rows")
    if not np.isfinite(np.asarray(X)).all() or not np.isfinite(np.asarray(y)).all():
        raise ValueError("Training data must be finite")
    if X.shape[1] != cgp_structure.n_inputs or cgp_structure.n_custom_weights != (
        cgp_structure.n_outputs + 1
    ) * y.shape[1]:
        raise ValueError("CGP dimensions do not match the training data/readout")
    if n_gens < 1 or n_pop < 1:
        raise ValueError("Generation and population counts must be positive")
    if sample_weights is not None:
        sample_weights = jnp.asarray(sample_weights, dtype=jnp.float32)
        weights = np.asarray(sample_weights)
        if weights.shape != (len(X),) or not np.isfinite(weights).all():
            raise ValueError("Weights must be finite with one value per training row")
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Weights must be non-negative with positive total")

    single_score = partial(_score_individual, X=X, y=y, cgp_structure=cgp_structure,
                           sample_weights=sample_weights)

    @jax.jit
    def score_population(
        genotypes: Genotype, key: RNGKey
    ) -> tuple[jax.Array, dict[str, Any]]:
        del key
        # Map individuals sequentially to avoid materializing population x data
        # x graph intermediates or running all SVD workspaces simultaneously.
        return jax.lax.map(single_score, genotypes)

    emitter = CustomMixingEmitter(
        mutation_fn=jax.jit(jax.vmap(cgp_structure.mutate, in_axes=(0, 0))),
        variation_fn=None, variation_percentage=0, batch_size=n_pop,
        selector=TournamentSelector(tournament_size=min(3, n_pop)),
    )
    ga = GeneticAlgorithmWithExtraScores(
        scoring_function=score_population, emitter=emitter,
        metrics_function=partial(
            custom_ga_metrics, extra_scores_metrics={"test_accuracy": jnp.ravel}
        ),
        lamarckian=True,
    )
    key, init_key, ga_key = jax.random.split(jax.random.key(seed), 3)
    initial = jax.jit(jax.vmap(cgp_structure.init))(jax.random.split(init_key, n_pop))
    repertoire, state, _ = ga.init(initial, n_pop, ga_key)
    history = []
    for generation in range(n_gens):
        if generation:
            key, update_key = jax.random.split(key)
            repertoire, state, _ = ga.update(repertoire, state, update_key)
        losses = -np.asarray(repertoire.fitnesses[:, 0])
        history.append({
            "generation": generation, "best_loss": float(np.min(losses)),
            "median_loss": float(np.median(losses)),
        })
        print(f"generation={generation} best_mse={np.min(losses):.6g}", flush=True)
    if not np.isfinite(losses).all():
        raise FloatingPointError("Non-finite feature-policy fitness in final population")
    best_index = int(np.argmin(losses))
    return {
        "genotype": jax.tree.map(lambda x: x[best_index], repertoire.genotypes),
        "repertoire": GARepertoire.init(
            genotypes=repertoire.genotypes, fitnesses=repertoire.fitnesses,
            population_size=n_pop,
        ),
        "history": history, "loss": float(losses[best_index]),
    }
