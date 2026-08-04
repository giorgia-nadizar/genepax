import functools
import time
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.custom_types import Genotype, RNGKey

from genepax.evolution.custom_emitters import CustomMixingEmitter
from genepax.evolution.evolution_metrics import custom_ga_metrics
from genepax.evolution.genetic_algorithm_extra_scores import (
    GeneticAlgorithmWithExtraScores,
)
from genepax.evolution.tournament_selector import TournamentSelector
from genepax.gp.cartesian_genetic_programming import CGP
from distillation.rollouts import sanitize_action


def gm_dagger_loss(
        expert_actions: jnp.ndarray,
        symbolic_actions: jnp.ndarray,
        expert_q: jnp.ndarray,
        symbolic_q: jnp.ndarray,
        alpha: float,
        epsilon: float,
        valid_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Computes the paper's per-state geometric-mean DAGGER loss.

    ``expert_q`` is used as the deterministic-teacher approximation of
    :math:`V^*(s)`. The paper assumes an optimal teacher, hence a non-negative
    Q gap. ``relu`` enforces that assumption when an imperfect learned critic
    happens to rank the symbolic action above the teacher action.
    """
    if alpha <= 0 or epsilon <= 0:
        raise ValueError("GM-DAGGER alpha and epsilon must be positive")

    q_gap = jnp.nan_to_num(
        expert_q - symbolic_q,
        nan=0.0,
        posinf=1e6,
        neginf=0.0,
    )
    performance_gap = jax.nn.relu(q_gap) + alpha
    fidelity_gap = (
        jnp.linalg.norm(symbolic_actions - expert_actions, axis=-1) + epsilon
    )
    per_sample_loss = jnp.sqrt(performance_gap * fidelity_gap)

    if valid_mask is None:
        valid_mask = jnp.ones_like(per_sample_loss)
    valid_mask = valid_mask.astype(per_sample_loss.dtype)
    normalizer = jnp.maximum(jnp.sum(valid_mask), 1.0)

    def masked_mean(values):
        # Multiplication is not a safe mask: NaN * 0 is still NaN.
        return jnp.sum(jnp.where(valid_mask > 0, values, 0.0)) / normalizer

    return masked_mean(per_sample_loss), {
        "performance_gap": masked_mean(performance_gap),
        "fidelity_gap": masked_mean(fidelity_gap),
    }


def single_genome_scoring_fn(
        genotype: Genotype,
        X: jnp.ndarray,
        y: jnp.ndarray,
        cgp_structure: CGP,
        q_value_estimator,
        alpha: float,
        epsilon: float,
        valid_mask: jnp.ndarray | None = None,
) -> Tuple:
    # Construct features
    y_pred = jax.vmap(cgp_structure.apply, in_axes=(None, 0))(genotype, X)

    # Sanitization
    y_pred = sanitize_action(y_pred)
    # The teacher action's Q-value approximates V*(s) for the deterministic
    # SAC policy used by this implementation.
    expert_q = q_value_estimator(
        X,
        y,
    )
    symbolic_q = q_value_estimator(
        X,
        y_pred,
    )
    loss, loss_components = gm_dagger_loss(
        expert_actions=y,
        symbolic_actions=y_pred,
        expert_q=expert_q,
        symbolic_q=symbolic_q,
        alpha=alpha,
        epsilon=epsilon,
        valid_mask=valid_mask,
    )

    # GA maximizes fitness
    fitness = -loss
    fitness = jnp.nan_to_num(fitness, nan=-jnp.inf)

    return jnp.asarray([fitness]), {
        "test_accuracy": fitness,
        "updated_params": genotype,
        **loss_components,
    }


def feature_construction_scoring_fn(
        genotypes: Genotype,
        key: RNGKey,
        X: jnp.ndarray,
        y: jnp.ndarray,
        cgp_structure: CGP,
        q_value_estimator,
        dataset_batch_size: int,
        alpha: float,
        epsilon: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Scores a population in chunks to bound peak device memory."""
    del key
    n_samples = X.shape[0]
    n_chunks = (n_samples + dataset_batch_size - 1) // dataset_batch_size
    padded_size = n_chunks * dataset_batch_size
    pad_count = padded_size - n_samples
    X = jnp.pad(X, ((0, pad_count), (0, 0)))
    y = jnp.pad(y, ((0, pad_count), (0, 0)))
    valid = jnp.concatenate(
        [jnp.ones((n_samples,)), jnp.zeros((pad_count,))]
    )
    X = X.reshape(n_chunks, dataset_batch_size, X.shape[-1])
    y = y.reshape(n_chunks, dataset_batch_size, y.shape[-1])
    valid = valid.reshape(n_chunks, dataset_batch_size)

    def score_chunk(carry, chunk):
        X_chunk, y_chunk, valid_chunk = chunk
        scoring_fn = partial(
            single_genome_scoring_fn,
            X=X_chunk,
            y=y_chunk,
            cgp_structure=cgp_structure,
            q_value_estimator=q_value_estimator,
            alpha=alpha,
            epsilon=epsilon,
            valid_mask=valid_chunk,
        )
        fitness, extra_scores = jax.vmap(scoring_fn)(genotypes)
        weight = jnp.sum(valid_chunk)
        fitness_total, performance_total, fidelity_total = carry
        updated_carry = (
            fitness_total + fitness * weight,
            performance_total + extra_scores["performance_gap"] * weight,
            fidelity_total + extra_scores["fidelity_gap"] * weight,
        )
        return updated_carry, None

    population_size = genotypes["genes"]["inputs1"].shape[0]
    initial_carry = (
        jnp.zeros((population_size, 1)),
        jnp.zeros((population_size,)),
        jnp.zeros((population_size,)),
    )
    (fitness, performance_gap, fidelity_gap), _ = jax.lax.scan(
        score_chunk,
        initial_carry,
        (X, y, valid),
    )
    # scan returns the final carry and per-chunk extra scores.  Recompute the
    # scalar extra score from the population fitness for GA logging.
    fitness = fitness / n_samples
    return fitness, {
        "test_accuracy": fitness[:, 0],
        "updated_params": genotypes,
        "performance_gap": performance_gap / n_samples,
        "fidelity_gap": fidelity_gap / n_samples,
    }


# def process_metrics_mtr(metrics: Dict, headers: List) -> Dict:
#     test_accuracy_values = metrics.pop("test_accuracy")
#     for idx, header in enumerate(headers):
#         metrics[header] = test_accuracy_values[idx]
#     return metrics


def fit_dataset(X, y, cgp_structure, seed=0, n_gens=100,
                q_value_estimator=None, bootstrap_repertoire=None, n_pop=100,
                dataset_batch_size=4096, alpha=0.1, epsilon=0.1):
    if q_value_estimator is None:
        raise ValueError("SPID scoring requires a Q-value estimator")
    if len(X) == 0:
        raise ValueError("SPID scoring requires a non-empty dataset")
    finite_rows = jnp.all(jnp.isfinite(X), axis=-1) & jnp.all(
        jnp.isfinite(y), axis=-1
    )
    invalid_rows = int(jnp.sum(~finite_rows))
    if invalid_rows:
        raise ValueError(
            f"SPID dataset contains {invalid_rows} non-finite transitions"
        )
    if bootstrap_repertoire is not None and len(bootstrap_repertoire.fitnesses) != n_pop:
        raise ValueError("Bootstrap repertoire size must match n_pop")
    key = jax.random.key(seed)

    # Init from the previous best population when available.  It must be
    # rescored on the newly aggregated DAgger data; simply overwriting a
    # repertoire after ``ga.init`` leaves stale fitness values behind.
    key, subkey = jax.random.split(key)
    if bootstrap_repertoire is None:
        init_keys = jax.random.split(key, n_pop)
        init_population = jax.jit(jax.vmap(cgp_structure.init))(init_keys)
    else:
        init_population = bootstrap_repertoire.genotypes

    # Define a metrics function
    metrics_function = functools.partial(
        custom_ga_metrics, extra_scores_metrics={"test_accuracy": jnp.ravel}
    )

    # Define emitter
    mutation_fn = jax.jit(jax.vmap(cgp_structure.mutate, in_axes=(0, 0)))
    tournament_selector = TournamentSelector()
    mixing_emitter = CustomMixingEmitter(
        mutation_fn=mutation_fn,
        variation_fn=None,
        variation_percentage=0,
        batch_size=n_pop,
        selector=tournament_selector,
    )

    # Prepare the scoring function
    scoring_fn = partial(
        feature_construction_scoring_fn,
        X=X, y=y, cgp_structure=cgp_structure,
        q_value_estimator=q_value_estimator,
        dataset_batch_size=dataset_batch_size,
        alpha=alpha,
        epsilon=epsilon,
    )
    # Instantiate GA
    ga = GeneticAlgorithmWithExtraScores(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
    )

    # Evaluate the initial population
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = ga.init(
        genotypes=init_population, population_size=n_pop, key=subkey
    )
    metrics = {
        k: jnp.array([])
        for k in ["iteration", "max_fitness", "time"]
    }

    # Set up init metrics
    # init_metrics = jax.tree.map(lambda x: jnp.array([x]) if x.shape == () else x, init_metrics)
    init_metrics["iteration"] = 0
    init_metrics["max_fitness"] = init_metrics["max_fitness"][0]
    init_metrics["time"] = 0.0  # No time recorded for initialization

    # Convert init_metrics to match the metrics dictionary structure
    # metrics = jax.tree.map(lambda metric, init_metric: jnp.concatenate([metric, init_metric], axis=0), metrics,
    #                        init_metrics)
    # csv_logger = CSVLogger(
    #     f'../results/{config["run_name"]}.csv', header=list(metrics.keys())
    # )

    # Log initial metrics
    # csv_logger.log(jax.tree.map(lambda x: x[-1], init_metrics))
    # csv_logger.log(process_metrics_mtr(init_metrics, test_accuracy_header))

    current_metrics = init_metrics

    # Iterations
    for iteration in range(1, n_gens):
        key, subkey, sample_key = jax.random.split(key, 3)

        start_time = time.time()

        repertoire, emitter_state, current_metrics = ga.update(
            repertoire=repertoire,
            emitter_state=emitter_state,
            key=subkey,
        )
        timelapse = time.time() - start_time

        # Metrics
        unwrapped_metrics = jax.tree.map(lambda x: jnp.ravel(x), current_metrics)
        unwrapped_metrics["iteration"] = iteration
        unwrapped_metrics["time"] = timelapse
        unwrapped_metrics["max_fitness"] = unwrapped_metrics["max_fitness"][0]

        # print(unwrapped_metrics)

        # Log
        # csv_logger.log(unwrapped_metrics)

    repertoire_to_store = GARepertoire.init(
        genotypes=repertoire.genotypes,
        fitnesses=repertoire.fitnesses,
        population_size=len(repertoire.fitnesses),
    )
    max_fitness = jnp.ravel(current_metrics["max_fitness"])[0]
    finite_fitness = jnp.isfinite(repertoire.fitnesses)
    best_idx = jnp.argmax(repertoire.fitnesses[:, 0])
    diagnostics = {
        "finite_fitness_fraction": jnp.mean(finite_fitness),
        "best_performance_gap": repertoire.extra_scores[
            "performance_gap"
        ][best_idx],
        "best_fidelity_gap": repertoire.extra_scores[
            "fidelity_gap"
        ][best_idx],
    }
    if not bool(jnp.all(finite_fitness)):
        raise FloatingPointError(
            "Non-finite CGP fitness detected; "
            f"finite fraction={float(diagnostics['finite_fitness_fraction']):.3f}"
        )
    return repertoire_to_store, -max_fitness, diagnostics
