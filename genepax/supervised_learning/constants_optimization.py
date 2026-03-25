from typing import Any, Callable, Dict, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree
from optax import GradientTransformation
from qdax.baselines.cmaes import CMAES, CMAESState
from qdax.custom_types import Genotype, RNGKey

from genepax.supervised_learning.dataset_utils import downsample_dataset
from genepax.supervised_learning.metrics import rmse
from genepax.supervised_learning.regularization import no_regularizer, no_snap


def optimize_constants_with_cmaes(
    graph_weights: Dict,
    genotype: Genotype,
    key: RNGKey,
    X: jnp.ndarray,
    y: jnp.ndarray,
    prediction_fn: Callable,
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = rmse,
    max_iter: int = 10,
    mini_batch_size: int = 32,
) -> Union[Dict, Any]:
    """
    Optimize the constants (weights) of a set of genotypes using CMA-ES.

    This function performs per-genotype optimization of weights in a
    computational graph using the Covariance Matrix Adaptation Evolution
    Strategy (CMA-ES). For each genome in the population, it flattens the
    weight pytree, runs CMA-ES for a specified number of iterations, and
    returns the optimized weights in the original pytree structure.

    Parameters
    ----------
    graph_weights : Dict
        A pytree (nested dictionary) of initial weights for the computational
        graph. Each leaf should be an array of shape `(n_genotypes, ...)`.
    genotype : Genotype
        A pytree representing the genotypes corresponding to the weights.
        Each leaf should have the same leading dimension as `graph_weights`.
    key : RNGKey
        A JAX random key used for sampling in CMA-ES.
    X : jnp.ndarray
        Input features for the prediction function.
    y : jnp.ndarray
        Target outputs corresponding to `X`.
    prediction_fn : Callable
        Function with signature `(X, genotype, graph_weights) -> y_pred` that
        computes predictions given inputs, a genotype, and a set of weights.
    loss_fn : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], optional
        Loss function comparing predictions to targets. Should return a scalar
        per sample. Default is `rmse`.
    max_iter : int, optional
        Number of CMA-ES iterations to run per genotype. Default is 10.
    mini_batch_size: int, optional
        Size of minibatches for each CMA-ES iteration. Default is 32.

    Returns
    -------
    Dict
        A pytree of the same structure as `graph_weights`, containing the
        optimized weights. Each leaf will have shape `(n_genotypes, ...)`,
        preserving the original batching of genotypes.

    Notes
    -----
    - Each genome is optimized independently with its own CMA-ES instance (in parallel).
    - The CMA-ES mean is used as the final optimized weight vector for each genome.
    """
    n_genotypes = jax.tree.leaves(graph_weights)[0].shape[0]
    weights_array_sample, weights_tree_def = ravel_pytree(
        jax.tree.map(lambda x: x[0], graph_weights)
    )
    search_dim = weights_array_sample.shape[0]
    pop_size = int(4 + jnp.floor(3 * jnp.log(search_dim)))
    mini_batch_size = min(mini_batch_size, X.shape[0])
    global_cmaes = CMAES(
        population_size=pop_size,
        search_dim=search_dim,
        fitness_function=lambda x: jnp.ones(
            (1,)
        ),  # dummy fitness function, will not be used
        mean_init=weights_array_sample,
    )

    def _single_genome_cmaes(
        single_genotype: Genotype, cmaes_state: CMAESState, single_key: RNGKey
    ) -> Union[Dict, Any]:
        def _single_weights_fitness_function(
            s_weights: jnp.ndarray, X_batch: jnp.ndarray, y_batch: jnp.ndarray
        ) -> jnp.ndarray:
            single_pytree_weights = weights_tree_def(s_weights)
            y_pred = prediction_fn(
                X_batch, single_genotype, graph_weights=single_pytree_weights
            )
            return loss_fn(y_batch, y_pred)

        def _weights_ranking_function(
            candidate_array_weights: jnp.ndarray, random_key: RNGKey
        ) -> jnp.ndarray:
            X_batch, y_batch = downsample_dataset(
                X, y, random_key=random_key, size=mini_batch_size
            )
            vmap_weights_fitness_fn = jax.vmap(
                _single_weights_fitness_function, in_axes=(0, None, None)
            )
            fitness_values = vmap_weights_fitness_fn(
                candidate_array_weights, X_batch, y_batch
            )
            idx_sorted = jnp.argsort(fitness_values)
            sorted_candidates = candidate_array_weights[
                idx_sorted[: global_cmaes._num_best]
            ]
            return sorted_candidates

        def _cmaes_body(
            i: int, carry: Tuple[RNGKey, CMAESState]
        ) -> Tuple[RNGKey, CMAESState]:
            thekey, state = carry
            key1, key2, thekey = jax.random.split(thekey, 3)
            current_genotypes = global_cmaes.sample(state, key1)
            sorted_samples = _weights_ranking_function(current_genotypes, key2)
            state = global_cmaes.update_state(state, sorted_samples)
            return thekey, state

        final_key, cmaes_state = jax.lax.fori_loop(
            0, max_iter, _cmaes_body, (single_key, cmaes_state)
        )
        final_weights = cmaes_state.mean
        return weights_tree_def(final_weights)

    cmaes_states = []
    for i in range(n_genotypes):
        current_weights = jax.tree.map(lambda x: x[i], graph_weights)  # noqa: B023
        current_state = CMAES(
            population_size=pop_size,
            search_dim=search_dim,
            fitness_function=lambda x: jnp.ones(
                (1,)
            ),  # dummy fitness function, will not be used
            mean_init=ravel_pytree(current_weights)[0],
        ).init()
        cmaes_states.append(current_state)
    pytree_cmaes_states = jax.tree.map(lambda *xs: jnp.stack(xs), *cmaes_states)

    keys = jax.random.split(key, n_genotypes)
    vmapped_cmaes = jax.vmap(_single_genome_cmaes, in_axes=(0, 0, 0))
    updated_weights = vmapped_cmaes(genotype, pytree_cmaes_states, keys)

    return updated_weights


def optimize_constants_with_lbfgs(
    graph_weights: Dict,
    genotype: Genotype,
    key: RNGKey,
    X: jnp.ndarray,
    y: jnp.ndarray,
    prediction_fn: Callable,
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = rmse,
    max_iter: int = 100,
    tol: float = 1e-3,
) -> Union[Dict, Any]:
    """
    Optimize the constant parameters (weights) of a batch of computational graphs
    using the L-BFGS quasi-Newton optimizer from Optax.

    Parameters
    ----------
    graph_weights : Dict
        A PyTree dictionary containing constant/weight parameters for each genome
        in the batch. The leading dimension corresponds to the batch size.
    genotype : Genotype
        The batch of genotypes whose constants are being optimized.
    key : RNGKey
        Unused RNG key included for API consistency with other optimizers.
    X : jnp.ndarray
        Input feature matrix of shape (n_samples, n_features).
    y : jnp.ndarray
        Target regression outputs of shape (n_samples,) or (n_samples, n_outputs).
    prediction_fn : Callable
        A function with signature `(X, genotype, graph_weights) -> predictions`
        that computes model outputs given a genome and weight PyTree.
    loss_fn : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], optional
        Differentiable loss function to minimize. Defaults to `rmse`.
    max_iter : int, optional
        Maximum number of L-BFGS optimization iterations. Defaults to 100.
    tol : float, optional
        Gradient-norm tolerance for convergence. Optimization stops early when
        ‖grad‖ < tol. Defaults to `1e-3`.

    Returns
    -------
    Dict
        A dictionary of optimized weight PyTrees, one for each genome, matching
        the structure of the input `graph_weights`.

    Notes
    -----
    - L-BFGS runs on flattened parameter vectors; trees are automatically
      flattened and reconstructed via `ravel_pytree`.
    - Convergence is determined by both iteration count and gradient norm.
    - Optimization is fully vectorized across genomes via `jax.vmap`.
    - The `key` argument is unused but maintained for functional symmetry with
      SGD/Adam-based optimization functions.
    """

    def _single_genome_lbfgs(
        single_weights: Dict, single_genotype: Genotype
    ) -> Union[Dict, Any]:
        init_flat_weights, weights_tree_def = ravel_pytree(single_weights)

        def _loss_fn(array_weights: jnp.ndarray) -> jnp.ndarray:
            pytree_weights = weights_tree_def(array_weights)
            y_pred = prediction_fn(X, single_genotype, graph_weights=pytree_weights)
            return loss_fn(y, y_pred)

        value_and_grad_fun = optax.value_and_grad_from_state(_loss_fn)

        def _step(carry: Tuple[Any, Any]) -> Tuple[Any, Any]:
            params, state = carry
            value, grad = value_and_grad_fun(params, state=state)
            updates, state = opt.update(
                grad, state, params, value=value, grad=grad, value_fn=_loss_fn
            )
            params = optax.apply_updates(params, updates)
            return params, state

        def _continuing_criterion(carry: Tuple[Any, Any]) -> Any:
            _, state = carry
            iter_num = optax.tree.get(state, "count")
            grad = optax.tree.get(state, "grad")
            err = optax.tree.norm(grad)
            return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

        opt = optax.lbfgs()
        init_carry = (init_flat_weights, opt.init(init_flat_weights))
        final_params, final_state = jax.lax.while_loop(
            _continuing_criterion, _step, init_carry
        )
        return weights_tree_def(final_params)

    vmapped_lbfgs = jax.vmap(jax.jit(_single_genome_lbfgs), in_axes=(0, 0))
    updated_weights = vmapped_lbfgs(graph_weights, genotype)

    return updated_weights


def optimize_constants_with_sgd(
    graph_weights: Dict,
    genotype: Genotype,
    key: RNGKey,
    X: jnp.ndarray,
    y: jnp.ndarray,
    prediction_fn: Callable,
    optimizer: GradientTransformation = optax.adam(1e-3),  # noqa: B008
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = rmse,
    regularization_loss_fn: Callable[[Dict], jnp.ndarray] = no_regularizer,
    regularization_update_fn: Callable[[Dict], Dict] = no_snap,
    n_gradient_steps: int = 100,
    batch_size: int = 32,
    regularization_strength: float = 1e-2,
) -> Dict:
    """
    Optimize the constant parameters (weights) of a batch of computational gp
    using mini-batch stochastic gradient descent with an Optax optimizer
    (default Adam), gradient clipping, and optional regularization.

    This function operates over a batch of genomes in parallel via `jax.vmap`,
    performing iterative gradient updates on each genome’s weight dictionary.
    During each optimization step, a random mini-batch is sampled, loss and
    gradients are computed, weights are updated, and optional regularization is applied.

    Parameters
    ----------
    graph_weights : Dict
        A PyTree-compatible dictionary containing the current constant/weight
        parameters for each genome in the batch. The leading dimension corresponds
        to the batch size (number of genomes).
    genotype : Genotype
        The batch of genotypes associated with the weight dictionaries. Used by
        the prediction function to interpret the weights.
    key : RNGKey
        JAX random key used for mini-batch sampling during optimization.
    X : jnp.ndarray
        Input feature matrix of shape (n_samples, n_features).
    y : jnp.ndarray
        Target regression outputs of shape (n_samples,) or (n_samples, n_outputs).
    prediction_fn : Callable
        A function with signature `(X, genotype, graph_weights) -> predictions`
        that computes regression model outputs given weights.
    optimizer : GradientTransformation, optional
        An Optax optimizer transformation used to update weights. Defaults to
        `optax.adam(1e-3)`. Gradient clipping is automatically prepended.
    loss_fn : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], optional
        A differentiable loss function comparing predicted and target outputs.
        Defaults to `rmse`.
    regularization_loss_fn : Callable[[Dict], jnp.ndarray], optional
        A differentiable regularization function that computes a scalar penalty
        from the current weights. Defaults to `no_regularizer` (no penalty).
    regularization_update_fn : Callable[[Dict], Dict], optional
        A function that optionally modifies weights after each optimizer step
        (e.g., snapping to discrete values). Defaults to `no_snap` (no modification).
    n_gradient_steps : int, optional
        Number of optimization steps to perform. Defaults to 100.
    batch_size : int or None, optional
        Mini-batch size. If None, uses the full dataset (i.e., full-batch gradient
        descent). Defaults to 32.
    regularization_strength : float, optional
        Scaling factor for the regularization loss. Defaults to 1e-2.

    Returns
    -------
    Dict
        A dictionary of optimized graph/constant weights, with the same structure
        as the input `graph_weights`, containing updated values after SGD/Adam
        training.

    Notes
    -----
    - Optimization is parallelized across genomes using `jax.vmap`.
    - Gradient clipping (`clip_by_global_norm`) is automatically added to prevent
      exploding gradients and reduce the likelihood of NaNs.
    - Mini-batch sampling is stochastic and driven by the provided PRNG key.
    - Regularization can be used to bias constants toward target values or enforce
      constraints via `regularization_loss_fn` and `regularization_update_fn`.
    """
    # add gradient clipping to the pipeline to prevent nan values
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optimizer,
    )

    opt_states = jax.vmap(optimizer.init)(graph_weights)

    batch_size = batch_size if batch_size is not None else X.shape[0]

    @jax.jit
    def _single_genome_loss(
        single_weights: Dict[str, jnp.ndarray],
        single_genotype: Genotype,
        X_batch: jnp.ndarray,
        y_batch: jnp.ndarray,
    ) -> jnp.ndarray:
        data_loss = loss_fn(
            y_batch,
            prediction_fn(X_batch, single_genotype, graph_weights=single_weights),
        )
        regularization_loss = regularization_loss_fn(single_weights)
        total_loss = data_loss + regularization_loss * regularization_strength
        return total_loss

    @jax.jit
    def _single_genome_gradient_step(
        single_weights: Dict[str, jnp.ndarray],
        single_genotype: Genotype,
        opt_st: Any,
        X_batch: jnp.ndarray,
        y_batch: jnp.ndarray,
    ) -> Tuple[Any, Any, jnp.ndarray]:
        loss, grads = jax.value_and_grad(_single_genome_loss)(
            single_weights, single_genotype, X_batch, y_batch
        )
        # clamp loss
        loss = jnp.where(jnp.isfinite(loss), loss, jnp.inf)
        # zero-out non-finite gradients to prevent nan constants
        grads = jax.tree.map(
            lambda g: jnp.where(jnp.isfinite(g), g, 0.0),
            grads,
        )
        weights_updates, new_opt_st = optimizer.update(grads, opt_st)
        updated_weights = optax.apply_updates(single_weights, weights_updates)
        updated_weights = regularization_update_fn(updated_weights)
        updated_weights = jax.tree.map(
            lambda w: jnp.clip(w, -1e4, 1e4),
            updated_weights,
        )
        return updated_weights, new_opt_st, loss

    step_fn = jax.vmap(
        jax.jit(_single_genome_gradient_step), in_axes=(0, 0, 0, None, None)
    )

    for _ in range(n_gradient_steps):
        key, subkey = jax.random.split(key)
        # sample a mini-batch
        X_batch, y_batch = downsample_dataset(X, y, random_key=subkey, size=batch_size)
        graph_weights, opt_states, train_losses = step_fn(
            graph_weights, genotype, opt_states, X_batch, y_batch
        )

    return graph_weights
