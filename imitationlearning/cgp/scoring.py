import jax
import jax.numpy as jnp

eps = 1e-8


def pearson_corr(x, y):
    x = x.ravel()
    y = y.ravel()

    x = x - jnp.mean(x)
    y = y - jnp.mean(y)

    numerator = jnp.sum(x * y)
    denominator = (
            jnp.sqrt(jnp.sum(x ** 2))
            * jnp.sqrt(jnp.sum(y ** 2))
            + eps
    )

    return numerator / denominator


def prepare_bc_scoring_fn(observations, actions, cgp_structure):
    """
    Returns a QDax-compatible scoring function.

    Fitness = - MSE(predicted_action, expert_action)
    """

    observations = jnp.asarray(observations)
    actions = jnp.asarray(actions)

    # ----------------------------------------
    # Evaluate ONE genome on full dataset
    # ----------------------------------------
    def eval_genome(genome):
        preds = jax.vmap(
            lambda x: cgp_structure.apply(genome, x)
        )(observations)
        corr = pearson_corr(preds, actions)

        # maximize correlation
        return jnp.asarray([corr])

        # loss = jnp.mean((preds - actions) ** 2)
        # return -jnp.asarray([loss])  # GA maximizes fitness

    # ----------------------------------------
    # Vectorize over population
    # ----------------------------------------
    def scoring_fn(genotypes, random_key=None):
        fitnesses = jax.vmap(eval_genome)(genotypes)

        # QDax expects (fitness, extra_scores)
        return jnp.nan_to_num(fitnesses, nan=-jnp.inf), {"updated_params": genotypes}

    return scoring_fn
