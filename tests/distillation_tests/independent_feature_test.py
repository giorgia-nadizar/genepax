import pickle

import jax
import jax.numpy as jnp
import numpy as np

from distillation.fit_feature_imitation import make_feature_cgp, fit_feature_imitation
from distillation.independent_feature_policy import independent_feature_action, independent_feature_mse


def test_independent_graphs_and_readouts_are_isolated_after_reload() -> None:
    cgp = make_feature_cgp(2, 1, k=8, n_nodes=4)
    graphs = []
    for action in range(3):
        graph = cgp.init(jax.random.key(action))
        graph['genes']['outputs'] = jnp.full(8, action % 2)
        readout = jnp.zeros(9).at[0].set(.2 * (action + 1)).at[-1].set(.1 * action)
        graphs.append(cgp.update_weights(graph, {'custom_weights': readout}))
    policy = pickle.loads(pickle.dumps({'actions': tuple(graphs)}))
    obs = jnp.array([.2, .3])
    before = jax.jit(lambda p: independent_feature_action(p, cgp, obs))(policy)
    np.testing.assert_allclose(before, [.04, .22, .32], atol=1e-6)
    # Change only action 1's feature graph, keeping every regression fixed.
    changed = pickle.loads(pickle.dumps(policy))
    changed['actions'][1]['genes']['outputs'] = jnp.zeros(8, dtype=jnp.int32)
    after = independent_feature_action(changed, cgp, obs)
    np.testing.assert_allclose(after, [.04, .18, .32], atol=1e-6)


def test_combined_weighted_loss_matches_separately_evolved_action_losses() -> None:
    cgp = make_feature_cgp(2, 1, k=8, n_nodes=4)
    X = jax.random.uniform(jax.random.key(2), (24, 2))
    y = jnp.stack([.2 * X[:, 0], -.3 * X[:, 1] + .1], axis=1)
    weights = jnp.arange(24, dtype=jnp.float32)
    fits = [fit_feature_imitation(X, y[:, i:i+1], cgp, seed=i * 10000,
                                  n_gens=2, n_pop=3, sample_weights=weights) for i in range(2)]
    policy = {'actions': tuple(fit['genotype'] for fit in fits)}
    np.testing.assert_allclose(independent_feature_mse(policy, cgp, X, y, weights),
                               np.mean([fit['loss'] for fit in fits]), atol=1e-7)
