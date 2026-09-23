import pickle

import jax
import jax.numpy as jnp
import numpy as np

from distillation.fit_feature_imitation import (
    feature_policy_action, feature_policy_mse, fit_feature_imitation,
    fit_readout, make_feature_cgp,
)


def test_readout_handles_redundant_features_and_multiple_actions() -> None:
    x = jnp.linspace(-1, 1, 40)
    features = jnp.stack([x, 2 * x, jnp.ones_like(x)], axis=1)
    actions = jnp.stack([0.3 * x + 0.1, -0.2 * x + 0.4], axis=1)
    readout = jax.jit(fit_readout)(features, actions)
    assert readout.shape == (4, 2)
    np.testing.assert_allclose(
        features @ readout[:-1] + readout[-1], actions, atol=1e-6
    )


def test_stored_readout_controls_actions_after_reload() -> None:
    cgp = make_feature_cgp(2, 2, k=8, n_nodes=4)
    genotype = cgp.init(jax.random.key(0))
    # Use observation coordinates as known graph outputs.
    genotype["genes"]["outputs"] = jnp.asarray([0, 1] * 4)
    readout = jnp.zeros((9, 2)).at[0, 0].set(2).at[1, 1].set(-3)
    readout = readout.at[-1].set(jnp.asarray([0.1, 0.2]))
    genotype = cgp.update_weights(genotype, {"custom_weights": readout.ravel()})
    restored = pickle.loads(pickle.dumps(genotype))
    action = jax.jit(lambda x: feature_policy_action(restored, cgp, x))(
        jnp.asarray([0.2, 0.5])
    )
    np.testing.assert_allclose(action, [0.5, -1], atol=1e-6)


def test_evolution_preserves_fitted_readouts_for_entire_population() -> None:
    cgp = make_feature_cgp(2, 2, k=8, n_nodes=4)
    X = jax.random.uniform(jax.random.key(2), (24, 2), minval=-0.5, maxval=0.5)
    y = X @ jnp.asarray([[0.2, -0.1], [0.1, 0.3]]) + 0.05
    fit = fit_feature_imitation(X, y, cgp, n_gens=2, n_pop=3)
    population = pickle.loads(pickle.dumps(fit["repertoire"]))
    assert population.genotypes["weights"]["custom_weights"].shape == (3, 18)
    for index in range(3):
        genotype = jax.tree.map(lambda value: value[index], population.genotypes)
        actual = feature_policy_mse(genotype, cgp, X, y)
        np.testing.assert_allclose(actual, -population.fitnesses[index, 0], atol=1e-7)
    np.testing.assert_allclose(
        feature_policy_mse(fit["genotype"], cgp, X, y), fit["loss"], atol=1e-7
    )
    assert fit["history"][-1]["best_loss"] <= fit["history"][0]["best_loss"]


def test_weighted_readout_matches_independent_least_squares() -> None:
    features = np.array([[0., 0.], [1., 2.], [2., 4.], [3., 6.], [9., 18.]], dtype=np.float32)
    actions = np.array([[.1, .3], [.4, .1], [.6, -.2], [.5, -.3], [50., 60.]], dtype=np.float32)
    weights = np.array([1., 4., 2., 1., 0.], dtype=np.float32)
    design = np.column_stack([features, np.ones(len(features))])
    expected = np.column_stack([
        np.linalg.lstsq(design * np.sqrt(weights[:, None]),
                        actions[:, action] * np.sqrt(weights), rcond=None)[0]
        for action in range(actions.shape[1])
    ])
    actual = fit_readout(jnp.asarray(features), jnp.asarray(actions), jnp.asarray(weights))
    np.testing.assert_allclose(design @ actual, design @ expected, atol=2e-6)


def test_weighted_evolution_stores_readout_used_by_weighted_fitness() -> None:
    cgp = make_feature_cgp(2, 2, k=8, n_nodes=4)
    X = jax.random.uniform(jax.random.key(7), (24, 2))
    y = X * .3 + .1
    weights = jnp.arange(24, dtype=jnp.float32)
    fit = fit_feature_imitation(X, y, cgp, n_gens=2, n_pop=3, sample_weights=weights)
    np.testing.assert_allclose(
        feature_policy_mse(fit["genotype"], cgp, X, y, weights), fit["loss"], atol=1e-7
    )


def test_feature_collector_executes_student_but_retains_expert_labels() -> None:
    from typing import NamedTuple
    from distillation_experiments.scripts.experiments.dagger_cgp_features import collect_feature_trajectories

    class State(NamedTuple):
        obs: object
        reward: object
        done: object

    class Environment:
        def reset(self, key: jax.Array) -> State:
            return State(jnp.zeros(2), jnp.asarray(0.), jnp.asarray(False))

        def step(self, state: State, action: jax.Array) -> State:
            obs = state.obs + action + 1
            return State(obs, jnp.asarray(1.), obs[0] >= 2)

    cgp = make_feature_cgp(2, 2, k=8, n_nodes=4)
    genotype = cgp.init(jax.random.key(0))
    genotype = cgp.update_weights(genotype, {"custom_weights": jnp.zeros(18)})
    X, y, diagnostics = collect_feature_trajectories(
        genotype, cgp, lambda obs, key: (obs * .2 + .1, {}), Environment(),
        seed=0, steps=4, trajectories=2,
    )
    np.testing.assert_allclose(X, [[0, 0], [1, 1], [0, 0], [1, 1]])
    np.testing.assert_allclose(y, X * .2 + .1)
    assert diagnostics["new_samples"] == 4
    assert diagnostics["invalid_transitions"] == 0
