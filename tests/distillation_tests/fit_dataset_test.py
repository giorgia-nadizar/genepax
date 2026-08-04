import jax.numpy as jnp
import pytest

from distillation.fit_dataset import gm_dagger_loss


def test_gm_dagger_loss_matches_paper_equation_and_mask():
    expert_actions = jnp.asarray([[0.0, 0.0], [1.0, 1.0]])
    symbolic_actions = jnp.asarray([[3.0, 4.0], [1.0, 2.0]])
    expert_q = jnp.asarray([10.0, 5.0])
    symbolic_q = jnp.asarray([6.0, 4.0])

    loss, components = gm_dagger_loss(
        expert_actions,
        symbolic_actions,
        expert_q,
        symbolic_q,
        alpha=0.2,
        epsilon=0.1,
        valid_mask=jnp.asarray([1, 0]),
    )

    assert loss == pytest.approx(jnp.sqrt(4.2 * 5.1))
    assert components["performance_gap"] == pytest.approx(4.2)
    assert components["fidelity_gap"] == pytest.approx(5.1)


def test_gm_dagger_loss_stays_finite_for_imperfect_critic():
    loss, components = gm_dagger_loss(
        expert_actions=jnp.asarray([[0.0]]),
        symbolic_actions=jnp.asarray([[1.0]]),
        expert_q=jnp.asarray([0.0]),
        symbolic_q=jnp.asarray([10.0]),
        alpha=0.1,
        epsilon=0.1,
    )

    assert jnp.isfinite(loss)
    assert loss > 0
    assert components["performance_gap"] == pytest.approx(0.1)


@pytest.mark.parametrize("alpha, epsilon", [(0.0, 0.1), (0.1, 0.0)])
def test_gm_dagger_regularizers_must_be_positive(alpha, epsilon):
    with pytest.raises(ValueError):
        gm_dagger_loss(
            jnp.zeros((1, 1)),
            jnp.zeros((1, 1)),
            jnp.zeros(1),
            jnp.zeros(1),
            alpha=alpha,
            epsilon=epsilon,
        )
