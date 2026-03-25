import jax
import jax.numpy as jnp
import pytest

from genepax.gp.cartesian_genetic_programming import CGP
from genepax.gp.linear_genetic_programming import LGP
from genepax.gp.tree_genetic_programming import TreeGP


def test_equal_behavior() -> None:
    key = jax.random.PRNGKey(0)
    cgp_key, lgp_key, tgp_key, data_key = jax.random.split(key, 4)
    n_inputs = 4
    n_outputs = 2
    cgp = CGP(n_inputs=n_inputs, n_outputs=n_outputs)
    lgp = LGP(n_inputs=n_inputs, n_outputs=n_outputs)
    tgp = TreeGP(n_inputs=n_inputs)
    cgp_genome = cgp.init(cgp_key)
    lgp_genome = lgp.init(lgp_key)
    tgp_genome = tgp.init(tgp_key, target_depth=2)
    data = jax.random.uniform(data_key, (n_inputs,))
    inputs_mapping = lambda i: f"x_{i}"
    outputs_mapping = lambda i: f"y_{i}"
    for model, genome in zip([cgp, lgp, tgp], [cgp_genome, lgp_genome, tgp_genome]):
        bound_model = model.bind(genome)

        # test prediction
        output1 = model.apply(genome, data)
        output2 = bound_model.apply(data)
        pytest.assume(jnp.allclose(output1, output2, rtol=1e-5, atol=1e-8))

        # test size
        size1 = model.size(genome)
        size2 = bound_model.size()
        pytest.assume(jnp.allclose(size1, size2, rtol=1e-5, atol=1e-8))

        # test readable representation
        string1 = model.get_readable_expression(genome, inputs_mapping=inputs_mapping)
        string2 = bound_model.get_readable_expression(inputs_mapping=inputs_mapping)
        pytest.assume(string1 == string2)

    for model, genome in zip([cgp, lgp], [cgp_genome, lgp_genome]):
        bound_model = model.bind(genome)
        # test readable representation
        string1 = model.get_readable_expression(
            genome, inputs_mapping=inputs_mapping, outputs_mapping=outputs_mapping
        )
        string2 = bound_model.get_readable_expression(
            inputs_mapping=inputs_mapping, outputs_mapping=outputs_mapping
        )
        pytest.assume(string1 == string2)


def test_unbind() -> None:
    key = jax.random.PRNGKey(0)
    cgp_key, lgp_key, tgp_key = jax.random.split(key, 3)
    n_inputs = 4
    n_outputs = 2
    cgp = CGP(n_inputs=n_inputs, n_outputs=n_outputs)
    lgp = LGP(n_inputs=n_inputs, n_outputs=n_outputs)
    tgp = TreeGP(n_inputs=n_inputs)
    cgp_genome = cgp.init(cgp_key)
    lgp_genome = lgp.init(lgp_key)
    tgp_genome = tgp.init(tgp_key, target_depth=2)
    for model, genome in zip([cgp, lgp, tgp], [cgp_genome, lgp_genome, tgp_genome]):
        bound_model = model.bind(genome)

        unbinded_model, unbinded_genome = bound_model.unbind()
        pytest.assume(model == unbinded_model)
        pytest.assume(genome == unbinded_genome)
