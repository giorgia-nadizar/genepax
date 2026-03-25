import functools

import jax
import jax.numpy as jnp
import pytest

from genepax.gp.cartesian_genetic_programming import CGP
from genepax.gp.ensemble_genetic_programming import EnsembleGP
from genepax.gp.tree_genetic_programming import TreeGP


def test_init_cgp() -> None:
    # define genome structure
    n_outputs = 4
    n_pop = 3
    n_nodes = 5
    cgp = CGP(n_inputs=2, n_outputs=1, n_nodes=n_nodes)
    ensemble_gp = EnsembleGP(n_outputs, cgp)

    key = jax.random.key(42)
    single_key, multi_key = jax.random.split(key)
    init_single_genome = ensemble_gp.init(single_key)
    pytest.assume(
        init_single_genome["genes"]["functions"].shape == (n_outputs, n_nodes)
    )

    multi_keys = jax.random.split(multi_key, n_pop)
    init_genomes = jax.jit(jax.vmap(ensemble_gp.init))(multi_keys)
    pytest.assume(
        init_genomes["genes"]["functions"].shape == (n_pop, n_outputs, n_nodes)
    )


def test_init_tree() -> None:
    # define genome structure
    n_outputs = 4
    n_pop = 3
    max_depth = 5
    tree_gp = TreeGP(n_inputs=2, max_depth=max_depth)
    ensemble_gp = EnsembleGP(n_outputs, tree_gp)

    key = jax.random.key(42)
    single_key, multi_key = jax.random.split(key)
    init_single_genome = ensemble_gp.init(single_key, target_depth=2)
    pytest.assume(
        init_single_genome["genes"]["tree"].shape == (n_outputs, tree_gp.n_nodes)
    )

    multi_keys = jax.random.split(multi_key, n_pop)
    partial_init = functools.partial(ensemble_gp.init, target_depth=2)
    init_genomes = jax.jit(jax.vmap(partial_init))(multi_keys)
    pytest.assume(
        init_genomes["genes"]["tree"].shape == (n_pop, n_outputs, tree_gp.n_nodes)
    )


def test_mutation_cgp() -> None:
    # define genome structure
    n_outputs = 4
    n_pop = 3
    n_nodes = 5
    cgp = CGP(n_inputs=2, n_outputs=1, n_nodes=n_nodes)
    ensemble_gp = EnsembleGP(n_outputs, cgp)

    key = jax.random.key(42)
    single_key, multi_key, mutate_key, multi_mutate_key = jax.random.split(key, 4)
    init_single_genome = ensemble_gp.init(single_key)
    mutated_single_genome = ensemble_gp.mutate(init_single_genome, mutate_key)
    pytest.assume(
        mutated_single_genome["genes"]["functions"].shape == (n_outputs, n_nodes)
    )

    multi_keys = jax.random.split(multi_key, n_pop)
    init_genomes = jax.jit(jax.vmap(ensemble_gp.init))(multi_keys)
    multi_mutate_keys = jax.random.split(multi_mutate_key, n_pop)
    mutated_genomes = jax.jit(jax.vmap(ensemble_gp.mutate))(
        init_genomes, multi_mutate_keys
    )
    pytest.assume(
        mutated_genomes["genes"]["functions"].shape == (n_pop, n_outputs, n_nodes)
    )


def test_mutation_tree() -> None:
    # define genome structure
    n_outputs = 4
    n_pop = 3
    max_depth = 5
    tree_gp = TreeGP(n_inputs=2, max_depth=max_depth)
    ensemble_gp = EnsembleGP(n_outputs, tree_gp)

    key = jax.random.key(42)
    single_key, multi_key, mutate_key, multi_mutate_key = jax.random.split(key, 4)
    init_single_genome = ensemble_gp.init(single_key, target_depth=2)
    mutated_single_genome = ensemble_gp.mutate(init_single_genome, mutate_key)
    pytest.assume(
        mutated_single_genome["genes"]["tree"].shape == (n_outputs, tree_gp.n_nodes)
    )

    multi_keys = jax.random.split(multi_key, n_pop)
    partial_init = functools.partial(ensemble_gp.init, target_depth=2)
    init_genomes = jax.jit(jax.vmap(partial_init))(multi_keys)
    multi_mutate_keys = jax.random.split(multi_mutate_key, n_pop)
    mutated_genomes = jax.jit(jax.vmap(ensemble_gp.mutate))(
        init_genomes, multi_mutate_keys
    )
    pytest.assume(
        mutated_genomes["genes"]["tree"].shape == (n_pop, n_outputs, tree_gp.n_nodes)
    )


def test_known_genome_execution() -> None:
    """Test that a CGP genome behaves as expected.
    The chosen genome takes as outputs:
    - input 0
    - constant 0
    - input 0 + input 1
    - (input 0 + input 1) * input 1
    All outputs are wrapped by the tanh function.
    """
    n_outputs = 4
    # define genome structure
    cgp = CGP(n_inputs=2, n_outputs=1, n_nodes=5)
    ensemble_gp = EnsembleGP(n_outputs, cgp)

    key = jax.random.key(42)
    fake_genome = ensemble_gp.init(key)

    ensemble_genome = {
        "genes": {
            "inputs1": jnp.asarray([[0, 0, 4, 0, 0] for _ in range(n_outputs)]),
            "inputs2": jnp.ones((n_outputs, cgp.n_nodes), dtype=jnp.int32),
            "functions": jnp.asarray([[0, 0, 2, 0, 0] for _ in range(n_outputs)]),
            "outputs": jnp.asarray([[0], [2], [4], [6]]),
        },
        "weights": fake_genome["weights"],
    }

    input_test_range = jnp.arange(start=-1, stop=1, step=0.2)
    for x in input_test_range:
        for y in input_test_range:
            inputs = jnp.asarray([x, y])
            outputs = ensemble_gp.apply(
                ensemble_genome,
                inputs,
            )
            expected_outputs = jnp.tanh(
                jnp.asarray(
                    [
                        x,
                        ensemble_genome["weights"]["program_inputs"][0][0],
                        x + y,
                        (x + y) * y,
                    ]
                )
            )
            pytest.assume(jnp.allclose(outputs, expected_outputs, rtol=1e-5, atol=1e-8))


def test_readable_expression_cgp():
    n_outputs = 4
    # define genome structure
    cgp = CGP(n_inputs=2, n_outputs=1, n_nodes=5)
    ensemble_gp = EnsembleGP(n_outputs, cgp)

    key = jax.random.key(42)
    fake_genome = ensemble_gp.init(key)

    ensemble_genome = {
        "genes": {
            "inputs1": jnp.asarray([[0, 0, 4, 0, 0] for _ in range(n_outputs)]),
            "inputs2": jnp.ones((n_outputs, cgp.n_nodes), dtype=jnp.int32),
            "functions": jnp.asarray([[0, 0, 2, 0, 0] for _ in range(n_outputs)]),
            "outputs": jnp.asarray([[0], [2], [4], [6]]),
        },
        "weights": fake_genome["weights"],
    }
    readable_expression = ensemble_gp.get_readable_expression(ensemble_genome)
    print(readable_expression)


def test_readable_expression_tree():
    n_outputs = 4
    max_depth = 5
    tree_gp = TreeGP(n_inputs=2, max_depth=max_depth)
    ensemble_gp = EnsembleGP(n_outputs, tree_gp)

    key = jax.random.key(42)
    ensemble_genome = ensemble_gp.init(key, target_depth=2)

    readable_expression = ensemble_gp.get_readable_expression(ensemble_genome)
    print(readable_expression)


def test_active_size():
    n_outputs = 4
    cgp = CGP(n_inputs=2, n_outputs=1, n_nodes=5)
    ensemble_gp = EnsembleGP(n_outputs, cgp)

    key = jax.random.key(42)
    fake_genome = ensemble_gp.init(key)

    ensemble_genome = {
        "genes": {
            "inputs1": jnp.asarray([[0, 0, 4, 0, 0] for _ in range(n_outputs)]),
            "inputs2": jnp.ones((n_outputs, cgp.n_nodes), dtype=jnp.int32),
            "functions": jnp.asarray([[0, 0, 2, 0, 0] for _ in range(n_outputs)]),
            "outputs": jnp.asarray([[0], [2], [4], [6]]),
        },
        "weights": fake_genome["weights"],
    }
    readable_expression = ensemble_gp.get_readable_expression(ensemble_genome)
    print(readable_expression)
    size = ensemble_gp.size(ensemble_genome)
    print(size)
