import jax
import jax.numpy as jnp
import optax
import pytest

from genepax.gp.linear_genetic_programming import LGP


def test_genome_bounds() -> None:
    """Test that a LGP genome has all elements in the correct bounds.
    Tests both at initialization and after mutation.
    """
    # define genome structure
    lgp = LGP(
        n_inputs=2,
        n_outputs=1,
    )
    key = jax.random.key(42)

    # define expected bounds
    lhs_lower_bound = lgp.n_inputs + lgp.n_input_constants
    assignments_upper_bounds = lgp.n_registers
    functions_bound = len(lgp.function_set)

    # init genome
    key, init_key = jax.random.split(key)
    initial_lgp_genome = lgp.init(init_key)

    # check if bounds are respected at initialization
    def _test_bounds(genome) -> None:
        pytest.assume(jnp.all(genome["genes"]["targets"] >= lhs_lower_bound))
        pytest.assume(jnp.all(genome["genes"]["targets"] < assignments_upper_bounds))
        pytest.assume(jnp.all(genome["genes"]["inputs1"] < assignments_upper_bounds))
        pytest.assume(jnp.all(genome["genes"]["inputs2"] < assignments_upper_bounds))
        pytest.assume(jnp.all(genome["genes"]["functions"] < functions_bound))
        pytest.assume(jnp.all(genome["weights"]["functions"] == 1))
        pytest.assume(jnp.all(genome["weights"]["inputs1"] == 1))
        pytest.assume(jnp.all(genome["weights"]["inputs2"] == 1))
        pytest.assume(jnp.all(genome["weights"]["functions_biases"] == 0))
        pytest.assume(jnp.all(genome["weights"]["inputs1_biases"] == 0))
        pytest.assume(jnp.all(genome["weights"]["inputs2_biases"] == 0))
        pytest.assume(
            jnp.allclose(genome["weights"]["program_inputs"], jnp.asarray([0.1, 1]))
        )

    _test_bounds(initial_lgp_genome)

    # mutate genome
    key, mut_key = jax.random.split(key)
    mutated_lgp_genome = lgp.mutate(
        genotype=initial_lgp_genome,
        rnd_key=mut_key,
    )

    # check if bounds are respected after mutation
    _test_bounds(mutated_lgp_genome)

    # init another genome and perform crossover
    key, another_init_key = jax.random.split(key)
    another_lgp_genome = lgp.init(another_init_key)
    key, xover_key = jax.random.split(key)
    crossed_genome = lgp.crossover(mutated_lgp_genome, another_lgp_genome, xover_key)

    # check if bounds are respected after crossover
    _test_bounds(crossed_genome)


@pytest.mark.parametrize("init_type", ["uniform", "natural"])
@pytest.mark.parametrize("weighted_functions", [True, False])
@pytest.mark.parametrize("weighted_inputs", [True, False])
def test_weights_initialization(init_type, weighted_functions, weighted_inputs) -> None:
    key = jax.random.PRNGKey(0)
    n_lines = 4
    n_input_constants = 5
    lgp = LGP(
        n_inputs=2,
        n_outputs=4,
        n_program_lines=n_lines,
        n_input_constants=n_input_constants,
        weighted_functions=weighted_functions,
        weighted_inputs=weighted_inputs,
        weights_initialization=init_type,
    )
    weights = lgp.init_weights(key)

    # Check shapes
    assert weights["program_inputs"].shape[0] == n_input_constants
    assert weights["functions"].shape[0] == n_lines
    assert weights["inputs1"].shape[0] == n_lines
    assert weights["inputs2"].shape[0] == n_lines

    # Check values for "natural"
    if init_type == "natural":
        assert jnp.all(weights["functions"] == 1)
        assert jnp.all(weights["inputs1"] == 1)
        assert jnp.all(weights["inputs2"] == 1)

    # Check values for "uniform"
    if init_type == "uniform":
        assert jnp.all(weights["functions"] <= 1)
        assert jnp.all(weights["functions"] >= -1)
        assert jnp.all(weights["inputs1"] <= 1)
        assert jnp.all(weights["inputs1"] >= -1)
        assert jnp.all(weights["inputs2"] <= 1)
        assert jnp.all(weights["inputs2"] >= -1)


def test_weights_changes_after_mutation() -> None:
    """Test that the weights of a LGP genome change (or not) correctly given the set flag."""
    for mut in [True, False]:
        # define genome structure
        lgp = LGP(
            n_inputs=2,
            n_outputs=1,
            weighted_inputs=True,
            weighted_functions=True,
            weighted_program_inputs=True,
            weights_mutation=mut,
        )
        key = jax.random.key(42)

        # init genome
        key, init_key = jax.random.split(key)
        initial_lgp_genome = lgp.init(init_key)

        # mutate genome
        key, mut_key = jax.random.split(key)
        mutated_lgp_genome = lgp.mutate(
            genotype=initial_lgp_genome,
            rnd_key=mut_key,
        )

        assert not mut * all(
            jax.tree.leaves(
                jax.tree.map(
                    lambda x, y: jnp.allclose(x, y),
                    initial_lgp_genome["weights"],
                    mutated_lgp_genome["weights"],
                )
            )
        )

    for mut_type in ["gaussian", "automl0", "other"]:
        # define genome structure
        lgp = LGP(
            n_inputs=2,
            n_outputs=1,
            weighted_inputs=True,
            weighted_functions=True,
            weighted_program_inputs=True,
            weights_mutation=True,
            weights_mutation_type=mut_type,
        )
        key = jax.random.key(42)

        # init genome
        key, init_key = jax.random.split(key)
        initial_lgp_genome = lgp.init(init_key)

        # mutate genome
        key, mut_key = jax.random.split(key)

        if mut_type in ["gaussian", "automl0"]:
            mutated_lgp_genome = lgp.mutate(
                genotype=initial_lgp_genome,
                rnd_key=mut_key,
            )

            assert not mut * all(
                jax.tree.leaves(
                    jax.tree.map(
                        lambda x, y: jnp.allclose(x, y),
                        initial_lgp_genome["weights"],
                        mutated_lgp_genome["weights"],
                    )
                )
            )
        else:
            with pytest.raises(NotImplementedError):
                lgp.mutate(
                    genotype=initial_lgp_genome,
                    rnd_key=mut_key,
                )


def test_known_genome_execution() -> None:
    """Test that a lGP genome behaves as expected.
    The chosen genome takes as outputs:
    - input 0
    - constant 0
    - input 0 + input 1
    - (input 0 + input 1) * input 1
    All outputs are wrapped by the tanh function.
    """
    # define genome structure
    lgp = LGP(n_inputs=2, n_outputs=4, n_program_lines=4, n_computation_registers=4)
    lgp_genome = {
        "genes": {
            "targets": jnp.asarray([8, 9, 10, 11]),
            "inputs1": jnp.asarray([0, 2, 0, 10]),
            "inputs2": jnp.asarray([3, 3, 1, 1]),
            "functions": jnp.asarray([2, 2, 0, 2]),
        },
        "weights": lgp.init_weights(key=jax.random.key(42)),
    }

    input_test_range = jnp.arange(start=-1, stop=1, step=0.2)
    for x in input_test_range:
        for y in input_test_range:
            inputs = jnp.asarray([x, y])
            outputs = lgp.apply(
                lgp_genome,
                inputs,
            )
            expected_outputs = jnp.tanh(
                jnp.asarray(
                    [x, lgp_genome["weights"]["program_inputs"][0], x + y, (x + y) * y]
                )
            )
            pytest.assume(jnp.allclose(outputs, expected_outputs, rtol=1e-5, atol=1e-8))


def test_descriptors() -> None:
    """Test that a LGP genome has the expected descriptors."""
    # define genome structure
    lgp = LGP(n_inputs=2, n_outputs=4, n_program_lines=4, n_computation_registers=4)
    lgp_genome = {
        "genes": {
            "targets": jnp.asarray([8, 9, 10, 11]),
            "inputs1": jnp.asarray([0, 2, 0, 10]),
            "inputs2": jnp.asarray([3, 3, 1, 1]),
            "functions": jnp.asarray([2, 2, 0, 2]),
        },
        "weights": lgp.init_weights(key=jax.random.key(42)),
    }
    complexity = lgp.compute_complexity(lgp_genome)
    arities = lgp.compute_function_arities(lgp_genome)
    pytest.assume(complexity == 1)
    pytest.assume(arities[0] == 0)
    pytest.assume(arities[1] == 1)


def test_active_lines() -> None:
    """Test that a LGP genomes has the correct active nodes."""
    # define genome structure
    lgp = LGP(n_inputs=2, n_outputs=4, n_program_lines=5, n_computation_registers=4)
    lgp_genome = {
        "genes": {
            "targets": jnp.asarray([8, 9, 10, 11, 4]),
            "inputs1": jnp.asarray([0, 2, 0, 10, 2]),
            "inputs2": jnp.asarray([3, 3, 1, 1, 10]),
            "functions": jnp.asarray([2, 2, 0, 2, 1]),
        },
        "weights": lgp.init_weights(key=jax.random.key(42)),
    }
    expected_active_lines = jnp.asarray([1, 1, 1, 1, 0])
    active_lines = lgp.compute_active_mask(lgp_genome)
    active_size = lgp.size(lgp_genome)
    pytest.assume(jnp.array_equal(active_lines, expected_active_lines))
    pytest.assume(jnp.array_equal(active_size, jnp.sum(expected_active_lines)))

    # define genome structure
    lgp2 = LGP(n_inputs=2, n_outputs=2, n_program_lines=2, n_computation_registers=4)
    lgp_genome2 = {
        "genes": {
            "targets": jnp.asarray([5, 9]),
            "inputs1": jnp.asarray([0, 0]),
            "inputs2": jnp.asarray([1, 5]),
            "functions": jnp.asarray([2, 5]),
        },
        "weights": lgp.init_weights(key=jax.random.key(42)),
    }
    expected_active_lines2 = jnp.asarray([0, 1])
    active_lines2 = lgp2.compute_active_mask(lgp_genome2)
    active_size2 = lgp2.size(lgp_genome2)
    pytest.assume(jnp.array_equal(active_lines2, expected_active_lines2))
    pytest.assume(jnp.array_equal(active_size2, jnp.sum(expected_active_lines2)))


def test_active_lines_jit() -> None:
    """Test that the computation of the active lines is jittable."""
    key = jax.random.key(42)
    lgp = LGP(
        n_inputs=3,
        n_outputs=2,
    )

    # Init the population of LGP genomes
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=10)
    init_lgp_genomes = jax.vmap(lgp.init)(keys)

    # Check it runs
    jax.vmap(jax.jit(lgp.compute_active_mask))(init_lgp_genomes)


def test_readable_program() -> None:
    """Test of different ways of obtaining readable LGP programs."""
    lgp = LGP(n_inputs=2, n_outputs=4, n_program_lines=5, n_computation_registers=4)
    lgp_genome = {
        "genes": {
            "targets": jnp.asarray([8, 9, 10, 11, 4]),
            "inputs1": jnp.asarray([0, 2, 0, 10, 2]),
            "inputs2": jnp.asarray([3, 3, 1, 1, 10]),
            "functions": jnp.asarray([2, 2, 0, 2, 1]),
        },
        "weights": lgp.init_weights(key=jax.random.key(42)),
    }
    print(lgp.get_readable_program(lgp_genome))

    lgp2 = LGP(n_inputs=2, n_outputs=2, n_program_lines=2, n_computation_registers=4)
    lgp_genome2 = {
        "genes": {
            "targets": jnp.asarray([5, 9]),
            "inputs1": jnp.asarray([0, 0]),
            "inputs2": jnp.asarray([1, 5]),
            "functions": jnp.asarray([2, 5]),
        },
        "weights": lgp.init_weights(key=jax.random.key(42)),
    }
    print(lgp2.get_readable_program(lgp_genome2))


def test_readable_expression() -> None:
    """Test of different ways of obtaining readable LGP expressions."""
    lgp = LGP(n_inputs=2, n_outputs=4, n_program_lines=5, n_computation_registers=4)
    lgp_genome = {
        "genes": {
            "targets": jnp.asarray([8, 9, 10, 11, 4]),
            "inputs1": jnp.asarray([0, 2, 0, 10, 2]),
            "inputs2": jnp.asarray([6, 3, 1, 1, 10]),
            "functions": jnp.asarray([2, 2, 0, 2, 1]),
        },
        "weights": lgp.init_weights(key=jax.random.key(42)),
    }
    print(lgp.get_readable_program(lgp_genome), "\n")

    print(lgp.get_readable_expression(lgp_genome), "\n")

    inputs_mapping_fn = lambda x: f"i_{{{x}}}"
    print(
        lgp.get_readable_expression(lgp_genome, inputs_mapping=inputs_mapping_fn), "\n"
    )

    inputs_mapping_dict = {0: "a", 1: "b"}
    print(
        lgp.get_readable_expression(lgp_genome, inputs_mapping=inputs_mapping_dict),
        "\n",
    )

    outputs_mapping_fn = lambda x: f"o_{{{x}}}"
    print(
        lgp.get_readable_expression(lgp_genome, outputs_mapping=outputs_mapping_fn),
        "\n",
    )

    outputs_mapping_dict = {0: "x", 1: "y"}
    print(
        lgp.get_readable_expression(lgp_genome, outputs_mapping=outputs_mapping_dict),
        "\n",
    )


def test_weights_update() -> None:
    """Test that the weights update works correctly."""
    lgp = LGP(
        n_inputs=3,
        n_input_constants=0,
        n_outputs=2,
        n_program_lines=4,
        weighted_functions=True,
        weighted_inputs=False,
    )
    # Init the population of lgp genomes
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=10)
    init_lgp_genomes = jax.vmap(lgp.init)(keys)

    # Change weights
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=10)
    init_weights = jax.vmap(jax.jit(lgp.init_weights))(keys)

    # Get the trainable weights
    original_weights = init_lgp_genomes["weights"]
    updated_genomes = jax.vmap(jax.jit(lgp.update_weights), in_axes=(0, 0))(
        init_lgp_genomes, init_weights
    )
    assert all(
        jax.tree.leaves(
            jax.tree.map(
                lambda x, y: jnp.allclose(x, y),
                init_weights,
                updated_genomes["weights"],
            )
        )
    )
    assert not all(
        jax.tree.leaves(
            jax.tree.map(
                lambda x, y: jnp.allclose(x, y), init_weights, original_weights
            )
        )
    )


def test_get_weights() -> None:
    """Test that the get weights function returns only the weights that count."""
    for w_fn in [True, False]:
        for w_in in [True, False]:
            for w_pr_in in [True, False]:
                for b_fn in [True, False]:
                    for b_in in [True, False]:
                        lgp = LGP(
                            n_inputs=3,
                            n_input_constants=0,
                            n_outputs=2,
                            n_program_lines=4,
                            weighted_functions=w_fn,
                            weighted_inputs=w_in,
                            weighted_program_inputs=w_pr_in,
                            biased_inputs=b_in,
                            biased_functions=b_fn,
                        )

                        # Init the population of CGP genomes
                        key = jax.random.key(0)
                        key, subkey = jax.random.split(key)
                        keys = jax.random.split(subkey, num=10)
                        init_lgp_genomes = jax.vmap(lgp.init)(keys)

                        # Get the trainable weights
                        weights = jax.vmap(jax.jit(lgp.get_weights))(init_lgp_genomes)

                        assert w_fn == ("functions" in weights)
                        assert w_in == ("inputs1" in weights)
                        assert w_in == ("inputs2" in weights)
                        assert w_pr_in == ("program_inputs" in weights)
                        assert b_fn == ("functions_biases" in weights)
                        assert b_in == ("inputs1_biases" in weights)
                        assert b_in == ("inputs2_biases" in weights)


def test_gradient_optimization_of_function_weights() -> None:
    # Generate genome
    lgp = LGP(
        n_inputs=3,
        n_input_constants=0,
        n_outputs=2,
        n_computation_registers=2,
        n_program_lines=4,
        weighted_functions=True,
        weighted_inputs=False,
    )
    target_weights = jnp.asarray([0.2, -0.5, 0.4, -0.3])
    lgp_genome = {
        "genes": {
            "targets": jax.lax.stop_gradient(jnp.asarray([4, 5, 3, 6])),
            "inputs1": jax.lax.stop_gradient(jnp.asarray([1, 0, 2, 4])),
            "inputs2": jax.lax.stop_gradient(jnp.asarray([2, 3, 4, 5])),
            "functions": jax.lax.stop_gradient(jnp.asarray([2, 6, 3, 0])),
        },
        "weights": lgp.init_weights(key=jax.random.key(42)),
    }
    active = lgp.compute_active_mask(lgp_genome)
    print(lgp.get_readable_expression(lgp_genome), "\n")
    print(target_weights * active)

    # Generate synthetic dataset
    n_samples = 500
    key = jax.random.key(0)
    x_key, y_key, z_key, noise_key, weights_key = jax.random.split(key, 5)
    x = jax.random.uniform(x_key, (n_samples,), minval=0, maxval=2 * jnp.pi)
    y = jax.random.normal(y_key, (n_samples,))
    z = jax.random.normal(z_key, (n_samples,))
    observations = jnp.vstack((x, y, z)).T
    noise = 0.01 * jax.random.normal(noise_key, (n_samples, lgp.n_outputs))
    target_outputs = (
        jax.vmap(jax.jit(lgp.apply), (None, 0, None))(
            lgp_genome, observations, {"functions": target_weights}
        )
        + noise
    )

    # Initialize weights to random values
    lgp_weights = (
        jax.random.uniform(key=weights_key, shape=(lgp.n_program_lines,)) * 2 - 1
    )

    # Loss = mean squared error
    def loss_fn(weights, genome, inputs, target_y):
        pred_y = jax.vmap(jax.jit(lgp.apply), (None, 0, None))(
            genome, inputs, {"functions": weights}
        )
        return jnp.mean((pred_y - target_y) ** 2)

    @jax.jit
    def step(genome, weights, opt_st, inputs, targets):
        loss, grads = jax.value_and_grad(loss_fn)(weights, genome, inputs, targets)
        updates, opt_st = optimizer.update(grads, opt_st)
        params = optax.apply_updates(weights, updates)
        return params, opt_state, loss

    # Optimizer
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(lgp_weights)

    # Training loop
    for _ in range(50_000):
        lgp_weights, opt_state, train_loss = step(
            lgp_genome, lgp_weights, opt_state, observations, target_outputs
        )
    print(lgp_weights * active)

    pytest.assume(train_loss < 0.01)


def test_gradient_optimization_of_input_weights() -> None:
    # Generate genome
    lgp = LGP(
        n_inputs=3,
        n_input_constants=0,
        n_outputs=2,
        n_computation_registers=2,
        n_program_lines=4,
        weighted_functions=False,
        weighted_inputs=True,
    )
    target_weights1 = jnp.asarray([0.2, -0.5, 0.4, -0.3])
    target_weights2 = jnp.asarray([-0.7, 0.6, 0.1, -1.0])
    lgp_genome = {
        "genes": {
            "targets": jax.lax.stop_gradient(jnp.asarray([4, 5, 3, 6])),
            "inputs1": jax.lax.stop_gradient(jnp.asarray([1, 0, 2, 4])),
            "inputs2": jax.lax.stop_gradient(jnp.asarray([2, 3, 4, 5])),
            "functions": jax.lax.stop_gradient(jnp.asarray([2, 6, 3, 0])),
        },
        "weights": lgp.init_weights(key=jax.random.key(42)),
    }
    active = lgp.compute_active_mask(lgp_genome)

    # Generate synthetic dataset
    n_samples = 500
    key = jax.random.key(0)
    x_key, y_key, z_key, noise_key, weights_key = jax.random.split(key, 5)
    x = jax.random.uniform(x_key, (n_samples,), minval=0, maxval=2 * jnp.pi)
    y = jax.random.normal(y_key, (n_samples,))
    z = jax.random.normal(z_key, (n_samples,))
    observations = jnp.vstack((x, y, z)).T
    target_outputs = jax.vmap(jax.jit(lgp.apply), (None, 0, None))(
        lgp_genome,
        observations,
        {"inputs1": target_weights1, "inputs2": target_weights2},
    )

    # Initialize weights to random values
    lgp_weights = (
        jax.random.uniform(key=weights_key, shape=(lgp.n_program_lines * 2,)) * 2 - 1
    )
    weights1, weights2 = jnp.split(lgp_weights, 2)
    optimizable_weights = {"inputs1": weights1, "inputs2": weights2}

    # Loss = mean squared error
    def loss_fn(weights_dict, genome, inputs, target_y):
        pred_y = jax.vmap(jax.jit(lgp.apply), (None, 0, None))(
            genome, inputs, weights_dict
        )
        return jnp.mean((pred_y - target_y) ** 2)

    @jax.jit
    def step(genome, weights_dict, opt_st, inputs, targets):
        loss, grads = jax.value_and_grad(loss_fn)(weights_dict, genome, inputs, targets)
        updates, opt_st = optimizer.update(grads, opt_st)
        weights_dict = optax.apply_updates(weights_dict, updates)
        return weights_dict, opt_state, loss

    # Optimizer
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(optimizable_weights)

    # Training loop
    train_loss = jnp.inf
    for _ in range(50_000):
        optimizable_weights, opt_state, train_loss = step(
            lgp_genome, optimizable_weights, opt_state, observations, target_outputs
        )
    print(train_loss)
    print(lgp.get_readable_expression(lgp_genome))
    print(optimizable_weights["inputs1"] * active, target_weights1 * active)
    print(optimizable_weights["inputs2"] * active, target_weights2 * active)
