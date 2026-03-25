import jax
import jax.numpy as jnp
from qdax.core.containers.ga_repertoire import GARepertoire

from genepax.evolution.elite_selector import EliteSelector
from genepax.evolution.tournament_selector import TournamentSelector
from genepax.gp.cartesian_genetic_programming import CGP
from genepax.gp.tree_genetic_programming import TreeGP


def test_tournament_with_tree_gp() -> None:
    """Test that tournament selection works with TGP."""

    # Init a random key
    key = jax.random.key(seed=0)

    # Init the CGP policy graph with default values
    tree_structure = TreeGP(
        n_inputs=8,
        max_depth=3,
        outputs_wrapper=lambda x: x,
    )

    # Init the population of CGP genomes and store in a GA repertoire
    pop_size = 20
    key, subkey = jax.random.split(key)
    init_trees = tree_structure.init_ramped_half_and_half(subkey, pop_size)
    ga_repertoire = GARepertoire.init(
        init_trees,
        jnp.ones(
            (pop_size, 1),
        ),
        population_size=pop_size,
    )

    # Create selector
    selector = TournamentSelector(tournament_size=3)
    key, subkey = jax.random.split(key)
    selected = selector.select(ga_repertoire, subkey, num_samples=3)
    print(selected)


def test_tournament_with_cgp() -> None:
    """Test that tournament selection works with CGP."""

    # Init a random key
    key = jax.random.key(seed=0)

    # Init the CGP policy graph with default values
    policy_graph = CGP(
        n_inputs=3,
        n_outputs=2,
        weighted_functions=True,
    )

    # Init the population of CGP genomes and store in a GA repertoire
    pop_size = 10
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=pop_size)
    init_cgp_genomes = jax.vmap(policy_graph.init)(keys)
    ga_repertoire = GARepertoire.init(
        init_cgp_genomes,
        jnp.ones(
            (pop_size, 1),
        ),
        population_size=pop_size,
    )

    # Create selector
    selector = TournamentSelector(tournament_size=3)
    key, subkey = jax.random.split(key)
    selected = selector.select(ga_repertoire, subkey, num_samples=3)
    print(selected)


def test_elitism_with_cgp() -> None:
    # Init a random key
    key = jax.random.key(seed=0)

    # Init the CGP policy graph with default values
    policy_graph = CGP(
        n_inputs=3,
        n_outputs=2,
        n_nodes=4,
    )

    # Init the population of CGP genomes and store in a GA repertoire
    pop_size = 10
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=pop_size)
    init_cgp_genomes = jax.vmap(policy_graph.init)(keys)
    fitnesses = jnp.expand_dims(jnp.arange(pop_size), axis=1)
    ga_repertoire = GARepertoire.init(
        init_cgp_genomes, fitnesses, population_size=pop_size
    )

    # Create selector
    selector = EliteSelector()
    key, subkey = jax.random.split(key)
    selected = selector.select(ga_repertoire, subkey, num_samples=3)

    print(selected)


def test_elite_selector_selects_top_k():
    key = jax.random.PRNGKey(0)
    selector = EliteSelector()

    fitnesses = jnp.array([[1.0], [5.0], [3.0], [7.0], [2.0]])  # shape (5, 1)
    genotypes = jnp.arange(5)  # easy to verify indices

    ga_repertoire = GARepertoire.init(
        genotypes, fitnesses, population_size=len(genotypes)
    )

    selected = selector.select(ga_repertoire, key, num_samples=2)

    # Expected: fitness values 7.0 (idx 3), 5.0 (idx 1)
    assert jnp.all(selected.fitnesses.squeeze() == jnp.array([7.0, 5.0]))
    assert jnp.all(selected.genotypes == jnp.array([3, 1]))


def test_elite_selector_single_sample():
    key = jax.random.PRNGKey(0)
    selector = EliteSelector()

    fitnesses = jnp.array([[1.0], [9.0], [3.0]])
    genotypes = jnp.array([10, 20, 30])

    repertoire = GARepertoire.init(
        genotypes=genotypes, fitnesses=fitnesses, population_size=len(genotypes)
    )

    selected = selector.select(repertoire, key, num_samples=1)

    assert selected.genotypes[0] == 20
    assert selected.fitnesses[0] == 9.0


def test_elite_selector_selects_all():
    key = jax.random.PRNGKey(0)
    selector = EliteSelector()

    fitnesses = jnp.array([[3.0], [1.0], [2.0]])
    genotypes = jnp.array([100, 200, 300])

    repertoire = GARepertoire.init(
        genotypes=genotypes, fitnesses=fitnesses, population_size=len(genotypes)
    )

    selected = selector.select(repertoire, key, num_samples=3)

    # descending fitness: indices [0, 2, 1] → genotypes [100, 300, 200]
    assert jnp.all(selected.genotypes == jnp.array([100, 300, 200]))


def test_elite_selector_arbitrary_pytree_genotypes():
    key = jax.random.PRNGKey(0)
    selector = EliteSelector()

    # Create a nested pytree of genotypes
    genotypes = {
        "weights": jnp.array([[1, 1], [2, 2], [3, 3], [4, 4]]),
        "biases": (jnp.array([10, 20, 30, 40]),),
        "metadata": [
            {"id": jnp.array([100, 200, 300, 400])},
            jnp.array([5.0, 6.0, 7.0, 8.0]),
        ],
    }

    # Fitness values choose indices 3, 1 (descending fitness: 100 → idx 3, 50 → idx 1)
    fitnesses = jnp.array([[10.0], [50.0], [20.0], [100.0]])

    repertoire = GARepertoire.init(
        genotypes=genotypes,
        fitnesses=fitnesses,
        population_size=len(fitnesses),
    )

    selected = selector.select(repertoire, key, num_samples=2)

    # Expected selected indices: [3, 1]
    expected_idx = jnp.array([3, 1])

    # Check that each leaf was correctly indexed
    assert jnp.all(selected.genotypes["weights"] == genotypes["weights"][expected_idx])
    assert jnp.all(
        selected.genotypes["biases"][0] == genotypes["biases"][0][expected_idx]
    )
    assert jnp.all(
        selected.genotypes["metadata"][0]["id"]
        == genotypes["metadata"][0]["id"][expected_idx]
    )
    assert jnp.all(
        selected.genotypes["metadata"][1] == genotypes["metadata"][1][expected_idx]
    )

    # Fitnesses should also be properly selected
    assert jnp.all(selected.fitnesses.squeeze() == fitnesses.squeeze()[expected_idx])
