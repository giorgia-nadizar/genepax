import jax
import jax.numpy as jnp
import pytest

from genepax.gp.tree_genetic_programming import TreeGP


@pytest.fixture
def simple_tree_gp():
    return TreeGP(n_inputs=3, max_depth=5, max_arity=2)


def test_tree_init(simple_tree_gp):
    key = jax.random.PRNGKey(0)
    genotype = simple_tree_gp.init(key, target_depth=2, full=True)

    # Check keys exist
    assert "genes" in genotype
    genes = genotype["genes"]
    for k in ["functions", "terminals", "constants", "tree"]:
        assert k in genes
        assert genes[k].shape[0] == simple_tree_gp.n_nodes

    # Check tree values are valid
    assert jnp.all(genes["tree"] >= 0)
    assert jnp.all(genes["tree"] <= 3)


def test_ramped_half_and_half(simple_tree_gp):
    key = jax.random.PRNGKey(1)
    pop_size = 10
    population = simple_tree_gp.init_ramped_half_and_half(key, pop_size)

    # Shape checks
    for k in ["functions", "terminals", "constants", "tree"]:
        assert population["genes"][k].shape[0] == pop_size


def test_tree_size(simple_tree_gp):
    genotype = {"genes": {"tree": jnp.array([1, 2, 0, -1, 3, 0])}}

    size = simple_tree_gp.size(genotype)

    assert int(size) == 3


def test_apply_returns_scalar(simple_tree_gp):
    key = jax.random.PRNGKey(2)
    genotype = simple_tree_gp.init(key, target_depth=2, full=True)
    x = jnp.array([0.5, -0.2, 1.0])

    out = simple_tree_gp.apply(genotype, x)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (1,)  # scalar output


def test_vmap_apply(simple_tree_gp):
    key = jax.random.PRNGKey(2)
    pop_size = 10
    population = simple_tree_gp.init_ramped_half_and_half(key, pop_size)
    x = jnp.array([0.5, -0.2, 1.0])
    outputs = jax.jit(jax.vmap(simple_tree_gp.apply, in_axes=(0, None)))(population, x)
    assert isinstance(outputs, jnp.ndarray)
    assert len(outputs) == pop_size


def test_known_genome_execution(simple_tree_gp):
    n_nodes = simple_tree_gp.n_nodes
    constants = jnp.ones(n_nodes, dtype=float) * 2
    tree = jnp.zeros(n_nodes, dtype=jnp.int32)
    functions = jnp.zeros(n_nodes, dtype=jnp.int32)
    terminals = jnp.zeros(n_nodes, dtype=jnp.int32)
    tree = tree.at[jnp.asarray([0, 1, 2, 5])].set(1)
    tree = tree.at[jnp.asarray([3, 4, 11])].set(2)
    tree = tree.at[6].set(3)
    functions = functions.at[0].set(2)
    functions = functions.at[1].set(0)
    functions = functions.at[2].set(3)
    functions = functions.at[5].set(8)
    terminals = terminals.at[4].set(2)
    terminals = terminals.at[11].set(1)
    genome = {
        "genes": {
            "tree": tree,
            "functions": functions,
            "terminals": terminals,
            "constants": constants,
        }
    }
    # print(simple_tree_gp.get_readable_expression(genome))
    random_points = jax.random.uniform(key=jax.random.PRNGKey(2), shape=(100, 3))
    for x in random_points:
        y = (x[0] + x[2]) * (jnp.log(x[1]) / 2)
        res = simple_tree_gp.apply(genome, x)
        assert jnp.isclose(y, res, 0.001)


def test_readable_expression(simple_tree_gp):
    key = jax.random.PRNGKey(3)
    genotype = simple_tree_gp.init(key, target_depth=2, full=True)
    expr = simple_tree_gp.get_readable_expression(genotype)

    assert isinstance(expr, str)
    # Expression should not be empty
    assert len(expr) > 0


def test_crossover_preserves_tree_shape(simple_tree_gp):
    key1, key2, xkey = jax.random.split(jax.random.PRNGKey(4), 3)
    g1 = simple_tree_gp.init(key1, target_depth=2, full=True)
    g2 = simple_tree_gp.init(key2, target_depth=2, full=False)

    offspring = simple_tree_gp.crossover(g1, g2, xkey)

    for k in ["functions", "terminals", "constants", "tree"]:
        assert offspring["genes"][k].shape == g1["genes"][k].shape


def test_subtree_mutation(simple_tree_gp):
    key = jax.random.PRNGKey(5)
    g = simple_tree_gp.init(key, target_depth=2, full=True)
    mutated = simple_tree_gp.subtree_mutation(g, key)

    # Check shape is preserved
    for k in ["functions", "terminals", "constants", "tree"]:
        assert mutated["genes"][k].shape == g["genes"][k].shape


def test_point_mutation(simple_tree_gp):
    key = jax.random.PRNGKey(6)
    g = simple_tree_gp.init(key, target_depth=2, full=True)
    mutated = simple_tree_gp.point_mutation(g, key, mutation_rate=0.5)

    # Check shape is preserved
    for k in ["functions", "terminals", "constants", "tree"]:
        assert mutated["genes"][k].shape == g["genes"][k].shape


def test_constants_mutation(simple_tree_gp):
    key = jax.random.PRNGKey(7)
    g = simple_tree_gp.init(key, target_depth=2, full=True)
    mutated = simple_tree_gp.constants_mutation(
        g, key, mutation_rate=0.5, reinit_rate=0.1
    )

    # Check shape is preserved
    assert mutated["genes"]["constants"].shape == g["genes"]["constants"].shape


def test_mutation(simple_tree_gp):
    key = jax.random.PRNGKey(7)
    g = simple_tree_gp.init(key, target_depth=2, full=True)
    mutated = simple_tree_gp.mutate(g, key)

    for k in ["functions", "terminals", "constants", "tree"]:
        assert mutated["genes"][k].shape == g["genes"][k].shape


def test_syntactic_equality(simple_tree_gp):
    key = jax.random.PRNGKey(8)
    g1 = simple_tree_gp.init(key, target_depth=2, full=True)
    g2 = g1.copy()
    assert simple_tree_gp.check_syntactic_equality(g1, g2)


def test_semantic_equality(simple_tree_gp):
    key = jax.random.PRNGKey(9)
    g1 = simple_tree_gp.init(key, target_depth=2, full=True)
    g2 = g1.copy()
    assert simple_tree_gp.check_semantic_equality(g1, g2)


def test_compute_subtree_heights(simple_tree_gp):
    key = jax.random.PRNGKey(11)
    g = simple_tree_gp.init(key, target_depth=2, full=True)
    heights = simple_tree_gp.compute_subtree_heights(g)
    assert heights.shape[0] == simple_tree_gp.n_nodes
    assert jnp.all(heights >= -1)


def test_subtree_mask_includes_root(simple_tree_gp):
    mask = simple_tree_gp.subtree_mask(root_idx=0)
    assert mask[0]
    assert mask.shape[0] == simple_tree_gp.n_nodes


def test_node_depth_and_children(simple_tree_gp):
    n = simple_tree_gp.n_nodes
    depths = jax.jit(jax.vmap(simple_tree_gp.node_depth))(jnp.arange(n))
    children = jax.jit(jax.vmap(simple_tree_gp.children_ids))(jnp.arange(n))
    assert depths.shape[0] == n
    assert children.shape[0] == n


def test_single_node_tree():
    tree = TreeGP(n_inputs=1, max_depth=0, max_arity=2)
    key = jax.random.PRNGKey(13)
    g = tree.init(key, target_depth=0, full=True)
    out = tree.apply(g, jnp.array([1.0]))
    assert isinstance(out, jnp.ndarray)
