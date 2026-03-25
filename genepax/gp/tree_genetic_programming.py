from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax.numpy as jnp
import jax.random
from flax import struct
from jax import random
from jax.lax import fori_loop
from qdax.custom_types import Genotype, RNGKey

from genepax.gp.functions import FunctionSet
from genepax.gp.genetic_programming import GP


def identity(x: Any) -> Any:
    return x


@struct.dataclass
class TreeGP(GP):
    """Tree-based Genetic Programming (TreeGP) representation.

    This class implements a fixed-size tree-based genetic programming genome.
    Nodes can be functions, terminals (inputs), or constants. Supports JAX-friendly
    evaluation, subtree crossover, point mutation, and constants mutation.

    Attributes:
        n_inputs (int): Number of input features.
        min_depth (int): Minimum depth of trees at initialization.
        max_depth (int): Maximum depth of trees.
        max_arity (int): Maximum arity (number of children) for functions.
        function_set (FunctionSet): Allowed set of functions in the tree.
        outputs_wrapper (Callable): Optional wrapper function applied to tree outputs.
        semantic_equality_points (Optional[jnp.ndarray]): Points for semantic equality check.
    """

    n_inputs: int
    min_depth: int = 1
    max_depth: int = 15  # 0 is the root
    max_arity: int = 2
    function_set: FunctionSet = FunctionSet()
    outputs_wrapper: Callable = identity
    semantic_equality_points: Optional[jnp.ndarray] = struct.field(
        pytree_node=False, default=None
    )

    def __init__(
        self,
        n_inputs: int,
        min_depth: int = 1,
        max_depth: int = 15,
        max_arity: int = 2,
        function_set: Optional[FunctionSet] = None,
        outputs_wrapper: Callable[[jnp.ndarray], jnp.ndarray] = identity,
        semantic_equality_points: Optional[jnp.ndarray] = None,
    ):
        """Initializes the TreeGP object.

        Generates a default semantic equality dataset if not provided.

        Args:
            n_inputs (int): Number of input features.
            min_depth (int, optional): Minimum tree depth. Defaults to 1.
            max_depth (int, optional): Maximum tree depth. Defaults to 15.
            max_arity (int, optional): Maximum function arity. Defaults to 2.
            function_set (FunctionSet, optional): Set of functions to use. Defaults to None.
            outputs_wrapper (Callable, optional): Function to wrap outputs. Defaults to None.
            semantic_equality_points (Optional[jnp.ndarray], optional): Points for semantic equality checks.
            Defaults to None.
        """
        _outputs_wrapper = outputs_wrapper
        object.__setattr__(self, "n_inputs", n_inputs)
        object.__setattr__(self, "min_depth", min_depth)
        object.__setattr__(self, "max_depth", max_depth)
        object.__setattr__(self, "max_arity", max_arity)
        object.__setattr__(self, "function_set", function_set or FunctionSet())
        object.__setattr__(self, "outputs_wrapper", _outputs_wrapper)
        if semantic_equality_points is None:
            key = jax.random.PRNGKey(42)
            semantic_equality_points = (
                2.0 * jax.random.uniform(key, (100, n_inputs)) - 1.0
            )
        object.__setattr__(self, "semantic_equality_points", semantic_equality_points)

    @property
    def n_nodes(self) -> int:
        """Compute the total number of nodes in a full tree of `max_depth`.

        Returns:
            int: Total number of nodes in the fixed-size tree.
        """
        return int((self.max_arity ** (self.max_depth + 1) - 1) // (self.max_arity - 1))

    # init methods
    def init(
        self,
        rnd_key: RNGKey,
        target_depth: int,
        full: bool = True,
    ) -> Genotype:
        """Initialize a single tree genotype randomly.

        Can use "full" method or partially filled trees. Assigns function IDs,
        terminals, and constants.

        Args:
            rnd_key (RNGKey): JAX random key.
            target_depth (int): Target depth for tree initialization.
            full (bool, optional): Whether to use full tree initialization. Defaults to True.

        Returns:
            Genotype: A newly initialized tree genotype.
        """
        f_key, t_key, c_key, ops_key = random.split(rnd_key, 4)

        depths = jax.jit(jax.vmap(self.node_depth))(jnp.arange(self.n_nodes))
        active = depths <= target_depth
        is_leaf = depths == target_depth

        sampled = jax.random.randint(ops_key, shape=(self.n_nodes,), minval=1, maxval=4)
        internal_nodes = jnp.where(
            full, jnp.ones(self.n_nodes, dtype=jnp.int32), sampled
        )
        terminal_nodes = jax.random.randint(
            ops_key, shape=(self.n_nodes,), minval=2, maxval=4
        )

        tree = jnp.where(is_leaf, terminal_nodes, internal_nodes)
        tree = jnp.where(active, tree, 0)

        functions_tree = jax.random.randint(
            f_key, shape=(self.n_nodes,), minval=0, maxval=len(self.function_set)
        )

        # tree semantics: 0 empty, 1 functions, 2 terminals, 3 constants
        genotype = {
            "genes": {
                "functions": functions_tree,
                "terminals": jax.random.randint(
                    t_key, shape=(self.n_nodes,), minval=0, maxval=self.n_inputs
                ),
                "constants": random.uniform(
                    key=c_key, shape=(self.n_nodes,), minval=-1, maxval=1
                ),
                "tree": tree,
            },
        }
        return self._clean_genotype(genotype)

    # noinspection PyMethodMayBeStatic
    def size(self, genotype: Genotype) -> jnp.ndarray:
        """Compute the actual size of a tree.

        Args:
            genotype: the tree

        Returns:
            jnp.ndarray: the size of the tree
        """
        return jnp.sum(genotype["genes"]["tree"] > 0)

    def init_ramped_half_and_half(
        self,
        rnd_key: RNGKey,
        pop_size: int,
    ) -> Genotype:
        """Initialize a population using the Ramped Half-and-Half method.

        The first half of the population is initialized using the "full" method,
        the second half uses "grow" method. Tree depths are distributed from min_depth
        to max_depth.

        Args:
            rnd_key (RNGKey): JAX PRNG key.
            pop_size (int): Number of genotypes to generate.

        Returns:
            Genotype: Array of genotypes of shape (pop_size, ...) representing the initial population.
        """
        keys = random.split(rnd_key, pop_size)
        half_pop = jnp.floor(pop_size / 2).astype(jnp.int32)
        full_mask = jnp.arange(pop_size) < half_pop
        depths_range_length = self.max_depth - self.min_depth + 1
        first_half_depths = self.min_depth + (
            jnp.arange(half_pop) % depths_range_length
        )
        second_half_depths = self.min_depth + (
            jnp.arange(pop_size - half_pop) % depths_range_length
        )
        depths = jnp.concatenate([first_half_depths, second_half_depths])
        return jax.jit(jax.vmap(self.init, in_axes=(0, 0, 0)))(keys, depths, full_mask)

    @staticmethod
    def _safe_int_cast(genotype: Genotype) -> Genotype:
        """Casts the int part of the genome for safety."""
        return {
            "genes": {
                "constants": genotype["genes"]["constants"],
                "tree": genotype["genes"]["tree"].astype(int),
                "terminals": genotype["genes"]["terminals"].astype(int),
                "functions": genotype["genes"]["functions"].astype(int),
            },
        }

    def _clean_genotype(self, genotype: Genotype) -> Genotype:
        """Cleans a genotype to enforce valid tree structure.

        All inactive children are set to 0, and child nodes exceeding parent arity
        are masked out.

        Args:
            genotype (Genotype): Input genotype.

        Returns:
            Genotype: Cleaned genotype with valid tree structure.
        """
        genotype = self._safe_int_cast(genotype)
        init_arities = jnp.where(
            genotype["genes"]["tree"] == 1,
            self.function_set.arities[genotype["genes"]["functions"]],
            0,
        )

        def _clean_tree(
            idx: int, carry: Tuple[jnp.ndarray, jnp.ndarray]
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            arities, g_tree = carry
            children = self.children_ids(idx)
            arities_range = jnp.arange(1, self.max_arity + 1)
            children_to_zero_out = jnp.where(
                arities_range > arities[idx], children, self.n_nodes
            )
            arities = arities.at[children_to_zero_out].set(0)
            g_tree = g_tree.at[children_to_zero_out].set(0)
            return arities, g_tree

        updated_arities, updated_tree = fori_loop(
            lower=0,
            upper=self.n_nodes,
            body_fun=_clean_tree,
            init_val=(init_arities, genotype["genes"]["tree"]),
        )

        return {
            **genotype,
            "genes": {
                **genotype["genes"],
                "tree": updated_tree,
            },
        }

    # tree structure utilities
    def children_ids(self, node_idx: int) -> jnp.ndarray:
        """Return indices of the children of a given node in the fixed-size tree.

        Args:
            node_idx (int): Index of the parent node.

        Returns:
            jnp.ndarray: Array of child node indices of length `max_arity`.
        """
        return (self.max_arity * node_idx + 1) + jnp.arange(self.max_arity)

    def node_depth(self, node_idx: int) -> jnp.ndarray:
        """Compute the depth of a given node in the tree.

        Depth of root is 0.

        Args:
            node_idx (int): Index of the node.

        Returns:
            jnp.ndarray: Depth of the node as integer.
        """
        return jnp.floor(
            jnp.log((node_idx * (self.max_arity - 1) + 1)) / jnp.log(self.max_arity)
        ).astype(jnp.int32)

    def compute_parent_id(self, node_idx: int) -> jnp.ndarray:
        """Compute the parent index of a given node. Root has parent -1.

        Args:
            node_idx (int): Node index.

        Returns:
            jnp.ndarray: Parent node index.
        """
        return jnp.where(
            node_idx == 0,
            -1,
            (node_idx - 1) // self.max_arity,
        )

    def subtree_mask(self, root_idx: int) -> jnp.ndarray:
        """Compute boolean mask of nodes belonging to a subtree.

        Args:
            root_idx (int): Root node index of the subtree.

        Returns:
            jnp.ndarray: Boolean array of shape (n_nodes,), True for nodes in the subtree.
        """

        n = self.n_nodes

        # initialize all nodes as NOT in subtree
        init_mask = jnp.zeros((n,), dtype=bool)

        def _update_mask(idx: int, mask: jnp.ndarray) -> jnp.ndarray:
            # root is always in its subtree
            is_root = idx == root_idx

            # parent of current node
            parent = self.compute_parent_id(idx)

            # node is in subtree if:
            # - it is the root, OR
            # - its parent is already marked as in subtree
            in_subtree = jnp.where(
                is_root,
                True,
                jnp.where(parent >= 0, mask[parent], False),
            )

            return mask.at[idx].set(in_subtree)

        # forward pass (parents always come before children)
        return jax.lax.fori_loop(
            0,
            n,
            _update_mask,
            init_mask,
        )

    def compute_subtree_heights(self, genotype: Genotype) -> jnp.ndarray:
        """Compute subtree heights for all nodes.

        height[i] = max distance from node i to an active leaf below it.

        Args:
            genotype (Genotype): Tree genotype.

        Returns:
            jnp.ndarray: Array of subtree heights of shape (n_nodes,).
        """
        # initialize all heights to -1 (inactive)
        init_heights = -jnp.ones((self.n_nodes,), dtype=jnp.int32)
        genotype = self._safe_int_cast(genotype)

        def _update_heights(idx: int, heights: jnp.ndarray) -> jnp.ndarray:
            node = genotype["genes"]["tree"][idx]

            def active_node(_: Any) -> jnp.ndarray:
                # leaf node
                def leaf() -> int:
                    return 0

                # function node
                def function() -> jnp.ndarray:
                    child_heights = heights[self.children_ids(idx)]
                    return 1 + jnp.max(child_heights)

                return jax.lax.cond(
                    node == 1,  # function
                    function,
                    leaf,
                )

            h = jax.lax.cond(
                node == 0,
                lambda _: -1,
                active_node,
                operand=None,
            )

            return heights.at[idx].set(h)

        # bottom-up traversal
        return jax.lax.fori_loop(
            0,
            self.n_nodes,
            lambda i, h: _update_heights(self.n_nodes - 1 - i, h),
            init_heights,
        )

    # checking methods
    # noinspection PyMethodMayBeStatic
    def check_syntactic_equality(
        self, genotype1: Genotype, genotype2: Genotype
    ) -> jnp.ndarray:
        """Check if two genotypes are syntactically identical.

        Compares tree structure, function IDs, terminal assignments, and constants.

        Args:
            genotype1 (Genotype): First genotype.
            genotype2 (Genotype): Second genotype.

        Returns:
            jnp.ndarray: True if the genotypes are syntactically identical, False otherwise.
        """
        genotype1 = self._safe_int_cast(genotype1)
        genotype2 = self._safe_int_cast(genotype2)

        def _equality(idx: int, param: str) -> jnp.ndarray:
            f1 = jnp.where(
                genotype1["genes"]["tree"] == idx, genotype1["genes"][param], 0
            )
            f2 = jnp.where(
                genotype2["genes"]["tree"] == idx, genotype2["genes"][param], 0
            )
            return jnp.allclose(f1, f2)

        return (
            _equality(0, "functions")
            & _equality(1, "terminals")
            & _equality(2, "constants")
        )

    def check_semantic_equality(
        self, genotype1: Genotype, genotype2: Genotype, data_points: jnp.ndarray = None
    ) -> jnp.ndarray:
        """Check if two genotypes are semantically equivalent on given input points.

        Args:
            genotype1 (Genotype): First genotype.
            genotype2 (Genotype): Second genotype.
            data_points (jnp.ndarray, optional): Points to evaluate. Defaults to `semantic_equality_points`.

        Returns:
            jnp.ndarray: True if outputs are element-wise close, False otherwise.
        """
        genotype1 = self._safe_int_cast(genotype1)
        genotype2 = self._safe_int_cast(genotype2)
        data_points = (
            self.semantic_equality_points if data_points is None else data_points
        )
        mapped_apply = jax.jit(jax.vmap(self.apply, in_axes=(None, 0)))
        output1 = mapped_apply(genotype1, data_points)
        output2 = mapped_apply(genotype2, data_points)

        # If shapes match, check elementwise closeness
        return jnp.all(jnp.isclose(output1, output2)) & (output1.shape == output2.shape)

    def apply(
        self,
        genotype: Genotype,
        obs: jnp.ndarray,
        weights: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        """Evaluate a tree genotype on input data in a JAX-friendly way.

        Uses a bottom-up evaluation with a buffer and supports function,
        terminal (input), and constant nodes.

        Args:
            genotype (Genotype): Tree genotype to evaluate.
            obs (jnp.ndarray): Input features array of shape (n_inputs,).
            weights (Optional[Dict[str, jnp.ndarray]]): Tree weights, currently not used.

        Returns:
            jnp.ndarray: Output value computed by the tree.
        """
        genotype = self._safe_int_cast(genotype)
        buffer = jnp.zeros((self.n_nodes,), dtype=jnp.float32)

        # noinspection PyUnusedLocal
        def _eval_empty(idx: int, buff: jnp.ndarray) -> float:
            return 0.0

        # function node
        def _eval_function(idx: int, buff: jnp.ndarray) -> jnp.ndarray:
            fn_id = genotype["genes"]["functions"].at[idx].get()
            children_values = buff.at[self.children_ids(idx)].get()
            return self.function_set.apply(fn_id, *children_values).astype(float)

        # terminal node
        # noinspection PyUnusedLocal
        def _eval_terminal(idx: int, buff: jnp.ndarray) -> jnp.ndarray:
            return obs[genotype["genes"]["terminals"][idx]].astype(float)

        # constant node
        # noinspection PyUnusedLocal
        def _eval_constant(idx: int, buff: jnp.ndarray) -> jnp.ndarray:
            return genotype["genes"]["constants"][idx].astype(float)

        def _eval_body(idx: int, values: jnp.ndarray) -> jnp.ndarray:
            node_value = jax.lax.switch(
                genotype["genes"]["tree"][idx],
                (_eval_empty, _eval_function, _eval_terminal, _eval_constant),
                idx,
                values,
            )
            return values.at[idx].set(node_value)

        buffer = jax.lax.fori_loop(
            0,
            self.n_nodes,
            lambda i, v: _eval_body(self.n_nodes - 1 - i, v),
            buffer,
        )

        return self.outputs_wrapper(jnp.asarray([buffer[0]]))

    def get_readable_expression(
        self,
        genotype: Genotype,
        inputs_mapping: Union[Dict[int, str], Callable[[int], str], None] = None,
    ) -> str:
        """Return a human-readable symbolic expression of the tree.

        Args:
            genotype (Genotype): Tree genotype.
            inputs_mapping: inputs_mapping (dict[int,str] | callable[[int], str], optional):
                Mapping from input indices to custom names.
                - If a dict, keys are input indices
                - If a callable, it is called with the input index and must
                  return the desired string
                Defaults to "x0", "x1", ...

        Returns:
            str: Expression string, recursively representing the tree.
        """
        genotype = self._safe_int_cast(genotype)
        inputs_mapping = inputs_mapping or {}
        if isinstance(inputs_mapping, dict):
            inputs_mapping_fn = lambda idx: inputs_mapping.get(idx, f"x{idx}")
        else:
            inputs_mapping_fn = inputs_mapping
        return "y = " + self._get_readable_expression(genotype, inputs_mapping_fn)

    def _get_readable_expression(
        self,
        genotype: Genotype,
        inputs_mapping_fn: Callable[[int], str],
        node_idx: int = 0,
    ) -> str:
        """Recursive worker for get_readable_expression.

        Args:
            genotype (Genotype): Tree genotype.
            node_idx (int, optional): Current node index. Defaults to 0 (root).

        Returns:
            str: Expression string for the subtree rooted at node_idx.
        """
        node_type = int(genotype["genes"]["tree"][node_idx])

        if node_type == 0:  # inactive node
            return ""
        if node_type == 2:  # terminal feature
            return inputs_mapping_fn(int(genotype["genes"]["terminals"][node_idx]))
        if node_type == 3:  # terminal constant
            return f"{float(genotype['genes']['constants'][node_idx]):.3f}"
        if node_type == 1:  # function
            fn_id = int(genotype["genes"]["functions"][node_idx])
            fn_name = list(self.function_set.function_set.values())[fn_id].symbol
            arity = self.function_set.arities[fn_id]
            children = self.children_ids(node_idx)

            args = []
            for k in range(arity):
                child_idx = int(children[k])
                arg = self._get_readable_expression(
                    genotype, inputs_mapping_fn, child_idx
                )
                args.append(arg)

            return f"{fn_name}(" + ", ".join(args) + ")"

        raise ValueError(f"Unknown node type {node_type}")

    def crossover(
        self,
        genotype1: Genotype,
        genotype2: Genotype,
        rnd_key: RNGKey,
    ) -> Genotype:
        """Perform subtree crossover between two tree genotypes.

        Randomly selects a crossover point in genotype1 and exchanges the subtree
        from genotype2 that fits depth constraints.

        Args:
            genotype1 (Genotype): First parent genotype.
            genotype2 (Genotype): Second parent genotype.
            rnd_key (RNGKey): JAX PRNG key.

        Returns:
            Genotype: Offspring genotype after crossover.
        """
        genotype1 = self._safe_int_cast(genotype1)
        genotype2 = self._safe_int_cast(genotype2)
        point_key_1, point_key_2 = jax.random.split(rnd_key, 2)
        admissible_xover_point1 = genotype1["genes"]["tree"] > 0
        sampling_probs1 = admissible_xover_point1 / jnp.sum(admissible_xover_point1)
        xover_point1 = jax.random.choice(point_key_1, self.n_nodes, p=sampling_probs1)
        max_height_xover_point2 = self.max_depth - self.node_depth(xover_point1)
        tree2_heights = self.compute_subtree_heights(genotype2)
        admissible_xover_point2 = jnp.logical_and(
            tree2_heights >= 0, tree2_heights <= max_height_xover_point2
        )
        sampling_probs2 = (admissible_xover_point2 > 0) / jnp.sum(
            admissible_xover_point2 > 0
        )
        xover_point2 = jax.random.choice(point_key_2, self.n_nodes, p=sampling_probs2)
        subtree_height = tree2_heights[xover_point2]

        depths = jax.jit(jax.vmap(self.node_depth))(jnp.arange(self.n_nodes))

        subtree_mask1 = self.subtree_mask(xover_point1)
        height_mask1 = depths <= (depths[xover_point1] + subtree_height)
        mask1 = jnp.logical_and(subtree_mask1, height_mask1)

        subtree_mask2 = self.subtree_mask(xover_point2)
        height_mask2 = depths <= (depths[xover_point2] + subtree_height)
        mask2 = jnp.logical_and(subtree_mask2, height_mask2)

        idx1 = jnp.where(mask1, size=mask1.shape[0], fill_value=self.n_nodes)[0]
        idx2 = jnp.where(mask2, size=mask2.shape[0], fill_value=self.n_nodes)[0]
        offspring = jax.tree.map(
            lambda x1, x2: x1.at[idx1].set(x2[idx2]),
            genotype1,
            genotype2,
        )

        return self._clean_genotype(offspring)

    def mutate(
        self,
        genotype: Genotype,
        rnd_key: RNGKey,
        p_subtree: float = 0.6,
        p_point: float = 0.2,
        p_constants: float = 0.2,
        mutation_probabilities: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> Genotype:
        """
        Apply exactly one mutation operator to a genotype, chosen stochastically
        among subtree mutation, point mutation, and constants mutation.

        Args:
            genotype: Parent genotype to mutate.
            rnd_key: JAX PRNG key.
            p_subtree: Probability of applying subtree mutation.
            p_point: Probability of applying point mutation.
            p_constants: Probability of applying constants mutation.
            mutation_probabilities: optional dictionary mapping genotype parts
                 to their mutation probabilities.

        Returns:
            Mutated genotype.
        """
        mutation_probabilities = mutation_probabilities or {}
        p_subtree = mutation_probabilities.get("subtree", p_subtree)
        p_point = mutation_probabilities.get("point", p_point)
        p_constants = mutation_probabilities.get("constants", p_constants)

        genotype = self._safe_int_cast(genotype)
        probs = jnp.array([p_subtree, p_point, p_constants], dtype=jnp.float32)
        probs = probs / jnp.sum(probs)
        choice_key, mut_key = jax.random.split(rnd_key)
        mutation_id = jax.random.choice(choice_key, a=3, p=probs)

        return jax.lax.switch(
            mutation_id,
            (self.subtree_mutation, self.point_mutation, self.constants_mutation),
            genotype,
            mut_key,
        )

    def subtree_mutation(self, genotype: Genotype, rnd_key: RNGKey) -> Genotype:
        """Perform subtree mutation.

        Randomly generates a donor tree and performs a subtree crossover to mutate.

        Args:
            genotype (Genotype): Original genotype.
            rnd_key (RNGKey): JAX PRNG key.

        Returns:
            Genotype: Mutated genotype.
        """
        genotype = self._safe_int_cast(genotype)
        xover_key, depth_key, donor_key = jax.random.split(rnd_key, 3)
        depth = jax.random.randint(
            depth_key, shape=(), minval=1, maxval=self.max_depth + 1
        )
        donor = self.init(donor_key, depth, full=False)
        return self.crossover(genotype, donor, xover_key)

    # noinspection PyMethodMayBeStatic
    def constants_mutation(
        self,
        genotype: Genotype,
        rnd_key: RNGKey,
        mutation_rate: float = 0.05,
        reinit_rate: float = 0.005,
        gaussian_sigma: float = 0.1,
    ) -> Genotype:
        """Mutate constants in a tree genotype.

        Adds Gaussian noise to some constants and randomly reinitializes a few constants.

        Args:
            genotype (Genotype): Original genotype.
            rnd_key (RNGKey): JAX PRNG key.
            mutation_rate (float, optional): Probability of adding Gaussian noise. Defaults to 0.05.
            reinit_rate (float, optional): Probability of reinitializing constant. Defaults to 0.005.
            gaussian_sigma (float, optional): Standard deviation of Gaussian noise. Defaults to 0.1.

        Returns:
            Genotype: Genotype with mutated constants.
        """
        genotype = self._safe_int_cast(genotype)
        points_noise_key, noise_key, points_reinit_key, reinit_key = jax.random.split(
            rnd_key, 4
        )
        constants_noise = gaussian_sigma * jax.random.normal(
            noise_key, shape=genotype["genes"]["constants"].shape
        )
        noisy_constant = genotype["genes"]["constants"] + constants_noise
        noising_constants_mask = (
            random.uniform(points_noise_key, shape=noisy_constant.shape) < mutation_rate
        )
        mutated_constants = jnp.where(
            noising_constants_mask, noisy_constant, genotype["genes"]["constants"]
        )

        # re-init some constants
        reinit_constants = jax.random.uniform(
            reinit_key, shape=mutated_constants.shape, minval=-1.0, maxval=1.0
        )
        reinit_constants_mask = (
            random.uniform(points_reinit_key, shape=reinit_constants.shape)
            < reinit_rate
        )
        mutated_constants = jnp.where(
            reinit_constants_mask, reinit_constants, mutated_constants
        )

        return {
            **genotype,
            "genes": {
                **genotype["genes"],
                "constants": mutated_constants,
            },
        }

    def point_mutation(
        self,
        genotype: Genotype,
        rnd_key: RNGKey,
        mutation_rate: float = 0.5,
        gaussian_sigma: float = 0.1,
    ) -> Genotype:
        """Perform point mutation on active nodes of a tree genotype.

        - Functions are replaced with other functions of same arity.
        - Terminal nodes may swap type or change terminal index.
        - Constants are mutated with Gaussian noise.

        Args:
            genotype (Genotype): Original genotype.
            rnd_key (RNGKey): JAX PRNG key.
            mutation_rate (float, optional): Probability to mutate each node. Defaults to 0.5.
            gaussian_sigma (float, optional): Standard deviation for constant mutation. Defaults to 0.1.

        Returns:
            Genotype: Mutated genotype.
        """
        genotype = self._safe_int_cast(genotype)
        points_key, loop_key = jax.random.split(rnd_key, 2)
        mutation_mask = jnp.logical_and(
            random.uniform(points_key, shape=(self.n_nodes,)) < mutation_rate,
            genotype["genes"]["tree"] > 0,
        )
        mutations_target = genotype["genes"]["tree"] * mutation_mask

        # noinspection PyUnusedLocal
        def _ignore(idx: int, genome: Genotype, key: RNGKey) -> Genotype:
            return genome

        # function node
        def _mutate_function(idx: int, genome: Genotype, key: RNGKey) -> Genotype:
            previous_function = genome["genes"]["functions"][idx]
            previous_arity = self.function_set.arities[previous_function]
            candidate_functions_mask = self.function_set.arities == previous_arity
            candidate_functions_mask = candidate_functions_mask.at[
                previous_function
            ].set(False)
            candidate_functions_probs = candidate_functions_mask / jnp.sum(
                candidate_functions_mask
            )
            new_function_id = jax.random.choice(
                key, candidate_functions_probs.shape[0], p=candidate_functions_probs
            )
            updated_functions = (
                genome["genes"]["functions"].at[idx].set(new_function_id)
            )
            return {
                **genome,
                "genes": {
                    **genome["genes"],
                    "functions": updated_functions,
                },
            }

        # leaf node (terminal or constant)
        def _mutate_leaf(idx: int, genome: Genotype, key: RNGKey) -> Genotype:
            flip_key, t_key, noise_key = jax.random.split(key, 3)
            node_type = genome["genes"]["tree"][idx]
            # swap leaf type with probability 1/2
            updated_node_type = jnp.where(
                jax.random.bernoulli(flip_key, 0.5), 5 - node_type, node_type
            )
            updated_tree = genome["genes"]["tree"].at[idx].set(updated_node_type)
            # change terminal in either case
            new_terminal = jax.random.randint(
                t_key, shape=(), minval=0, maxval=self.n_inputs
            )
            updated_terminals = genome["genes"]["terminals"].at[idx].set(new_terminal)
            # add gaussian noise to constant in any case
            mutated_constant = genome["genes"]["constants"][
                idx
            ] + gaussian_sigma * jax.random.normal(noise_key)
            updated_constants = (
                genome["genes"]["constants"].at[idx].set(mutated_constant)
            )
            return {
                **genome,
                "genes": {
                    **genome["genes"],
                    "tree": updated_tree,
                    "terminals": updated_terminals,
                    "constants": updated_constants,
                },
            }

        def _mutation_body(
            idx: int, carry: Tuple[jnp.ndarray, Genotype, RNGKey]
        ) -> Tuple[jnp.ndarray, Genotype, RNGKey]:
            mutation_tree, mutated_genotype, inner_key = carry
            mut_key, inner_key = jax.random.split(inner_key)
            mutated_genotype = jax.lax.switch(
                mutation_tree[idx],
                (_ignore, _mutate_function, _mutate_leaf),
                idx,
                mutated_genotype,
                mut_key,
            )
            return mutation_tree, mutated_genotype, inner_key

        return jax.lax.fori_loop(
            0,
            self.n_nodes,
            lambda i, v: _mutation_body(i, v),
            (mutations_target, genotype, loop_key),
        )[1]
