"""Core components of Cartesian Genetic Programming (CGP) for graph evolution."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax import jit, random
from jax.lax import fori_loop
from qdax.custom_types import Genotype, Mask, RNGKey

from genepax.gp.graph_genetic_programming import GGP, _mutate_subgenome


@struct.dataclass
class CGP(GGP):
    """Cartesian Genetic Programming (CGP).

    Uses a fixed-length integer genome describing a directed acyclic graph
    of computational nodes arranged in a row (1D grid).

    Extra Args:
        n_nodes: number of computational nodes in the graph.
        fixed_outputs: whether output nodes are fixed (last nodes) or can be evolved.
    """

    n_nodes: int = 50
    fixed_outputs: bool = False

    @property
    def buffer_size(self) -> int:
        """Size of the computation buffer used by CGP."""
        return self.n_inputs + self.n_input_constants + self.n_nodes

    @property
    def n_functions(self) -> int:
        """Max number of functions that can be performed by CGP."""
        return self.n_nodes

    def init(
        self,
        rnd_key: RNGKey,
        *args: Any,
    ) -> Genotype:
        """Initializes a random CGP genome.

        Args:
            rnd_key: JAX PRNG key used to generate random genome values.
            *args: Unused additional arguments for API compatibility.

        Returns:
            A dictionary containing the `"genes"` key with genome sections as
            integer JAX arrays:
                - `"inputs1"`
                - `"inputs2"`
                - `"functions"`
                - `"outputs"`
            and the `"weights`" key with weights sections as floating point JAX arrays:
                - `"inputs1"`
                - `"inputs2"`
                - `"functions"`
                - `"program_inputs"`
                - `"inputs1_biases"`
                - `"inputs2_biases"`
                - `"functions_biases"`
            The encoding is inspired by that of MLPs.
        """
        # determine bounds for genes for each section of the genome
        in_mask = jnp.arange(
            self.n_inputs + self.n_input_constants,
            self.n_inputs + self.n_input_constants + self.n_nodes,
        )
        f_mask = len(self.function_set) * jnp.ones(self.n_nodes)
        out_mask = (self.n_inputs + self.n_input_constants + self.n_nodes) * jnp.ones(
            self.n_outputs
        )

        # generate the random float values for each section of the genome
        n_key, out_key, weights_key = random.split(rnd_key, 3)
        random_n = random.uniform(key=n_key, shape=(self.n_nodes * 3,))
        random_x, random_y, random_f = jnp.split(random_n, 3)
        random_out = random.uniform(key=out_key, shape=out_mask.shape)

        # rescale, cast to integer and store the random genome parts
        return {
            "genes": {
                "inputs1": jnp.floor(random_x * in_mask).astype(int),
                "inputs2": jnp.floor(random_y * in_mask).astype(int),
                "functions": jnp.floor(random_f * f_mask).astype(int),
                "outputs": (
                    out_mask
                    if self.fixed_outputs
                    else jnp.floor(random_out * out_mask).astype(int)
                ),
            },
            "weights": self.init_weights(weights_key),
        }

    def apply(
        self,
        genotype: Genotype,
        obs: jnp.ndarray,
        weights: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        """Evaluates a CGP genome on a given input observation.

        This method interprets the integer-encoded genome to construct and
        execute the corresponding computational graph. Node values are computed
        sequentially and stored in a buffer, starting from the provided inputs and constants.

        Args:
            genotype: dictionary of CGP genome parameters.
            weights: dictionary of weights for nodes and/or connections,
                defaults to the CGP weights (or 1 if not weighted).
            obs: problem inputs/observation.

        Returns:
            Array of processed outputs after evaluating the genome and applying
            the output wrapper.
        """

        weights_dict = weights or {}
        weights_dict = {**genotype["weights"], **weights_dict}
        genotype = {
            "weights": weights_dict,
            "genes": jax.tree.map(lambda x: x.astype(int), genotype["genes"]),
        }

        # define function to update buffer in a certain position: get inputs from the x and y connections
        # then apply the function
        @jit
        def _update_buffer(
            buffer_idx: int, carry: Tuple[Genotype, jnp.ndarray]
        ) -> Tuple[Genotype, jnp.ndarray]:
            cgp_genes, buff = carry
            n_in = len(buff) - len(cgp_genes["genes"]["inputs1"])
            idx = buffer_idx - n_in
            return self._update_memory(cgp_genes, weights_dict, buff, idx, buffer_idx)

        # initialize the buffer with inputs and constants and use zeros as placeholders for computation
        input_constants = genotype["weights"]["program_inputs"]
        buffer = jnp.concatenate([obs, input_constants, jnp.zeros(self.n_nodes)])
        # apply the buffer update function for all positions of the buffer to update it
        _, buffer = fori_loop(
            lower=self.n_inputs + len(input_constants),
            upper=len(buffer),
            body_fun=_update_buffer,
            init_val=(genotype, buffer),
        )
        outputs = jnp.take(buffer, genotype["genes"]["outputs"])

        # apply wrapper to constraint the outputs in the correct domain
        return self.outputs_wrapper(outputs)

    def compute_active_mask(
        self,
        genotype: Genotype,
    ) -> Mask:
        """
        Compute the mask of active (expressed) nodes in a CGP genome.
        This method identifies which nodes are active by starting from the output
        connections and recursively marking all nodes that contribute to them.

        Args:
            genotype: the CGP genome parameters.

        Returns:
            Mask: a binary mask (1 = active, 0 = inactive) of length `n_nodes`,
            indicating which nodes are used in producing the final outputs.
        """
        genotype = {
            "weights": genotype["weights"],
            "genes": jax.tree.map(lambda x: x.astype(int), genotype["genes"]),
        }

        active_buffer = jnp.zeros(self.buffer_size)
        active_buffer = active_buffer.at[genotype["genes"]["outputs"]].set(1)

        # define function to mark if a buffer is active in a certain position
        def _compute_active_nodes(
            opposite_idx: int,
            carry: Tuple[Genotype, Mask],
        ) -> Tuple[Genotype, Mask]:
            cgp_genes, active = carry
            n_in = len(active) - len(cgp_genes["genes"]["inputs1"])
            idx = len(active) - opposite_idx - 1
            x_idx = cgp_genes["genes"]["inputs1"].at[idx - n_in].get()
            y_idx = cgp_genes["genes"]["inputs2"].at[idx - n_in].get()
            arity = self.function_set.arities.at[
                cgp_genes["genes"]["functions"][idx - n_in]
            ].get()
            active = active.at[x_idx].set(
                jnp.logical_or(active.at[x_idx].get(), active.at[idx].get())
            )
            active = active.at[y_idx].set(
                jnp.logical_or(
                    active.at[y_idx].get(),
                    jnp.logical_and(active.at[idx].get(), arity == 2),
                )
            )
            return cgp_genes, active

        _, active_buffer = fori_loop(
            lower=0,
            upper=self.n_nodes,
            body_fun=_compute_active_nodes,
            init_val=(genotype, active_buffer),
        )
        return active_buffer[-self.n_nodes :].astype(int)

    def mutate(
        self,
        genotype: Genotype,
        rnd_key: RNGKey,
        *,
        p_mut_inputs: float = 0.1,
        p_mut_functions: float = 0.1,
        p_mut_outputs: float = 0.3,
        weights_mut_sigma: float = 0.1,
        mutation_probabilities: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> Genotype:
        """Mutates a CGP genome using int-flip mutation. If the genome is weighted, the weights
        are mutated with Gaussian mutation.

        This mutation is implemented as a form of crossover with a newly
        generated "donor" genome: for each gene, the value is taken from the
        donor with a low probability, otherwise kept from the original genome.
        This ensures that all mutated genes remain valid (i.e., within the
        correct index ranges for their respective genome section).

        The function is compatible with standard emitters when wrapped using
        `functools.partial`.

        Mutation probabilities and sigma can be specified either via individual arguments or by
        passing a dictionary to `mutation_probabilities`, the dictionary values override
        the individual arguments.

        Args:
            genotype: the CGP genome parameters to mutate.
            rnd_key: JAX PRNG key for randomness.
            p_mut_inputs: probability of mutating each input connection gene
                (ignored if overridden via `mutation_probabilities`).
            p_mut_functions: probability of mutating each function gene
                (ignored if overridden via `mutation_probabilities`).
            p_mut_outputs: probability of mutating each output connection gene
                (ignored if overridden via `mutation_probabilities`).
            weights_mut_sigma: mutation step for weights Gaussian mutation
                (ignored if overridden via `mutation_probabilities`).
            mutation_probabilities: optional dictionary mapping `"inputs"`,
                `"functions"`, `"outputs"`, and `"weights_sigma"` to their mutation probabilities.

        Returns:
            The mutated genome.
        """
        mutation_probabilities = mutation_probabilities or {}
        out_key, super_key = random.split(rnd_key, 2)
        temporary_genotype, donor_genotype = super()._mutate(
            genotype,
            super_key,
            p_mut_inputs,
            p_mut_functions,
            weights_mut_sigma,
            mutation_probabilities,
        )
        p_mut_outputs = mutation_probabilities.get("outputs", p_mut_outputs)
        return {
            "genes": {
                "inputs1": temporary_genotype["genes"]["inputs1"],
                "inputs2": temporary_genotype["genes"]["inputs2"],
                "functions": temporary_genotype["genes"]["functions"],
                "outputs": _mutate_subgenome(
                    genotype["genes"]["outputs"],
                    donor_genotype["genes"]["outputs"],
                    out_key,
                    p_mut_outputs,
                ),
            },
            "weights": temporary_genotype["weights"],
        }

    def _get_readable_expression(
        self,
        genotype: Genotype,
        inputs_mapping_fn: Callable[[int], str],
        outputs_mapping_fn: Callable[[int], str],
    ) -> List[str]:
        """Worker class for computing the readable symbolic representation of a CGP genotype."""
        n_in = self.n_inputs + self.n_input_constants
        targets = []
        input_constants = genotype["weights"]["program_inputs"]

        def _replace_cgp_expression(cgp_genes: Genotype, idx: int) -> str:
            if idx < self.n_inputs:
                return inputs_mapping_fn(int(idx))
            elif idx < n_in:
                return str(input_constants[idx - self.n_inputs])
            functions = list(self.function_set.function_set.values())
            gene_idx = idx - n_in
            function = functions[cgp_genes["genes"]["functions"][gene_idx]]
            node_weight, x_weight, y_weight = self._weights_representations(
                cgp_genes, gene_idx
            )
            node_bias, x_bias, y_bias = self._biases_representations(
                cgp_genes, gene_idx
            )
            n_p1, n_p2 = ("(", ")") if self.biased_functions else ("", "")
            i_p1, i_p2 = ("(", ")") if self.biased_inputs else ("", "")
            if function.arity == 1:
                return (
                    f"{n_p1}{node_weight}{function.symbol}({x_weight}"
                    f"{_replace_cgp_expression(cgp_genes, int(cgp_genes['genes']['inputs1'][gene_idx]))}{x_bias})"
                    f"{node_bias}{n_p2}"
                )
            else:
                return (
                    f"{n_p1}{node_weight}({i_p1}{x_weight}"
                    f"{_replace_cgp_expression(cgp_genes, int(cgp_genes['genes']['inputs1'][gene_idx]))}"
                    f"{x_bias}{i_p2}{function.symbol}{i_p1}{y_weight}"
                    f"{_replace_cgp_expression(cgp_genes, int(cgp_genes['genes']['inputs2'][gene_idx]))}"
                    f"{y_bias}{i_p2}){node_bias}{n_p2}"
                )

        for i, out in enumerate(genotype["genes"]["outputs"].tolist()):
            targets.append(
                f"{outputs_mapping_fn(int(i))} = {self.outputs_wrapper.__name__}({_replace_cgp_expression(genotype, out)})"
            )

        return targets
