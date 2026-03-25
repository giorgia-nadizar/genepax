"""Core components of Linear Genetic Programming (LGP) for graph evolution."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
import jax.random
from flax import struct
from jax import jit, random
from jax.lax import fori_loop
from qdax.custom_types import Genotype, Mask, RNGKey

from genepax.gp.graph_genetic_programming import GGP, _mutate_subgenome


@struct.dataclass
class LGP(GGP):
    """Linear Genetic Programming (LGP).

    Uses a sequence of program lines (instructions) that operate on registers.
    The program executes sequentially, line by line, with later instructions
    potentially overwriting earlier results.

    Extra Args:
        n_computation_registers: number of internal registers for intermediate computations.
        n_program_lines: number of instructions in the program.
    """

    n_computation_registers: int = 5
    n_program_lines: int = 15

    @property
    def n_registers(self) -> int:
        """Total number of registers used by LGP."""
        return self.n_inputs + self.n_input_constants + self.n_assignable_registers

    @property
    def n_assignable_registers(self) -> int:
        """Number of registers that can be assigned by LGP."""
        return self.n_computation_registers + self.n_outputs

    @property
    def n_functions(self) -> int:
        """Max number of functions that can be performed by LGP."""
        return self.n_program_lines

    def init(
        self,
        rnd_key: RNGKey,
        *args: Any,
    ) -> Genotype:
        """Initializes a random LGP genome.

        Args:
            rnd_key: JAX PRNG key used to generate random genome values.
            *args: Unused additional arguments for API compatibility.

        Returns:
            A dictionary containing the `"genes"` key with genome sections as
            integer JAX arrays:
                - `"targets"`
                - `"x_arguments"`
                - `"y_arguments"`
                - `"functions"`
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
        lhs_mask = self.n_assignable_registers * jnp.ones(self.n_program_lines)
        lhs_offset = (self.n_inputs + self.n_input_constants) * jnp.ones(
            self.n_program_lines
        )
        f_mask = len(self.function_set) * jnp.ones(self.n_program_lines)
        rhs_mask = self.n_registers * jnp.ones(self.n_program_lines)

        # generate the random float values for each section of the genome
        lines_key, weights_key = random.split(rnd_key, 2)
        random_lines = random.uniform(key=lines_key, shape=(self.n_program_lines * 4,))
        random_targets, random_x, random_y, random_f = jnp.split(random_lines, 4)

        # rescale, cast to integer and store the random genome parts
        return {
            "genes": {
                "targets": (jnp.floor(random_targets * lhs_mask) + lhs_offset).astype(
                    int
                ),
                "inputs1": jnp.floor(random_x * rhs_mask).astype(int),
                "inputs2": jnp.floor(random_y * rhs_mask).astype(int),
                "functions": jnp.floor(random_f * f_mask).astype(int),
            },
            "weights": self.init_weights(weights_key),
        }

    def apply(
        self,
        genotype: Genotype,
        obs: jnp.ndarray,
        weights: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        """Evaluates a LGP genome on a given input observation.

        This method interprets the integer-encoded genome to construct and
        execute the corresponding program. Program lines results are computed
        sequentially and stored in the registers, starting from the provided inputs and constants.

        Args:
            genotype: dictionary of LGP genome parameters.
            weights: dictionary of weights for nodes and/or connections,
                defaults to the CGP weights (or 1 if not weighted).
            obs: problem inputs/observation.

        Returns:
            Array of processed outputs after evaluating the genome and applying
            the output wrapper.
        """

        # take provided weights and replace with genome ones if missing
        weights_dict = weights or {}
        weights_dict = {**genotype["weights"], **weights_dict}
        genotype = {
            "weights": weights_dict,
            "genes": jax.tree.map(lambda x: x.astype(int), genotype["genes"]),
        }

        input_constants = genotype["weights"]["program_inputs"]

        # define function to update the registers following the instructions of a
        # given program line: get inputs from the x and y connections, then apply the function
        # and store the result in the target register
        @jit
        def _update_registers(
            line_idx: int, carry: Tuple[Genotype, jnp.ndarray]
        ) -> Tuple[Genotype, jnp.ndarray]:
            lgp_genes, regs = carry
            target_register_idx = lgp_genes["genes"]["targets"].at[line_idx].get()
            return self._update_memory(
                lgp_genes, weights_dict, regs, line_idx, target_register_idx
            )

        # initialize the registers with inputs and constants and zeros for remaining registers
        registers = jnp.concatenate(
            [obs, input_constants, jnp.zeros(self.n_assignable_registers)]
        )
        # apply the registers update function for all program lines
        _, registers = fori_loop(
            lower=0,
            upper=self.n_program_lines,
            body_fun=_update_registers,
            init_val=(genotype, registers),
        )
        outputs = registers[-self.n_outputs :]

        # apply wrapper to constraint the outputs in the correct domain
        return self.outputs_wrapper(outputs)

    def compute_active_mask(
        self,
        genotype: Genotype,
    ) -> Mask:
        """
        Compute the mask of active (expressed) program lines in a LGP genome.
        This method identifies which lines are active by starting from the output
        registers and recursively marking all lines that contribute to them.

        Args:
            genotype: the CGP genome parameters.

        Returns:
            Mask: a binary mask (1 = active, 0 = inactive) of length `n_program_lines`,
            indicating which lines are used in producing the final outputs.
        """
        genotype = {
            "weights": genotype["weights"],
            "genes": jax.tree.map(lambda x: x.astype(int), genotype["genes"]),
        }

        active_lines = jnp.zeros(self.n_program_lines)
        registers_mask = jnp.where(
            jnp.arange(self.n_registers) >= (self.n_registers - self.n_outputs), 1, 0
        )

        # define function to mark if a line is active
        def _compute_active_lines(
            opposite_idx: int, carry: Tuple[Genotype, Mask, Mask]
        ) -> Tuple[Genotype, Mask, Mask]:
            lgp_genes, active, regs_mask = carry
            line_idx = len(active) - opposite_idx - 1
            line_use = regs_mask.at[
                lgp_genes["genes"]["targets"].at[line_idx].get()
            ].get()
            active = active.at[line_idx].set(line_use)

            x_reg = lgp_genes["genes"]["inputs1"].at[line_idx].get()
            y_reg = lgp_genes["genes"]["inputs2"].at[line_idx].get()
            arity = self.function_set.arities.at[
                lgp_genes["genes"]["functions"][line_idx]
            ].get()
            regs_mask = regs_mask.at[line_idx].set(0)
            regs_mask = regs_mask.at[x_reg].set(
                jnp.logical_or(line_use, regs_mask.at[x_reg].get())
            )
            regs_mask = regs_mask.at[y_reg].set(
                jnp.logical_or(
                    regs_mask.at[y_reg].get(), jnp.logical_and(line_use, arity == 2)
                )
            )

            return lgp_genes, active, regs_mask

        _, active_lines, _ = fori_loop(
            lower=0,
            upper=self.n_program_lines,
            body_fun=_compute_active_lines,
            init_val=(genotype, active_lines, registers_mask),
        )
        return active_lines.astype(int)

    def mutate(
        self,
        genotype: Genotype,
        rnd_key: RNGKey,
        *,
        p_mut_targets: float = 0.3,
        p_mut_inputs: float = 0.1,
        p_mut_functions: float = 0.1,
        weights_mut_sigma: float = 0.1,
        mutation_probabilities: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> Genotype:
        """Mutates a LGP genome using int-flip mutation.

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
            p_mut_targets: probability of mutating each target assignment gene
                (ignored if overridden via `mutation_probabilities`).
            p_mut_inputs: probability of mutating each input connection gene
                (ignored if overridden via `mutation_probabilities`).
            p_mut_functions: probability of mutating each function gene
                (ignored if overridden via `mutation_probabilities`).
            weights_mut_sigma: mutation step for weights Gaussian mutation
                (ignored if overridden via `mutation_probabilities`).
            mutation_probabilities: optional dictionary mapping `"inputs"`,
                `"functions"`, `"targets"`, and `"weights_sigma"` to their mutation probabilities.

        Returns:
            The mutated genome.
        """

        mutation_probabilities = mutation_probabilities or {}
        targets_key, super_key = random.split(rnd_key, 2)
        temporary_genotype, donor_genotype = super()._mutate(
            genotype,
            super_key,
            p_mut_inputs,
            p_mut_functions,
            weights_mut_sigma,
            mutation_probabilities,
        )
        p_mut_targets = mutation_probabilities.get("targets", p_mut_targets)

        return {
            "genes": {
                "targets": _mutate_subgenome(
                    genotype["genes"]["targets"],
                    donor_genotype["genes"]["targets"],
                    targets_key,
                    p_mut_targets,
                ),
                "inputs1": temporary_genotype["genes"]["inputs1"],
                "inputs2": temporary_genotype["genes"]["inputs2"],
                "functions": temporary_genotype["genes"]["functions"],
            },
            "weights": temporary_genotype["weights"],
        }

    def crossover(
        self,
        genotype1: Genotype,
        genotype2: Genotype,
        rnd_key: RNGKey,
    ) -> Genotype:
        """Performs one-point crossover between two LGP genomes, weights included.

        A crossover point is chosen uniformly at random among the program lines.
        Genes before the crossover point are inherited from the first parent,
        while genes from the crossover point onward are inherited from the
        second parent. This is applied consistently across all genome sections.

        The operation produces a valid LGP genome of the same structure as the
        parents.

        Args:
            genotype1: first LGP parent genome.
            genotype2: second LGP parent genome.
            rnd_key: JAX PRNG key for randomness.

        Returns:
            Genotype: the offspring genome created by crossover.
        """

        cross_idx = jax.random.randint(rnd_key, (1,), 0, self.n_program_lines) + 1
        genes_ids = jnp.arange(self.n_program_lines)
        mask = genes_ids < cross_idx
        # crossover each sub-part of the genome
        return {
            "genes": {
                "targets": jnp.where(
                    mask, genotype1["genes"]["targets"], genotype2["genes"]["targets"]
                ),
                "inputs1": jnp.where(
                    mask, genotype1["genes"]["inputs1"], genotype2["genes"]["inputs1"]
                ),
                "inputs2": jnp.where(
                    mask, genotype1["genes"]["inputs2"], genotype2["genes"]["inputs2"]
                ),
                "functions": jnp.where(
                    mask,
                    genotype1["genes"]["functions"],
                    genotype2["genes"]["functions"],
                ),
            },
            "weights": {
                "program_inputs": genotype1["weights"]["program_inputs"],
                "inputs1": jnp.where(
                    mask,
                    genotype1["weights"]["inputs1"],
                    genotype2["weights"]["inputs1"],
                ),
                "inputs2": jnp.where(
                    mask,
                    genotype1["weights"]["inputs2"],
                    genotype2["weights"]["inputs2"],
                ),
                "functions": jnp.where(
                    mask,
                    genotype1["weights"]["functions"],
                    genotype2["weights"]["functions"],
                ),
                "inputs1_biases": jnp.where(
                    mask,
                    genotype1["weights"]["inputs1_biases"],
                    genotype2["weights"]["inputs1_biases"],
                ),
                "inputs2_biases": jnp.where(
                    mask,
                    genotype1["weights"]["inputs2_biases"],
                    genotype2["weights"]["inputs2_biases"],
                ),
                "functions_biases": jnp.where(
                    mask,
                    genotype1["weights"]["functions_biases"],
                    genotype2["weights"]["functions_biases"],
                ),
            },
        }

    def get_readable_program(self, genotype: Genotype) -> str:
        """Generate a human-readable Python-like representation of an LGP program.

        The LGP genome is unrolled into a sequence of instructions that
        operate on registers. Inputs are first copied into registers, then
        program lines are expanded into assignment statements. Only active
        lines (contributing to the outputs) are included.

        Unary functions are printed in the form:
            r[target] = f(r[x])
        Binary functions are printed in the form:
            r[target] = r[x] op r[y]
        where `op` is the function symbol (e.g., `+`, `*`, `sin`).

        The outputs are read from the last `n_outputs` registers.

        Args:
            genotype: LGP genotype.

        Returns:
            str: A string representing the program as a Python function,
            showing register initialization, executed instructions, and
            the final outputs.

        Example:
            def program(inputs):
                r[[0, 1]] = inputs
                r[[2]] = [0.1]
                r[3] = r[0] + r[2]
                r[4] = tanh(r[3])
                outputs = r[[3, 4]]
                return outputs
        """
        # header and inputs copy into registers
        input_constants = genotype["weights"]["program_inputs"]
        program_lines = [
            "def program(inputs):",
            f"r[{list(range(self.n_inputs))}] = inputs",
            f"r[{list(range(self.n_inputs, self.n_inputs + self.n_input_constants))}] = {input_constants}",
        ]

        functions = list(self.function_set.function_set.values())
        active_lines = self.compute_active_mask(genotype)

        # execution
        for line_idx in range(self.n_program_lines):
            if active_lines[line_idx]:
                function = functions[genotype["genes"]["functions"][line_idx]]
                line_weight, x_weight, y_weight = self._weights_representations(
                    genotype, line_idx
                )
                line_bias, x_bias, y_bias = self._biases_representations(
                    genotype, line_idx
                )
                target_reg = genotype["genes"]["targets"][line_idx]
                x_reg = genotype["genes"]["inputs1"][line_idx]
                y_reg = genotype["genes"]["inputs2"][line_idx]
                i_p1, i_p2 = ("(", ")") if self.biased_inputs else ("", "")
                if function.arity > 1:
                    program_lines.append(
                        f"r[{target_reg}] = {line_weight}({i_p1}{x_weight}r[{x_reg}]{x_bias}{i_p2} "
                        f"{function.symbol} "
                        f"{i_p2}{y_weight}r[{y_reg}]{y_bias}{i_p2}){line_bias}"
                    )
                else:
                    program_lines.append(
                        f"r[{target_reg}] = {line_weight}{function.symbol}"
                        f"({x_weight}r[{x_reg}]{x_bias}){line_bias}"
                    )

        # output selection
        program_lines.append(
            f"outputs = r[{list(range(self.n_registers - self.n_outputs, self.n_registers))}]"
        )
        program_lines.append(f"return {self.outputs_wrapper.__name__}(outputs)")
        return "\n\t".join(program_lines)

    def _get_readable_expression(
        self,
        genotype: Genotype,
        inputs_mapping_fn: Callable[[int], str],
        outputs_mapping_fn: Callable[[int], str],
    ) -> List[str]:
        """Worker class for computing the readable symbolic representation of a LGP genotype."""
        n_in = self.n_inputs + self.n_input_constants
        targets = []

        def _replace_lgp_expression(
            lgp_genes: Genotype,
            reg_idx: int,
            max_row_idx: int,
        ) -> str:
            functions = list(self.function_set.function_set.values())
            input_constants = genotype["weights"]["program_inputs"]
            for row_idx in range(max_row_idx - 1, -1, -1):
                if int(lgp_genes["genes"]["targets"][row_idx]) == reg_idx:
                    function = functions[lgp_genes["genes"]["functions"][row_idx]]
                    line_weight, x_weight, y_weight = self._weights_representations(
                        lgp_genes, row_idx
                    )
                    line_bias, x_bias, y_bias = self._biases_representations(
                        lgp_genes, row_idx
                    )
                    n_p1, n_p2 = ("(", ")") if self.biased_functions else ("", "")
                    i_p1, i_p2 = ("(", ")") if self.biased_inputs else ("", "")
                    if function.arity == 1:
                        return (
                            f"{n_p1}{line_weight}{function.symbol}({x_weight}"
                            f"{_replace_lgp_expression(lgp_genes, int(lgp_genes['genes']['inputs1'][row_idx]), row_idx)}"
                            f"{x_bias})"
                            f"{line_bias}{n_p2}"
                        )
                    else:
                        return (
                            f"{n_p1}{line_weight}({i_p1}{x_weight}"
                            f"{_replace_lgp_expression(lgp_genes, int(lgp_genes['genes']['inputs1'][row_idx]), row_idx)}"
                            f"{x_bias}{i_p2}{function.symbol}{i_p1}{y_weight}"
                            f"{_replace_lgp_expression(lgp_genes, int(lgp_genes['genes']['inputs2'][row_idx]), row_idx)}"
                            f"{y_bias}{i_p2}){line_bias}{n_p2}"
                        )
            if reg_idx < self.n_inputs:
                return inputs_mapping_fn(int(reg_idx))
            elif reg_idx < n_in:
                return str(input_constants[reg_idx - self.n_inputs])
            else:
                return "0"

        for output_idx in range(self.n_outputs):
            register_idx = self.n_registers - self.n_outputs + output_idx
            targets.append(
                f"{outputs_mapping_fn(output_idx)} = {self.outputs_wrapper.__name__}("
                f"{_replace_lgp_expression(genotype, register_idx, self.n_program_lines)})"
            )

        return targets
