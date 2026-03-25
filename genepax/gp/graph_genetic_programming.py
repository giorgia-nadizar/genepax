from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import jax.random
from flax import struct
from jax import random
from jax.lax import fori_loop
from qdax.custom_types import Genotype, Mask, RNGKey

from genepax.gp.functions import FunctionSet
from genepax.gp.genetic_programming import GP


@struct.dataclass
class GGP(GP):
    """Base class for Graph-based Genetic Programming (GGP) representations.

    Common parameters for GP encodings:

    Args:
        n_inputs: number of input values provided to the program/graph (excluding constants).
            Typically set to the environment’s observation size, e.g., `env.observation_size`.
        n_outputs: number of outputs produced by the GP individual.
            Typically set to the environment’s action size, e.g., `env.action_size`.
        function_set: set of allowed functions that nodes/instructions can use.
        n_input_constants: number of constant values that can be used as additional inputs.
        outputs_wrapper: function applied to the outputs before returning them
            (e.g., `tanh` to bound outputs).
        weighted_program_inputs: whether the genotype will contain optimizable inputs for the program.
        weighted_functions: whether the genotype will contain weighting factors for each node/program line.
        biased_functions: whether the genotype will contain the biases (additive) for each node/program line.
        weighted_inputs: whether the genotype will contain weighting factors for each connection.
        biased_inputs: whether the genotype will contain the biases (additive) for each connection.
        weights_initialization: how to initialize the weights before their optimization (either `uniform` or `natural`).
        weights_mutation: whether weights will undergo mutation or not.
        weights_mutation_type: what type of mutation is used (if weights_mutation is True) to mutate the weights.
    """

    n_inputs: int
    n_outputs: int
    function_set: FunctionSet = FunctionSet()
    n_input_constants: int = 2
    outputs_wrapper: Callable = jnp.tanh
    weighted_program_inputs: bool = False
    weighted_functions: bool = False
    biased_functions: bool = False
    weighted_inputs: bool = False
    biased_inputs: bool = False
    weights_initialization: str = "uniform"
    weights_mutation: bool = True
    weights_mutation_type: str = "gaussian"

    @property
    def n_functions(self) -> int:
        """Max number of functions that can be performed by GGP."""
        raise NotImplementedError

    def init(self, rnd_key: RNGKey, *args: Any) -> Any:
        """Initialize a random genotype (to be implemented by subclasses)."""
        raise NotImplementedError

    def apply(
        self,
        genotype: Genotype,
        obs: jnp.ndarray,
        weights: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        """Evaluate a genotype on an input observation (subclass-specific)."""
        raise NotImplementedError

    def compute_active_mask(
        self,
        genotype: Genotype,
    ) -> Mask:
        """Compute the mask of active (expressed) elements in a genotype (subclass-specific)."""
        raise NotImplementedError

    def size(self, genotype: Genotype) -> jnp.ndarray:
        """Compute the number of active (expressed) elements in a genotype."""
        return jnp.sum(self.compute_active_mask(genotype))

    def mutate(
        self,
        genotype: Genotype,
        rnd_key: RNGKey,
        *,
        p_mut_inputs: float = 0.1,
        p_mut_functions: float = 0.1,
        weights_mut_sigma: float = 0.1,
        mutation_probabilities: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> Genotype:
        """Mutates a GGP genotype using int-flip mutation. If the genotype is weighted, the weights
        are mutated with Gaussian mutation.

        This mutation is implemented as a form of crossover with a newly
        generated "donor" genotype: for each gene, the value is taken from the
        donor with a low probability, otherwise kept from the original genotype.
        This ensures that all mutated genes remain valid (i.e., within the
        correct index ranges for their respective genotype section).

        The function is compatible with standard emitters when wrapped using
        `functools.partial`.

        Mutation probabilities and sigma can be specified either via individual arguments or by
        passing a dictionary to `mutation_probabilities`, the dictionary values override
        the individual arguments.

        Args:
            genotype: the CGP genotype parameters to mutate.
            rnd_key: JAX PRNG key for randomness.
            p_mut_inputs: probability of mutating each input connection gene
                (ignored if overridden via `mutation_probabilities`).
            p_mut_functions: probability of mutating each function gene
                (ignored if overridden via `mutation_probabilities`).
            weights_mut_sigma: mutation step for weights Gaussian mutation
                (ignored if overridden via `mutation_probabilities`).
            mutation_probabilities: optional dictionary mapping genotype parts
             to their mutation probabilities.

        Returns:
            The mutated genotype.
        """
        return self._mutate(
            genotype,
            rnd_key,
            p_mut_inputs,
            p_mut_functions,
            weights_mut_sigma,
            mutation_probabilities,
        )[0]

    def _mutate(
        self,
        genotype: Genotype,
        rnd_key: RNGKey,
        p_mut_inputs: float = 0.1,
        p_mut_functions: float = 0.1,
        weights_mut_sigma: float = 0.1,
        mutation_probabilities: Optional[Dict[str, float]] = None,
    ) -> Tuple[Genotype, Genotype]:
        """Worker class for mutation that returns both the mutated genotype and the donor."""
        # extract mutation probabilities if passed through a dictionary
        mutation_probabilities = mutation_probabilities or {}
        p_mut_inputs = mutation_probabilities.get("inputs", p_mut_inputs)
        p_mut_functions = mutation_probabilities.get("functions", p_mut_functions)
        weights_mut_sigma = mutation_probabilities.get(
            "weights_sigma", weights_mut_sigma
        )

        new_key, x_key, y_key, f_key, weights_key = random.split(rnd_key, 5)
        # generate the donor genotype -> only few genes from this will be used
        donor_genotype = self.init(new_key)

        return {
            "genes": {
                "inputs1": _mutate_subgenome(
                    genotype["genes"]["inputs1"],
                    donor_genotype["genes"]["inputs1"],
                    x_key,
                    p_mut_inputs,
                ),
                "inputs2": _mutate_subgenome(
                    genotype["genes"]["inputs2"],
                    donor_genotype["genes"]["inputs2"],
                    y_key,
                    p_mut_inputs,
                ),
                "functions": _mutate_subgenome(
                    genotype["genes"]["functions"],
                    donor_genotype["genes"]["functions"],
                    f_key,
                    p_mut_functions,
                ),
            },
            "weights": self._mutate_weights(
                genotype["weights"], weights_key, weights_mut_sigma
            ),
        }, donor_genotype

    def _mutate_weights(
        self, weights: Dict, key: RNGKey, weights_mut_sigma: float
    ) -> Dict:
        if self.weights_mutation_type == "gaussian":
            weights_key1, weights_key2 = random.split(key)
            weights_noise = (
                weights_mut_sigma
                * self.weights_mutation
                * random.normal(weights_key1, shape=(self.n_functions * 6,))
            )
            fn_w_noise, i1_w_noise, i2_w_noise, fn_b_noise, i1_b_noise, i2_b_noise = (
                jnp.split(weights_noise, 6)
            )
            progr_in_noise = (
                weights_mut_sigma
                * self.weighted_program_inputs
                * self.weights_mutation
                * random.normal(weights_key2, shape=(self.n_input_constants,))
            )
            return {
                "program_inputs": weights["program_inputs"] + progr_in_noise,
                "inputs1": weights["inputs1"] + self.weighted_inputs * i1_w_noise,
                "inputs2": weights["inputs2"] + self.weighted_inputs * i2_w_noise,
                "functions": weights["functions"]
                + self.weighted_functions * fn_w_noise,
                "inputs1_biases": weights["inputs1_biases"]
                + self.biased_inputs * i1_b_noise,
                "inputs2_biases": weights["inputs2_biases"]
                + self.biased_inputs * i2_b_noise,
                "functions_biases": weights["functions_biases"]
                + self.biased_functions * fn_b_noise,
            }
        elif self.weights_mutation_type == "automl0":

            def _automl0_mutation(
                weights_array: jnp.ndarray, w_key: RNGKey, mutate: bool
            ) -> jnp.ndarray:
                sample_key1, sample_key2, bern_key1, bern_key2 = random.split(w_key, 4)
                double_values_array = jax.random.uniform(
                    sample_key1, shape=weights_array.shape, minval=1, maxval=2
                )
                half_values_array = jax.random.uniform(
                    sample_key2, shape=weights_array.shape, minval=0.5, maxval=2
                )
                mask = jax.random.bernoulli(bern_key1, 0.5, weights_array.shape)
                multipliers = jnp.where(mask, double_values_array, half_values_array)
                signs = jnp.where(
                    jax.random.bernoulli(bern_key2, 0.5, weights_array.shape), 1, -1
                )
                final_multiplier = multipliers * signs * mutate + (
                    1 - mutate
                ) * jnp.ones_like(weights_array)
                return weights_array * final_multiplier

            w_key1, w_key2, w_key3, w_key4, w_key5, w_key6, w_key7 = random.split(
                key, 7
            )
            return {
                "program_inputs": _automl0_mutation(
                    weights["program_inputs"],
                    w_key1,
                    self.weighted_program_inputs and self.weights_mutation,
                ),
                "inputs1": _automl0_mutation(
                    weights["inputs1"],
                    w_key2,
                    self.weighted_inputs and self.weights_mutation,
                ),
                "inputs2": _automl0_mutation(
                    weights["inputs2"],
                    w_key3,
                    self.weighted_inputs and self.weights_mutation,
                ),
                "functions": _automl0_mutation(
                    weights["functions"],
                    w_key4,
                    self.weighted_functions and self.weights_mutation,
                ),
                "inputs1_biases": _automl0_mutation(
                    weights["inputs1_biases"],
                    w_key5,
                    self.biased_inputs and self.weights_mutation,
                ),
                "inputs2_biases": _automl0_mutation(
                    weights["inputs2_biases"],
                    w_key6,
                    self.biased_inputs and self.weights_mutation,
                ),
                "functions_biases": _automl0_mutation(
                    weights["functions_biases"],
                    w_key7,
                    self.biased_functions and self.weights_mutation,
                ),
            }
        else:
            raise NotImplementedError(
                f"Mutation not available for {self.weights_mutation_type}"
            )

    def get_readable_expression(
        self,
        genotype: Genotype,
        inputs_mapping: Optional[Union[Dict[int, str], Callable[[int], str]]] = None,
        outputs_mapping: Optional[Union[Dict[int, str], Callable[[int], str]]] = None,
    ) -> str:
        """Generate a human-readable symbolic representation of a GGP genotype.

        Unary functions are printed in the form:
            f(x)
        Binary functions are printed in the form:
            (x op y)
        where `op` is the function symbol (e.g., `+`, `*`, `sin`).

        Args:
            genotype: GGP genotype.
            inputs_mapping (dict[int,str] | callable[[int], str], optional):
                Mapping from input indices to custom names.
                - If a dict, keys are input indices
                - If a callable, it is called with the input index and must
                  return the desired string
                Defaults to "i0", "i1", ...
            outputs_mapping (dict[int,str] | callable[[int], str], optional):
                Mapping from output indices to custom names.
                - If a dict, keys are output indices
                - If a callable, it is called with the output index and must
                  return the desired string
                Defaults to "o0", "o1", ...

        Returns:
            str: A multi-line string, with one line per output, showing the
            symbolic expression computed for each CGP output node.

        Example:
            o0 = (i0+i1)
            o1 = sin(i2)
        """
        genotype = {
            "weights": genotype["weights"],
            "genes": jax.tree.map(lambda x: x.astype(int), genotype["genes"]),
        }

        inputs_mapping = inputs_mapping or {}
        if isinstance(inputs_mapping, dict):
            inputs_mapping_fn = lambda idx: inputs_mapping.get(idx, f"i{idx}")
        else:
            inputs_mapping_fn = inputs_mapping

        outputs_mapping = outputs_mapping or {}
        if isinstance(outputs_mapping, dict):
            outputs_mapping_fn = lambda idx: outputs_mapping.get(idx, f"o{idx}")
        else:
            outputs_mapping_fn = outputs_mapping

        targets = self._get_readable_expression(
            genotype, inputs_mapping_fn, outputs_mapping_fn
        )
        return "\n".join(targets)

    def _get_readable_expression(
        self,
        genotype: Genotype,
        inputs_mapping_fn: Callable[[int], str],
        outputs_mapping_fn: Callable[[int], str],
    ) -> List[str]:
        """Worker class for computing the readable symbolic representation of a GGP genotype."""
        raise NotImplementedError

    def _weights_representations(
        self, genotype: Genotype, gene_idx: int
    ) -> Tuple[str, str, str]:
        input_weight = (
            f"{genotype['weights']['functions'][gene_idx]:.2f}*"
            if self.weighted_functions
            else ""
        )
        x_weight = (
            f"{genotype['weights']['inputs1'][gene_idx]:.2f}*"
            if self.weighted_inputs
            else ""
        )
        y_weight = (
            f"{genotype['weights']['inputs2'][gene_idx]:.2f}*"
            if self.weighted_inputs
            else ""
        )
        return input_weight, x_weight, y_weight

    def _biases_representations(
        self, genotype: Genotype, gene_idx: int
    ) -> Tuple[str, str, str]:
        input_bias = (
            f"+{genotype['weights']['functions_biases'][gene_idx]:.2f}"
            if self.biased_functions
            else ""
        )
        x_bias = (
            f"+{genotype['weights']['inputs1_biases'][gene_idx]:.2f}"
            if self.biased_inputs
            else ""
        )
        y_bias = (
            f"+{genotype['weights']['inputs2_biases'][gene_idx]:.2f}"
            if self.biased_inputs
            else ""
        )
        return input_bias, x_bias, y_bias

    def init_weights(self, key: RNGKey) -> Dict[str, jnp.ndarray]:
        """Initialize the weights' dictionary."""
        key1, key2 = random.split(key)
        if self.weights_initialization == "uniform":
            randoms = random.uniform(key=key1, shape=(self.n_functions * 6,)) * 2 - 1
            random_weights, random_biases = jnp.split(randoms, 2)
        elif self.weights_initialization == "natural":
            random_weights = jnp.ones(shape=(self.n_functions * 3,), dtype=jnp.float32)
            random_biases = jnp.zeros(shape=(self.n_functions * 3,), dtype=jnp.float32)
        else:
            raise NotImplementedError
        node_weights, input_weights1, input_weights2 = jnp.split(random_weights, 3)
        node_biases, input_biases1, input_biases2 = jnp.split(random_biases, 3)
        program_inputs = (
            random.uniform(key=key2, shape=(self.n_input_constants,)) * 2 - 1
        )
        if not self.weighted_program_inputs:
            program_inputs = program_inputs.at[:2].set(jnp.asarray([0.1, 1.0]))
        return {
            "program_inputs": program_inputs,
            "functions": (
                node_weights if self.weighted_functions else jnp.ones_like(node_weights)
            ),
            "inputs1": (
                input_weights1
                if self.weighted_inputs
                else jnp.ones_like(input_weights1)
            ),
            "inputs2": (
                input_weights2
                if self.weighted_inputs
                else jnp.ones_like(input_weights2)
            ),
            "functions_biases": (
                node_biases if self.biased_functions else jnp.zeros_like(node_biases)
            ),
            "inputs1_biases": (
                input_biases1 if self.biased_inputs else jnp.zeros_like(input_biases1)
            ),
            "inputs2_biases": (
                input_biases2 if self.biased_inputs else jnp.zeros_like(input_biases2)
            ),
        }

    def get_weights(self, genotype: Genotype) -> Dict[str, jnp.ndarray]:
        """Retrieve the trainable weights from a genotype based on the graph configuration.

        The returned weights depend on the flags `weighted_inputs`, `weighted_functions` and `
        weighted_program_inputs`:
          - If `weighted_inputs` is True, returns the weights associated with input connections:
            - "inputs1"
            - "inputs2"
          - If `weighted_functions` is True, returns the weights associated with function nodes:
            - "functions"
          - If `weighted_program_inputs` is True, returns the weights associated with the program inputs:
            - "program_inputs"
          - If `biased_inputs` is True, returns the biases associated with input connections:
            - "inputs1_biases"
            - "inputs2_biases"
          - If `biased_functions` is True, returns the biases associated with function nodes:
            - "functions_biases"
          - If neither flag is set, returns an empty dictionary.

        Args:
            genotype (Genotype): A genotype object containing the "weights" dictionary.

        Returns:
            Dict[str, jnp.ndarray]: A dictionary mapping weight types to their corresponding JAX arrays.
        """
        return_dictionary: Dict[str, jnp.ndarray] = {}
        if self.weighted_inputs:
            return_dictionary = return_dictionary | {
                "inputs1": genotype["weights"]["inputs1"],
                "inputs2": genotype["weights"]["inputs2"],
            }
        if self.weighted_functions:
            return_dictionary = return_dictionary | {
                "functions": genotype["weights"]["functions"],
            }
        if self.weighted_program_inputs:
            return_dictionary = return_dictionary | {
                "program_inputs": genotype["weights"]["program_inputs"],
            }
        if self.biased_functions:
            return_dictionary = return_dictionary | {
                "functions_biases": genotype["weights"]["functions_biases"],
            }
        if self.biased_inputs:
            return_dictionary = return_dictionary | {
                "inputs1_biases": genotype["weights"]["inputs1_biases"],
                "inputs2_biases": genotype["weights"]["inputs2_biases"],
            }

        return return_dictionary

    # noinspection PyMethodMayBeStatic
    def update_weights(
        self, genotype: Genotype, weights: Dict[str, jnp.ndarray]
    ) -> Genotype:
        """Update the weights of a genotype with the passed values.

        This method returns a new genotype dictionary where each weight type
        ("inputs1", "inputs2", "functions", "inputs1_biases", "inputs2_biases",
        "functions_biases") is replaced by the corresponding array
        in the provided `weights` dictionary if present. If a weight type is not
        provided, the original value from the input genotype is retained.

        Args:
            genotype (Genotype): The original genotype containing "genes" and "weights".
            weights (Dict[str, jnp.ndarray]): A dictionary of weights to update. Keys can include:
                - "inputs1"
                - "inputs2"
                - "functions"
                - "program_inputs"
                - "inputs1_biases"
                - "inputs2_biases"
                - "functions_biases"

        Returns:
            Genotype: A new genotype dictionary with updated weights.
        """
        return {
            "genes": genotype["genes"],
            "weights": {
                "inputs1": weights.get("inputs1", genotype["weights"]["inputs1"]),
                "inputs2": weights.get("inputs2", genotype["weights"]["inputs2"]),
                "functions": weights.get("functions", genotype["weights"]["functions"]),
                "program_inputs": weights.get(
                    "program_inputs", genotype["weights"]["program_inputs"]
                ),
                "inputs1_biases": weights.get(
                    "inputs1_biases", genotype["weights"]["inputs1_biases"]
                ),
                "inputs2_biases": weights.get(
                    "inputs2_biases", genotype["weights"]["inputs2_biases"]
                ),
                "functions_biases": weights.get(
                    "functions_biases", genotype["weights"]["functions_biases"]
                ),
            },
        }

    def _update_memory(
        self,
        genotype: Genotype,
        weights: Dict[str, jnp.ndarray],
        memory: jnp.ndarray,
        gene_idx: int,
        memory_idx: Union[int, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Updates the memory at a given index computing the function at the genotype index."""
        f_idx = genotype["genes"]["functions"].at[gene_idx].get()
        x_w = weights["inputs1"].at[gene_idx].get()
        y_w = weights["inputs2"].at[gene_idx].get()
        f_w = weights["functions"].at[gene_idx].get()
        x_b = weights["inputs1_biases"].at[gene_idx].get()
        y_b = weights["inputs2_biases"].at[gene_idx].get()
        f_b = weights["functions_biases"].at[gene_idx].get()
        x_arg = (
            memory.at[genotype["genes"]["inputs1"].at[gene_idx].get()].get() * x_w + x_b
        )
        y_arg = (
            memory.at[genotype["genes"]["inputs2"].at[gene_idx].get()].get() * y_w + y_b
        )
        f_computed = self.function_set.apply(f_idx, x_arg, y_arg) * f_w + f_b
        memory = memory.at[memory_idx].set(f_computed)
        return genotype, memory

    # descriptors that can be used with MAP Elites
    def compute_complexity(self, genotype: Genotype) -> jnp.ndarray:
        """Compute the relative complexity of the graph/program.
        Relative complexity is measured as the fraction of computing power used w.r.t.
        that allowed. For CGP this boils down to the amount of used nodes, for LGP the
        amount of program lines used.
        """
        return jnp.expand_dims(jnp.mean(self.compute_active_mask(genotype)), axis=0)

    def compute_function_count(self, genotype: Genotype) -> jnp.ndarray:
        """Compute the number of functions of each type used by the graph/program."""
        active_mask = self.compute_active_mask(genotype)
        f_genes = genotype["genes"]["functions"]

        def _count_functions(
            idx: int,
            f_counter: jnp.ndarray,
        ) -> jnp.ndarray:
            f_id = f_genes.at[idx].get()
            f_counter = f_counter.at[f_id].set(
                f_counter.at[f_id].get() + active_mask.at[idx].get()
            )
            return f_counter

        functions_count = fori_loop(
            lower=0,
            upper=len(f_genes),
            body_fun=_count_functions,
            init_val=(jnp.zeros(len(self.function_set))),
        )
        return functions_count

    def compute_function_arities(self, genotype: Genotype) -> jnp.ndarray:
        """Compute the fraction of one/two arity functions employed in the graph/program."""
        functions_count = self.compute_function_count(genotype)
        one_arity_total = jnp.sum(
            jnp.where(self.function_set.arities == 1, functions_count, 0)
        )
        two_arity_total = jnp.sum(
            jnp.where(self.function_set.arities == 2, functions_count, 0)
        )
        return jnp.asarray([one_arity_total, two_arity_total]) / self.n_functions


def _mutate_subgenome(
    x1: jnp.ndarray, x2: jnp.ndarray, key: RNGKey, p_mut: float
) -> jnp.ndarray:
    """Performs elementwise mutation of a genotype section.

    For each gene, a random number in [0, 1) is drawn. If the number is
    greater than `p_mut`, the gene is kept from the original subgenome (`x1`);
    otherwise, it is replaced with the corresponding gene from the donor
    subgenome (`x2`).

    Args:
        x1: Original subgenome array.
        x2: Donor subgenome array (must be the same shape as `x1`).
        key: JAX PRNG key used to generate mutation probabilities.
        p_mut: Probability of replacing each gene with the donor's value.

    Returns:
        The mutated subgenome array.
    """
    mutation_probs = random.uniform(key=key, shape=x1.shape)
    return jnp.where(mutation_probs > p_mut, x1, x2).astype(int)
