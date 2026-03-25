import functools
from typing import Any, Callable, Dict, Optional, Union

import jax.numpy as jnp
import jax.random
from flax import struct
from qdax.custom_types import Genotype, RNGKey

from genepax.gp.genetic_programming import GP
from genepax.gp.graph_genetic_programming import GGP
from genepax.gp.tree_genetic_programming import TreeGP


@struct.dataclass
class EnsembleGP(GP):
    """Ensemble wrapper for multiple GP programs.

    This class implements an ensemble of independent GP programs that share
    the same base representation but evolve separately per output dimension.
    It is typically used when multiple outputs must be produced (e.g., multi-action
    control policies) while maintaining diversity across program instances.

    Args:
        n_outputs: Number of independent GP programs in the ensemble.
        base_gp_model: Base GP model used to instantiate and evaluate each
            program in the ensemble.
    """

    n_outputs: int
    base_gp_model: Union[TreeGP, GGP]

    def init(self, rnd_key: RNGKey, **kwargs: Any) -> Genotype:
        """Initialize an ensemble of random genotypes.

        Each output dimension corresponds to an independent GP program, thus
        requiring independent random initialization.

        Args:
            rnd_key: JAX PRNG key used to seed genotype initialization.
            **kwargs: Additional arguments.

        Returns:
            Genotype: A stacked ensemble genotype where each element corresponds
            to a GP program for one output dimension.
        """
        init_keys = jax.random.split(rnd_key, self.n_outputs)
        init_fn: Callable[..., Any] = self.base_gp_model.init
        partial_init = functools.partial(init_fn, **kwargs)
        return jax.jit(jax.vmap(partial_init, in_axes=(0,)))(init_keys)

    def apply(
        self,
        genotype: Genotype,
        obs: jnp.ndarray,
        weights: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        """Evaluate the ensemble of GP programs on an observation.

        Each GP program in the ensemble is evaluated independently and their
        outputs are concatenated into a flat vector.

        Args:
            genotype: Ensemble genotype containing one GP program per output.
            obs: Observation input array.
            weights: Optional dictionary of trainable weights to use during evaluation.

        Returns:
            jnp.ndarray: Flattened vector of ensemble program outputs.
        """
        mapped_apply_fn = jax.jit(
            jax.vmap(self.base_gp_model.apply, in_axes=(0, None, 0))
        )
        nested_outputs = mapped_apply_fn(genotype, obs, weights)
        return jnp.ravel(nested_outputs)

    def mutate(
        self,
        genotype: Genotype,
        rnd_key: RNGKey,
        mutation_probabilities: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> Genotype:
        """Mutate an ensemble genotype.

        Each GP program in the ensemble is mutated independently using the
        mutation operator of the base GP model.

        Args:
            genotype: Ensemble genotype to mutate.
            rnd_key: JAX PRNG key used for stochastic mutation.
            mutation_probabilities: Optional dictionary overriding mutation
                probabilities for genotype components.

        Returns:
            Genotype: Mutated ensemble genotype.
        """
        mutate_keys = jax.random.split(rnd_key, self.n_outputs)
        partial_mutate = functools.partial(
            self.base_gp_model.mutate, mutation_probabilities=mutation_probabilities
        )
        return jax.jit(jax.vmap(partial_mutate, in_axes=(0, 0)))(genotype, mutate_keys)

    def get_readable_expression(
        self,
        genotype: Genotype,
        inputs_mapping: Union[Dict[int, str], Callable[[int], str], None] = None,
        outputs_mapping: Union[Dict[int, str], Callable[[int], str], None] = None,
    ) -> str:
        """Generate a human-readable symbolic representation of a GP ensemble genotype.

        Unary functions are printed in the form:
            f(x)
        Binary functions are printed in the form:
            (x op y)
        where `op` is the function symbol (e.g., `+`, `*`, `sin`).

        Args:
            genotype: EnsembleGP genotype.
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
            symbolic expression computed for each output.

        Example:
            o0 = (i0+i1)
            o1 = sin(i2)
        """
        expressions = [
            self.base_gp_model.get_readable_expression(
                jax.tree.map(lambda x: x[i], genotype), inputs_mapping  # noqa: B023
            )
            for i in range(self.n_outputs)
        ]

        outputs_mapping = outputs_mapping or {}
        if isinstance(outputs_mapping, dict):
            outputs_mapping_fn: Callable[[int], str] = lambda idx: outputs_mapping.get(
                idx, f"o{idx}"
            )
        else:
            outputs_mapping_fn = outputs_mapping
        processed_expressions = []
        for i, expr in enumerate(expressions):
            _, content = expr.split("=")
            processed_expressions.append(f"{outputs_mapping_fn(i)} = {content.strip()}")

        return "\n".join(processed_expressions)

    def size(self, genotype: Genotype) -> jnp.ndarray:
        """Compute the average number of active (expressed) elements in a genotype."""
        sizes = jax.jit(jax.vmap(self.base_gp_model.size, in_axes=0))(genotype)
        return jnp.sum(sizes)
