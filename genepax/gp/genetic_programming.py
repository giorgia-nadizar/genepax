from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import struct
from qdax.custom_types import Genotype, RNGKey


@struct.dataclass
class GP(ABC):

    @abstractmethod
    def init(self, *args: Any, **kwargs: Any) -> Any:
        """Initialize a random genotype."""
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        genotype: Genotype,
        obs: jnp.ndarray,
        weights: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        """Evaluate the GP program on an observation.

        Args:
            genotype: GP genotype.
            obs: Observation input array.
            weights: Optional dictionary of trainable weights to use during evaluation.

        Returns:
            jnp.ndarray: vector of program output(s).
        """
        raise NotImplementedError

    @abstractmethod
    def size(self, genotype: Genotype) -> jnp.ndarray:
        """Compute the number of active (expressed) elements in a genotype.

        Args:
            genotype: GP genotype.

        Returns:
            jnp.ndarray: the size of a genotype.
        """

        raise NotImplementedError

    @abstractmethod
    def mutate(
        self,
        genotype: Genotype,
        rnd_key: RNGKey,
        *,
        mutation_probabilities: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> Genotype:
        """Mutate a genotype.

        Args:
            genotype: GP genotype to mutate.
            rnd_key: JAX PRNG key used for stochastic mutation.
            mutation_probabilities: Optional dictionary overriding mutation
                probabilities for genotype components.

        Returns:
            Genotype: Mutated GP genotype.
        """
        raise NotImplementedError

    def vmap_mutate(
        self,
        genotype: Genotype,
        rnd_key: RNGKey,
        *,
        mutation_probabilities: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> Genotype:
        """Mutate a batch of genotypes.

        Args:
            genotype: a batch of GP genotypes to mutate.
            rnd_key: JAX PRNG key used for stochastic mutation.
            mutation_probabilities: Optional dictionary overriding mutation
                probabilities for genotype components.

        Returns:
            Genotype: Mutated GP genotypes batch.
        """
        batch_size = jax.tree.leaves(genotype)[0].shape[0]
        mutation_keys = jax.random.split(rnd_key, batch_size)

        def _mutation_fn(genotype_i: Genotype, key_i: RNGKey) -> Genotype:
            return self.mutate(
                genotype_i,
                key_i,
                mutation_probabilities=mutation_probabilities,
                **kwargs,
            )

        vmap_mutation_fn = jax.jit(jax.vmap(_mutation_fn, in_axes=(0, 0)))
        return vmap_mutation_fn(
            genotype,
            mutation_keys,
        )

    @abstractmethod
    def get_readable_expression(
        self,
        genotype: Genotype,
        *,
        inputs_mapping: Union[Dict[int, str], Callable[[int], str], None] = None,
    ) -> str:
        """Generate a human-readable symbolic representation of a GP genotype.

        Unary functions are printed in the form:
            f(x)
        Binary functions are printed in the form:
            (x op y)
        where `op` is the function symbol (e.g., `+`, `*`, `sin`).

        Args:
            genotype: GP genotype.
            inputs_mapping (dict[int,str] | callable[[int], str], optional):
                Mapping from input indices to custom names.
                - If a dict, keys are input indices
                - If a callable, it is called with the input index and must
                  return the desired string
                Defaults to "i0", "i1", ...

        Returns:
            str: A showing the symbolic expression computed for the genotype.

        Example:
            y = ((i0+i1) * sin(i2))
        """
        raise NotImplementedError

    def bind(self, genotype: Genotype) -> GPInstance:
        """Create an interactive GP instance by binding a genotype to this model.

        This method returns a `GPInstance` in which the provided
        ``genotype`` is bound to the current GP model. The resulting object
        represents a concrete, executable instance of the model and can be
        used to perform operations that require both the model definition
        and a specific genotype.

        This method provides a convenient way to work with a stateful
        instance directly.

        Args:
            genotype: The genotype to bind to this GP model.

        Returns:
            GPInstance: An instance combining this GP model with the
            provided genotype.
        """

        return GPInstance(gp_model=self, genotype=genotype)  # type: ignore


@struct.dataclass
class GPInstance:
    """
    A bound instance of a `GP` model with a fixed `Genotype`.

    `GPInstance` acts as a convenient wrapper around a `GP` model,
    automatically binding a specific `Genotype`. Once bound, all
    methods operate on this fixed genotype, eliminating the need
    to pass it explicitly every time.

    Attributes
    ----------
    gp_model : GP
        The genetic programming model that defines the operations.
    genotype : Genotype
        The genotype bound to this instance, used by all method calls.
    """

    gp_model: GP
    genotype: Genotype

    def apply(
        self, obs: jnp.ndarray, weights: Optional[Dict[str, jnp.ndarray]] = None
    ) -> jnp.ndarray:
        """
        Evaluate the GP model on a given observation using the bound genotype.

        Parameters
        ----------
        obs : jnp.ndarray
            Input observation for which to evaluate the GP model.
        weights : Optional[Dict[str, jnp.ndarray]]
            Optional weights to override any default or learned parameters
            in the GP model during evaluation.

        Returns
        -------
        jnp.ndarray
            The output of the GP model for the given observation.
        """
        return self.gp_model.apply(self.genotype, obs, weights)

    def size(self) -> jnp.ndarray:
        """
        Compute the size or complexity of the bound genotype.

        Returns
        -------
        jnp.ndarray
            The size (e.g., number of nodes or operations) of the genotype.
        """
        return self.gp_model.size(self.genotype)

    def mutate(
        self,
        rnd_key: RNGKey,
        *,
        mutation_probabilities: Optional[Dict[str, float]] = None,
    ) -> Genotype:
        """
        Apply mutation to the bound genotype to produce a new genotype.

        Parameters
        ----------
        rnd_key : RNGKey
            A random key for reproducible mutation.
        mutation_probabilities : Optional[Dict[str, float]]
            Optional dictionary specifying per-operation mutation probabilities.
            If not provided, default probabilities are used.

        Returns
        -------
        Genotype
            A new genotype resulting from mutation of the bound genotype.
        """
        return self.gp_model.mutate(
            self.genotype, rnd_key, mutation_probabilities=mutation_probabilities
        )

    def get_readable_expression(
        self,
        **kwargs: Any,
    ) -> str:
        """
        Generate a human-readable representation of the bound genotype.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments to customize formatting.

        Returns
        -------
        str
            A string representation of the genotype in a readable form.
        """
        return self.gp_model.get_readable_expression(self.genotype, **kwargs)

    def unbind(self) -> Tuple[GP, Genotype]:
        """
        Return the original GP model and the bound genotype.

        Returns
        -------
        tuple
            A tuple containing:
            - gp_model : GP
                The original GP model.
            - genotype : Genotype
                The genotype bound to this instance.
        """
        return self.gp_model, self.genotype
