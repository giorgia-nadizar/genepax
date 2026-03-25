from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.core.emitters.emitter import EmitterState
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.custom_types import ExtraScores, Genotype, RNGKey


class CustomMixingEmitter(MixingEmitter):

    def emit(  # type: ignore
        self,
        repertoire: GARepertoire,
        emitter_state: Optional[EmitterState],
        key: RNGKey,
    ) -> Tuple[Genotype, ExtraScores]:
        """
        Emitter that performs both mutation and variation. Two batches of
        variation_percentage * batch_size genotypes are sampled in the repertoire,
        copied and cross-over to obtain new offsprings. One batch of
        (1.0 - variation_percentage) * batch_size genotypes are sampled in the
        repertoire, copied and mutated.

        Note: this emitter has no state. A fake none state must be added
        through a function redefinition to make this emitter usable with MAP-Elites.

        Params:
            repertoire: the MAP-Elites repertoire to sample from
            emitter_state: void
            key: a jax PRNG random key

        Returns:
            a batch of offsprings
        """
        n_variation = int(self._batch_size * self._variation_percentage)
        n_mutation = self._batch_size - n_variation
        var_key, mut_key = jax.random.split(key)

        if n_variation > 0:
            sample_key_1, sample_key_2, variation_key = jax.random.split(var_key, 3)
            x1 = repertoire.select(
                sample_key_1, n_variation, selector=self._selector
            ).genotypes
            x2 = repertoire.select(
                sample_key_2, n_variation, selector=self._selector
            ).genotypes
            variation_keys = jax.random.split(variation_key, n_variation)
            x_variation = self._variation_fn(x1, x2, variation_keys)

        if n_mutation > 0:
            sample_key, mutation_key = jax.random.split(mut_key)
            x1 = repertoire.select(
                sample_key, n_mutation, selector=self._selector
            ).genotypes
            mutation_keys = jax.random.split(mutation_key, n_mutation)
            x_mutation = self._mutation_fn(x1, mutation_keys)

        if n_variation == 0:
            genotypes = x_mutation
        elif n_mutation == 0:
            genotypes = x_variation
        else:
            genotypes = jax.tree.map(
                lambda x_1, x_2: jnp.concatenate([x_1, x_2], axis=0),
                x_variation,
                x_mutation,
            )

        return genotypes, {}
