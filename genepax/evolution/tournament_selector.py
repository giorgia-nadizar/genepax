from functools import partial

import jax
import jax.numpy as jnp
from flax import struct
from jax import vmap
from qdax.core.emitters.repertoire_selectors.selector import (
    GARepertoireT,
    Selector,
    unfold_repertoire,
)
from qdax.custom_types import Fitness, Genotype, RNGKey


@struct.dataclass
class TournamentSelector(Selector):
    """A selector using tournament to sample individuals from the population."""

    tournament_size: int = 3

    def select(
        self,
        repertoire: GARepertoireT,
        key: RNGKey,
        num_samples: int,
    ) -> GARepertoireT:
        def _tournament(
            sample_key: RNGKey,
            genomes: Genotype,
            fitness_values: Fitness,
        ) -> jnp.ndarray:
            indexes = jax.random.choice(
                sample_key,
                jnp.arange(start=0, stop=len(genomes)),
                shape=[self.tournament_size],
                replace=True,
            )
            mask = -jnp.inf * jnp.ones_like(fitness_values)
            mask = mask.at[indexes].set(1)
            positive_fitnesses = fitness_values + jnp.abs(
                jnp.minimum(jnp.min(fitness_values), 0)
            )
            fitness_values_for_selection = positive_fitnesses * mask
            return jnp.argmax(fitness_values_for_selection)

        sample_keys = jax.random.split(key, num_samples)
        partial_single_tournament = partial(
            _tournament,
            genomes=repertoire.genotypes,
            fitness_values=repertoire.fitnesses,
        )
        vmap_tournament = vmap(partial_single_tournament)
        selected_indexes = vmap_tournament(sample_key=sample_keys)

        repertoire_unfolded = unfold_repertoire(repertoire)
        selected: GARepertoireT = jax.tree.map(
            lambda x: x[selected_indexes],
            repertoire_unfolded,
        )

        return selected
