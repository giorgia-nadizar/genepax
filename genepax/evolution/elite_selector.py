import jax
import jax.numpy as jnp
from flax import struct
from qdax.core.emitters.repertoire_selectors.selector import (
    GARepertoireT,
    Selector,
    unfold_repertoire,
)
from qdax.custom_types import RNGKey


@struct.dataclass
class EliteSelector(Selector):
    """A selector sampling the best num_samples individuals from the population."""

    def select(
        self,
        repertoire: GARepertoireT,
        key: RNGKey,
        num_samples: int,
    ) -> GARepertoireT:
        top_n_idx = jnp.argsort(repertoire.fitnesses.squeeze())[
            -num_samples:
        ]  # sorted ascending → take last n
        selected_indexes = top_n_idx[::-1]

        repertoire_unfolded = unfold_repertoire(repertoire)
        selected: GARepertoireT = jax.tree.map(
            lambda x: x[selected_indexes],
            repertoire_unfolded,
        )

        return selected
