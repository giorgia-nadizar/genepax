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
class SequentialGP(GP):
    first_gp_model: Union[TreeGP, GGP]
    second_gp_model: Union[TreeGP, GGP]

    def init(self, rnd_key: RNGKey, **kwargs: Any) -> Genotype:
        init_key1, init_key2 = jax.random.split(rnd_key)
        g1 = self.first_gp_model.init(init_key1)
        g2 = self.second_gp_model.init(init_key2)
        return {
            'g1': g1,
            'g2': g2,
        }

    def apply(
            self,
            genotype: Genotype,
            obs: jnp.ndarray,
            weights: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        g1 = genotype['g1']
        g2 = genotype['g2']
        intermediate_result = self.first_gp_model.apply(g1, obs)
        return self.second_gp_model.apply(g2, intermediate_result)

    def mutate(
            self,
            genotype: Genotype,
            rnd_key: RNGKey,
            mutation_probabilities: Optional[Dict[str, float]] = None,
            **kwargs: Any,
    ) -> Genotype:
        mutate_key1, mutate_key2 = jax.random.split(rnd_key)
        g1 = genotype['g1']
        g2 = genotype['g2']
        g1_mutated = self.first_gp_model.mutate(g1, mutate_key1, **kwargs)
        g2_mutated = self.second_gp_model.mutate(g2, mutate_key2, **kwargs)
        return {
            'g1': g1_mutated,
            'g2': g2_mutated,
        }

    def get_readable_expression(
            self,
            genotype: Genotype,
            inputs_mapping: Union[Dict[int, str], Callable[[int], str], None] = None,
            outputs_mapping: Union[Dict[int, str], Callable[[int], str], None] = None,
    ) -> str:
        raise NotImplementedError

    def size(self, genotype: Genotype) -> jnp.ndarray:
        """Compute the average number of active (expressed) elements in a genotype."""
        s1 = self.first_gp_model.size(genotype['g1'])
        s2 = self.second_gp_model.size(genotype['g2'])
        return s1 + s2
