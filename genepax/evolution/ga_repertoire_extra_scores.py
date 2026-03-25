"""Defines a repertoire for simple genetic algorithms."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.custom_types import ExtraScores, Fitness, Genotype


class GARepertoireExtraScores(GARepertoire):
    """Class for a simple repertoire for a simple genetic
    algorithm with proper handling of extra scores.

    Args:
        genotypes: a PyTree containing the genotypes of the
            individuals in the population. Each leaf has the
            shape (population_size, num_features).
        fitnesses: an array containing the fitness of the individuals
            in the population. With shape (population_size, fitness_dim).
            The implementation of GARepertoire was thought for the case
            where fitness_dim equals 1 but the class can be herited and
            rules adapted for cases where fitness_dim is greater than 1.
        extra_scores: extra scores resulting from the evaluation of the genotypes
        keys_extra_scores: keys of the extra scores to store in the repertoire
    """

    def add(  # type: ignore
        self,
        batch_of_genotypes: Genotype,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: Optional[ExtraScores] = None,
    ) -> GARepertoire:
        """Implements the repertoire addition rules.

        Parents and offsprings are gathered and only the population_size
        bests are kept. The others are killed.

        Args:
            batch_of_genotypes: new genotypes that we try to add.
            batch_of_fitnesses: fitness of those new genotypes.
            batch_of_extra_scores: extra scores of those new genotypes.

        Returns:
            The updated repertoire.
        """
        if batch_of_extra_scores is None:
            batch_of_extra_scores = {}

        filtered_batch_of_extra_scores = self.filter_extra_scores(batch_of_extra_scores)

        # gather individuals and fitnesses
        candidates = jax.tree.map(
            lambda x, y: jnp.concatenate((x, y), axis=0),
            self.genotypes,
            batch_of_genotypes,
        )
        candidates_fitnesses = jnp.concatenate(
            (self.fitnesses, batch_of_fitnesses), axis=0
        )
        candidates_extra_scores = jax.tree.map(
            lambda x, y: jnp.concatenate((x, y), axis=0),
            self.extra_scores,
            filtered_batch_of_extra_scores,
        )

        # sort by fitnesses
        indices = jnp.argsort(jnp.sum(candidates_fitnesses, axis=1))[::-1]

        # keep only the best ones
        survivor_indices = indices[: self.size]

        # keep only the best ones
        new_candidates = jax.tree.map(lambda x: x[survivor_indices], candidates)
        new_extra_scores = jax.tree.map(
            lambda x: x[survivor_indices], candidates_extra_scores
        )
        new_repertoire = self.replace(
            genotypes=new_candidates,
            fitnesses=candidates_fitnesses[survivor_indices],
            extra_scores=new_extra_scores,
        )

        return new_repertoire  # type: ignore
