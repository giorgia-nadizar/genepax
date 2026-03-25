from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax
from qdax.baselines.genetic_algorithm import GeneticAlgorithm
from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.custom_types import ExtraScores, Fitness, Genotype, Metrics, RNGKey

from genepax.evolution.ga_repertoire_extra_scores import GARepertoireExtraScores


class GeneticAlgorithmWithExtraScores(GeneticAlgorithm):
    """
    Extends GeneticAlgorithm to track additional metrics ('extra scores')
    for each individual.

    In addition, it has
    - a lamarckian flag to state whether the genomes are
    replaced by their updated version before entering the repertoire
    - a rescoring_function to rescore the repertoire upon update if desired

    The main GA behavior is unchanged.
    """

    def __init__(
        self,
        scoring_function: Callable[[Genotype, RNGKey], Tuple[Fitness, ExtraScores]],
        emitter: Emitter,
        metrics_function: Callable[[GARepertoire], Metrics],
        lamarckian: bool = False,
        rescoring_function: Optional[Callable[[Genotype, RNGKey], Fitness]] = None,
    ):
        super().__init__(scoring_function, emitter, metrics_function)
        self._lamarckian = lamarckian
        self._rescoring_function = rescoring_function

    def init(
        self, genotypes: Genotype, population_size: int, key: RNGKey
    ) -> Tuple[GARepertoireExtraScores, Optional[EmitterState], Metrics]:
        """Initialize a GARepertoire with an initial population of genotypes.

        Args:
            genotypes: the initial population of genotypes
            population_size: the maximal size of the repertoire
            key: a random key to handle stochastic operations

        Returns:
            The initial repertoire, an initial emitter state and a new random key.

        Note: it differs from the original GA as it stores the extra_scores in the
            GARepertoire.
        """

        # score initial genotypes
        key, subkey = jax.random.split(key)
        fitnesses, extra_scores = self._scoring_function(genotypes, subkey)
        genotypes = jax.lax.cond(
            self._lamarckian,
            lambda _: extra_scores["updated_params"],
            lambda _: genotypes,
            operand=None,
        )

        # init the repertoire
        repertoire = GARepertoireExtraScores.init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            population_size=population_size,
            extra_scores=extra_scores,
            keys_extra_scores=extra_scores.keys(),
        )

        # get initial state of the emitter
        key, subkey = jax.random.split(key)
        emitter_state = self._emitter.init(
            key=subkey,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=None,
            extra_scores=extra_scores,
        )

        # calculate the initial metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics

    def update(
        self,
        repertoire: GARepertoireExtraScores,
        emitter_state: Optional[EmitterState],
        key: RNGKey,
        rescore_repertoire: bool = False,
    ) -> Tuple[GARepertoireExtraScores, Optional[EmitterState], Metrics]:
        """
        Performs one iteration of a Genetic algorithm.
        1. A batch of genotypes is sampled in the repertoire and the genotypes
            are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the repertoire.

        Args:
            repertoire: a repertoire
            emitter_state: state of the emitter
            key: a jax PRNG random key
            rescore_repertoire: a flag to determine whether the repertoire needs
            to be rescored before merging with the offspring

        Returns:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key

        Note: it differs from the original GA as it stores the extra_scores in the
            GARepertoire.
        """

        # generate offsprings
        key, subkey = jax.random.split(key)
        offspring, extra_info = self._emitter.emit(repertoire, emitter_state, subkey)

        # score the offsprings
        key, subkey = jax.random.split(key)
        fitnesses, extra_scores = self._scoring_function(offspring, subkey)

        offspring = jax.lax.cond(
            self._lamarckian,
            lambda _: extra_scores["updated_params"],
            lambda _: offspring,
            operand=None,
        )

        # optionally rescore the repertoire
        if rescore_repertoire:
            rescoring_fn = (
                self._rescoring_function
                if self._rescoring_function
                else lambda g, r: self._scoring_function(g, r)[0]
            )
            key, rescore_subkey = jax.random.split(key)
            rescored_fitnesses = rescoring_fn(repertoire.genotypes, rescore_subkey)
            repertoire = GARepertoireExtraScores.init(
                genotypes=repertoire.genotypes,
                fitnesses=rescored_fitnesses,
                population_size=repertoire.size,
                extra_scores=repertoire.extra_scores,
                keys_extra_scores=repertoire.keys_extra_scores,
            )

        # update the repertoire
        repertoire = repertoire.add(offspring, fitnesses, extra_scores)

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=offspring,
            fitnesses=fitnesses,
            descriptors=None,
            extra_scores={**extra_scores, **extra_info},
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics  # type: ignore

    def replace_scoring_fns(
        self,
        scoring_fn: Callable[[Genotype, RNGKey], Tuple[Fitness, ExtraScores]],
        rescoring_fn: Optional[Callable[[Genotype, RNGKey], Fitness]] = None,
    ) -> GeneticAlgorithmWithExtraScores:
        """
        Return a new genetic algorithm instance that uses the given scoring function.

        The new instance keeps the same emitter and metrics function as the current
        one but replaces the scoring function with `scoring_fn`.

        Parameters
        ----------
        scoring_fn : Callable[[Genotype, RNGKey], Tuple[Fitness, ExtraScores]]
            The scoring function to use in the new algorithm.
        rescoring_fn: Callable[[Genotype, RNGKey], Tuple[Fitness, ExtraScores]]
            The function use to rescore the repertoire if needed

        Returns
        -------
        GeneticAlgorithmWithExtraScores
            A copy of the algorithm with the updated scoring functions.
        """
        return GeneticAlgorithmWithExtraScores(
            scoring_fn,
            self._emitter,
            self._metrics_function,
            self._lamarckian,
            rescoring_fn,
        )
