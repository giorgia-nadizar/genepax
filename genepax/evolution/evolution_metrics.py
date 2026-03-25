from typing import Callable, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.custom_types import Metrics


def custom_ga_metrics(
    repertoire: GARepertoire,
    extra_scores_metrics: Optional[Union[str, List[str], Dict[str, Callable]]] = None,
) -> Metrics:
    """Compute the max fitness of the repertoire.

    Args:
        repertoire: a GA repertoire
        extra_scores_metrics: a list of extra scores metrics
            to be extracted from the repertoire
    Returns:
        a dictionary containing the max fitness of the
            repertoire and the related extra scores
    """
    # check metric type
    if isinstance(extra_scores_metrics, str):
        extra_scores_metrics = [extra_scores_metrics]
    if isinstance(extra_scores_metrics, list):
        extra_scores_metrics = {k: lambda x: x for k in extra_scores_metrics}

    # get metrics
    max_fitness = jnp.max(repertoire.fitnesses, axis=0)
    metrics_dict = {
        "max_fitness": max_fitness,
    }

    if extra_scores_metrics is not None:
        best_idx = jnp.argmax(repertoire.fitnesses, axis=0)
        best_extra_scores = jax.tree.map(lambda x: x[best_idx], repertoire.extra_scores)
        selected_extra_scores = {
            k: fn(best_extra_scores[k])
            for k, fn in extra_scores_metrics.items()
            if k in best_extra_scores
        }
        metrics_dict = metrics_dict | selected_extra_scores

    return metrics_dict
