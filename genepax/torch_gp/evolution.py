"""A small, native PyTorch genetic algorithm for scalar-loss minimization."""

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor

from genepax.torch_gp.initialization import check_generator
from genepax.torch_gp.program import TorchProgram


def evolve(
    population: Sequence[TorchProgram],
    loss_fn: Callable[[TorchProgram], float | Tensor],
    *,
    generations: int = 100,
    elite_count: int = 1,
    tournament_size: int = 3,
    crossover_probability: float = 0.0,
    mutation_kwargs: dict[str, Any] | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Minimize a deterministic scalar loss using tournaments and elitism.

    Evaluate one model at a time under no_grad. The caller's models are copied
    and never modified. NaN/inf losses rank last; an entirely invalid initial
    population raises an error. History includes the initial population at
    generation zero. The result contains best_model, best_loss, population,
    losses, and history. Create a new optimizer for any returned offspring.
    """
    check_generator(generator)
    if not population or generations < 0 or tournament_size < 1:
        raise ValueError(
            "A population, non-negative generation count, and positive tournament size are required"
        )
    if not 1 <= elite_count <= len(population):
        raise ValueError("elite_count must lie between 1 and the population size")
    if not math.isfinite(crossover_probability) or not 0 <= crossover_probability <= 1:
        raise ValueError("crossover_probability must lie between 0 and 1")
    mutation_kwargs = dict(mutation_kwargs or {})
    if "generator" in mutation_kwargs:
        raise ValueError("Supply the generator to evolve(), not in mutation_kwargs")
    models: list[TorchProgram] = [model.clone() for model in population]

    def score(model: TorchProgram) -> float:
        with torch.no_grad():
            value = loss_fn(model)
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise ValueError("loss_fn must return a scalar")
            value = float(value.detach())
        result = float(value)
        return result if math.isfinite(result) else math.inf

    losses = [score(model) for model in models]
    if not any(math.isfinite(loss) for loss in losses):
        raise ValueError("The entire initial population has non-finite losses")

    def select() -> TorchProgram:
        candidates = torch.randint(
            len(models), (tournament_size,), generator=generator
        ).tolist()
        selected: TorchProgram = models[min(candidates, key=losses.__getitem__)]
        return selected

    history = []
    for generation in range(generations + 1):
        order = sorted(range(len(models)), key=lambda index: losses[index])
        models, losses = [models[index] for index in order], [
            losses[index] for index in order
        ]
        history.append(
            {
                "generation": generation,
                "best_loss": losses[0],
                "mean_loss": sum(losses) / len(losses),
            }
        )
        if generation == generations:
            break
        offspring = [model.clone() for model in models[:elite_count]]
        offspring_losses = list(losses[:elite_count])

        while len(offspring) < len(models):
            parent = select()
            if float(torch.rand((), generator=generator)) < crossover_probability:
                parent = parent.crossover(select(), generator=generator)
            child = parent.mutate(generator=generator, **mutation_kwargs)
            offspring.append(child)
            offspring_losses.append(score(child))
        models, losses = offspring, offspring_losses
    return {
        "best_model": models[0].clone(),
        "best_loss": losses[0],
        "population": models,
        "losses": losses,
        "history": history,
    }
