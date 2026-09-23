"""Native CPU random sampling, independent of the evaluation device."""

from collections.abc import Sequence

import torch
from torch import Tensor


def check_generator(generator: torch.Generator | None) -> None:
    if generator is not None and generator.device.type != "cpu":
        raise ValueError("Use a CPU torch.Generator, including for CUDA models")


def sample_indices(bounds: Tensor, generator: torch.Generator | None = None) -> Tensor:
    check_generator(generator)
    return (torch.rand(bounds.shape, generator=generator) * bounds).long()


def initial_weights(
    count: int,
    n_constants: int,
    trainable_weights: Sequence[str],
    initialization: str,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> dict[str, Tensor]:
    check_generator(generator)
    if count < 1 or n_constants < 0:
        raise ValueError(
            "Instruction count must be positive and constant count non-negative"
        )
    if initialization not in ("natural", "uniform"):
        raise ValueError("weights_initialization must be 'natural' or 'uniform'")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Protected GP programs require float32 or float64")
    constants = torch.rand(n_constants, generator=generator, dtype=dtype) * 2 - 1
    if "program_inputs" not in trainable_weights:
        defaults = torch.tensor([0.1, 1.0], dtype=dtype)
        constants[: min(2, n_constants)] = defaults[: min(2, n_constants)]
    weights = {"program_inputs": constants}
    for name in (
        "inputs1",
        "inputs2",
        "functions",
        "inputs1_biases",
        "inputs2_biases",
        "functions_biases",
    ):
        if name in trainable_weights and initialization == "uniform":
            weights[name] = torch.rand(count, generator=generator, dtype=dtype) * 2 - 1
        else:
            weights[name] = torch.full(
                (count,), 0.0 if "biases" in name else 1.0, dtype=dtype
            )
    return weights
