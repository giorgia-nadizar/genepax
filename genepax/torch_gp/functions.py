"""PyTorch equivalents of the built-in, protected GP primitives."""

import math
from collections.abc import Callable

import torch
from torch import Tensor

EPS = 1e-6
MAX_ABS_VALUE = 1e6


def finite_value(value: Tensor) -> Tensor:
    return torch.nan_to_num(
        value, nan=0.0, posinf=MAX_ABS_VALUE, neginf=-MAX_ABS_VALUE
    ).clamp(-MAX_ABS_VALUE, MAX_ABS_VALUE)


def protected_division(x: Tensor, y: Tensor) -> Tensor:
    x, y = finite_value(x), finite_value(y)
    small = y.abs() < EPS
    denominator = torch.where(small, torch.ones_like(y), y)
    return finite_value(torch.where(small, torch.zeros_like(x), x / denominator))


def protected_power(x: Tensor, y: Tensor) -> Tensor:
    x, y = finite_value(x), finite_value(y)
    log_magnitude = y * torch.log(x.abs() + EPS)
    bound = math.log(MAX_ABS_VALUE)
    return finite_value(x.sign() * log_magnitude.clamp(-bound, bound).exp())


# Order matches genepax.gp.functions.function_set_numeric. Function indices
# refer to this order unless a different function_names sequence is supplied.
NUMERIC_FUNCTIONS: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "plus": lambda x, y: finite_value(finite_value(x) + finite_value(y)),
    "minus": lambda x, y: finite_value(finite_value(x) - finite_value(y)),
    "times": lambda x, y: finite_value(finite_value(x) * finite_value(y)),
    "prot_div": protected_division,
    "abs": lambda x, y: finite_value(x).abs(),
    "safe_exp": lambda x, y: finite_value(finite_value(x).clamp(-20, 20).exp()),
    "sin": lambda x, y: finite_value(x).sin(),
    "cos": lambda x, y: finite_value(x).cos(),
    "prot_log": lambda x, y: finite_value((finite_value(x).abs() + EPS).log()),
    "sqrt": lambda x, y: (finite_value(x).abs() + EPS).sqrt(),
    "pow": protected_power,
    "identity": lambda x, y: finite_value(x),
}

BOOLEAN_FUNCTIONS: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "and": torch.logical_and,
    "or": torch.logical_or,
    "xor": torch.logical_xor,
    "and_not": lambda x, y: torch.logical_and(x, torch.logical_not(y)),
}

FUNCTIONS = {**NUMERIC_FUNCTIONS, **BOOLEAN_FUNCTIONS}
ARITIES = {
    name: (
        1
        if name in {"abs", "safe_exp", "sin", "cos", "prot_log", "sqrt", "identity"}
        else 2
    )
    for name in FUNCTIONS
}
