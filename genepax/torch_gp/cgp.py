"""Cartesian GP evaluation using PyTorch autograd."""

from collections.abc import Callable, Sequence
from typing import Any, Self

import torch
from torch import Tensor

from genepax.torch_gp.functions import NUMERIC_FUNCTIONS
from genepax.torch_gp.initialization import initial_weights, sample_indices
from genepax.torch_gp.program import TorchProgram


class TorchCGP(TorchProgram):
    """Execute a fixed CGP genotype, optionally optimizing its numeric weights."""

    def __init__(
        self,
        genotype: dict[str, dict[str, Any]],
        n_inputs: int,
        n_outputs: int,
        *,
        function_names: Sequence[str] = tuple(NUMERIC_FUNCTIONS),
        output_transform: str | Callable[[Tensor], Tensor] = "tanh",
        trainable_weights: Sequence[str] = (),
        dtype: torch.dtype = torch.float32,
        fixed_outputs: bool = False,
    ) -> None:
        super().__init__(
            genotype,
            n_inputs,
            n_outputs,
            function_names=function_names,
            output_transform=output_transform,
            trainable_weights=trainable_weights,
            dtype=dtype,
        )
        self._validate_arrays("outputs", n_outputs)
        self.n_nodes = self.gene("functions").numel()
        self.fixed_outputs = fixed_outputs
        n_initial = n_inputs + self.n_input_constants
        bounds = torch.arange(n_initial, n_initial + self.n_nodes)
        self._check_bounds("inputs1", 0, bounds)
        self._check_bounds("inputs2", 0, bounds)
        self._check_bounds("outputs", 0, n_initial + self.n_nodes)

    @classmethod
    def random(
        cls,
        n_inputs: int,
        n_outputs: int,
        *,
        n_nodes: int = 50,
        n_input_constants: int = 2,
        fixed_outputs: bool = False,
        function_names: Sequence[str] = tuple(NUMERIC_FUNCTIONS),
        output_transform: str | Callable[[Tensor], Tensor] = "tanh",
        trainable_weights: Sequence[str] = (),
        weights_initialization: str = "uniform",
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> Self:
        """Initialize a CGP with PyTorch randomness; fixed outputs use the last nodes."""
        if min(n_inputs, n_outputs, n_nodes) < 1 or n_input_constants < 0:
            raise ValueError("Invalid input, output, node, or constant count")
        if not function_names or (fixed_outputs and n_outputs > n_nodes):
            raise ValueError(
                "Functions are required; fixed outputs require n_outputs <= n_nodes"
            )
        n_initial = n_inputs + n_input_constants
        bounds = torch.arange(n_initial, n_initial + n_nodes)
        genes = {
            "inputs1": sample_indices(bounds, generator),
            "inputs2": sample_indices(bounds, generator),
            "functions": sample_indices(
                torch.full((n_nodes,), len(function_names)), generator
            ),
            "outputs": (
                torch.arange(n_initial + n_nodes - n_outputs, n_initial + n_nodes)
                if fixed_outputs
                else sample_indices(
                    torch.full((n_outputs,), n_initial + n_nodes), generator
                )
            ),
        }
        weights = initial_weights(
            n_nodes,
            n_input_constants,
            trainable_weights,
            weights_initialization,
            dtype,
            generator,
        )
        return cls(
            {"genes": genes, "weights": weights},
            n_inputs,
            n_outputs,
            function_names=function_names,
            output_transform=output_transform,
            trainable_weights=trainable_weights,
            dtype=dtype,
            fixed_outputs=fixed_outputs,
        )

    def _gene_bounds(self) -> dict[str, tuple[int, Tensor]]:
        n_initial = self.n_inputs + self.n_input_constants
        return {
            "inputs1": (0, torch.arange(n_initial, n_initial + self.n_nodes)),
            "inputs2": (0, torch.arange(n_initial, n_initial + self.n_nodes)),
            "functions": (0, torch.full((self.n_nodes,), len(self.function_names))),
            "outputs": (0, torch.full((self.n_outputs,), n_initial + self.n_nodes)),
        }

    def _targets_and_outputs(self) -> tuple[list[int], list[int]]:
        start = self.n_inputs + self.n_input_constants
        return list(range(start, start + self.n_nodes)), self.gene("outputs").tolist()

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "fixed_outputs": self.fixed_outputs}

    def forward(self, observations: Tensor) -> Tensor:
        memory = self._memory(observations)
        instructions = zip(
            self.gene("inputs1").tolist(),
            self.gene("inputs2").tolist(),
            self.gene("functions").tolist(),
        )
        for index, (x_index, y_index, function) in enumerate(instructions):
            memory.append(self._instruction(memory, index, x_index, y_index, function))
        return self._outputs([memory[index] for index in self.gene("outputs").tolist()])
