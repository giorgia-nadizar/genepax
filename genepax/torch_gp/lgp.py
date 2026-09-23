"""Linear GP evaluation using PyTorch autograd."""

from collections.abc import Callable, Sequence
from typing import Any, Self

import torch
from torch import Tensor

from genepax.torch_gp.functions import NUMERIC_FUNCTIONS
from genepax.torch_gp.initialization import initial_weights, sample_indices
from genepax.torch_gp.program import TorchProgram


class TorchLGP(TorchProgram):
    """Execute a fixed LGP genotype without in-place tensor register updates."""

    def __init__(
        self,
        genotype: dict[str, dict[str, Any]],
        n_inputs: int,
        n_outputs: int,
        *,
        n_computation_registers: int = 5,
        function_names: Sequence[str] = tuple(NUMERIC_FUNCTIONS),
        output_transform: str | Callable[[Tensor], Tensor] = "tanh",
        trainable_weights: Sequence[str] = (),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if n_computation_registers < 0:
            raise ValueError("The computation register count cannot be negative")
        super().__init__(
            genotype,
            n_inputs,
            n_outputs,
            function_names=function_names,
            output_transform=output_transform,
            trainable_weights=trainable_weights,
            dtype=dtype,
        )
        self.n_program_lines = self.gene("functions").numel()
        self._validate_arrays("targets", self.n_program_lines)
        self.n_computation_registers = n_computation_registers
        n_initial = n_inputs + self.n_input_constants
        self.n_registers = n_initial + n_computation_registers + n_outputs
        self._check_bounds("inputs1", 0, self.n_registers)
        self._check_bounds("inputs2", 0, self.n_registers)
        self._check_bounds("targets", n_initial, self.n_registers)

    @classmethod
    def random(
        cls,
        n_inputs: int,
        n_outputs: int,
        *,
        n_program_lines: int = 15,
        n_computation_registers: int = 5,
        n_input_constants: int = 2,
        function_names: Sequence[str] = tuple(NUMERIC_FUNCTIONS),
        output_transform: str | Callable[[Tensor], Tensor] = "tanh",
        trainable_weights: Sequence[str] = (),
        weights_initialization: str = "uniform",
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> Self:
        """Initialize instructions and registers with native PyTorch randomness."""
        if (
            min(n_inputs, n_outputs, n_program_lines) < 1
            or min(n_input_constants, n_computation_registers) < 0
        ):
            raise ValueError(
                "Invalid input, output, instruction, register, or constant count"
            )
        if not function_names:
            raise ValueError("At least one function is required")
        n_initial = n_inputs + n_input_constants
        n_assignable = n_computation_registers + n_outputs
        bounds = torch.full((n_program_lines,), n_initial + n_assignable)
        genes = {
            "inputs1": sample_indices(bounds, generator),
            "inputs2": sample_indices(bounds, generator),
            "functions": sample_indices(
                torch.full((n_program_lines,), len(function_names)), generator
            ),
            "targets": n_initial
            + sample_indices(torch.full((n_program_lines,), n_assignable), generator),
        }
        weights = initial_weights(
            n_program_lines,
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
            n_computation_registers=n_computation_registers,
            function_names=function_names,
            output_transform=output_transform,
            trainable_weights=trainable_weights,
            dtype=dtype,
        )

    def _gene_bounds(self) -> dict[str, tuple[int, Tensor]]:
        n_initial = self.n_inputs + self.n_input_constants
        bounds = torch.full((self.n_program_lines,), self.n_registers)
        return {
            "inputs1": (0, bounds),
            "inputs2": (0, bounds),
            "functions": (
                0,
                torch.full((self.n_program_lines,), len(self.function_names)),
            ),
            "targets": (n_initial, bounds),
        }

    def _targets_and_outputs(self) -> tuple[list[int], list[int]]:
        return self.gene("targets").tolist(), list(
            range(self.n_registers - self.n_outputs, self.n_registers)
        )

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "n_computation_registers": self.n_computation_registers,
        }

    def forward(self, observations: Tensor) -> Tensor:
        memory = self._memory(
            observations, self.n_computation_registers + self.n_outputs
        )
        instructions = zip(
            self.gene("inputs1").tolist(),
            self.gene("inputs2").tolist(),
            self.gene("functions").tolist(),
            self.gene("targets").tolist(),
        )
        for index, (x_index, y_index, function, target) in enumerate(instructions):
            # Replace a Python list entry, preserving the tensor that previously
            # occupied the register for any backward computation that needs it.
            memory[target] = self._instruction(
                memory, index, x_index, y_index, function
            )
        return self._outputs(memory[-self.n_outputs :])
