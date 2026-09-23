"""Shared storage and evaluation helpers for fixed GP programs."""

import copy
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from torch import Tensor, nn

from genepax.torch_gp.functions import ARITIES, FUNCTIONS, NUMERIC_FUNCTIONS
from genepax.torch_gp.initialization import check_generator, sample_indices

WEIGHT_NAMES = (
    "program_inputs",
    "inputs1",
    "inputs2",
    "functions",
    "inputs1_biases",
    "inputs2_biases",
    "functions_biases",
)


def _copy_tensor(value: Any, dtype: torch.dtype) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().to(device="cpu", dtype=dtype).clone()
    # Copy JAX/NumPy arrays so neither backend shares mutable storage.
    return torch.tensor(np.asarray(value).copy(), dtype=dtype)


class TorchProgram(nn.Module):
    """A fixed integer program with optional trainable numeric parameters.

    Inputs have shape (..., n_inputs), outputs (..., n_outputs). Genes and
    frozen weights are buffers; trainable weights are Parameters. Construct on
    CPU, then use the usual module.to(device=..., dtype=...) API.
    """

    output_transform: str | Callable[[Tensor], Tensor]

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
    ) -> None:
        super().__init__()
        if n_inputs < 1 or n_outputs < 1:
            raise ValueError("Input and output counts must be positive")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Protected GP programs require float32 or float64")
        self.n_inputs, self.n_outputs = n_inputs, n_outputs
        self.function_names = tuple(function_names)
        if not self.function_names or any(
            name not in FUNCTIONS for name in self.function_names
        ):
            raise ValueError("function_names must contain supported GP primitives")
        self._functions = tuple(FUNCTIONS[name] for name in self.function_names)
        if isinstance(output_transform, str):
            if output_transform not in ("tanh", "identity"):
                raise ValueError(
                    "output_transform must be 'tanh', 'identity', or callable"
                )
            self.output_transform = output_transform
        elif callable(output_transform):
            self.output_transform = output_transform
        else:
            raise TypeError("output_transform must be 'tanh', 'identity', or callable")
        if set(trainable_weights) - set(WEIGHT_NAMES):
            raise ValueError("Unknown trainable weight names")
        if set(genotype) != {"genes", "weights"}:
            raise ValueError("A genotype must contain exactly 'genes' and 'weights'")
        if set(genotype["weights"]) != set(WEIGHT_NAMES):
            raise ValueError("Expected the seven standard GGP weight arrays")
        if not {"inputs1", "inputs2", "functions"}.issubset(genotype["genes"]):
            raise ValueError("Missing input or function gene arrays")

        self.genes = nn.Module()
        for name, values in genotype["genes"].items():
            raw = _copy_tensor(values, torch.float64)
            if (
                raw.ndim != 1
                or not torch.isfinite(raw).all()
                or (raw != raw.round()).any()
            ):
                raise ValueError(f"Gene array {name!r} must be a finite integer vector")
            self.genes.register_buffer(name, raw.to(torch.long))
        self.weights = nn.Module()
        for name in WEIGHT_NAMES:
            value = _copy_tensor(genotype["weights"][name], dtype)
            if value.ndim != 1 or not torch.isfinite(value).all():
                raise ValueError(f"Weight array {name!r} must be a finite vector")
            if name in trainable_weights:
                self.weights.register_parameter(name, nn.Parameter(value))
            else:
                self.weights.register_buffer(name, value)
        self.n_input_constants = self.weight("program_inputs").numel()

    def gene(self, name: str) -> Tensor:
        value: Tensor = getattr(self.genes, name)
        return value

    def weight(self, name: str) -> Tensor:
        value: Tensor = getattr(self.weights, name)
        return value

    def _validate_arrays(self, extra_gene: str, extra_length: int) -> None:
        expected = {"inputs1", "inputs2", "functions", extra_gene}
        if set(dict(self.genes.named_buffers())) != expected:
            raise ValueError(f"Expected gene arrays {sorted(expected)}")
        count = self.gene("functions").numel()
        if count < 1:
            raise ValueError("A program must contain at least one instruction")
        for name in expected:
            length = extra_length if name == extra_gene else count
            if self.gene(name).shape != (length,):
                raise ValueError(f"Incorrect shape for gene array {name!r}")
        for name in WEIGHT_NAMES[1:]:
            if self.weight(name).shape != (count,):
                raise ValueError(f"Expected {count} values for weight array {name!r}")
        self._check_bounds("functions", 0, len(self.function_names))

    def _check_bounds(self, name: str, lower: int, upper: int | Tensor) -> None:
        values = self.gene(name)
        if ((values < lower) | (values >= upper)).any():
            raise ValueError(f"Gene array {name!r} contains out-of-range indices")

    def _memory(self, observations: Tensor, n_registers: int = 0) -> list[Tensor]:
        if observations.ndim < 1 or observations.shape[-1] != self.n_inputs:
            raise ValueError(
                f"Expected observations with last dimension {self.n_inputs}"
            )
        constants = self.weight("program_inputs")
        if observations.dtype not in (torch.float32, torch.float64):
            raise ValueError("Observations must use float32 or float64")
        if (
            observations.dtype != constants.dtype
            or observations.device != constants.device
        ):
            raise ValueError(
                "Observations and module must have the same dtype and device"
            )
        shape = observations.shape[:-1]
        return (
            list(observations.unbind(-1))
            + [constant.expand(shape) for constant in constants.unbind()]
            + [observations.new_zeros(shape) for _ in range(n_registers)]
        )

    def _instruction(
        self,
        memory: list[Tensor],
        index: int,
        x_index: int,
        y_index: int,
        function: int,
    ) -> Tensor:
        x = (
            memory[x_index] * self.weight("inputs1")[index]
            + self.weight("inputs1_biases")[index]
        )
        y = (
            memory[y_index] * self.weight("inputs2")[index]
            + self.weight("inputs2_biases")[index]
        )
        return (
            self._functions[function](x, y) * self.weight("functions")[index]
            + self.weight("functions_biases")[index]
        )

    def _outputs(self, outputs: list[Tensor]) -> Tensor:
        result = torch.stack(outputs, dim=-1)
        if isinstance(self.output_transform, str):
            return result.tanh() if self.output_transform == "tanh" else result
        return self.output_transform(result)

    def to_genotype(self) -> dict[str, dict[str, np.ndarray]]:
        """Export independent CPU NumPy arrays, including optimized weights.

        These arrays can be passed to the matching JAX structure, or converted
        with jax.tree.map(jax.numpy.asarray, genotype).
        """
        return {
            "genes": {
                name: value.detach().cpu().numpy().copy()
                for name, value in self.genes.named_buffers()
            },
            "weights": {
                name: self.weight(name).detach().cpu().numpy().copy()
                for name in WEIGHT_NAMES
            },
        }

    def tensor_genotype(self) -> dict[str, dict[str, Tensor]]:
        """Export detached tensor copies on the model's device."""
        return {
            "genes": {
                name: value.detach().clone()
                for name, value in self.genes.named_buffers()
            },
            "weights": {
                name: self.weight(name).detach().clone() for name in WEIGHT_NAMES
            },
        }

    def clone(self) -> Self:
        """Copy a program without sharing storage, gradients, or optimizer state."""
        return copy.deepcopy(self)

    def get_config(self) -> dict[str, Any]:
        """Constructor options needed to interpret this model's genotype."""
        return {
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
            "function_names": self.function_names,
            "output_transform": self.output_transform,
            "trainable_weights": tuple(
                name
                for name, value in self.weights.named_parameters()
                if value.requires_grad
            ),
            "dtype": self.weight("program_inputs").dtype,
        }

    def _gene_bounds(self) -> dict[str, tuple[int, Tensor]]:
        raise NotImplementedError

    def _targets_and_outputs(self) -> tuple[list[int], list[int]]:
        raise NotImplementedError

    def mutate(
        self,
        *,
        generator: torch.Generator | None = None,
        p_mut_inputs: float = 0.1,
        p_mut_functions: float = 0.1,
        p_mut_outputs: float = 0.3,
        p_mut_targets: float = 0.3,
        weights_mut_sigma: float = 0.1,
        weights_mutation: bool = True,
        weights_mutation_type: str = "gaussian",
        mutation_probabilities: dict[str, float] | None = None,
    ) -> Self:
        """Return an independent offspring; the parent and its optimizer stay intact.

        Genes are resampled within their legal bounds (the value may stay the
        same). Only trainable numeric weights are mutated. Random draws use a
        CPU generator even when the model is on CUDA.
        """
        check_generator(generator)
        probabilities = {
            "inputs": p_mut_inputs,
            "functions": p_mut_functions,
            "outputs": p_mut_outputs,
            "targets": p_mut_targets,
            "weights_sigma": weights_mut_sigma,
        }
        if mutation_probabilities is not None:
            if set(mutation_probabilities) - set(probabilities):
                raise ValueError("Unknown mutation probability keys")
            probabilities.update(mutation_probabilities)
        sigma = probabilities.pop("weights_sigma")
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
            raise ValueError(
                "Mutation probabilities must be finite and between 0 and 1"
            )
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("weights_mut_sigma must be finite and non-negative")
        if weights_mutation_type not in ("gaussian", "automl0"):
            raise ValueError("weights_mutation_type must be 'gaussian' or 'automl0'")
        child = self.clone()
        with torch.no_grad():
            for name, (lower, upper) in self._gene_bounds().items():
                if name == "outputs" and getattr(self, "fixed_outputs", False):
                    continue
                probability = probabilities[
                    "inputs" if name.startswith("inputs") else name
                ]
                mask = torch.rand(upper.shape, generator=generator) < probability
                donor = sample_indices(upper - lower, generator) + lower
                destination = child.gene(name)
                destination.copy_(
                    torch.where(
                        mask.to(destination.device),
                        donor.to(destination.device),
                        destination,
                    )
                )
            if weights_mutation:
                for _, value in child.weights.named_parameters():
                    if not value.requires_grad:
                        continue
                    if weights_mutation_type == "gaussian":
                        noise = torch.randn(
                            value.shape, generator=generator, dtype=value.dtype
                        )
                        value.add_(noise.to(value.device) * sigma)
                    else:
                        # Randomly double or halve the magnitude and optionally flip its sign.
                        factor = 1 + torch.rand(
                            value.shape, generator=generator, dtype=value.dtype
                        )
                        halve = torch.rand(value.shape, generator=generator) < 0.5
                        factor = torch.where(halve, factor.reciprocal(), factor)
                        sign = torch.where(
                            torch.rand(value.shape, generator=generator) < 0.5, -1, 1
                        )
                        value.mul_((factor * sign).to(value.device))
        return child

    def crossover(
        self, other: Self, *, generator: torch.Generator | None = None
    ) -> Self:
        """One-point crossover of instructions/nodes and their numeric weights.

        Constants come from the first parent. CGP output connections are chosen
        independently from either parent (or retained when fixed_outputs=True).
        Parents must have identical configurations, shapes, and devices.
        """
        check_generator(generator)
        if type(self) is not type(other) or self.get_config() != other.get_config():
            raise ValueError("Crossover requires matching program configurations")
        left, right = self.tensor_genotype(), other.tensor_genotype()
        for section in left:
            for name, value in left[section].items():
                peer = right[section][name]
                if value.shape != peer.shape or value.device != peer.device:
                    raise ValueError(
                        "Crossover requires matching genotype shapes and devices"
                    )
        child = self.clone()
        count = self.gene("functions").numel()
        cut = int(torch.randint(1, count + 1, (), generator=generator))
        with torch.no_grad():
            for name, _ in self.genes.named_buffers():
                if name == "outputs":
                    if not getattr(self, "fixed_outputs", False):
                        mask = torch.rand((self.n_outputs,), generator=generator) < 0.5
                        child.gene(name).copy_(
                            torch.where(
                                mask.to(self.gene(name).device),
                                self.gene(name),
                                other.gene(name),
                            )
                        )
                else:
                    child.gene(name)[cut:].copy_(other.gene(name)[cut:])
            for name in WEIGHT_NAMES[1:]:
                child.weight(name)[cut:].copy_(other.weight(name)[cut:])
        return child

    def compute_active_mask(self) -> Tensor:
        """Structural liveness of nodes/instructions, accounting for unary functions."""
        targets, outputs = self._targets_and_outputs()
        needed = set(outputs)
        active = [False] * len(targets)
        x_indices, y_indices = (
            self.gene("inputs1").tolist(),
            self.gene("inputs2").tolist(),
        )
        functions = self.gene("functions").tolist()
        for index in reversed(range(len(targets))):
            if targets[index] in needed:
                active[index] = True
                # Kill the overwritten value before adding this instruction's reads.
                needed.remove(targets[index])
                needed.add(x_indices[index])
                if ARITIES[self.function_names[functions[index]]] == 2:
                    needed.add(y_indices[index])
        return torch.tensor(
            active, dtype=torch.bool, device=self.gene("functions").device
        )

    def size(self) -> int:
        return int(self.compute_active_mask().sum())

    def compute_complexity(self) -> float:
        return float(self.size() / self.gene("functions").numel())

    def compute_function_count(self) -> Tensor:
        return torch.bincount(
            self.gene("functions")[self.compute_active_mask()],
            minlength=len(self.function_names),
        )

    def compute_function_arities(self) -> Tensor:
        counts = self.compute_function_count()
        arities = torch.tensor(
            [ARITIES[name] for name in self.function_names], device=counts.device
        )
        return (
            torch.stack([counts[arities == arity].sum() for arity in (1, 2)])
            / self.gene("functions").numel()
        )

    def get_readable_program(self) -> str:
        """Display active instructions, weights, constants, and output transformation.

        Primitive names denote the protected operations in torch_gp.functions;
        the returned text is for inspection, not an executable serialization.
        """
        targets, outputs = self._targets_and_outputs()
        initial = self.n_inputs + self.n_input_constants
        register_count = max(initial, max(targets) + 1, max(outputs) + 1)
        lines = [f"r = zeros({register_count})"]
        lines += [f"r[{index}] = x{index}" for index in range(self.n_inputs)]
        lines += [
            f"r[{self.n_inputs + index}] = {value:.6g}"
            for index, value in enumerate(
                self.weight("program_inputs").detach().cpu().tolist()
            )
        ]

        def affine(expression: str, weight: float, bias: float) -> str:
            if weight != 1:
                expression = f"({weight:.6g} * {expression})"
            if bias != 0:
                expression = f"({expression} + {bias:.6g})"
            return expression

        weights = {
            name: self.weight(name).detach().cpu().tolist() for name in WEIGHT_NAMES[1:]
        }
        for index, active in enumerate(self.compute_active_mask().tolist()):
            if not active:
                continue
            name = self.function_names[int(self.gene("functions")[index])]
            arguments = []
            for key in ("inputs1", "inputs2")[: ARITIES[name]]:
                arguments.append(
                    affine(
                        f"r[{int(self.gene(key)[index])}]",
                        weights[key][index],
                        weights[f"{key}_biases"][index],
                    )
                )
            expression = affine(
                f"{name}({', '.join(arguments)})",
                weights["functions"][index],
                weights["functions_biases"][index],
            )
            lines.append(f"r[{targets[index]}] = {expression}")
        transform = (
            self.output_transform
            if isinstance(self.output_transform, str)
            else getattr(self.output_transform, "__name__", "custom_transform")
        )
        lines.append(
            f"return {transform}([{', '.join(f'r[{index}]' for index in outputs)}])"
        )
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Save tensors and configuration; no JAX objects or arbitrary callables."""
        if not isinstance(self.output_transform, str):
            raise ValueError(
                "save() requires 'tanh' or 'identity'; use state_dict for custom transforms"
            )
        genotype = self.tensor_genotype()
        genotype = {
            section: {name: value.cpu() for name, value in arrays.items()}
            for section, arrays in genotype.items()
        }
        torch.save(
            {
                "version": 1,
                "kind": type(self).__name__,
                "config": self.get_config(),
                "genotype": genotype,
                "training": self.training,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device = "cpu") -> Self:
        """Restore a complete program written by save(), including trainable weights."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("version") != 1 or checkpoint.get("kind") != cls.__name__:
            raise ValueError(f"Not a supported {cls.__name__} checkpoint")
        model = cls(checkpoint["genotype"], **checkpoint["config"])
        model.to(device)
        model.train(checkpoint["training"])
        return model

    @classmethod
    def from_jax(
        cls,
        structure: Any,
        genotype: dict[str, dict[str, Any]],
        *,
        output_transform: str | Callable[[Tensor], Tensor] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Self:
        """Copy a JAX CGP/LGP and respect its trainable-weight flags.

        Supports built-in primitives in any order/subset. A non-default JAX
        output wrapper requires an explicit PyTorch output_transform; Python
        lambdas and arbitrary JAX functions cannot be translated automatically.
        """
        import jax.numpy as jnp

        from genepax.gp.cartesian_genetic_programming import CGP
        from genepax.gp.functions import function_set_boolean, function_set_numeric
        from genepax.gp.linear_genetic_programming import LGP
        from genepax.torch_gp.cgp import TorchCGP

        expected_type = CGP if issubclass(cls, TorchCGP) else LGP
        if not isinstance(structure, expected_type):
            raise TypeError(
                f"{cls.__name__}.from_jax requires {expected_type.__name__}"
            )
        builtins = {**function_set_numeric, **function_set_boolean}
        for name, function in structure.function_set.function_set.items():
            if name not in builtins or function is not builtins[name]:
                raise ValueError(
                    f"No automatic PyTorch translation for function {name!r}"
                )
        if output_transform is None:
            if structure.outputs_wrapper is not jnp.tanh:
                raise ValueError(
                    "Supply output_transform for a non-default JAX output wrapper"
                )
            output_transform = "tanh"
        options: dict[str, Any] = {}
        if isinstance(structure, LGP):
            options["n_computation_registers"] = structure.n_computation_registers
        else:
            options["fixed_outputs"] = structure.fixed_outputs
        return cls(
            genotype,
            n_inputs=structure.n_inputs,
            n_outputs=structure.n_outputs,
            function_names=tuple(structure.function_set.function_set),
            output_transform=output_transform,
            trainable_weights=tuple(structure.get_weights(genotype)),
            dtype=dtype,
            **options,
        )
