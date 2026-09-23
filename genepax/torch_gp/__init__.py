"""Standalone PyTorch construction, evolution, and training of CGP/LGP programs."""

from genepax.torch_gp.cgp import TorchCGP
from genepax.torch_gp.evolution import evolve
from genepax.torch_gp.lgp import TorchLGP

__all__ = ["TorchCGP", "TorchLGP", "evolve"]
