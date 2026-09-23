"""Run from the repository root: python -m examples.torch_gp

Requires: pip install '.[torch]'
"""

import torch

from genepax.torch_gp import TorchCGP, TorchLGP


def main() -> None:
    # Five data points, each with two inputs. The target is x0**2 + x1.
    x = torch.tensor(
        [
            [-1.0, 0.0],
            [-0.5, 0.25],
            [0.0, 0.5],
            [0.5, 0.75],
            [1.0, 1.0],
        ]
    )
    y = x[:, :1].square() + x[:, 1:2]  # shape: (5, 1)

    for program_type in (TorchCGP, TorchLGP):
        rng = torch.Generator().manual_seed(42)
        size_options = (
            {"n_nodes": 8} if program_type is TorchCGP else {"n_program_lines": 8}
        )
        options = {
            "n_inputs": 2,
            "n_outputs": 1,
            "function_names": ("plus", "minus", "times", "sin"),
            "output_transform": "identity",  # raw regression outputs
            **size_options,
        }

        # 1. Generate one program and evaluate all data points in one call.
        program = program_type.random(**options, generator=rng)
        with torch.no_grad():
            predictions = program(x)  # shape: (5, 1)
            mse = (predictions - y).square().mean()
        print(f"\n{program_type.__name__}")
        print("Single-program predictions:", predictions.flatten().tolist())
        print("Single-program MSE:", mse.item())

        # 2. Generate a population of ten independent programs.
        population = [program_type.random(**options, generator=rng) for _ in range(10)]
        with torch.no_grad():
            # Each program evaluates the entire batch; programs run sequentially.
            predictions = torch.stack([model(x) for model in population])
            # predictions shape: (10 programs, 5 points, 1 output).
            losses = (predictions - y).square().mean(dim=(1, 2))
        best_index = int(losses.argmin())
        best = population[best_index]
        print("Population MSEs:", losses.tolist())
        print("Best program index:", best_index)

        # 3. Mutate the best program. Mutation returns a new model.
        # The parent stays unchanged; these defaults evolve structure only
        # because no numeric weights were marked as trainable.
        child = best.mutate(generator=rng, p_mut_inputs=0.2, p_mut_functions=0.2)
        with torch.no_grad():
            child_loss = (child(x) - y).square().mean()
        print("Mutated child MSE:", child_loss.item())

        # Or mutate every member and evaluate the offspring population.
        offspring = [model.mutate(generator=rng) for model in population]
        with torch.no_grad():
            offspring_predictions = torch.stack([model(x) for model in offspring])
            offspring_losses = (offspring_predictions - y).square().mean(dim=(1, 2))
        print("Offspring MSEs:", offspring_losses.tolist())


if __name__ == "__main__":
    main()
