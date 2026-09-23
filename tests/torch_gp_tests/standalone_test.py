"""Native workflows: these tests must run with no JAX or QDax installed."""

import math
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from genepax.torch_gp import TorchCGP, TorchLGP, evolve  # noqa: E402
from genepax.torch_gp.program import WEIGHT_NAMES  # noqa: E402


def seeded(seed: int = 3) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def assert_same(left, right) -> None:
    assert left.get_config() == right.get_config()
    for group, arrays in left.tensor_genotype().items():
        for name, values in arrays.items():
            torch.testing.assert_close(
                values, right.tensor_genotype()[group][name], rtol=0, atol=0
            )


@pytest.mark.parametrize("cls", [TorchCGP, TorchLGP])
@pytest.mark.parametrize("n_constants", [0, 1, 2, 5])
@pytest.mark.parametrize("initialization", ["uniform", "natural"])
def test_native_initialization_and_reproducibility(
    cls, n_constants: int, initialization: str
) -> None:
    options = {
        "n_inputs": 3,
        "n_outputs": 2,
        "n_input_constants": n_constants,
        "weights_initialization": initialization,
        "trainable_weights": ("inputs1",),
    }
    model = cls.random(**options, generator=seeded())
    assert_same(model, cls.random(**options, generator=seeded()))
    # Reconstructing validates every gene bound and weight dimension.
    cls(model.tensor_genotype(), **model.get_config())
    assert torch.isfinite(model(torch.ones(4, 3))).all()
    assert set(dict(model.named_parameters())) == {"weights.inputs1"}
    torch.testing.assert_close(
        model.weight("inputs2"), torch.ones_like(model.weight("inputs2"))
    )
    torch.testing.assert_close(
        model.weight("functions_biases"),
        torch.zeros_like(model.weight("functions_biases")),
    )
    if initialization == "natural":
        torch.testing.assert_close(
            model.weight("inputs1"), torch.ones_like(model.weight("inputs1"))
        )
    if n_constants:
        torch.testing.assert_close(
            model.weight("program_inputs")[:2], torch.tensor([0.1, 1.0])[:n_constants]
        )


@pytest.mark.parametrize("cls", [TorchCGP, TorchLGP])
@pytest.mark.parametrize("mutation", ["gaussian", "automl0"])
def test_mutation_is_valid_reproducible_and_preserves_parent(
    cls, mutation: str
) -> None:
    model = cls.random(
        3, 2, trainable_weights=("inputs1", "functions_biases"), generator=seeded()
    )
    original = model.clone()
    options = {
        "p_mut_inputs": 1.0,
        "p_mut_functions": 1.0,
        "p_mut_outputs": 1.0,
        "p_mut_targets": 1.0,
        "weights_mutation_type": mutation,
    }
    child = model.mutate(generator=seeded(5), **options)
    assert_same(child, model.mutate(generator=seeded(5), **options))
    assert_same(model, original)
    assert not torch.equal(child.weight("inputs1"), model.weight("inputs1"))
    assert not torch.equal(child.gene("inputs1"), model.gene("inputs1"))
    for name in (
        "inputs2",
        "functions",
        "inputs1_biases",
        "inputs2_biases",
        "program_inputs",
    ):
        torch.testing.assert_close(
            child.weight(name), model.weight(name), rtol=0, atol=0
        )
    # Repeated mutation keeps graph connections and register writes valid.
    for seed in range(12):
        child = child.mutate(generator=seeded(seed), **options)
        cls(child.tensor_genotype(), **child.get_config())
    assert torch.isfinite(child(torch.zeros(3))).all()


@pytest.mark.parametrize("cls", [TorchCGP, TorchLGP])
def test_disabled_mutation_and_independent_storage(cls) -> None:
    model = cls.random(2, 1, trainable_weights=WEIGHT_NAMES, generator=seeded())
    child = model.mutate(
        mutation_probabilities={
            "inputs": 0,
            "functions": 0,
            "outputs": 0,
            "targets": 0,
            "weights_sigma": 0,
        }
    )
    assert_same(model, child)
    for name in WEIGHT_NAMES:
        assert model.weight(name).data_ptr() != child.weight(name).data_ptr()
    with torch.no_grad():
        child.weight("functions").add_(5)
    assert not torch.equal(child.weight("functions"), model.weight("functions"))


def test_fixed_outputs_remain_fixed() -> None:
    model = TorchCGP.random(2, 3, n_nodes=7, fixed_outputs=True, generator=seeded())
    torch.testing.assert_close(model.gene("outputs"), torch.tensor([8, 9, 10]))
    child = model.mutate(generator=seeded(6), p_mut_outputs=1)
    torch.testing.assert_close(child.gene("outputs"), model.gene("outputs"))
    peer = TorchCGP.random(2, 3, n_nodes=7, fixed_outputs=True, generator=seeded(8))
    torch.testing.assert_close(
        model.crossover(peer).gene("outputs"), model.gene("outputs")
    )


@pytest.mark.parametrize("cls", [TorchCGP, TorchLGP])
def test_crossover_transfers_aligned_instruction_weights(cls) -> None:
    options = {"n_inputs": 2, "n_outputs": 1, "trainable_weights": WEIGHT_NAMES}
    left, right = cls.random(**options, generator=seeded()), cls.random(
        **options, generator=seeded(8)
    )
    with torch.no_grad():
        for name in WEIGHT_NAMES:
            left.weight(name).fill_(1)
            right.weight(name).fill_(2)
    child = left.crossover(right, generator=seeded(12))
    assert_same(child, left.crossover(right, generator=seeded(12)))
    first = child.weight("functions") == 1
    assert first[0]
    assert not ((~first[:-1]) & first[1:]).any()
    for name in WEIGHT_NAMES[1:]:
        torch.testing.assert_close(
            child.weight(name),
            torch.where(first, left.weight(name), right.weight(name)),
        )
    for name in ("inputs1", "inputs2", "functions"):
        torch.testing.assert_close(
            child.gene(name), torch.where(first, left.gene(name), right.gene(name))
        )
    torch.testing.assert_close(
        child.weight("program_inputs"), left.weight("program_inputs")
    )
    cls(child.tensor_genotype(), **child.get_config())
    with pytest.raises(ValueError, match="configurations"):
        left.crossover(cls.random(3, 1))


def test_active_nodes_ignore_unused_unary_argument() -> None:
    model = TorchCGP.random(
        1,
        1,
        n_nodes=3,
        n_input_constants=0,
        function_names=("plus", "identity"),
        output_transform="identity",
    )
    model.gene("inputs1").copy_(torch.tensor([0, 0, 1]))
    model.gene("inputs2").copy_(torch.tensor([0, 0, 2]))
    model.gene("functions").copy_(torch.tensor([0, 0, 1]))
    model.gene("outputs").copy_(torch.tensor([3]))
    assert model.compute_active_mask().tolist() == [True, False, True]
    assert model.size() == 2
    assert model.compute_complexity() == pytest.approx(2 / 3)
    assert model.compute_function_count().tolist() == [1, 1]
    torch.testing.assert_close(
        model.compute_function_arities(), torch.tensor([1 / 3, 1 / 3])
    )
    text = model.get_readable_program()
    assert "r[2] =" not in text
    assert "return identity([r[3]])" in text


def test_lgp_liveness_kills_overwritten_values() -> None:
    model = TorchLGP.random(
        1,
        1,
        n_input_constants=0,
        n_computation_registers=1,
        n_program_lines=4,
        function_names=("plus", "identity"),
        output_transform="identity",
    )
    # r1=x+x; r2=r1+x; r2=x; r2=r2+x. First two writes are dead.
    model.gene("inputs1").copy_(torch.tensor([0, 1, 0, 2]))
    model.gene("inputs2").copy_(torch.tensor([0, 0, 1, 0]))
    model.gene("targets").copy_(torch.tensor([1, 2, 2, 2]))
    model.gene("functions").copy_(torch.tensor([0, 0, 1, 0]))
    assert model.compute_active_mask().tolist() == [False, False, True, True]
    torch.testing.assert_close(model(torch.tensor([0.3])), torch.tensor([0.6]))
    assert "identity" in model.get_readable_program()


@pytest.mark.parametrize("cls", [TorchCGP, TorchLGP])
def test_native_gradient_training_after_mutation(cls) -> None:
    options = (
        {"n_nodes": 3, "fixed_outputs": True}
        if cls is TorchCGP
        else {"n_program_lines": 3}
    )
    model = cls.random(
        1,
        1,
        function_names=("plus",),
        output_transform="identity",
        trainable_weights=("functions_biases",),
        weights_initialization="natural",
        generator=seeded(),
        **options,
    ).mutate(generator=seeded(1))
    if cls is TorchLGP:
        model.gene("targets")[-1] = model.n_registers - 1
    x = torch.linspace(-0.5, 0.5, 8)[:, None]
    target = model(x).detach() + 1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    initial = float((model(x).detach() - target).square().mean())
    for _ in range(50):
        optimizer.zero_grad()
        loss = (model(x) - target).square().mean()
        loss.backward()
        optimizer.step()
    assert float((model(x).detach() - target).square().mean()) < initial / 10


@pytest.mark.parametrize("cls", [TorchCGP, TorchLGP])
def test_complete_checkpoint_roundtrip(cls, tmp_path: Path) -> None:
    model = cls.random(
        2,
        2,
        trainable_weights=("program_inputs", "inputs2"),
        dtype=torch.float64,
        function_names=("sin", "plus"),
        output_transform="identity",
        generator=seeded(),
    ).eval()
    path = tmp_path / "program.pt"
    model.save(path)
    restored = cls.load(path)
    assert not restored.training
    assert_same(model, restored)
    x = torch.ones(4, 2, dtype=torch.float64)
    torch.testing.assert_close(model(x), restored(x))
    other = TorchLGP if cls is TorchCGP else TorchCGP
    with pytest.raises(ValueError, match="checkpoint"):
        other.load(path)


@pytest.mark.parametrize("cls", [TorchCGP, TorchLGP])
def test_evolution_minimizes_loss_and_preserves_inputs(cls) -> None:
    population = [
        cls.random(
            1,
            1,
            function_names=("plus", "times"),
            output_transform="identity",
            trainable_weights=("functions_biases",),
            generator=seeded(seed),
        )
        for seed in range(6)
    ]
    originals = [model.clone() for model in population]
    x = torch.linspace(-1, 1, 8)[:, None]

    def loss_fn(model) -> torch.Tensor:
        return (model(x) - x.square()).square().mean()

    result = evolve(
        population,
        loss_fn,
        generations=5,
        crossover_probability=0.5,
        generator=seeded(),
    )
    repeat = evolve(
        population,
        loss_fn,
        generations=5,
        crossover_probability=0.5,
        generator=seeded(),
    )
    assert result["history"] == repeat["history"]
    assert_same(result["best_model"], repeat["best_model"])
    assert len(result["history"]) == 6
    best_losses = [row["best_loss"] for row in result["history"]]
    assert best_losses == sorted(best_losses, reverse=True)
    assert float(loss_fn(result["best_model"]).detach()) == pytest.approx(
        result["best_loss"]
    )
    for original, current in zip(originals, population):
        assert_same(original, current)
    for model, loss in zip(result["population"], result["losses"]):
        assert float(loss_fn(model).detach()) == pytest.approx(loss)


def test_invalid_settings_and_nonfinite_losses() -> None:
    with pytest.raises(ValueError):
        TorchCGP.random(1, 2, n_nodes=1, fixed_outputs=True)
    with pytest.raises(ValueError):
        TorchLGP.random(1, 1, n_input_constants=-1)
    model = TorchCGP.random(1, 1)
    for options in (
        {"p_mut_inputs": -0.1},
        {"p_mut_functions": math.nan},
        {"weights_mut_sigma": math.inf},
        {"weights_mutation_type": "unknown"},
    ):
        with pytest.raises(ValueError):
            model.mutate(**options)
    with pytest.raises(ValueError, match="non-finite"):
        evolve([model], lambda _: math.nan)
    with pytest.raises(ValueError, match="scalar"):
        evolve([model], lambda _: torch.ones(2))
    result = evolve([model, model], lambda _: 2.0, generations=0)
    assert result["best_loss"] == 2.0


def test_no_jax_imports_in_native_workflow(tmp_path: Path) -> None:
    script = """
import importlib.abc
import sys
class BlockJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax', 'jaxlib', 'qdax', 'flax', 'optax'}:
            raise ImportError('Forbidden dependency: ' + fullname)
sys.meta_path.insert(0, BlockJax())
import torch
from genepax.torch_gp import TorchCGP, TorchLGP, evolve
for cls in (TorchCGP, TorchLGP):
    rng = torch.Generator().manual_seed(0)
    population = [cls.random(2, 1, generator=rng) for _ in range(3)]
    x = torch.ones(2, 2)
    result = evolve(population, lambda model: model(x).square().mean(), generations=2, generator=rng)
    best = result['best_model']
    best.save(sys.argv[1])
    restored = cls.load(sys.argv[1])
    torch.testing.assert_close(best(x), restored(x))
assert 'jax' not in sys.modules and 'qdax' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "model.pt")],
        check=True,
        timeout=60,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("cls", [TorchCGP, TorchLGP])
def test_cuda_native_operators(cls, tmp_path: Path) -> None:
    model = cls.random(2, 1, trainable_weights=WEIGHT_NAMES, generator=seeded()).cuda()
    child = model.mutate(generator=seeded()).crossover(model, generator=seeded())
    child.save(tmp_path / "cuda.pt")
    restored = cls.load(tmp_path / "cuda.pt", device="cuda")
    torch.testing.assert_close(
        child(torch.ones(2, device="cuda")), restored(torch.ones(2, device="cuda"))
    )
    assert all(value.device.type == "cuda" for value in restored.parameters())
