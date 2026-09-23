"""Check backend parity, differentiability, and the PyTorch module lifecycle."""

import io

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from genepax.gp.cartesian_genetic_programming import CGP  # noqa: E402
from genepax.gp.functions import (  # noqa: E402
    FunctionSet,
    JaxFunction,
    function_set_boolean,
    function_set_numeric,
)
from genepax.gp.linear_genetic_programming import LGP  # noqa: E402

torch = pytest.importorskip("torch")

from genepax.torch_gp import TorchCGP, TorchLGP  # noqa: E402
from genepax.torch_gp.functions import FUNCTIONS  # noqa: E402
from genepax.torch_gp.program import WEIGHT_NAMES  # noqa: E402


def example_genotype(kind: str) -> dict:
    # CGP: a = x + c; b = a*a; output = b+x.
    # LGP computes the same function, overwriting its sole output register.
    genes = {
        "inputs1": [0, 2, 2],
        "inputs2": [1, 2, 0],
        "functions": [0, 2, 0],
    }
    if kind == "cgp":
        genes["inputs1"][2] = 3
        genes["outputs"] = [4]
    else:
        genes["targets"] = [2, 2, 2]
    return {
        "genes": genes,
        "weights": {
            name: (
                [0.25]
                if name == "program_inputs"
                else [0.0 if "biases" in name else 1.0] * 3
            )
            for name in WEIGHT_NAMES
        },
    }


def make_model(kind: str, **kwargs) -> TorchCGP | TorchLGP:
    cls = TorchCGP if kind == "cgp" else TorchLGP
    options = {} if kind == "cgp" else {"n_computation_registers": 0}
    return cls(example_genotype(kind), 1, 1, **options, **kwargs)


@pytest.mark.parametrize("kind", ["cgp", "lgp"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("seed", [0, 3, 7])
def test_jax_parity(kind: str, weighted: bool, seed: int) -> None:
    cls, torch_cls = (CGP, TorchCGP) if kind == "cgp" else (LGP, TorchLGP)
    options = {"n_nodes": 8} if kind == "cgp" else {"n_program_lines": 8}
    structure = cls(
        n_inputs=3,
        n_outputs=2,
        **options,
        weighted_inputs=weighted,
        weighted_functions=weighted,
        weighted_program_inputs=weighted,
        biased_inputs=weighted,
        biased_functions=weighted,
    )
    genotype = structure.init(jax.random.key(seed))
    model = torch_cls.from_jax(structure, genotype)
    x = np.random.default_rng(10).normal(size=(2, 4, 3)).astype(np.float32)
    expected = jax.vmap(structure.apply, in_axes=(None, 0))(
        genotype, jnp.asarray(x.reshape(-1, 3))
    ).reshape(2, 4, 2)
    np.testing.assert_allclose(
        model(torch.tensor(x)).detach(), expected, rtol=3e-4, atol=3e-5
    )
    expected_names = {f"weights.{name}" for name in structure.get_weights(genotype)}
    assert set(dict(model.named_parameters())) == expected_names


@pytest.mark.parametrize("name", list(FUNCTIONS))
def test_primitive_parity_and_finite_gradients(name: str) -> None:
    x = np.array(
        [-np.inf, -1e20, -2.0, -0.3, 0.0, 0.2, 3.0, 1e20, np.inf, np.nan], np.float32
    )
    y = np.array(
        [0.0, -1e20, 2.0, 0.7, 1e-8, 0.5, -2.0, 1e20, np.nan, np.inf], np.float32
    )
    reference = {**function_set_numeric, **function_set_boolean}[name]
    expected = reference(jnp.asarray(x), jnp.asarray(y))
    actual = FUNCTIONS[name](torch.tensor(x), torch.tensor(y))
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-5)
    if name in function_set_boolean:
        return
    tx = torch.tensor([-2.0, -0.3, 0.0, 0.2, 3.0], requires_grad=True)
    ty = torch.tensor([0.0, 0.7, 1e-8, 0.5, -2.0], requires_grad=True)
    FUNCTIONS[name](tx, ty).sum().backward()
    assert torch.isfinite(tx.grad).all()
    if ty.grad is not None:
        assert torch.isfinite(ty.grad).all()


@pytest.mark.parametrize("kind", ["cgp", "lgp"])
@pytest.mark.parametrize("transform", ["identity", "tanh"])
def test_analytic_outputs_input_and_weight_gradients(kind: str, transform: str) -> None:
    model = make_model(
        kind,
        output_transform=transform,
        trainable_weights=WEIGHT_NAMES,
        dtype=torch.float64,
    )
    x = torch.tensor([[-0.3], [0.7]], dtype=torch.float64, requires_grad=True)
    expected = (x + 0.25).square() + x
    if transform == "tanh":
        expected = expected.tanh()
    torch.testing.assert_close(model(x), expected)
    parameters = dict(model.named_parameters())

    def evaluate(obs: torch.Tensor, *weights: torch.Tensor) -> torch.Tensor:
        return torch.func.functional_call(model, dict(zip(parameters, weights)), (obs,))

    assert torch.autograd.gradcheck(evaluate, (x, *parameters.values()))
    model(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert not any(g.requires_grad for g in model.genes.buffers())


@pytest.mark.parametrize("kind", ["cgp", "lgp"])
def test_training_and_genotype_roundtrip(kind: str) -> None:
    model = make_model(
        kind, output_transform="identity", trainable_weights=("functions_biases",)
    )
    original = model.to_genotype()
    x = torch.linspace(-0.5, 0.5, 20)[:, None]
    targets = model(x).detach() + 0.4
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    initial = torch.nn.functional.mse_loss(model(x), targets).item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), targets)
        loss.backward()
        optimizer.step()
    assert torch.nn.functional.mse_loss(model(x), targets).item() < initial / 10
    exported = model.to_genotype()
    for name in WEIGHT_NAMES:
        if name != "functions_biases":
            np.testing.assert_array_equal(
                exported["weights"][name], original["weights"][name]
            )
    cls = CGP if kind == "cgp" else LGP
    options = (
        {"n_nodes": 3}
        if kind == "cgp"
        else {"n_computation_registers": 0, "n_program_lines": 3}
    )
    structure = cls(
        n_inputs=1,
        n_outputs=1,
        n_input_constants=1,
        outputs_wrapper=lambda a: a,
        **options,
    )
    genotype = jax.tree.map(jnp.asarray, exported)
    expected = jax.vmap(structure.apply, in_axes=(None, 0))(
        genotype, jnp.asarray(x.numpy())
    )
    np.testing.assert_allclose(model(x).detach(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kind", ["cgp", "lgp"])
def test_state_dict_batch_shapes_and_dtype(kind: str) -> None:
    source = make_model(kind, trainable_weights=("inputs1",)).double()
    target = make_model(kind, trainable_weights=("inputs1",)).double()
    # The restored topology must be used, even when destination genes differ.
    target.gene("functions").zero_()
    stream = io.BytesIO()
    torch.save(source.state_dict(), stream)
    stream.seek(0)
    target.load_state_dict(torch.load(stream, weights_only=True))
    for shape in [(1,), (4, 1), (2, 3, 1), (0, 1)]:
        x = torch.full(shape, 0.3, dtype=torch.float64)
        assert target(x).shape == shape
        torch.testing.assert_close(target(x), source(x))
    assert all(g.dtype == torch.long for g in target.genes.buffers())
    assert all(w.dtype == torch.float64 for w in target.weights.buffers())


@pytest.mark.parametrize("kind", ["cgp", "lgp"])
def test_reordered_functions_and_explicit_wrapper(kind: str) -> None:
    cls, torch_cls = (CGP, TorchCGP) if kind == "cgp" else (LGP, TorchLGP)
    structure = cls(
        n_inputs=2,
        n_outputs=1,
        outputs_wrapper=lambda x: x,
        function_set=FunctionSet(
            {name: function_set_numeric[name] for name in ["sin", "plus"]}
        ),
    )
    genotype = structure.init(jax.random.key(2))
    with pytest.raises(ValueError, match="output_transform"):
        torch_cls.from_jax(structure, genotype)
    model = torch_cls.from_jax(structure, genotype, output_transform="identity")
    x = np.array([0.2, 0.7], np.float32)
    np.testing.assert_allclose(
        model(torch.tensor(x)), structure.apply(genotype, jnp.asarray(x)), atol=1e-6
    )


def test_custom_jax_function_is_not_silently_replaced() -> None:
    structure = CGP(
        n_inputs=1,
        n_outputs=1,
        function_set=FunctionSet({"plus": JaxFunction(lambda x, y: x - y, 2)}),
    )
    with pytest.raises(ValueError, match="No automatic PyTorch translation"):
        TorchCGP.from_jax(structure, structure.init(jax.random.key(0)))


@pytest.mark.parametrize("kind", ["cgp", "lgp"])
def test_reject_invalid_genes_and_inputs(kind: str) -> None:
    model = make_model(kind)
    with pytest.raises(ValueError, match="last dimension"):
        model(torch.ones(2))
    with pytest.raises(ValueError, match="dtype and device"):
        model(torch.ones(1, dtype=torch.float64))
    genotype = example_genotype(kind)
    genotype["genes"]["inputs1"][0] = -1
    cls = TorchCGP if kind == "cgp" else TorchLGP
    with pytest.raises(ValueError, match="out-of-range"):
        cls(genotype, 1, 1)


def test_reject_cgp_forward_reference_and_lgp_input_assignment() -> None:
    genotype = example_genotype("cgp")
    genotype["genes"]["inputs1"][0] = 2
    with pytest.raises(ValueError, match="out-of-range"):
        TorchCGP(genotype, 1, 1)
    genotype = example_genotype("lgp")
    genotype["genes"]["targets"][0] = 0
    with pytest.raises(ValueError, match="out-of-range"):
        TorchLGP(genotype, 1, 1, n_computation_registers=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("kind", ["cgp", "lgp"])
def test_cuda_forward_backward(kind: str) -> None:
    cpu_model = make_model(kind, trainable_weights=WEIGHT_NAMES)
    model = make_model(kind, trainable_weights=WEIGHT_NAMES).cuda()
    x = torch.tensor([[0.2], [0.7]], device="cuda", requires_grad=True)
    output = model(x)
    torch.testing.assert_close(output.cpu(), cpu_model(x.detach().cpu()))
    output.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
