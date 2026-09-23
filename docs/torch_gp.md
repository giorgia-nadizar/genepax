# Standalone PyTorch CGP and LGP

`genepax.torch_gp` provides native initialization, mutation, crossover,
population search, batched evaluation, and gradient training. These workflows
need PyTorch and NumPy, with no JAX, QDax, Flax, or Optax dependency.

```bash
pip install '.[torch]'
```

For a complete example that creates individual programs and populations,
evaluates their predictions and losses, and mutates them, run
`python -m examples.torch_gp` from the repository root.

JAX users should install `.[jax]`; both backends can be installed together with
`.[jax,torch]`. The base installation no longer installs a backend implicitly.
The existing conda environment still includes the JAX dependencies.

## Create and evaluate programs

```python
import torch
from genepax.torch_gp import TorchCGP, TorchLGP, evolve

rng = torch.Generator().manual_seed(42)
cgp = TorchCGP.random(
    n_inputs=3,
    n_outputs=1,
    n_nodes=20,
    fixed_outputs=True,
    trainable_weights=("inputs1", "inputs2", "functions_biases"),
    weights_initialization="natural",
    generator=rng,
)
lgp = TorchLGP.random(
    n_inputs=3,
    n_outputs=1,
    n_program_lines=20,
    n_computation_registers=5,
    trainable_weights=("inputs1", "inputs2", "functions_biases"),
    generator=rng,
)
observations = torch.randn(32, 3)
cgp_predictions = cgp(observations)
lgp_predictions = lgp(observations)
```

Both are `torch.nn.Module` instances. Inputs have shape `(..., n_inputs)` and
outputs `(..., n_outputs)`; single observations and arbitrary leading batch
dimensions are supported. One module represents one genotype.

`random()` accepts a CPU `torch.Generator`. Reusing a generator advances its
state; new generators with the same seed reproduce the same draws. When no
generator is provided, the global PyTorch CPU generator is used. Random draws
stay on the CPU even when evolving a CUDA model.

Initialization options shared by both representations:

- `n_input_constants=2`: zero or more constant inputs. Frozen constants begin
  with 0.1 and 1.0; additional constants are sampled uniformly in [-1, 1].
  Trainable constants are all sampled uniformly in [-1, 1].
- `function_names`: the ordered primitive names that integer function genes
  refer to. The default is the complete numeric function set.
- `output_transform="tanh"`: use `"identity"` for raw outputs, or provide a
  PyTorch callable.
- `weights_initialization="uniform"`: trainable connection/node weights and
  biases start uniformly in [-1, 1]. `"natural"` starts weights at one and
  biases at zero. Frozen weights always use these natural values.
- `trainable_weights=()`: numeric arrays to enable for gradient optimization
  and numeric mutation. Available names are `inputs1`, `inputs2`, `functions`,
  `inputs1_biases`, `inputs2_biases`, `functions_biases`, and `program_inputs`.
- `dtype=torch.float32`: float32 and float64 are supported.

For CGP, `fixed_outputs=True` connects outputs to the last `n_outputs` nodes
and preserves those connections during mutation and crossover. It requires
`n_outputs <= n_nodes`. Otherwise output connections are random and evolvable.
LGP always reads its final `n_outputs` registers.

## Mutation and crossover

```python
child = cgp.mutate(
    generator=rng,
    p_mut_inputs=0.1,
    p_mut_functions=0.1,
    p_mut_outputs=0.3,
    weights_mut_sigma=0.05,
)
lgp_child = lgp.mutate(generator=rng, p_mut_targets=0.3)
crossed = lgp.crossover(lgp_child, generator=rng)
```

These operations return independent models and leave the parents unchanged.
Mutation resamples selected integer genes within their valid bounds; a resample
can select the current value. Numeric mutation affects only trainable weights.
The default `weights_mutation_type="gaussian"` adds Gaussian noise;
`"automl0"` multiplies weights by a random factor in [0.5, 2] and may flip signs.
Pass `weights_mutation=False` to evolve only structure.

The optional `mutation_probabilities` dictionary overrides the individual
arguments using keys `inputs`, `functions`, `outputs`, `targets`, and
`weights_sigma`. Output probabilities apply to CGP, target probabilities to LGP.

Both representations support one-point crossover of nodes/instructions and
associated numeric weights. Constants come from the first parent. CGP output
connections are independently inherited from either parent, unless fixed.
Parents must have the same representation, configuration, array shapes, and
device. Optimizer state is not transferred to offspring: construct a new
optimizer for a new model.

## Optimize numeric parameters

```python
optimizer = torch.optim.Adam(cgp.parameters(), lr=1e-3)
targets = torch.randn(32, 1).tanh()
optimizer.zero_grad()
loss = torch.nn.functional.mse_loss(cgp(observations), targets)
loss.backward()
optimizer.step()
```

Integer genes are buffers; enabled numeric arrays are parameters. All other
weights are frozen buffers. Numeric optimization keeps the structure fixed.
Inputs can also receive gradients. LGP register writes replace references to
tensors without overwriting values needed for backpropagation.

Unused parameters may have no gradient. Some random programs connect outputs
directly to inputs/constants or never write an output register; their losses
may not depend on any trainable parameter. Inspect `loss.requires_grad` before
calling `backward()` on arbitrary evolved programs. The CGP example above uses
fixed node outputs and trainable node biases to ensure a parameter dependency.

## Evolve a population

```python
x = torch.linspace(-1, 1, 32)[:, None]
y = x.square()
population = [
    TorchCGP.random(
        1, 1, n_nodes=8,
        function_names=("plus", "times", "identity"),
        output_transform="identity", generator=rng,
    )
    for _ in range(12)
]
result = evolve(
    population,
    lambda model: (model(x) - y).square().mean(),
    generations=10,
    elite_count=2,
    tournament_size=3,
    crossover_probability=0.2,
    mutation_kwargs={"p_mut_inputs": 0.2},
    generator=rng,
)
best = result["best_model"]
print(result["best_loss"])
```

The same search function accepts an LGP population. It **minimizes** a scalar
loss; negate rewards to maximize them. Selection uses tournaments with
replacement and retains elites. The initial population is copied. The result
contains `best_model`, `best_loss`, sorted `population` and corresponding
`losses`, plus `history` (including generation zero).

Fitness is evaluated under `torch.no_grad()`, one model at a time. Supply a
fixed, deterministic evaluation for comparable fitness values; retained elites
reuse their scores. Non-finite losses rank last, and an entirely non-finite
initial population raises an error. This helper does not perform gradient
optimization inside the search. You can fine-tune a returned model separately.

## Inspect and save

```python
print(best.get_readable_program())
active = best.compute_active_mask()
size = best.size()
complexity = best.compute_complexity()
function_counts = best.compute_function_count()
arity_fractions = best.compute_function_arities()

best.save("program.pt")
restored = TorchCGP.load("program.pt")
torch.testing.assert_close(best(x), restored(x))
```

Active masks follow structural dependencies, ignore unused unary arguments,
and account for overwritten LGP registers. Size counts active nodes or
instructions; complexity divides that count by program length. Function counts
follow `function_names`; arity fractions report active unary and binary counts
divided by program length. Inspection uses protected primitive names and is
human-readable text, not executable serialization.

`save()` stores the genotype and constructor configuration, including function
order, output transform, trainability, dtype, and register settings. `load()`
uses `torch.load(weights_only=True)` and does not need an existing model.
Use `TorchLGP.load()` for LGP checkpoints. Optimizer state is not saved.

Custom callable output transforms require the ordinary PyTorch `state_dict()`
workflow and explicit reconstruction of the same configuration; `save()` only
accepts the built-in `tanh` and `identity` transforms.

`tensor_genotype()` exports independent, detached tensor copies on the model's
device. `to_genotype()` exports CPU NumPy copies for interoperability.
`clone()` copies the model without sharing tensor storage or optimizer state.
The constructors still accept explicit genotype dictionaries:
`TorchCGP(cgp.tensor_genotype(), **cgp.get_config())`.

## Devices and numerical behavior

Use `model.to(device="cuda", dtype=torch.float64)` and provide observations
on the same device and with the same dtype. Native mutation/crossover preserve
the model's dtype and device. Checkpoints load onto CPU by default; pass
`device="cuda"` to `load()` for another device.

Protected primitives bound intermediate numeric values as in the JAX backend,
including signed protected power (not ordinary exponentiation). Gradients at
nonsmooth boundaries can differ between backends, and saturated clipping
regions have zero gradients. Boolean primitives are available but do not have
useful gradients through their logical decisions.

The interpreter uses Python instruction loops with tensor operations batched
over observations. Integer indices are read on the host each forward pass.
This is not a fused GPU interpreter or a vectorized population evaluator.

## Optional JAX interoperability

Install `.[jax,torch]` to convert existing JAX models:

```python
import jax
from genepax.gp.cartesian_genetic_programming import CGP

structure = CGP(n_inputs=3, n_outputs=1, weighted_inputs=True)
genotype = structure.init(jax.random.key(0))
converted = TorchCGP.from_jax(structure, genotype)
updated_genotype = jax.tree.map(jax.numpy.asarray, converted.to_genotype())
```

`TorchLGP.from_jax()` works analogously. Conversion respects the JAX
trainable-weight flags and recognizes built-in primitives in any order/subset.
A non-default JAX output wrapper requires an explicit PyTorch
`output_transform`. Custom JAX primitives are not translated automatically.
Only these optional conversion methods import JAX.
