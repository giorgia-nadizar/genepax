<div align="center">
    <img src="docs/img/logo.png" alt="genepax_logo" width="300"></img>
</div>

# genepax
![CI](https://github.com/giorgia-nadizar/genepax/actions/workflows/ci.yml/badge.svg)
[![Coverage](https://codecov.io/gh/giorgia-nadizar/genepax/branch/main/graph/badge.svg)](https://codecov.io/gh/giorgia-nadizar/genepax)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://giorgia-nadizar.github.io/genepax/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/giorgia-nadizar.github.io/genepax/blob/main/LICENSE)

Implementations of graph-based GP variants in JAX and PyTorch.

## Installation
To install directly from source and set up the Python environment:

```bash
git clone https://github.com/giorgia-nadizar/genepax.git
cd genepax
conda env create -f environment.yml
```
For standalone PyTorch CGP and LGP initialization, evolution, and gradient
training, install `pip install '.[torch]'` and see the
[PyTorch guide](docs/torch_gp.md). This does not install JAX or QDax.

For the JAX backend with pip, use `pip install '.[jax]'`; for both backends,
use `pip install '.[jax,torch]'`. The base pip installation no longer installs
a backend implicitly. The conda environment above still includes JAX.

## Branches

- `feat/constants` contains results of experiments on symbolic regression with constants optimization
