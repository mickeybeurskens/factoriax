# Installation

The most recent Factoriax release can be installed through PyPi:

```bash
pip install factoriax
```

or if you prefer to use uv:

```bash
uv add factoriax
```

## GPU Support
By default Factoriax installs the CPU version of JAX. 
Read [the JAX documentation](https://docs.jax.dev/en/latest/installation.html#installation) to find which version of JAX you need to install for your particular hardware configuration.
If your system supports CUDA13 for example, run the following command after installing Factoriax:

```bash
pip install "jax[cuda13]"
```

or with uv:

```bash
uv pip install "jax[cuda13]"
```

## Most Recent Build For Research
Install an editable version if you want the most recent commit on main, or if you want to edit the source code while working on your projects:
```bash
git clone https://github.com/mickeybeurskens/factoriax.git
pip install -e ./factoriax
```

or with uv:

```bash
git clone https://github.com/mickeybeurskens/factoriax.git
uv add --editable ./factoriax
```

## Development Build
Install the development version with `uv sync` to automatically keep development packages synchronized. 
This is not suitable if you have your own project and are including factoriax as a dependency.

```bash
git clone https://github.com/mickeybeurskens/factoriax.git
uv sync
```

Installing this way automatically uninstalls the GPU version unless you specifically install it with uv:

```bash
uv sync --extra cuda
```

## Building The Docs
Go to the `docs/` directory and use `make` to build the documentation from scratch:

```bash
cd docs
make clean html
```