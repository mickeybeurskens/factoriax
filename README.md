# Factoriax

> "The Factory must grow!"

Factoriax is a GPU accelerated factory building simulator for reinforcement learning research in the style of the game [Factorio](https://www.factorio.com/), written in JAX.

<p align="center">
  <img src="https://raw.githubusercontent.com/mickeybeurskens/factoriax/main/docs/_static/title_image.png"
       alt="A Factoriax factory: miners on ore patches, belts that carry ore to furnaces and assemblers, and a player among them" />
</p>

[Read the documentation](https://mickeybeurskens.github.io/factoriax/)
for the tutorial, the API reference, and the design notes. To build the
documentation as HTML, run `cd docs && make html`.

## Why Factoriax?


Factorio is quite a cool game with an enormous amount of emergent gameplay complexity, and it is interesting to use it as a benchmark to evaluate AI systems. However, even though the game is famous for being [carefully optimized](https://www.factorio.com/blog/post/fff-421), training Reinforcement Learning agents on the base game is very slow. Factoriax is basically a simplified version of the core Factorio game mechanics that can run millions of game ticks per second on higher end GPUs (as of writing), while also running quickly enough to train models on local laptop GPUs. This allows researchers to test their RL ideas in Factoriax at scale, while allowing students to get familiar with RL concepts without the need for enormous GPU clusters.

And honestly, Factorio is just a lot of fun. Even though the Factoriax implementation of the original game mechanics is quite minimal, I hope it does some justice to the original game.

## Installation

To install the most recent release from PyPi, run this command:

```bash
pip install factoriax
```

To install it with uv, run this command instead:

```bash
uv add factoriax
```

### GPU Support

Factoriax installs the CPU version of JAX by default. Read
[the JAX documentation](https://docs.jax.dev/en/latest/installation.html#installation)
to find the version of JAX for your hardware. If your system supports CUDA13,
run this command after you install Factoriax:

```bash
pip install "jax[cuda13]"
```

To do the same with uv, run this command:

```bash
uv pip install "jax[cuda13]"
```

### Most Recent Build For Research

Install an editable build if you want the most recent commit on main. Install
an editable build also if you want to change the source code of Factoriax
while you work on your own project.

```bash
git clone https://github.com/mickeybeurskens/factoriax.git
pip install -e ./factoriax
```

To do the same with uv, run these commands:

```bash
git clone https://github.com/mickeybeurskens/factoriax.git
uv add --editable ./factoriax
```

### Development Build

The development build keeps the development packages in sync. Do not use it if
you include Factoriax as a dependency of your own project.

```bash
git clone https://github.com/mickeybeurskens/factoriax.git
uv sync
```

`uv sync` removes the GPU version of JAX. To keep the GPU version, run this
command instead:

```bash
uv sync --extra cuda
```

## Play The Game 

Factoriax has a human playable interface, called the playground. It contains a human playable version of Factoriax and a level editor.

To open the playground, run this command:

```bash
python -m factoriax.playground
```

More information is available [in the documentation](https://mickeybeurskens.github.io/factoriax/start_here/playing_manually.html).

## Build An Environment

```python
import jax

from factoriax.make import env_from_name

env, params = env_from_name("EasyRocket-v1")
obs, state = env.reset_env(jax.random.PRNGKey(0), params)
```

Three scenarios have an id: `MinerBootstrap-v1`, `EasyRocket-v1`, and
`Rocket-v1`. The function `factoriax.engine.envs.registry.list_scenarios`
returns each id with its specification.

## Development

To run the tests, the linter, and the documentation build, run these commands:

```bash
uv run pytest
uv run ruff check factoriax tests
```

## Citation

Factoriax was initially submitted in a workshop paper at the 19th European Workshop on
Reinforcement Learning (EWRL 2026). If you use Factoriax in your research then please cite:

```bibtex
@inproceedings{beurskens2026factoriax,
  title     = {{Factoriax - A GPU-Accelerated Factory Building Simulator In The Style Of Factorio}},
  author    = {Beurskens, Mickey and Tomilin, Tristan and Sim{\~a}o, Thiago D.},
  booktitle = {19th European Workshop on Reinforcement Learning (EWRL)},
  year      = {2026},
  address   = {Lille, France},
  url       = {https://github.com/mickeybeurskens/factoriax}
}
```

