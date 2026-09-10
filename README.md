# Factoriax

> "The Factory must grow!"

Factoriax is a factory building simulator for research into reinforcement
learning (training an agent by reward). It runs on a GPU and follows the style
of the game [Factorio](https://www.factorio.com/). It is written in JAX.

<p align="center">
  <img src="https://raw.githubusercontent.com/mickeybeurskens/factoriax/main/docs/_static/title_image.png"
       alt="A Factoriax factory: miners on ore patches, belts that carry ore to furnaces and assemblers, and a player among them" />
</p>

The state of the game is a set of JAX arrays. The `step` and `reset` functions
compile to one XLA graph. `jax.vmap` then runs a batch of environments on the
GPU. No Python code runs in the inner loop.

[Read the documentation](https://github.com/mickeybeurskens/factoriax/blob/main/docs/index.md)
for the tutorial, the API reference, and the design notes. To build the
documentation as HTML, run `cd docs && make html`.

## Why Factoriax?

Factorio is a game with a large amount of emergent gameplay complexity. That
makes it interesting as a benchmark for AI systems. The game is
[carefully optimized](https://www.factorio.com/blog/post/fff-421). Training
reinforcement learning agents on the base game is still very slow.

Factoriax is a simplified version of the core Factorio game mechanics. On a
higher end GPU it runs millions of game ticks per second, as of writing. It
also runs fast enough to train models on a laptop GPU. Researchers can
therefore test their ideas at scale. Students can learn reinforcement learning
without a large GPU cluster.

Factorio is also a lot of fun. The Factoriax version of the original game
mechanics is minimal. I hope that it does some justice to the original game.

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

## Play The Game First

Factoriax has a human interface, called the playground. Play it for a few
minutes before you train a model. The observation and the action space are
easier to read after you mine an ore patch and place a miner yourself.

To open the playground, run this command:

```bash
python -m factoriax.playground
```

The launcher shows four entries: Play, Editor, Settings, and Quit.

Play generates a world and puts you in it. Move with the `WASD` keys. Hold
`Space` to mine the ore under the player. Press `I` to craft. Press `E` to
place the machine that you crafted. Press `?` for the full list of controls.

Editor opens the level editor. Paint terrain and place machines to author the
fixed maps that scenarios load. Press `F5` to play-test the map. Press
`Ctrl+S` to save the map to the `levels/` directory.

[Playing A Game Manually](https://github.com/mickeybeurskens/factoriax/blob/main/docs/start_here/playing_manually.md)
describes both in full.

## Build An Environment

```python
import jax

from factoriax.make import env_from_name

env, params = env_from_name("EasyRocket-v1")
obs, state = env.reset_env(jax.random.PRNGKey(0), params)
```

Five scenarios have an id: `Mining-v1`, `MinerBootstrap-v1`,
`ScienceTiers-v1`, `EasyRocket-v1`, and `Rocket-v1`. The function
`factoriax.engine.envs.registry.list_scenarios` returns each id with its
specification.

## Development

To run the tests, the linter, and the documentation build, run these commands:

```bash
uv run pytest
uv run ruff check factoriax tests
cd docs && make html
```

The documentation build writes the pages to `docs/_build/html`. The build does
not run the notebooks, because `nb_execution_mode` is `"off"`. A notebook page
therefore holds the source of each cell and the figures in its `_images`
directory. To add a figure, run the notebook and commit the file that the
notebook writes.

The `tests/` directory mirrors the package. The path
`tests/<area>/test_<module>.py` maps to `factoriax/<area>/<module>.py`. Three
directories sit beside the mirror, for the tests that belong to no single
module:

- `tests/contracts/` holds a test for a rule that no single module owns.
- `tests/integration/` holds a test that needs two or more subpackages.
- `tests/benchmarks/` holds scripts. Pytest does not collect them.

## Citation

Factoriax comes from a workshop paper at the 19th European Workshop on
Reinforcement Learning (EWRL 2026). If you use Factoriax in your research,
cite that paper:

```bibtex
@inproceedings{beurskens2026factoriax,
  title     = {{Factoriax - A GPU-Accelerated Factory Building Simulator In The Style Of Factorio}},
  author    = {Beurskens, Mickey and Tomilin, Tristan and Sim{\~a}o, Thiago D.},
  booktitle = {19th European Workshop on Reinforcement Learning (EWRL)},
  year      = {2026},
  address   = {Lille, France},
  url       = {https://ewrl-org.github.io/ewrl-2026/poster_148.html}
}
```

The paper page holds the abstract and the poster session:
<https://ewrl-org.github.io/ewrl-2026/poster_148.html>
