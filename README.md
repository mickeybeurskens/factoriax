# Factoriax

A grid world for reinforcement learning research, in the style of Factorio,
written in JAX.

The state is a set of JAX arrays. `step` and `reset` compile to one XLA graph,
so `jax.vmap` runs a batch of environments on the GPU, with no Python in the
inner loop.

<p align="center">
  <img src="docs/guides/_images/getting_started_state.png" width="420"
       alt="A generated world, with ore patches and a player" />
</p>

## Install

```bash
uv sync
```

## Run the playground

The playground is the human interface. It holds the game and the level editor.

```bash
uv run python -m factoriax.playground
```

The launcher opens with three entries: Play, Editor, and Settings.

## Build an environment

```python
import jax

from factoriax.make import env_from_name

env, params = env_from_name("EasyRocket-v1")
obs, state = env.reset_env(jax.random.PRNGKey(0), params)
```

Five scenarios carry an id: `Mining-v1`, `MinerBootstrap-v1`,
`ScienceTiers-v1`, `EasyRocket-v1`, and `Rocket-v1`.
`factoriax.engine.envs.registry.list_scenarios` returns each id with its spec.

## Action design

The action set follows three rules.

**One action, one outcome.** A player crafts a miner with the single action
`CRAFT_MINER`. It does not cycle through a recipe list and then press a craft
key. The action space holds 87 actions, and no action hides a sequence.

**The observation tells the agent what works.** An affordability bit for each
item says whether the player can craft that item now. The agent therefore
reads what an action does before it takes that action. The bit belongs to an
item, and not to a recipe.

**Context does not change an action.** A craft takes items from the inventory
and adds the result. A mine adds an item, and a place takes one. There is no
separate crafting mode that changes what a key does.

## Test and lint

```bash
uv run pytest
uv run ruff check factoriax tests
```

## Build the documentation

```bash
cd docs && make html
```

The build writes the pages to `docs/_build/html`.

The build does not run the notebooks. `nb_execution_mode` is `"off"`, and the
`html` target removes the stored output first. A guide page therefore holds
the source of each cell, and the figures that `docs/guides/_images` holds. To
make a new figure, run the notebook yourself and commit the file that it
writes.

## Where to read next

- [`docs/guides/getting_started.ipynb`](docs/guides/getting_started.ipynb):
  the environment, the observation, the action space, and a random rollout.
- [`docs/guides/ppo_training_example.ipynb`](docs/guides/ppo_training_example.ipynb):
  train a PPO agent on a Factoriax task.
- [`docs/api/index.rst`](docs/api/index.rst): the reference for every public
  module. Sphinx builds it from the docstrings.
- [`ISSUES.md`](ISSUES.md): the known defects that stay open.
