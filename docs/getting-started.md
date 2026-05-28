# Getting Started with FactoriaX

FactoriaX is a JAX grid-world for factory-building research. Every
state field is a JAX array; every step compiles to one XLA graph.
You can drive an env from a notebook in five minutes, then run the
same env across hundreds of seeds with `jax.vmap`.

This guide walks through install, the launcher, building your first
level, and running a random policy. Each section ends with a
pointer to a runnable file under [`examples/`](../examples) — those
are the canonical "shape" of each pattern; copy and adapt.

## Install

FactoriaX uses [`uv`](https://docs.astral.sh/uv/) for environment
management. From a fresh clone:

```bash
uv sync
make install-hooks
```

`uv sync` builds the virtualenv at `.venv/` from `pyproject.toml`
plus `uv.lock`. `make install-hooks` symlinks `scripts/hooks/pre-commit`
into `.git/hooks/` so commits run lint + the fast test suite +
the coverage gate before they land.

## Launch the game

```bash
python -m factoriax
```

The launcher menu opens with three tiles: **Play**, **Editor**,
**Settings**. Play runs an interactive game on the default level;
Editor opens a level-authoring canvas; Settings has env params,
keyboard / controller bindings, and display options. Player 0
controls with WASD by default; the [controls
screen](../factoriax/menu/controls_menu.py) lets you rebind.

## Build your first level

Levels are pure-Python `Level` objects. The fluent
[`LevelBuilder`](../factoriax/levels.py) lets you author one in a
dozen lines and round-trip it through JSON for sharing:

```python
from factoriax import LevelBuilder, BlockType, Direction
from factoriax.constants import MachineType

level = (
    LevelBuilder(12, 12)
    .fill_rect(1, 1, 3, 3, BlockType.IRON, resources=200)
    .fill_rect(8, 1, 3, 3, BlockType.COPPER, resources=200)
    .place_machine(2, 4, int(MachineType.MINER), direction=int(Direction.UP))
    .build("two_patches")
)
```

Pass it to `factoriax.make(level)` and the resulting env resets to
your layout. The [Editor](../factoriax/editor) lets you build a level
visually and save it to JSON; `factoriax.load_level(path)` reads it
back into a `Level` object. Full round-trip example:
[examples/custom_level.py](../examples/custom_level.py).

## Run a random-policy agent

The factory function `factoriax.make()` builds the canonical wrapper
stack. Without args you get a bare `FactoriaXEnv`; pass `obs="local"`,
`auto_reset=True`, or `blocked_actions=(...)` and the right wrappers
compose around it in the documented outermost-to-innermost order:
`AutoReset → ActionMask → LocalObservation → FactoriaXEnv`.

```python
import jax
import factoriax

env, params = factoriax.make()
rng = jax.random.PRNGKey(0)
rng, key_reset = jax.random.split(rng)
_, state = env.reset_env(key_reset, params)
n_actions = int(env.action_space(params).n)

for _ in range(1000):
    rng, key_act, key_step = jax.random.split(rng, 3)
    action = jax.random.randint(key_act, (), 0, n_actions)
    _, state, reward, done, _ = env.step_env(key_step, state, action, params)
```

That's the full random-policy loop. To go batched, wrap the rollout
in a `jax.lax.scan` and `jax.vmap` over a batch of PRNG keys —
[examples/batched_evaluation.py](../examples/batched_evaluation.py)
shows the pattern over 16 seeds.

## What to read next

- [`docs/cookbook.md`](cookbook.md) — small recipes for common
  research moves: custom rewards, custom levels, batched eval,
  achievement tracking.
- [`docs/api-reference.md`](api-reference.md) — every public symbol,
  its signature, and its stability tier. Auto-generated from
  `factoriax.__all__`.
- [`baselines/easy_rocket/ppo/train_ppo.py`](../baselines/easy_rocket/ppo/train_ppo.py)
  — a complete PPO run against the easy rocket achievement benchmark.

If something doesn't fit a pattern in `examples/`, the test suite
under [`tests/`](../tests) is the next-best reference — most public
APIs have a dedicated test module that doubles as a usage example.
