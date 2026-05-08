# FactoriaX

A Factorio-inspired RL environment built on JAX. Mine ore, craft
machines, automate your factory, launch a rocket. Every step
compiles to a single XLA graph, so you can `vmap` across hundreds
of parallel envs at the speed of your GPU instead of waiting on
Python loops.

<p align="center">
  <img src="docs/media/factory.png" width="480" alt="A mid-game factory with miners, conveyor belts, and an assembler" />
</p>

## Quick start

```bash
uv sync
make install-hooks
python -m factoriax
```

That gets you the launcher menu — Play, Editor, Settings.

## Design principles

Every action in the environment follows three properties that make
the benchmark RL-native.

**Atomic.** One action, one outcome. Crafting a miner is one action
(`CRAFT_MINER`), not "cycle recipe three times, then press craft."
There is no hidden sequencing the agent has to discover.

**Observable.** The agent can see whether an action will succeed
before taking it, and the result afterward. Per-recipe affordability
signals tell the agent which recipes are available right now, not
which one happens to be selected in a menu.

**Composable.** Actions work the same way regardless of context.
Crafting consumes inventory and adds the result the same way mining
adds and placing consumes. There is no separate "crafting mode."

<p align="center">
  <img src="docs/media/mining.gif" width="320" alt="A trained agent mining through an iron patch" />
</p>

## Where to read next

- [`docs/getting-started.md`](docs/getting-started.md) — install,
  the launcher, building a first level, running a random policy.
  ~10 minutes end-to-end.
- [`docs/cookbook.md`](docs/cookbook.md) — short recipes for the
  moves a research project actually makes (custom rewards, batched
  eval, achievement tracking, action masking, …).
- [`docs/api-reference.md`](docs/api-reference.md) — every public
  symbol with its stability tier. Auto-generated; CI-validated.
- [`examples/`](examples) — five runnable scripts under 80 lines
  each, one per pattern, that the cookbook recipes link into.
- [`baselines/rocket/train_ppo.py`](baselines/rocket/train_ppo.py)
  — a complete PPO run against the rocket achievement benchmark.
- [`scripts/README.md`](scripts/README.md) — what each
  command-line tool does and where its output goes (benchmarks,
  profiling, atlas regen, api-reference regen).
