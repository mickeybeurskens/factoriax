# FactoriaX

A Factorio-inspired grid-world RL environment, written in JAX.
State lives in JAX arrays; `step` and `reset` compile to a single
XLA graph, so `jax.vmap` over a batch of envs runs on GPU without
Python in the inner loop.

<p align="center">
  <img src="docs/media/factory.png" width="480" alt="A mid-game factory with miners, conveyor belts, and an assembler" />
</p>

## Quick start

```bash
uv sync
make install-hooks
python -m factoriax
```

Opens the launcher (Play, Editor, Settings).

## Action design

Three properties the action set is built around.

**Atomic.** One action, one outcome. Crafting a miner is one action
(`CRAFT_MINER`), not "cycle recipe three times, then press craft."
No hidden sequencing.

**Observable.** Whether an action will succeed is in the observation
before it's taken. Per-recipe affordability bits expose which
recipes are available now, rather than which one happens to be
selected in a menu.

**Composable.** Actions behave the same regardless of context.
Crafting consumes inventory and adds the result the same way mining
adds and placing consumes — no separate "crafting mode."

<p align="center">
  <img src="docs/media/mining.gif" width="320" alt="Agent mining an iron patch" />
</p>

## Where to read next

- [`docs/getting-started.md`](docs/getting-started.md) — install,
  the launcher, building a first level, running a random policy.
  ~10 minutes end-to-end.
- [`docs/cookbook.md`](docs/cookbook.md) — short recipes for
  common research moves (custom rewards, batched eval, achievement
  tracking, action masking, …).
- [`docs/api-reference.md`](docs/api-reference.md) — every public
  symbol with its stability tier. Auto-generated; CI-validated.
- [`examples/`](examples) — five runnable scripts under 80 lines
  each, one per pattern, that the cookbook recipes link into.
- [`scripts/README.md`](scripts/README.md) — what each
  command-line tool does and where its output goes (benchmarks,
  profiling, atlas regen, api-reference regen).
