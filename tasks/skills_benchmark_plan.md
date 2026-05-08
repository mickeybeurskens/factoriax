# Implementation Plan: Skills Challenge Benchmark

Spec: `SPEC_SKILLS_BENCHMARK.md`. Read it first — this plan assumes
familiarity with the curriculum, scoring formula, per-level masks, and
the layout-varying seed.

## Overview

Build the `SkillsBenchmark` and its eight scripted baselines as a
vertically-sliced curriculum. Each curriculum level is one self-contained
task that adds: an achievement bit + condition, a level builder with
seed-varied layout, a per-level action mask, a scripted policy that
solves it, and tests. Foundation (protocol extension + skeleton) lands
first so each subsequent slice has somewhere to plug into. Cutover
(delete the old ad-hoc skill envs) happens last so nothing breaks
mid-flight.

## Architecture decisions

These are locked. If any need to change during implementation, stop and
update the spec first.

1. **One vertical slice per level.** Each `L.N` task touches the
   achievements module, the levels module, the scripted baseline
   directory, and the test suite — but only for that one level. Avoids
   "do all achievements, then all levels, then all scripted" horizontal
   slicing that leaves the system half-broken between phases.
2. **Per-level action mask via `BenchmarkLevel.blocked_actions`.** Adds
   an optional field to the existing `BenchmarkLevel` dataclass. The
   runner reads it and rebuilds its JIT'd env when the level's mask
   differs from the currently-wrapped one. Backwards compatible —
   `RocketBenchmark` keeps its class-level mask and triggers the same
   rebuild path with a single mask shared across all (one) of its
   levels.
3. **Achievement bits indexed by curriculum order.** Bit 0 = navigate,
   bit 1 = mine, …, bit 7 = mini_factory. The `score()` aggregate looks
   up bit `i` for level `i`. `MAX_ACHIEVEMENTS` is already large enough
   (rocket uses 38, plenty of headroom).
4. **`build_*_level(seed)` returns `(Level, EnvParams, frozenset[int])`.**
   Mask is part of the level definition, not implicit. Level builders
   are pure functions of `seed`. Default seed `0` is the canonical
   evaluation layout.
5. **Time-weighted score formula** is the only score. No binary mode,
   no toggle — the spec locked it as
   `solved * (max_timesteps - timesteps_used + 1) / max_timesteps`.
6. **Scripted policies are state-readers, not obs-readers.** Per the
   spec they're allowed to peek at `EnvState` directly — they're a
   baseline, not an agent under eval. This keeps each scripted file
   short and obvious instead of forcing them to decode an observation
   tensor.
7. **Cutover, not coexistence.** The new skills package replaces the
   old per-skill gymnax wrappers wholesale at the end of Phase D. No
   "keep the old code around just in case" — it's spec'd out.

## Risks and mitigations

| Risk                                                                                            | Impact | Mitigation                                                                                                                                                  |
| ----------------------------------------------------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Per-level mask triggers JIT recompile on every level switch — slow benchmark runs                | Med    | Measure once after F.1 lands; if recompile cost dominates, cache the JIT'd step per-mask in the runner. Add A/B in `scripts/bench_skills_runner.py`         |
| Achievement condition not pure-state — needs goal coords or recipe IDs that don't fit `EnvState` | High   | Levels are designed in the spec to be layout-invariant for their conditions. If a condition genuinely needs more, stop and revise the level — don't extend `EnvState` |
| Scripted baseline doesn't actually solve its level                                              | Med    | Test `tests/benchmarks/skills/test_skills_scripted_solves.py` is the load-bearing CI guard. Each level slice doesn't merge until its scripted test passes  |
| Mini-factory scripted baseline drifts into rocket-scripted complexity                            | Med    | Hard cap mini-factory scripted at 200 LOC. If it can't fit, the level is wrong — simplify the layout, not the policy                                       |
| Random-policy floor accidentally too high (≥ 0.1)                                               | Low    | If a level is solved by random more often than expected, tighten its mask or shrink the step budget. Caught by `test_skills_random_floor.py`               |
| GPU OOM during PPO training on the larger levels                                                 | Low    | Per memory note, GPU is ~6 GB. Default `num_envs=1024` is fine for 5x5; mini_factory at 16x16 may need `num_envs=256`. Document per-level config in T.1     |

## Phase F — Foundation

No curriculum yet. Lay rails so each subsequent slice plugs into the
same shape.

### F.1: Extend `BenchmarkLevel` with optional `blocked_actions` field

**Description:** Add `blocked_actions: frozenset[int] | None = None`
to `BenchmarkLevel` (dataclass in `factoriax/benchmarks/core.py`).
Update `BenchmarkRunner._run_level` to consult `bench_level.blocked_actions`
*before* falling back to `getattr(benchmark, "blocked_actions", ())`. When
the resolved mask differs from `self._current_blocked`, rebuild
`self._env` and `self._jit_step` via `_build_env`. Run-batched path
(`run_batched`) gets the same treatment.

**Acceptance criteria:**

- [ ] `BenchmarkLevel.blocked_actions` field exists and defaults to `None`
- [ ] `RocketBenchmark` continues to pass `tests/benchmarks/test_rocket_benchmark.py` unchanged (its class-level mask still wins when no per-level mask is set)
- [ ] A new test asserts that a `BenchmarkLevel(blocked_actions=frozenset({Action.MINE}))` causes `Action.MINE` to be rewritten to `NOOP` for that level only — i.e. running two levels with different masks back-to-back honors each
- [ ] `mypy factoriax` clean; `ruff check factoriax` clean

**Verification:**

```
pytest -q tests/benchmarks/test_runner.py tests/benchmarks/test_rocket_benchmark.py
mypy factoriax/benchmarks
ruff check factoriax/benchmarks
```

**Dependencies:** None — first task.

**Files:**

- `factoriax/benchmarks/core.py`
- `factoriax/benchmarks/runner.py`
- `tests/benchmarks/test_runner.py`

**Scope:** S (3 files).

### F.2: Skills package skeleton

**Description:** Create the new skills package. Files are real but
contain only stubs — no curriculum yet. Empty `skills_conditions` returns
all-False `(MAX_ACHIEVEMENTS,)` array. Empty `SKILLS_ACHIEVEMENT_INFO`
list. `SkillsBenchmark.levels()` returns `[]`. `score()` implements the
time-weighted aggregate but trivially returns `0.0` on empty input.

**Acceptance criteria:**

- [ ] `factoriax/benchmarks/skills/__init__.py` exports `SkillsBenchmark`, `skills_conditions`, `skills_reward`, and `SKILLS_ACHIEVEMENT_INFO`
- [ ] `factoriax/benchmarks/skills/achievements.py` defines `skills_conditions(state)` returning `jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_)`, plus an empty `SKILLS_ACHIEVEMENT_INFO: list[AchievementInfo] = []`
- [ ] `factoriax/benchmarks/skills/levels.py` is empty (stubs added per slice)
- [ ] `factoriax/benchmarks/skills/reward.py` defines `skills_reward(prev_state, new_state, params)` returning the time-weighted scalar (newly-unlocked-bits × `(max_timesteps - new_state.timestep) / max_timesteps`)
- [ ] `SkillsBenchmark` implements the `Benchmark` protocol: `name = "skills"`, `num_players = 1`, `achievement_fn = staticmethod(skills_conditions)`, empty `levels()`, time-weighted `score()`, `isinstance(SkillsBenchmark(), Benchmark)` is `True`
- [ ] `factoriax/benchmarks/__init__.py` re-exports `SkillsBenchmark`

**Verification:**

```
python -c "from factoriax.benchmarks import SkillsBenchmark; b = SkillsBenchmark(); assert b.levels() == []; print(b.score([]))"
mypy factoriax/benchmarks/skills
ruff check factoriax/benchmarks/skills
```

**Dependencies:** F.1.

**Files:**

- `factoriax/benchmarks/skills/__init__.py` (new)
- `factoriax/benchmarks/skills/achievements.py` (new)
- `factoriax/benchmarks/skills/levels.py` (new)
- `factoriax/benchmarks/skills/reward.py` (new)
- `factoriax/benchmarks/__init__.py`

**Scope:** M (5 files).

> **Naming collision:** the path `factoriax/benchmarks/skills/` already
> exists as a package containing `mining.py`, `place_miner.py`, and
> `__init__.py`. F.2 *replaces the contents* of `__init__.py` with the
> new exports and adds the new files alongside. The old
> `mining.py` / `place_miner.py` stay temporarily — they get deleted in
> D.1 once nothing imports them.

### F.3: Per-level action mask constants

**Description:** Define eight `frozenset[int]` constants in
`achievements.py`, one per skill, each containing the action ids to
*block*. Derive each from the spec table's "allowed actions" (mask is
the complement). Group helpers like `_movement_only()` and
`_all_craft_actions()` keep duplication low.

**Acceptance criteria:**

- [ ] `NAVIGATE_BLOCKED_ACTIONS`, `MINE_BLOCKED_ACTIONS`, `CRAFT_MINER_BLOCKED_ACTIONS`, `PLACE_MINER_BLOCKED_ACTIONS`, `FUEL_AND_COLLECT_BLOCKED_ACTIONS`, `BELT_LINE_BLOCKED_ACTIONS`, `ARM_TRANSFER_BLOCKED_ACTIONS`, `MINI_FACTORY_BLOCKED_ACTIONS` defined as `frozenset[int]`
- [ ] Each contains every action id *not* listed in the spec's "allowed actions" column for that skill
- [ ] `NAVIGATE_BLOCKED_ACTIONS` allows only `MOVE_*` and `NOOP`; verified by `assert int(Action.MINE) in NAVIGATE_BLOCKED_ACTIONS`
- [ ] `MINI_FACTORY_BLOCKED_ACTIONS` blocks every `CRAFT_*` action and nothing else
- [ ] No magic numbers — masks are built from `Action` enum members

**Verification:**

```
pytest -q tests/benchmarks/skills/test_action_masks.py
```

(New test file enumerating each mask's expected contents.)

**Dependencies:** F.2.

**Files:**

- `factoriax/benchmarks/skills/achievements.py`
- `tests/benchmarks/skills/test_action_masks.py` (new)

**Scope:** S (2 files).

### Checkpoint F

- [ ] `pytest -q tests/benchmarks/` green (rocket + runner + skills skeleton)
- [ ] `mypy factoriax baselines` clean
- [ ] `ruff check . && ruff format --check .` clean
- [ ] `python -c "from factoriax.benchmarks import SkillsBenchmark; print(SkillsBenchmark().score([]))"` prints `0.0`
- [ ] **Review with user** before starting Phase L. Specifically: confirm the per-level rebuild approach in F.1 is the shape they want, and the action mask groupings look right.

## Phase L — Curriculum (eight vertical slices)

Each `L.k` task adds one full level. Same shape every time:

1. Append a new `AchievementInfo` to `SKILLS_ACHIEVEMENT_INFO` at index `k - 1`
2. Append the level's condition expression to `skills_conditions` at the same index
3. Add `build_<name>_level(seed: int = 0)` to `levels.py`
4. Append the corresponding `BenchmarkLevel` to `SkillsBenchmark.levels()` (canonical seed = 0)
5. Add `baselines/skills/scripted/<name>.py` with a tiny solver
6. Register the scripted policy in `baselines/skills/scripted/__init__.py`'s `SCRIPTED_POLICIES` registry
7. Add the slice's tests to `tests/benchmarks/skills/test_skills_scripted_solves.py` (parametrized) and `test_skills_levels.py` (level-shape sanity)

The level slice is "done" when its scripted policy scores its level (in
the time-weighted sense) above 0.9 in CI.

### L.1: navigate

**Description:** 5x5 grass map. Goal is fixed at `(4, 4)`. Spawn varies
with seed (any non-goal tile). Achievement: player at `(map_h-1, map_w-1)`.
Mask: `NAVIGATE_BLOCKED_ACTIONS`. Scripted policy: walk right then down
(or use a tiny BFS — pick whichever is shorter to read).

**Acceptance criteria:**

- [ ] `build_navigate_level(seed=0)` and `build_navigate_level(seed=42)` return distinct `Level` objects with different player positions but identical map and goal
- [ ] `skills_conditions(state)[0]` is `True` exactly when `state.player_y[0] == 4 and state.player_x[0] == 4`
- [ ] Scripted policy solves seed 0 in under 10 ticks (manhattan distance ≤ 8)
- [ ] `SkillsBenchmark().levels()` now has length 1
- [ ] `BenchmarkRunner.run(SkillsBenchmark(), [navigate_policy])` returns aggregate ≥ 0.95

**Verification:**

```
pytest -q tests/benchmarks/skills/test_skills_scripted_solves.py::test_navigate
pytest -q tests/benchmarks/skills/test_skills_levels.py::test_navigate_layout_seed_variation
```

**Dependencies:** F.1, F.2, F.3.

**Files:**

- `factoriax/benchmarks/skills/achievements.py`
- `factoriax/benchmarks/skills/levels.py`
- `factoriax/benchmarks/skills/__init__.py` (extend `levels()`)
- `baselines/skills/scripted/navigate.py` (new)
- `baselines/skills/scripted/__init__.py` (registry; new on first slice)
- `tests/benchmarks/skills/test_skills_scripted_solves.py` (new on first slice)
- `tests/benchmarks/skills/test_skills_levels.py` (new on first slice)

**Scope:** M (7 files first time, 4–5 thereafter).

### L.2: mine

**Description:** 5x5 map with `ore_fraction=0.4` of mineable tiles
(iron/copper/coal mix). Player spawns near centre. Achievement: any ore
in `state.player_inventory[0]`. Mask: `MINE_BLOCKED_ACTIONS`. Scripted:
BFS to nearest mineable tile, face it, `MINE`.

**Acceptance criteria:**

- [ ] `build_mine_level(seed)` produces ≥ 5 ore tiles in the 5x5 map for any seed in `range(10)`
- [ ] Scripted policy solves seed 0 in under 30 ticks
- [ ] `skills_conditions(state)[1]` is `True` when any ore item count in the player inventory is ≥ 1
- [ ] Aggregate score across `SkillsBenchmark` is now ≥ 0.95 with [navigate_policy, mine_policy]

**Verification:**

```
pytest -q tests/benchmarks/skills/test_skills_scripted_solves.py::test_mine
```

**Dependencies:** L.1.

**Files:** as L.1, minus the new test/registry files (already created).

**Scope:** M.

### L.3: craft_miner

**Description:** 5x5 map with iron, copper, coal pre-placed within
2 tiles of spawn. Player starts with 0 inventory; mines and crafts.
Achievement: `state.player_inventory[0, ItemType.MINER] >= 1`.
Mask: `CRAFT_MINER_BLOCKED_ACTIONS` (only `CRAFT_MINER`, no other
crafts). Scripted: walk-mine-walk-mine sequence then `CRAFT_MINER`.
Recipe inputs come from `BASE_RECIPE_BOOK` — no rebalance needed.

**Acceptance criteria:**

- [ ] Level provides enough ore tiles to satisfy the miner recipe (≥ 1 of each input)
- [ ] Scripted policy solves seed 0 in under 60 ticks
- [ ] Random policy fails 19/20 seeds (sanity floor)

**Verification:**

```
pytest -q tests/benchmarks/skills/test_skills_scripted_solves.py::test_craft_miner
```

**Dependencies:** L.2.

**Files:** as L.2.

**Scope:** M.

### L.4: place_miner

**Description:** 5x5 map with one 2x2 ore patch off-centre. Player
starts with 5 miners in inventory. Achievement: any miner entity sitting
on an ore tile (reuse `count_miners_on_ore` from the existing
deprecated `factoriax/benchmarks/skills/__init__.py` — copy the function
into `achievements.py` before the old module gets deleted in D.1). Mask:
`PLACE_MINER_BLOCKED_ACTIONS` (movement, facing, place).

**Acceptance criteria:**

- [ ] `count_miners_on_ore(state) >= 1` triggers the achievement
- [ ] Scripted policy solves seed 0 in under 25 ticks
- [ ] Distinct from L.2 — `mine` doesn't accidentally satisfy `place_miner` and vice versa

**Verification:**

```
pytest -q tests/benchmarks/skills/test_skills_scripted_solves.py::test_place_miner
```

**Dependencies:** L.3.

**Files:** as L.2.

**Scope:** M.

### Checkpoint L1

After L.1–L.4 (foundational tier):

- [ ] `pytest -q tests/benchmarks/skills/` green
- [ ] `BenchmarkRunner.run(SkillsBenchmark(), [combined_scripted_policy])` aggregate ≥ 0.9 across the 4 levels (use a dispatching policy that picks the right scripted impl by level name)
- [ ] Random-policy aggregate ≤ 0.1 across the 4 levels (early sanity check before adding more)
- [ ] JIT recompile cost per level transition ≤ 5s (measure manually; if higher, add a runner-side cache before continuing)
- [ ] **Review with user.** Confirm achievement-bit indexing and scripted policy style are what they want before doing the harder Tier 3+ levels.

### L.5: fuel_and_collect

**Description:** 8x8 map. Pre-placed miner on a coal patch *adjacent
to* an iron patch (so a single placed miner is on coal + facing iron).
Player has 5 coal in inventory. Achievement: ore (any of iron/copper)
in `state.player_inventory[0]` AND at least one miner is currently
placed (distinguishes from L.2 — L.2 requires only that ore is held,
L.5 requires that the player got it via a placed miner). Implementation
detail: at episode start, set a flag in
`state.achievements_unlocked` to whatever — actually no, the achievement
is checked from state alone, so use: ore in inventory AND no `MINE`
action available (since the mask blocks it). Cleaner approach: reset
ore tile resources to 0 on the player's path so it can't be mined
manually. Mask: `FUEL_AND_COLLECT_BLOCKED_ACTIONS` (no MINE — that's
the trick). Scripted: face miner, `DEPOSIT` coal, walk away, walk back,
`WITHDRAW`.

**Acceptance criteria:**

- [ ] `MINE` is in `FUEL_AND_COLLECT_BLOCKED_ACTIONS` (player can't mine directly)
- [ ] Scripted policy solves seed 0 in under 200 ticks
- [ ] Pre-placed miner has zero coal in its fuel buffer at start (forces the agent to deposit)

**Verification:**

```
pytest -q tests/benchmarks/skills/test_skills_scripted_solves.py::test_fuel_and_collect
```

**Dependencies:** L.4.

**Scope:** M.

### L.6: belt_line

**Description:** 12x1-ish corridor map (12 wide, 5 tall). One ore tile
on the left edge with a placed miner. Player has 10 belts. Achievement:
ore on the rightmost-edge belt tile (`state.belt_item_type[map_h//2,
map_w-1] != ItemType.NONE`). Mask: `BELT_LINE_BLOCKED_ACTIONS`.
Scripted: face right, `PLACE` belt, step right, repeat.

**Acceptance criteria:**

- [ ] Goal coord is fixed at `(map_h // 2, map_w - 1)` regardless of seed (per spec — layout-invariant condition)
- [ ] Scripted policy solves seed 0 in under 80 ticks
- [ ] Achievement triggers only when an item *arrives at* the goal belt — not when a belt is merely placed there

**Verification:**

```
pytest -q tests/benchmarks/skills/test_skills_scripted_solves.py::test_belt_line
```

**Dependencies:** L.5.

**Scope:** M.

### L.7: arm_transfer

**Description:** 8x8 map. Pre-placed miner adjacent to a pre-placed
chest. Player has 1 arm in inventory. Achievement: chest contains ≥ 1
ore. Mask: `ARM_TRANSFER_BLOCKED_ACTIONS`. Scripted: face the gap, place
arm rotated correctly, wait.

**Acceptance criteria:**

- [ ] Pre-placed miner is on an ore tile and starts with coal in fuel
- [ ] Scripted policy solves seed 0 in under 200 ticks
- [ ] Achievement reads `state.ent_buf_count[chest_idx] >= 1` (chest is a `MachineType.PALLET` per the engine; verify in the level builder)

**Verification:**

```
pytest -q tests/benchmarks/skills/test_skills_scripted_solves.py::test_arm_transfer
```

**Dependencies:** L.6.

**Scope:** M.

### L.8: mini_factory

**Description:** 16x16 map. Single iron patch + single coal column on
the left, furnace + assembler pre-placed near spawn (rocket-style),
hand-crafting masked. Player must place a miner on iron, fuel it, run
belts to the furnace, withdraw plates from furnace, deposit into
assembler, withdraw output. Achievement: any assembler holds output
(reuse `_any_assembler_has_output` helper from `rocket.py` — promote to
a shared helper in `factoriax/state.py` or a new
`factoriax/benchmarks/_state_helpers.py` if both rocket and skills need
it). Mask: `MINI_FACTORY_BLOCKED_ACTIONS` (all `CRAFT_*` blocked).

Hard cap on scripted policy: 200 LOC. If it doesn't fit, the level is
too complex — simplify the layout, not the policy. The scripted policy
here *can* cheat by reading internal state heavily (knowing exact tile
coords, etc.) — it's a baseline, not an agent.

**Acceptance criteria:**

- [ ] Scripted policy solves seed 0 in under 1500 ticks (well under the 2000 budget)
- [ ] Scripted policy file ≤ 200 LOC including imports
- [ ] Test asserts the assembler output buffer increments at least once
- [ ] `mini_factory_policy` does *not* fire `CRAFT_*` actions (would be NOOP'd anyway, but flag if found)

**Verification:**

```
pytest -q tests/benchmarks/skills/test_skills_scripted_solves.py::test_mini_factory
wc -l baselines/skills/scripted/mini_factory.py  # ≤ 200
```

**Dependencies:** L.7.

**Scope:** M-L (largest scripted policy of the eight).

### Checkpoint L2

After L.1–L.8 (full curriculum):

- [ ] `pytest -q tests/benchmarks/skills/` green; all 8 scripted-solves tests pass
- [ ] `BenchmarkRunner.run(SkillsBenchmark(), [combined_scripted_policy])` aggregate ≥ 0.9
- [ ] Random-policy aggregate ≤ 0.1 (`test_skills_random_floor.py` from T.2 — implement that test now if it isn't already)
- [ ] No `factoriax/benchmarks/skills/{mining,place_miner}.py` imports anywhere outside the old module's own files (grep confirms; everything new uses `factoriax.benchmarks.skills.{achievements,levels,reward}`)
- [ ] **Review with user.** This is the right place to evaluate scripted style across all 8 — if any one feels too dense, simplify before moving to docs/training.

## Phase T — Training & integration

Phase T is the **interactive phase**. Training runs take real GPU time
(per `gpu_memory_ceiling` and `feedback_wait_before_running`), so each
run is gated by explicit user approval — the agent does *not* kick off
training in a loop. Every run logs to wandb with structured tags so the
user can compare across runs/levels/git SHAs without context-switching.

### Wandb conventions for this phase

These apply to T.1, T.2, and T.3 — defined once here so individual tasks
just say "log per the conventions".

**Project:** `factoriax_skills_benchmark` (one project for all skills
work — easier to filter than the per-trainer projects in flight today).

**Mandatory tags on every run:**

- `skills` — high-level area filter
- `<phase>` — one of `train`, `floor`, `bench`
- `<level_name>` — `navigate`, `mine`, …, `mini_factory`, or `all`
- `<git_sha_short>` — first 7 chars of `git rev-parse HEAD`, programmatically captured
- `<role>` — `train` runs add `ppo`; `bench` runs add `baseline` or `candidate`; `floor` runs add `random`

**Mandatory config logged on `wandb.init`:**

- The full training `Config` dataclass via `dataclasses.asdict`
- Git SHA (full) and dirty-tree status
- Level name, seed, max_timesteps, blocked_actions count
- Hardware: `jax.devices()`, GPU memory budget if known

**Mandatory artifacts/media:** every training run uploads:

- Final eval video (`runs/skills_<level>/<level>.mp4`)
- Action proportion plot
- Inventory composition plot
- Final benchmark `LevelResult.weighted_score` and `timesteps_used`

**No-wandb local mode:** if the user runs without `--use-wandb`,
training still works but the script logs a one-line warning at startup
("wandb disabled — results not shareable") so the user always knows
what mode they're in.

### User-in-the-loop protocol

Before any training run in this phase, the agent must:

1. **Ask the user explicitly** before starting any PPO training run. Per
   `feedback_wait_before_running`, the user may have GPU work in flight
   — never start a JAX/PPO process without confirming.
2. **Share the wandb URL** as soon as `wandb.init` returns. The user
   should be able to click into the live run from the message.
3. **Wait for user sign-off after each level** before moving to the
   next. The plan does not auto-iterate `--all` until the user has
   reviewed at least the `navigate` and `mining` runs and confirmed the
   trainer is doing what they want.
4. **Save a video on completion** (already in the existing
   `_evaluate()` flow) and surface it to the user via the wandb run.

### T.1: Rewrite `baselines/skills/train_ppo.py`

**Description:** Replace the per-skill `MiningSkill` / `PlaceMinerSkill`
gymnax-wrapper dispatch with a single PPO trainer driven by
`SkillsBenchmark`. The trainer takes a level name (or `all`), looks up
the corresponding `BenchmarkLevel` from `SkillsBenchmark().levels()`,
constructs the env using `factoriax.make()` + `ActionMaskWrapper`, and
trains with `skills_reward` as the reward function. CLI flags match the
existing rocket trainer's shape.

**Wandb behaviour (locked, not optional):**

- Default `--use-wandb=True`. To turn it off, the user must pass
  `--no-wandb` explicitly.
- Project: `factoriax_skills_benchmark`.
- Tags built per the *Wandb conventions* section above. For an `all`
  run, each level gets its own wandb run (eight runs, all with the
  shared `<git_sha_short>` tag so they group naturally in the wandb UI).
- Run name: `ppo_<level>_<git_sha_short>_<seed>` — collision-safe and
  reproducible.
- Periodic logging frequency: every iteration (already the existing
  trainer's behaviour, kept).

**Acceptance criteria:**

- [ ] `python -m baselines.skills.train_ppo navigate --total-steps 100000` runs to completion, prints the wandb URL on startup, and uploads video + plots on completion
- [ ] Tags on the wandb run include `skills`, `train`, `navigate`, `ppo`, and the 7-char git SHA
- [ ] `python -m baselines.skills.train_ppo all --total-steps 100000` iterates through all 8 levels in order, creating 8 wandb runs sharing the SHA tag
- [ ] No imports of the old `MiningSkill` / `PlaceMinerSkill` classes
- [ ] Per-level `num_envs` / `rollout_steps` defaults respect the `gpu_memory_ceiling` rule (≤ 1024 envs for 5x5 levels, ≤ 256 for the 16x16 mini_factory)
- [ ] Per-level training writes to `runs/skills_<level>/`
- [ ] When wandb is unavailable / unauthenticated, training still runs and logs a warning, not an error

**Verification (interactive — agent must ask first):**

Step T.1.a (initial smoke run, navigate only):

```
python -m baselines.skills.train_ppo navigate --total-steps 100000 --no-anneal-lr
```

Agent:
- asks user before running
- pastes the wandb URL into the conversation as soon as it appears
- waits for user review of the wandb run before proceeding

Step T.1.b (second smoke run, mining):

```
python -m baselines.skills.train_ppo mining --total-steps 100000 --no-anneal-lr
```

Same gate: ask, run, share URL, wait for sign-off.

Step T.1.c (full curriculum sweep — only after user approves T.1.a/T.1.b):

```
python -m baselines.skills.train_ppo all --total-steps 1_000_000
```

(Manually run, not in pytest — per `feedback_no_perf_tests`.)

**Dependencies:** L.8 (full curriculum).

**Files:**

- `baselines/skills/train_ppo.py` (full rewrite)

**Scope:** M.

### T.2: Random-policy floor — test + tagged wandb run

**Description:** Two artefacts:

1. `tests/benchmarks/skills/test_skills_random_floor.py` — pytest test
   asserting random-policy aggregate ≤ 0.1, JIT-heavy, marked
   `@pytest.mark.slow`.
2. `scripts/run_skills_random_floor.py` — CLI that runs the same floor
   evaluation but logs the per-level scores to wandb under the
   conventions above (tagged `skills`, `floor`, `random`,
   `<level_name>` per run, plus `<git_sha_short>`). The CLI is the
   shareable artefact; the pytest test is the CI guard.

**Acceptance criteria:**

- [ ] Pytest test exists and is registered under `pytest -m slow`
- [ ] Asserts `result.aggregate_score <= 0.1` for `seed=0`
- [ ] Asserts no individual level scores ≥ 0.5 from random play
- [ ] Documents which levels (if any) get a random hit, in a comment
- [ ] CLI script exists, defaults `--use-wandb=True`, tags per conventions
- [ ] Running the CLI produces a wandb run that the user can compare directly to T.1 PPO runs (same project, shared SHA tag, different `<role>`)

**Verification:**

```
pytest -q -m slow tests/benchmarks/skills/test_skills_random_floor.py
```

Then, **after asking the user**:

```
python scripts/run_skills_random_floor.py --num-seeds 5
```

Agent shares the wandb URL and waits for user review.

**Dependencies:** L.8.

**Files:**

- `tests/benchmarks/skills/test_skills_random_floor.py` (new)
- `scripts/run_skills_random_floor.py` (new)

**Scope:** S.

### T.3: Throughput A/B harness

**Description:** Write `scripts/bench_skills_runner.py` — standalone CLI
that runs `BenchmarkRunner.run(SkillsBenchmark(), [scripted])` `N` times
under two git refs and reports steps/sec. Logs per-level and aggregate
throughput numbers to wandb under the conventions above (tags `skills`,
`bench`, `baseline` or `candidate`, plus the SHA of the ref it ran
against). Mirrors the existing rocket throughput script's CLI.

**Acceptance criteria:**

- [ ] `python scripts/bench_skills_runner.py --num-runs 5` reports per-level and aggregate steps/sec
- [ ] `--baseline HEAD~1 --candidate HEAD` produces an A/B comparison table *and* two wandb runs (one per ref) tagged `baseline` and `candidate` respectively, sharing a `<comparison_id>` tag so they group in the wandb UI
- [ ] No new dependencies
- [ ] Not imported from anywhere; pure CLI

**Verification (after asking user):**

```
python scripts/bench_skills_runner.py --num-runs 3
```

Agent shares both wandb URLs and waits for user review.

**Dependencies:** L.8.

**Files:**

- `scripts/bench_skills_runner.py` (new)
- `scripts/README.md` (one-line entry)

**Scope:** S.

### Checkpoint T

- [ ] User has reviewed at least the `navigate` and `mining` PPO runs from T.1.a/T.1.b on wandb and approved continuing to the full curriculum
- [ ] Full-curriculum PPO sweep (T.1.c) completed; user has reviewed the eight runs grouped by SHA tag in wandb
- [ ] Random-policy floor test green; floor wandb run uploaded
- [ ] Throughput A/B baseline captured for HEAD; both baseline and candidate wandb runs visible in a single grouped view
- [ ] **Final user review for the phase.** Confirm PPO config defaults, the per-level training loop shape, and the wandb tagging scheme before moving to documentation.

## Phase D — Documentation & cutover

### D.1: Delete obsolete skills code

**Description:** Now that `factoriax.benchmarks.skills` (the new
package) is fully wired up and nothing internal imports the old per-skill
gymnax wrappers, delete:

- `factoriax/benchmarks/skills/mining.py`
- `factoriax/benchmarks/skills/place_miner.py`
- The old `MiningSkill` and `PlaceMinerSkill` exports from any
  `factoriax/benchmarks/skills/__init__.py` re-exports
- The `count_miners_on_ore` helper in the old `__init__.py` *only* if
  it has been copied into `achievements.py` already (per L.4)
- Old test files that import these classes (under
  `tests/benchmarks/skills/test_*` from before this work — grep for
  `MiningSkill` / `PlaceMinerSkill` usages)

Update `runs/skills_ppo_*` is *not* touched — old training run outputs
stay as historical reference.

**Acceptance criteria:**

- [ ] `git grep -l "MiningSkill\|PlaceMinerSkill"` returns no hits
- [ ] `git grep -l "from factoriax.benchmarks.skills import.*\(mining\|place_miner\)"` returns no hits
- [ ] `pytest -q` green (default suite)
- [ ] `mypy factoriax baselines` clean

**Verification:**

```
pytest -q
mypy factoriax baselines
git grep -l "MiningSkill"  # expect: empty
git grep -l "PlaceMinerSkill"  # expect: empty
```

**Dependencies:** T.1.

**Files:** deletions only.

**Scope:** S.

### D.2: User-facing benchmark doc

**Description:** Write `docs/skills_benchmark.md`. Sections: what the
benchmark measures, the curriculum table (copy from the spec), how
scoring works (time-weighted formula), how to run it (scripted, random,
PPO). Link from `docs/getting-started.md` as the next step after the
basic env setup.

**Explicitly NOT in scope for D.2:** No baseline training results
(scripted scores, random floor, PPO returns, sample episode lengths).
Those numbers come later, after the curriculum has settled and the
training runs are reproducible. The doc describes the benchmark
structurally and points the reader at the commands they'd run; it does
not bake in any specific scores. Where a results table would normally
go, leave a brief "Run the commands above to produce these numbers
yourself" note.

**Acceptance criteria:**

- [ ] `docs/skills_benchmark.md` exists, reads coherently start to finish without referencing the spec
- [ ] Includes a copy-pasteable code block that runs the scripted baseline (the *command*, not the *output*)
- [ ] Includes a copy-pasteable code block for training PPO on a single skill (command only, no reported scores)
- [ ] Links from `docs/getting-started.md` in the "next steps" / "what to do after install" section
- [ ] No marketing language (per `CLAUDE.md` writing style)
- [ ] **No baseline results, no example scores, no example training curves** — this content is deferred to a later spec/PR

**Verification:** `markdownlint docs/skills_benchmark.md` clean (if
markdownlint is configured); manual read-through.

**Dependencies:** D.1.

**Files:**

- `docs/skills_benchmark.md` (new)
- `docs/getting-started.md`

**Scope:** S.

### D.3: Cookbook entry — "Build your own skills challenge"

**Description:** Add a new section to `docs/cookbook.md` walking
through the navigate level end-to-end as the worked example for
building a custom benchmark. Sections: (1) define an achievement bit,
(2) write `build_*_level()`, (3) pick a `blocked_actions` mask, (4)
implement a `Benchmark` class with `levels()` and `score()`, (5) test
it. Reference the actual `factoriax/benchmarks/skills/*` code as
canonical.

**Acceptance criteria:**

- [ ] Section exists, ≤ 400 lines
- [ ] Code blocks run as written (test by copy-pasting into a scratch script)
- [ ] References real engine APIs only — no hypothetical or planned-but-not-shipped functions
- [ ] Stability tier of every referenced symbol is documented or stable per `factoriax/_stability.py`

**Verification:**

```
# Manual: copy each code block into /tmp/test_cookbook.py and run it.
```

**Dependencies:** D.2.

**Files:**

- `docs/cookbook.md`

**Scope:** M.

### D.4: Public API & stability manifest

**Description:** Update `factoriax/__init__.py` to re-export
`SkillsBenchmark` (and possibly the per-level builders if we want them
on the top-level `factoriax.*` namespace). Update
`factoriax/_stability.py` with the appropriate tier (probably
`stable` for `SkillsBenchmark`, `experimental` for the per-level
builders since they may grow seed-related options).

**Acceptance criteria:**

- [ ] `from factoriax import SkillsBenchmark` works
- [ ] `tests/test_public_api.py` green (asserts every public symbol has a tier)
- [ ] `tests/test_api_reference_fresh.py` green (auto-generated docs catch the new symbols)

**Verification:**

```
pytest -q tests/test_public_api.py tests/test_api_reference_fresh.py
python scripts/generate_api_reference.py --check  # if the script supports a check mode
```

**Dependencies:** D.3.

**Files:**

- `factoriax/__init__.py`
- `factoriax/_stability.py`
- `docs/api-reference.md` (regenerated)

**Scope:** S.

### Checkpoint D — Final

- [ ] Full `pytest -q` green; `pytest -q -m slow` green
- [ ] `mypy factoriax baselines` clean; `ruff check . && ruff format --check .` clean
- [ ] `python -m baselines.skills.scripted.runner --benchmark skills` aggregate ≥ 0.9
- [ ] Coverage floor unchanged or raised
- [ ] All success criteria from `SPEC_SKILLS_BENCHMARK.md` checked off
- [ ] `git grep -l 'MiningSkill\|PlaceMinerSkill'` empty (cutover complete)
- [ ] **Final review with user** before merging.

## Parallelization opportunities

- **Within Phase L:** Each `L.k` slice is mostly independent. After
  L.1 lands the test/registry scaffolding, L.2-L.8 can be drafted in
  any order — but to keep the achievement bit indices sequential, they
  should *land* in curriculum order. Drafting in parallel and merging
  in order is fine.
- **Phase T tasks** (T.1, T.2, T.3) are independent of each other and
  can land in any order after L.8.
- **D.2 (skills_benchmark.md) and D.3 (cookbook)** can be drafted in
  parallel — they reference the same code but are different audiences.
- **Sequential bottlenecks:** F.1 → F.2 → F.3 (foundation must land in
  order). L.1 → L.2 (need scaffolding from L.1). D.1 must land before
  D.2/D.3 reference the new module structure.

## Open questions for the user

None blocking — everything material is in the spec. Surfacing only:

1. **Phase L review cadence.** Plan has checkpoints after L.4 and L.8.
   Want one earlier (after L.1 — first full slice landed) to validate
   the slicing pattern, or is "after L.4" early enough?
2. **`tasks/plan.md` vs `tasks/skills_benchmark_plan.md`.** This plan
   was saved to a distinct filename to preserve the existing
   library-refactor `tasks/plan.md`. If you'd rather rotate the old
   plan to an archive name and reuse `tasks/plan.md` as the canonical
   path, say so.

## Decisions resolved during plan review

- **No baseline results in docs (D.2).** User-facing benchmark doc
  describes structure + commands only. Reported scores, training
  curves, and example outputs are deferred to a later spec. Don't bake
  numbers into the doc that will go stale before the curriculum settles.
