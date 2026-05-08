# Spec: Skills Challenge Benchmark

The factoriax engine ships with one benchmark today — `RocketBenchmark`, a
full end-to-end run of the production chain. That's the right capstone, but
it's a wall for a brand-new researcher: 38 weighted achievements, an 8000-step
budget, and no signal until ore comes out of a placed miner. The skills
challenge sits below it. It is the first thing a new user runs against
factoriax to confirm the engine works on their machine, the first thing an
RL agent should clear before being pointed at the rocket, and the worked
example of how to build a custom benchmark on top of the `Benchmark`
protocol.

This spec defines that benchmark.

## Objective

Ship a `SkillsBenchmark` that:

1. **Tests the foundational skills** an agent needs before the rocket is
   tractable — navigation, mining, crafting, machine placement, automated
   production, belts, arms, and a small composite factory.
2. **Mirrors the `RocketBenchmark` shape** so downstream tooling
   (`BenchmarkRunner`, scoring, `LevelResult.achievements_unlocked`) works
   identically. Anyone who has integrated the rocket benchmark integrates
   this one with no API changes.
3. **Reads as the canonical example** of authoring a benchmark. A researcher
   reading `factoriax/benchmarks/skills/` should be able to copy the pattern
   and write their own multi-level challenge in an afternoon.
4. **Has a scripted baseline per level**. Each scripted policy is a small,
   readable file that solves its level deterministically. Three jobs at
   once: solvability proof, score floor for RL to beat, tutorial for engine
   API use.
5. **Exposes each level for direct training.** Researchers can import a
   single level — `(Level, EnvParams, blocked_actions)` — and train an
   agent on just that skill, without going through `BenchmarkRunner`.
   The benchmark wrapper is for evaluation; the level builders are for
   training. Same data, two access patterns.
6. **Masks irrelevant actions per level.** Each level carries its own
   `blocked_actions` mask that hides actions outside its skill. The
   `navigate` level exposes only `MOVE_*` and `NOOP`; the `mine` level
   adds `MINE`; the `craft_miner` level adds only the one needed `CRAFT_*`
   action; and so on. This collapses the per-level action space (often
   from 40+ actions down to 5–8), which is what makes each skill a
   tractable training problem in isolation. Users who want the full
   action space can override the mask to `frozenset()`.

### Target user

A researcher (or curious engineer) who has just installed factoriax. They
read the README, run the skills benchmark, see that 8 levels exist with
scripted baselines that score 8/8, then point an RL agent at it.
That's the success path.

### Curriculum

Eight levels, ordered by tier. The tier is the same one used in the
`mechanic_tiers` planning memo — Tier 1 foundational, Tier 2 single-machine,
Tier 3 automated production, Tier 4 composite chain.

| #   | Name               | Tier | Skill tested                         | Solve condition                         | Step budget | Allowed actions (per-level mask)                                       |
| --- | ------------------ | ---- | ------------------------------------ | --------------------------------------- | ----------- | ---------------------------------------------------------------------- |
| 1   | `navigate`         | T1   | Move to a target tile                | Player on goal tile                     | 200         | `MOVE_*`, `NOOP`                                                       |
| 2   | `mine`             | T1   | Stand on ore + MINE                  | At least one ore in inventory           | 200         | `MOVE_*`, `MINE`, `NOOP`                                               |
| 3   | `craft_miner`      | T1   | Hand-craft from materials            | One miner held in inventory             | 400         | `MOVE_*`, `MINE`, `CRAFT_MINER`, `NOOP`                                |
| 4   | `place_miner`      | T2   | Place a machine on ore               | Miner sitting on ore tile               | 300         | `MOVE_*`, `FACE_*`, `PLACE`, `NOOP`                                    |
| 5   | `fuel_and_collect` | T3   | Fuel a miner, withdraw output        | Hold ore that came from a placed miner  | 800         | `MOVE_*`, `FACE_*`, `DEPOSIT`, `WITHDRAW`, `SELECT_*`, `NOOP`          |
| 6   | `belt_line`        | T3   | Place belts that move an item        | Item on belt at goal coord              | 600         | `MOVE_*`, `FACE_*`, `PLACE`, `ROTATE`, `MINE`, `NOOP`                  |
| 7   | `arm_transfer`     | T3   | Arm moves item between machine/chest | Chest contains item the arm transferred | 800         | `MOVE_*`, `FACE_*`, `PLACE`, `ROTATE`, `DEPOSIT`, `NOOP`               |
| 8   | `mini_factory`     | T4   | Miner → belt → assembler             | Assembler produces 1 plate              | 2000        | every action **except** all `CRAFT_*` (rocket-style pipeline)         |

The "Allowed actions" column is the *complement* of `blocked_actions` for
that level. The exact `Action` enum members get pinned in
`achievements.py` alongside the achievement bits, so the mask and the
solve condition live next to each other.

Levels 5–8 mask all hand-crafting (`CRAFT_*`) and pre-place the
machinery the agent needs — miner+coal for `fuel_and_collect`,
belts+source for `belt_line`, arm+chest+source for `arm_transfer`,
furnace+assembler for `mini_factory`. This mirrors the rocket benchmark's
"production must flow through machines" stance and means a policy that
solves levels 5–8 transfers cleanly to rocket. The `craft_miner` level
(level 3) is the only level that exercises hand-crafting directly, by
exposing `CRAFT_MINER` and nothing else from the craft suite.

Research (T4) and repair (T5) are explicitly out of scope. They belong in a
later, advanced benchmark — the skills benchmark is the on-ramp, not the
exhaustive eval suite.

### Scoring

Score is **time-weighted**, not binary. A level scores higher the faster
it's solved:

```
level_score = solved * (max_timesteps - timesteps_used + 1) / max_timesteps
```

Solve on tick 1 → score ≈ 1.0. Solve on the last tick → score ≈ 1 /
max_timesteps. Don't solve → 0. Aggregate is the mean across all eight
levels, range `[0, 1]`. The "+ 1" keeps the on-time-limit case strictly
above zero so we can distinguish "solved at the buzzer" from "didn't
solve". This shape:

- Preserves the binary "solved at all?" floor — multiplied by `solved`,
  failed levels still score zero.
- Penalises wasted steps, which is what we want from a curriculum: an
  agent that loiters and eventually solves should rank below one that
  solves quickly.
- Keeps the aggregate in `[0, 1]`, same range as the rocket
  benchmark's normalised score, so the two are visually comparable.

The same shape is mirrored in the training reward (see `reward.py` —
sparse on the unlock tick, scaled by remaining time) so the policy
optimises toward fast completion, not "score the achievement late and
loiter".

### Success criteria

- `factoriax.benchmarks.SkillsBenchmark` exists and implements the
  `Benchmark` protocol exactly the way `RocketBenchmark` does.
- `BenchmarkRunner.run(SkillsBenchmark(), policies)` returns a
  `BenchmarkResult` with eight `LevelResult` entries and an aggregate score
  in `[0, 1]` (time-weighted, see *Scoring* above).
- All eight scripted baselines score above 0.9 aggregate against
  `SkillsBenchmark` (each scripted policy solves its level in well under
  the step budget). Tested in CI under `tests/benchmarks/`.
- A random policy scores below 0.1 aggregate — sanity floor confirming
  the benchmark isn't trivial.
- `baselines/skills/train_ppo.py` is rewritten to take `SkillsBenchmark` as
  the single source of truth (not the deleted ad-hoc gymnax wrappers) and
  trains per-level with the time-discounted achievement reward.
- Each level builder takes a `seed: int = 0` argument that varies the
  layout while keeping the target achievement reachable. Default seed `0`
  is the canonical evaluation seed used by `SkillsBenchmark.levels()`;
  callers can pass any other seed for training-time variation. Reproduced
  identically across machines for a given seed.
- Every level builder is publicly importable from
  `factoriax.benchmarks.skills` — e.g.
  `from factoriax.benchmarks.skills import build_navigate_level` returns
  `(Level, EnvParams, blocked_actions: frozenset[int])` and accepts a
  `seed: int = 0` for layout variation. Sufficient for a user to
  construct a single-level training env without instantiating
  `SkillsBenchmark` at all.
- `BenchmarkLevel` is extended with an optional
  `blocked_actions: frozenset[int] | None` field. When non-`None`, the
  runner wraps the env with `ActionMaskWrapper(blocked_actions)` *before*
  rolling out that level. Defaults to `None` for backward compatibility
  with `RocketBenchmark` (which sets the mask at the benchmark class level).
- `docs/cookbook.md` gains a "Build your own skills challenge" section that
  walks through one of the level files end-to-end as a worked example.

## Tech Stack

Locked by the project:

- **Python 3.12** with `uv` for package management
- **JAX** (`jax`, `jax.numpy`) for the simulation core, fully JIT-able
- **gymnax** for the env protocol the engine already conforms to
- **`factoriax.benchmarks`** internal protocol (`Benchmark`,
  `BenchmarkLevel`, `BenchmarkRunner`, `LevelResult`) — already shipped
- **`optax` + `flax`** for the PPO baseline (already in `baselines/ppo/`)
- **`pytest`** for tests, **`ruff`** for lint/format, **`mypy`** for types

No new dependencies. The benchmark and its baselines must build entirely
on what the repo already has.

## Commands

```
# Install
uv sync

# Run the skills benchmark with scripted baselines (target: 8/8)
python -m baselines.skills.scripted run --benchmark skills

# Train PPO on one skill level (or all)
python -m baselines.skills.train_ppo navigate
python -m baselines.skills.train_ppo all --use-wandb

# Tests (default: skip slow JIT-heavy examples)
pytest -q -m "not slow"
pytest -q tests/benchmarks/test_skills_benchmark.py

# Lint, format, type-check
ruff check . --fix
ruff format .
mypy factoriax baselines

# Throughput A/B (perf-impacting changes only — NOT in pytest)
python scripts/bench_skills_runner.py --baseline HEAD~1 --candidate HEAD
```

## Project Structure

New and modified paths only. Anything not listed stays as-is.

```
factoriax/benchmarks/
├── core.py                                  (MODIFIED — add optional
│                                             blocked_actions field to
│                                             BenchmarkLevel)
├── runner.py                                (MODIFIED — read per-level
│                                             blocked_actions when set,
│                                             fall back to benchmark-class
│                                             attr otherwise)
├── rocket.py                                (unchanged behaviour; still
│                                             uses class-level
│                                             blocked_actions)
├── __init__.py                              (export SkillsBenchmark + per-level builders)
└── skills/                                  (REPLACED — see migration below)
    ├── __init__.py                          (SkillsBenchmark + public re-exports
    │                                         of every build_*_level helper so
    │                                         users can train on individual levels)
    ├── achievements.py                      (skills_conditions + 8 AchievementInfo
    │                                         + per-level blocked_actions table)
    ├── levels.py                            (8 level builders. Each returns
    │                                         (Level, EnvParams, frozenset[int]))
    └── reward.py                            (sparse achievement reward shim)

baselines/skills/
├── __init__.py
├── train_ppo.py                             (REWRITTEN — driven by SkillsBenchmark)
└── scripted/
    ├── __init__.py                          (registry: name → scripted policy fn)
    ├── runner.py                            (small CLI that drives any policy through
    │                                         SkillsBenchmark and reports per-level pass/fail)
    ├── navigate.py                          (tiny per-level scripted policies)
    ├── mine.py
    ├── craft_miner.py
    ├── place_miner.py
    ├── fuel_and_collect.py
    ├── belt_line.py
    ├── arm_transfer.py
    └── mini_factory.py

tests/benchmarks/
├── test_skills_benchmark.py                 (Benchmark protocol conformance + level shapes)
├── test_skills_scripted_solves.py           (each scripted policy scores its level = pass)
└── test_skills_random_floor.py              (random policy floor)

docs/
├── cookbook.md                              (UPDATED — add "Build your own skills challenge")
└── skills_benchmark.md                      (NEW — user-facing benchmark doc)

scripts/
└── bench_skills_runner.py                   (NEW — A/B throughput harness for runner)
```

### Migration: what gets deleted

The existing skills code is per-skill ad-hoc gymnax wrappers with custom
dense rewards. It does not implement the `Benchmark` protocol and it
duplicates effort. It gets replaced wholesale.

Delete:
- `factoriax/benchmarks/skills/mining.py` — replaced by
  `levels.py::build_mine_level` + `achievements.py::skills_conditions`
- `factoriax/benchmarks/skills/place_miner.py` — replaced by
  `levels.py::build_place_miner_level` + the same conditions function
- The `MiningSkill` / `PlaceMinerSkill` gymnax wrapper classes — gone.
  Training reads observations directly from the inner `FactoriaXEnv`
  wrapped via `factoriax.make()`, with reward computed by
  `reward.py::skills_reward` from achievement deltas.
- The `SKILL_ENVS` dict in `baselines/skills/train_ppo.py` — replaced by
  iterating over `SkillsBenchmark().levels()`.

The old PPO numbers in `runs/skills_ppo_*` stay as historical reference
but the new training script writes to `runs/skills_<level_name>/`.

## Code Style

The conventions in `CLAUDE.md` apply (PEP 8, 88-col, snake_case, full type
hints, ruff + mypy clean, no `Any` unless forced, docstrings on public
APIs). One concrete example showing the shape we want — this is the
authoring template for the benchmark module:

```python
# factoriax/benchmarks/skills/levels.py

"""Level builders for the skills challenge.

Each builder is a pure function returning a (Level, EnvParams) tuple.
Levels are deterministic — no PRNG state inside — so the benchmark is
fully reproducible. EnvParams carry the per-level step budget.

Layouts are kept small (5x5 to 16x16) so JIT compile and rollout cost
is low and the benchmark runs end-to-end in seconds, not minutes.
"""

from __future__ import annotations

from factoriax.constants import BlockType, Direction, ItemType, MachineType
from factoriax.levels import Level, LevelBuilder
from factoriax.state import EnvParams


def build_navigate_level(
    seed: int = 0,
) -> tuple[Level, EnvParams, frozenset[int]]:
    """Smallest possible level: walk to the bottom-right corner.

    A 5x5 grass map. The goal tile is fixed at ``(map-1, map-1)`` so the
    achievement function (``SKILL_NAVIGATE``) is a pure
    ``state``-only check — no per-level closure state. Only the spawn
    position varies with *seed*; that's enough to prevent an agent from
    overfitting to "always step right then down" without needing to
    plumb a goal coordinate through ``EnvState``.

    Action mask: only ``MOVE_*`` and ``NOOP`` are exposed. Every other
    action is blocked, so the agent cannot accidentally crash into a
    crafting menu or pick up a tile while learning to walk.

    Args:
        seed: Numpy RNG seed for spawn position. Default ``0`` is the
            canonical seed used by :class:`SkillsBenchmark`.

    Returns:
        Tuple of (level, params, blocked_actions). Pass *blocked_actions*
        to :class:`ActionMaskWrapper` (or rely on
        :class:`BenchmarkRunner` to do it for you).
    """
    rng = np.random.default_rng(seed)
    builder = LevelBuilder(5, 5)
    # Spawn anywhere except the goal tile.
    while True:
        spawn_y, spawn_x = int(rng.integers(0, 5)), int(rng.integers(0, 5))
        if (spawn_y, spawn_x) != (4, 4):
            break
    builder.set_player_position(spawn_x, spawn_y)
    level = builder.build(f"skills_navigate_seed{seed}")
    params = EnvParams(map_width=5, map_height=5, num_players=1, max_timesteps=200)
    return level, params, NAVIGATE_BLOCKED_ACTIONS
```

Note the design constraint: achievement conditions must be pure
functions of `EnvState`, and `EnvState` doesn't carry "goal" metadata.
So levels keep their target conditions **layout-invariant**:

- `navigate`: goal is always the bottom-right corner.
- `mine`: goal is "any ore in inventory" (works for any ore-patch layout).
- `belt_line`: goal is "item on the right-edge belt tile" (fixed
  destination, varied source/path).

What the seed varies is the *starting state* and any distractors, not
the achievement target. That gives the RL agent meaningful layout
variety without expanding the state schema or changing `EnvState`.

A user training on just this level — possibly with a different seed per
parallel env — looks like this. No `SkillsBenchmark`, no
`BenchmarkRunner`, just direct env construction:

```python
import factoriax
from factoriax.benchmarks.skills import build_navigate_level
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper

level, params, blocked = build_navigate_level(seed=42)
env, _ = factoriax.make(level)
env = ActionMaskWrapper(env, blocked)
# ... train PPO against env, params ...
```

Scripted policies follow the same shape — one file, one function, no
classes unless the policy needs internal state across many ticks:

```python
# baselines/skills/scripted/navigate.py

"""Scripted baseline for the navigate skill."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from factoriax.constants import Action
from factoriax.state import EnvState


def navigate_policy(obs: jax.Array, state: EnvState) -> jax.Array:
    """Walk the player toward the bottom-right goal corner.

    Reads player position from ``state`` (scripted policies are allowed
    to peek at state — they're a baseline, not an agent under eval).
    """
    py, px = state.player_y[0], state.player_x[0]
    goal_y, goal_x = state.map.shape[0] - 1, state.map.shape[1] - 1
    return jnp.where(
        px < goal_x, jnp.int32(Action.MOVE_RIGHT),
        jnp.where(py < goal_y, jnp.int32(Action.MOVE_DOWN), jnp.int32(Action.NOOP)),
    )
```

Things this style does *not* allow:

- No emoji or unicode emoji-likes (per `CLAUDE.md`).
- No `print(...)` for errors — `logger.error(...)`.
- No mutable default arguments. No bare `except:`.
- No comments restating what the code does. Comments only when *why* is
  non-obvious.
- No "future-proofing" abstractions for skills not in the curriculum yet.

## Testing Strategy

Three test files cover the benchmark, all under `tests/benchmarks/`.

**`test_skills_benchmark.py`** — protocol conformance.

- Each level's `EnvParams.map_width/height` matches the level's declared
  shape.
- `len(SkillsBenchmark().levels()) == 8` and names are unique.
- `isinstance(SkillsBenchmark(), Benchmark)` — runtime-checkable.
- `SkillsBenchmark().score([])` returns `0.0` (empty input edge case).
- Aggregate score with all-zero `LevelResult` masks returns `0.0`.
- Aggregate score with all-success masks returns `1.0`.

**`test_skills_scripted_solves.py`** — each scripted policy solves its
level. Runs each (level, scripted_policy) pair through `BenchmarkRunner`
and asserts `LevelResult.achievements_unlocked[i] == True` for the level's
target achievement bit. This is the load-bearing test — if a scripted
policy regresses, CI fails. JIT-heavy, marked `@pytest.mark.slow`.

**`test_skills_random_floor.py`** — sanity floor. Runs a uniform random
policy across all 8 levels with a fixed seed; asserts the aggregate score
is at most 2/8 (random shouldn't accidentally solve `arm_transfer` or
`mini_factory`). Catches the failure mode where an achievement condition
is too lax.

What we *don't* do in pytest:

- No throughput / sps benchmarks. Per the project's `feedback_no_perf_tests`
  rule, throughput A/Bs go in `scripts/bench_skills_runner.py` as
  standalone CLIs and are run manually before perf-impacting commits.
- No PPO training tests. PPO convergence is too noisy for unit tests; the
  training script lives in `baselines/` and is exercised via `scripts/`.

Coverage target: respect the project floor (currently 37) — adding ~600
lines of benchmark code with full test coverage should *raise* the floor,
not lower it.

## Boundaries

**Always:**

- Implement levels as deterministic builders **given a seed**. The seed
  is an explicit argument; once fixed, the layout must be reproducible
  across runs and machines. `SkillsBenchmark.levels()` always uses
  seed `0` for evaluation. PRNG use inside builders is fine, but only
  when threaded through the seed argument — never `np.random` global
  state, never `time.time()`, never anything that varies across machines.
- Achievement conditions stay inside the JAX-pure path:
  `Callable[[EnvState], jax.Array]` of shape `(MAX_ACHIEVEMENTS,)`. They
  must be JIT-traceable.
- All state mutation goes through the existing JAX state machinery
  (per `feedback_state_management`). The benchmark is a *consumer* of the
  engine — it never touches `EnvState` fields outside `achievement_fn`.
- Run scripted-baseline tests before committing any change to a level
  file or to the conditions. If the scripted policy stops solving its
  level, the level changed in a way that broke its target achievement —
  fix it before merging.

**Ask first:**

- Adding a 9th skill or extending into Tier 4-research / Tier 5-repair —
  scope creep, gets a separate spec.
- Changing the `Benchmark` protocol itself (`core.py`). The skills
  benchmark must conform to the existing protocol; if conforming is
  awkward, that's a signal worth discussing, not a quiet protocol change.
- The one extension already approved in this spec is adding an optional
  `blocked_actions: frozenset[int] | None` field to `BenchmarkLevel`.
  Anything beyond that needs sign-off.
- Adding new dependencies. Build on `jax`, `gymnax`, `optax`, `flax`,
  `pytest`, `ruff`, `mypy` — anything else needs sign-off.
- Tuning RecipeBalance for the `craft_miner` or `mini_factory` levels.
  Recipes are shared with rocket; rebalancing for skills could regress
  the rocket benchmark.
- Changing per-level action masks after they're set. The mask defines
  what skill the level tests; loosening it changes the benchmark's
  meaning. Tightening it can break the scripted baseline.

**Never:**

- Make a level non-deterministic *for a given seed*. Two calls to
  `build_navigate_level(seed=42)` must produce byte-identical levels.
  (This is the no-`np.random`-global-state rule restated.)
- Make achievement conditions depend on something other than `EnvState`.
  Conditions are pure functions of state; per-level metadata (goal
  coords, target items) gets baked into either the achievement function
  itself or into layout-invariant reads of state. No closure-captured
  goal coordinates that change per level.
- Add dense per-step reward shaping to `SkillsBenchmark` itself. The
  benchmark's reward signal is sparse-on-unlock, scaled by remaining
  time. Dense shaping (visit-count bonuses, distance-to-goal, etc.) is a
  *training-side* opt-in that lives in `baselines/skills/`, not in the
  benchmark module — otherwise the benchmark stops measuring the same
  thing across users.
- Drop items silently from the player inventory in any level — overflow
  must surface as an action failure (per `feedback_no_item_loss`). Levels
  that hand the agent items must size `max_machines` and inventory caps
  to fit the worst case.
- Run pygame, SDL, or any rendering side-effect from inside the benchmark
  module — rendering is a `factoriax/ui/` concern (per
  `feedback_subsystem_self_init`).
- Skip the perf A/B (`scripts/bench_skills_runner.py`) when changing
  anything that touches the runner inner loop. Per the project rule,
  game-logic perf changes need a benchmark before commit.

## Decisions resolved during spec review

These were open at draft time; user-confirmed:

1. **Layout-varying seed:** **in scope**. Each `build_*_level()` takes
   `seed: int = 0`. Seed `0` is the canonical evaluation layout used by
   `SkillsBenchmark.levels()`; arbitrary seeds are available for
   training-time variation. Achievement conditions are
   layout-invariant so the seed only changes spawn / source positions /
   distractors, not the target.
2. **Time-weighted scoring:** **in scope**. Score is
   `solved * (max_timesteps - timesteps_used + 1) / max_timesteps`,
   per-level, mean across levels for the aggregate. Same shape mirrored
   in the training reward. (See *Scoring*.)
3. **Mask hand-crafting on levels 5–8:** **in scope**. Levels 5–8 block
   all `CRAFT_*` actions and pre-place the machinery the agent needs.
   Level 3 is the only level that exposes hand-crafting (just
   `CRAFT_MINER`). Mirrors the rocket benchmark and keeps a policy
   trained on skills transferable to rocket.
4. **`mini_factory` capstone:** **kept**. It's the only level that
   exercises composition end-to-end and is the structural bridge to
   rocket.

## Open Questions

None blocking. Anything that comes up during planning gets recorded in
`tasks/skills_benchmark_plan.md` rather than added back here.

---

## Verification checklist

Before this spec is locked:

- [x] All six core areas covered (objective / commands / structure /
      style / testing / boundaries)
- [x] Success criteria specific and testable
- [x] Boundaries split into Always / Ask first / Never
- [ ] Spec reviewed and approved by user
- [ ] Spec committed to the repo

Next phase after lock: write `tasks/skills_benchmark_plan.md` per the
`feedback_plan_before_implement` rule. No code edits until that plan
exists and is reviewed.
