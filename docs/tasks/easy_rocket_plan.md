# Implementation Plan: Easy Rocket Scenario

This plan implements `docs/specs/2026_easy_rocket_scenario.md`. The
spec defines *what* and *why*; this document defines *how* — the
order of work, the file-level scope of each step, the verification
gate between steps, and the risks.

## Overview

The easy-rocket scenario is fully scenario-local: a new file under
`factoriax/scenarios/easy_rocket.py`, a small refactor to lift two
shared helpers out of `factoriax/scenarios/rocket.py` into a private
common module, and one test file. No engine touches, no recipe-table
changes, no new enum values. The machine-efficiency multipliers and
the four belt-network achievements are deferred to their own specs
(`docs/specs/2026_machine_efficiency_multipliers.md` and
`docs/specs/2026_entity_connection_graph.md`); this plan ships v1
with nine active achievements and four graph-gated stubs.

Six vertical slices, each leaves the codebase in a working state
with tests green. The slicing is bottom-up by component
(recipe book → level → helpers → conditions → reward → scenario)
because the scenario class at the top is a thin shell that
composes the others. Tests live with each slice rather than as a
trailing phase, matching the spec's "every test runs as a fast
unit test against pure functions" discipline.

## Architecture Decisions (locked in the spec)

- The scenario is self-contained in
  `factoriax/scenarios/easy_rocket.py`. No engine edits, no recipe
  module edits.
- The 13-achievement list is locked. Equal weights (1.0 each),
  `MAX_EASY_ROCKET_SCORE = 13.0`. Graph-gated achievements (8, 10,
  11, 12) ship as stubs that always read `False` and carry a `TODO`
  pointing at the connection-graph spec.
- Procgen runs at scenario construction time, keyed off a `seed`
  kwarg. The resulting `Level` is deterministic from `(seed, scenario
  version)` and stable across `env.reset` calls. The spec's
  "Procgen runs once per `reset`" boundary is aspirational and
  would require an engine hook; we defer that to a follow-up if a
  consumer ever actually needs per-reset variation. Document the
  gap in the implementation docstring and the plan's open
  questions.
- Recipe book ships with only the eight user-owned outputs
  (`MINER`, `ASSEMBLER`, `CONVEYOR_BELT`, `SPLITTER`, `CROSSING`,
  `HULL`, `ENGINE_UNIT`, `ROCKET`). Sane default inputs derived from
  `BASE_RECIPES` (`ROCKET` rewired to use `ENGINE_UNIT` since
  `ROCKET_CORE` and `AVIONICS` are dropped). Smelts and intermediates
  are intentionally absent — the user (Mickey) adds them post-merge
  along with any balance adjustments, as the spec's recipe-book
  section calls out. The book validates uniquely (no input-type-set
  collisions among the eight), but the agent cannot reach the
  rocket end-to-end until the user fills in the missing chain. That
  gap is by design.
- Observations reuse `factoriax.observations.local_array` with
  `radius=5`. The scenario does not override the observation
  function.
- Hand-craft actions stay available. `blocked_actions = frozenset()`.
- Tests run at the unit level against `state_factory`-built
  fixtures. No `env.reset` / `env.step` calls anywhere in
  `tests/scenarios/test_easy_rocket.py`. The existing
  `tests/scenarios/test_rocket_scenario.py` already proves this
  pattern works.

## Dependency Graph

```
   ┌──────────────────────────────────────────────────────┐
   │  Slice 1. Recipe book (data + validation)            │
   │  Slice 2. Level builder (procgen + invariants)       │
   │  Slice 3. Shared condition helpers (lift + extend)   │
   └─────────────────────┬────────────────────────────────┘
                         ▼
            ┌────────────────────────────┐
            │  Slice 4. Condition fn     │
            │  (v1 wired, graph stubbed) │
            └────────────┬───────────────┘
                         ▼
            ┌────────────────────────────┐
            │  Slice 5. Reward + scoring │
            └────────────┬───────────────┘
                         ▼
            ┌────────────────────────────┐
            │  Slice 6. Scenario class + │
            │  exports + obs shape test  │
            └────────────────────────────┘
```

Hard constraints:

- 1, 2, 3 can be done in any order — they share no code.
- 4 needs 3 (uses the lifted helpers).
- 5 needs 4 (reward weights index by achievement id).
- 6 needs 1, 2, 4, 5 (the scenario class composes all four).

Parallel-safe (independent file edits):

- Slices 1, 2, 3 touch disjoint files and can run in parallel if
  multiple sessions are available.

## Task List

### Slice 1 — Recipe book

#### Task 1.1: Define `EASY_ROCKET_RECIPE_BOOK` and its `RecipeTable`

**Description:** Construct a fresh `RecipeBook` from a hand-picked
tuple of `Recipe` objects. This is plain data construction — the
only constraints `RecipeBook.__post_init__` enforces are output
uniqueness and per-`(machine_type, arity)` uniqueness of the
unordered input type-set, both already exercised by
`tests/test_recipe_book.py`. Recipe identity (output item, input
item types, machine type) is freely choosable at construction;
`RecipeOverride` is the only layer that's identity-frozen, and we
do not use it here.

The book contains exactly eight recipes, all running on
`ASSEMBLER`:

1. **Seven recipes copied from `BASE_RECIPES` verbatim** — `MINER`,
   `ASSEMBLER`, `CONVEYOR_BELT`, `SPLITTER`, `CROSSING`, `HULL`,
   `ENGINE_UNIT`. Their input types match the spec's intended tier
   layout in the base book; the agent will need plates and
   intermediates (`IRON_PLATE`, `WIRE`, `FRAME`, `CIRCUIT`,
   `MOTOR`, `COAL`) to make any of them. Those plates and
   intermediates are not in the book.
2. **One rewired recipe** — `ROCKET`. Base takes
   `((HULL, 6), (ROCKET_CORE, 4))`; easy-rocket drops `ROCKET_CORE`
   and `AVIONICS`, so we construct a fresh
   `Recipe(output=ROCKET, inputs=((HULL, 1), (ENGINE_UNIT, 1)),
   ticks=50, name="Rocket")` as a sane default. Tight inputs match
   the "stripped-down for fast RL iteration" theme.

All eight have distinct unordered input type-sets on the assembler,
so `RecipeBook.__post_init__` validates cleanly.

`AVIONICS` and `ROCKET_CORE` are not in the book and never become
outputs of any easy-rocket recipe. The rewired `ROCKET` is the
only chain that references the rocket capstone.

**Scope caveat — book is intentionally gapped.** With only eight
recipes and no smelts/intermediates, the agent cannot actually
mine ore, smelt plates, build wire, or progress beyond raw input
buffers. The book constructs, the `RecipeTable` projects, the
scenario instantiates, and the engine runs — but no end-to-end
agent can score above the early mining achievements until the user
adds the missing recipes. That gap is by design: the spec hands
smelts and intermediates to the user as their own forcing function
for thinking through the chain. The user owns that follow-up;
this slice is complete when the eight user-owned recipes are
landed and the book constructs.

**Acceptance:**
- `EASY_ROCKET_RECIPE_BOOK = RecipeBook(recipes=(...))` constructs
  without raising.
- `EASY_ROCKET_RECIPE_TABLE = RecipeTable.from_book(EASY_ROCKET_RECIPE_BOOK)`
  succeeds.
- The book contains exactly the eight user-owned outputs listed
  above and no others.
- `AVIONICS` and `ROCKET_CORE` are not outputs of any recipe in
  the book.
- The rewired `ROCKET` recipe lists `HULL` and `ENGINE_UNIT` as
  its two inputs.

**Verification:**
- New test `test_recipe_book_constructs` instantiates the book and
  asserts the exact set of eight outputs.
- New test `test_recipe_book_rocket_is_rewired` confirms the
  `ROCKET` recipe's input items are `(HULL, ENGINE_UNIT)` and that
  `AVIONICS` / `ROCKET_CORE` are absent from every recipe's input
  list as well as from the output set.

**Files:** `factoriax/scenarios/easy_rocket.py` (new — book + table
constants), `tests/scenarios/test_easy_rocket.py` (new — first
two tests).

**Size:** S.

**Note on the spec's "empty inputs template":** the spec's recipe
book section ships an `EASY_ROCKET_RECIPES_TEMPLATE` with
`inputs=()` on every recipe and frames it as a forcing function
the user fills in by hand. Reading "user" literally as the human
maintainer, this slice ships working defaults for the eight
recipes so the book validates and downstream slices have a stable
`RecipeTable` to compose against; the user owns inputs/balance
adjustments and the smelt/intermediate fill-in as their own work.

### Slice 2 — Level builder

#### Task 2.1: Implement `build_easy_rocket_level(key)`

**Description:** Pure function returning a `Level`. Takes a
`jax.Array` PRNG key and produces a 16x16 map with a player at the
centre `(8, 8)`, no pre-placed machines, and six 2x2 ore patches
(iron, copper, tin, silicon, coal, limestone) placed by uniform sampling with
rejection. Reject samples where (a) the patch overlaps another
placed patch, (b) the patch overlaps the spawn or its
`forbid_radius=1` Chebyshev ring.

Use a `LevelBuilder` (already in `factoriax/levels.py`). Per-tile
resources: 3000 for ore patches. The map background is `DIRT`. The
sampling loop runs host-side at scenario construction time (not
inside JIT), so use `jax.random.split` + `jax.random.randint` on
the host. Cap the rejection-sample loop at, say, 1000 attempts per
patch; raise `RuntimeError` if exceeded (sanity guard against a
mis-tuned `forbid_radius`; mean attempts will be ~1).

**Acceptance:**
- `build_easy_rocket_level(jax.random.PRNGKey(0))` returns a `Level`
  with `map_width == map_height == 16`, no `place_machine` calls,
  and a player at `(8, 8)`.
- Two calls with the same key produce equal `Level` objects
  (`np.array_equal` across every array field).
- Two calls with different keys produce different ore positions in
  at least one patch on most samples.
- All six ore block types are present in the returned level's
  block map.
- No patch overlaps the spawn tile or its 3x3 forbidden zone.

**Verification:**
- `test_build_level_dimensions` — checks shapes.
- `test_build_level_determinism` — same key → equal `Level`.
- `test_build_level_keys_vary` — across 5 keys, at least one patch
  position differs from key 0.
- `test_build_level_invariants` — across 5 keys, all six ores
  present, no patch overlaps spawn, no patches overlap each other.

**Files:** `factoriax/scenarios/easy_rocket.py` (extended),
`tests/scenarios/test_easy_rocket.py` (extended).

**Size:** M.

### Slice 3 — Shared condition helpers

#### Task 3.1: Lift `_holds_item`, `_count_machines`, `_any_entity_buf_nonempty`, `_any_assembler_has_output` to `factoriax/scenarios/_common.py`

**Description:** Create a new private module that hosts the
condition-function helpers currently living in
`factoriax/scenarios/rocket.py`. The lift is pure code motion —
function bodies copy verbatim, signatures unchanged.
`factoriax/scenarios/rocket.py` then imports the helpers from the
new module rather than defining them locally. The `RocketScenario`
class and the `rocket_conditions` function stay untouched at the
public-API level; only the internal helper imports move. This
respects the spec's "Keep `RocketScenario` untouched" boundary
(the public surface is unchanged) while avoiding duplicating the
helpers into easy-rocket.

**Acceptance:**
- `factoriax/scenarios/_common.py` exists with the four helpers.
- `factoriax/scenarios/rocket.py` imports them and contains no
  duplicate definitions.
- `rocket_conditions` behaviour is byte-identical (covered by the
  existing `tests/scenarios/test_rocket_scenario.py` suite).

**Verification:**
- `uv run pytest tests/scenarios/test_rocket_scenario.py -q` passes
  unchanged.
- `uv run mypy factoriax/scenarios/_common.py factoriax/scenarios/rocket.py`
  passes.

**Files:** `factoriax/scenarios/_common.py` (new),
`factoriax/scenarios/rocket.py` (helper definitions deleted,
imports added).

**Size:** S.

#### Task 3.2: Add `_block_under_entity` and `_distinct_blocks_under_machines` helpers

**Description:** Two new helpers in `_common.py` needed by easy-rocket
achievements 7 and 9.

- `_block_under_entity(state, ent_idx) -> int` reads
  `state.world_blocks[ent_y, ent_x]` for one active entity. Used by
  achievement 7 ("Place miner on ore") — caller iterates over active
  miners and asks "is the block under this miner an ore type?".
- `_distinct_blocks_under_machines(state, machine_type, block_set) ->
  jax.Array` counts the number of distinct block types from
  `block_set` that appear underneath active entities of the given
  `machine_type`. Used by achievement 9 ("Have miners on three ore
  types") — caller asks "are there ≥3 distinct ore blocks under
  miners?".

Both are JIT-pure; the second uses a small `jnp.unique` reduction
or a loop over the bounded `block_set` size.

**Acceptance:**
- Helpers live in `_common.py`, type-annotated, JIT-traced under
  `jax.jit` without raising.
- Direct unit tests exercise each helper against `state_factory`
  fixtures (miner-on-iron, miner-on-dirt, three-miners-three-ores,
  three-miners-one-ore).

**Verification:**
- New tests `test_common_block_under_entity` and
  `test_common_distinct_blocks_under_machines` in
  `tests/scenarios/test_common.py` (new file) pass.

**Files:** `factoriax/scenarios/_common.py` (extended),
`tests/scenarios/test_common.py` (new).

**Size:** S.

### Slice 4 — Condition function

#### Task 4.1: Implement `easy_rocket_conditions(state)`

**Description:** Pure function returning a `(MAX_ACHIEVEMENTS,)`
bool array. The first 13 slots correspond to the spec's
achievement list; the rest are zero-padded.

V1 wires up achievements 1, 2, 3, 4, 5, 6, 7, 9, 13 using the
shared helpers. Graph-gated achievements 8, 10, 11, 12 are wired
to `jnp.zeros(1, dtype=bool)` constants with a `# TODO(connection-graph
spec)` comment pointing at the spec path.

Achievement 3 uses the four miner-craft-relevant ore types (iron,
copper, tin, coal — derived from the miner recipe trace, with
silicon excluded). Spell the type list out as a module-level
constant `_MINER_CRAFT_ORES` so the reasoning is visible at the
call site.

Achievement 2 uses all five raw-ore types present on the easy-rocket
map (`_EASY_ROCKET_ORE_TYPES = (IRON_ORE, COPPER_ORE, TIN_ORE,
SILICON, COAL)`).

**Acceptance:**
- `easy_rocket_conditions(state)` returns a
  `(MAX_ACHIEVEMENTS,)` `bool_` array.
- For each v1 achievement, a positive fixture (state where the
  condition should hold) returns `True` at the correct index, and
  a negative fixture returns `False` at the same index.
- For each graph-gated achievement, any fixture (positive or
  negative) returns `False` at the corresponding index. The
  positive case asserts the stub behaviour so a future implementer
  knows to update the test alongside the wiring.

**Verification:**
- Parametrised tests in `tests/scenarios/test_easy_rocket.py`:
  - `test_conditions_inventory_thresholds` — covers achievements
    1, 2, 3 with multiple inventory fixtures.
  - `test_conditions_item_holding` — covers 4, 5, 6 (craft a miner
    / assembler / belt).
  - `test_conditions_placement_on_blocks` — covers 7
    (miner-on-ore) and 9 (three distinct ore types).
  - `test_conditions_rocket_placed` — covers 13.
  - `test_conditions_graph_stubs_always_false` — asserts 8, 10,
    11, 12 always return False (including in fixtures designed to
    look like they should unlock, e.g. belts adjacent to miners).

**Files:** `factoriax/scenarios/easy_rocket.py` (extended),
`tests/scenarios/test_easy_rocket.py` (extended).

**Size:** M.

### Slice 5 — Reward and scoring

#### Task 5.1: Define `EASY_ROCKET_ACHIEVEMENT_WEIGHTS`, `MAX_EASY_ROCKET_SCORE`, `easy_rocket_reward`

**Description:** Mirror the rocket scenario's pattern.

```python
NUM_EASY_ROCKET_ACHIEVEMENTS = 13
_TIER_WEIGHTS = [1.0] * NUM_EASY_ROCKET_ACHIEVEMENTS
EASY_ROCKET_ACHIEVEMENT_WEIGHTS = (
    jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.float32)
    .at[:NUM_EASY_ROCKET_ACHIEVEMENTS]
    .set(jnp.array(_TIER_WEIGHTS, dtype=jnp.float32))
)
MAX_EASY_ROCKET_SCORE: float = float(sum(_TIER_WEIGHTS))  # 13.0

def easy_rocket_reward(prev, new, params):
    return achievement_reward(prev, new, params,
                               weights=EASY_ROCKET_ACHIEVEMENT_WEIGHTS)
```

**Acceptance:**
- The three module symbols exist with the documented values.
- `easy_rocket_reward(prev, new, params)` returns a `float32`
  scalar when a single achievement bit flips, and that scalar equals
  the per-achievement weight (1.0).

**Verification:**
- `test_reward_single_unlock` — build `prev` with mask all zeros,
  `new` with one bit set, assert reward == 1.0.
- `test_reward_no_unlock_is_zero` — both states identical, assert
  reward == 0.0.

**Files:** `factoriax/scenarios/easy_rocket.py` (extended),
`tests/scenarios/test_easy_rocket.py` (extended).

**Size:** S.

### Slice 6 — Scenario class, exports, observation shape

#### Task 6.1: Implement `EasyRocketScenario`

**Description:** Add the class to `factoriax/scenarios/easy_rocket.py`.
Class attributes: `name = "easy_rocket"`, `num_players = 1`,
`achievement_fn = staticmethod(easy_rocket_conditions)`,
`blocked_actions = frozenset()`. The constructor takes one kwarg:
`seed: int = 0`, used to construct the PRNG key passed to
`build_easy_rocket_level`. `levels()` returns a single
`ScenarioLevel` with `env_params = EnvParams(max_timesteps=2000,
map_width=16, map_height=16, num_players=1,
recipe_table=EASY_ROCKET_RECIPE_TABLE)`. `score_level` returns
`0.0` (the placeholder pattern from rocket scenario, since per-level
scoring needs the achievement mask which only `score()` sees).
`score()` returns the mean weighted-achievement score across
`LevelResult`s.

**Acceptance:**
- `EasyRocketScenario()` instantiates without raising.
- `isinstance(EasyRocketScenario(), Scenario)` is `True`.
- `EasyRocketScenario().levels()` returns a one-element list whose
  `ScenarioLevel.env_params` matches the spec values.
- `EasyRocketScenario(seed=1).levels()[0].level` differs from
  `EasyRocketScenario(seed=0).levels()[0].level` in at least one
  ore patch position.

**Verification:**
- `test_scenario_construction` — instantiate, check protocol,
  check class attrs.
- `test_scenario_levels_shape` — check `env_params` fields and the
  level dimensions.
- `test_scenario_seed_changes_layout` — two seeds give different
  patch positions.
- `test_score_returns_max_with_full_mask` — build a synthetic
  `LevelResult` whose `achievements_unlocked[:13]` is all True;
  assert `score([result]) == MAX_EASY_ROCKET_SCORE`.
- `test_score_returns_zero_with_no_mask` — empty mask path.

**Files:** `factoriax/scenarios/easy_rocket.py` (extended),
`tests/scenarios/test_easy_rocket.py` (extended).

**Size:** M.

#### Task 6.2: Add the observation-shape regression test

**Description:** Build a synthetic `EnvState` matching the
easy-rocket level dimensions (16x16) via the existing
`state_factory` fixture, call
`local_array(state, params, 0, radius=5)`, and assert the returned
shape and dtype. The formula is `10 spatial channels * 11 * 11 +
NUM_PLAYER_SCALARS`; pin both numbers so a future channel addition
forces a test update.

**Acceptance:**
- The test passes today and would fail if `local_array` changed
  channel count or output shape.
- No `env.reset` / `env.step` calls — fixture comes from
  `state_factory`.

**Verification:**
- `test_observation_shape` runs in milliseconds.

**Files:** `tests/scenarios/test_easy_rocket.py` (extended).

**Size:** S.

#### Task 6.3: Export public symbols from `factoriax/scenarios/__init__.py`

**Description:** Add `EasyRocketScenario`,
`EASY_ROCKET_ACHIEVEMENT_WEIGHTS`,
`EASY_ROCKET_RECIPE_BOOK`, `EASY_ROCKET_RECIPE_TABLE`,
`MAX_EASY_ROCKET_SCORE`, `build_easy_rocket_level`,
`easy_rocket_conditions`, `easy_rocket_reward` to the
`scenarios/__init__.py` imports and `__all__` list, alphabetised
to match the existing pattern.

**Acceptance:**
- `from factoriax.scenarios import EasyRocketScenario` works at
  the top level.
- `factoriax.scenarios.__all__` lists every new public symbol.

**Verification:**
- New test `test_easy_rocket_exports` in
  `tests/scenarios/test_easy_rocket.py` imports each symbol and
  asserts it is non-None.

**Files:** `factoriax/scenarios/__init__.py`,
`tests/scenarios/test_easy_rocket.py` (extended).

**Size:** S.

### Checkpoint — Pre-merge gate

- [ ] All Slice 1–6 tasks complete.
- [ ] `uv run ruff check factoriax/scenarios/ tests/scenarios/`
      green.
- [ ] `uv run ruff format --check factoriax/scenarios/ tests/scenarios/`
      green.
- [ ] `uv run mypy factoriax/scenarios/easy_rocket.py
      factoriax/scenarios/_common.py
      factoriax/scenarios/rocket.py` green.
- [ ] `uv run pytest tests/scenarios/test_easy_rocket.py
      tests/scenarios/test_common.py
      tests/scenarios/test_rocket_scenario.py -q` green.
- [ ] `uv run pytest tests/scenarios/test_easy_rocket.py -q
      --durations=10` shows total wall-clock under 1 second.
- [ ] Manual smoke: `uv run python -m factoriax` lets the user pick
      the easy-rocket scenario from the menu and play a few steps
      without crashing. (This is the only manual check; the rest
      are pure unit assertions.)
- [ ] **Review with human.**

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Rocket recipe rewiring breaks `RocketScenario` | High | The easy-rocket book builds a NEW `Recipe(output=ROCKET, ...)` inside a NEW `RecipeBook` rather than mutating `BASE_RECIPES` or `BASE_RECIPE_BOOK`. The base book and `ROCKET_RECIPE_BOOK` stay untouched. Run the existing rocket-scenario test suite after Slice 1 to catch any accidental shared-state damage. |
| Easy-rocket scenario looks broken to a future reader | Medium | The book ships with only 8 user-owned recipes; smelts and intermediates are missing by design. The agent can run but cannot make plates or intermediates from raw ore, so it cannot earn any post-mining achievements. Document the gap in the easy-rocket module docstring and in `EASY_ROCKET_RECIPE_BOOK`'s comment so the next reader knows it is intentional and the user owns the fill-in. |
| Level builder's rejection sampler infinite-loops | Medium | Cap attempts per patch at 1000 with a `RuntimeError` on overflow. Expected mean attempts is ~1 since 20 patch tiles fit easily in 256-tile map. The cap is a sanity guard. |
| Helper lift to `_common.py` regresses `rocket_conditions` | Medium | Lift is pure code motion (no signature or body changes). Existing rocket-scenario tests run as the regression guard; landed only if they stay green. |
| Spec boundary "Procgen runs once per `reset`" not met | Low | We build at scenario construction, not at reset. Documented in the implementation docstring as a known gap; flagged in the plan's open questions for the spec author to decide if it needs an engine change. |
| Graph-gated stubs misread as bugs | Low | Test `test_conditions_graph_stubs_always_false` codifies the stub behaviour. `# TODO(connection-graph spec)` comment in the source points at the unblocking spec. |
| Achievement 3's "ores needed for crafting a miner" interpretation | Medium | Spelled out as `_MINER_CRAFT_ORES = (IRON_ORE, COPPER_ORE, TIN_ORE, COAL)` constant with a docstring explaining the trace. If the spec author meant something different (e.g. all five ores), the constant is a one-line fix. |

## Parallelization Opportunities

If multiple sessions are available:

- **Slices 1, 2, 3** touch disjoint files and have no shared
  state. They can be done in parallel.
- **Slices 4 and 5** must be sequential.
- **Slice 6** must wait for 1, 2, 4, 5 to finish.

Strictly sequential within a slice: each task within Slice 6 (6.1
→ 6.2 → 6.3) builds on the previous.

## Open questions

These are not blockers for the plan — they get resolved during
implementation or surfaced to the spec author if they need
re-spec.

- **Per-reset procgen vs per-construction procgen.** The spec's
  boundary says procgen runs per `env.reset` and is keyed off the
  env's PRNG. The current `FactoriaXEnv` reset path does not
  support a procgen hook — it picks between
  `generate_state(key, params)` (the engine's general procgen) and
  `build_state(self._level, params)` (deterministic from a static
  level). Plumbing per-reset easy-rocket procgen would need a new
  `level_fn: Callable[[jax.Array, EnvParams], Level] | None`
  constructor argument on `FactoriaXEnv`, which is "Ask first"
  territory. v1 builds at construction time and accepts that the
  same `EasyRocketScenario` instance gives the same layout across
  resets; if researchers need per-reset variation, they construct
  multiple `EasyRocketScenario` instances with different seeds.
  Flag at plan-review.
- **Achievement 3 ore list.** Spelled out as `(IRON_ORE,
  COPPER_ORE, TIN_ORE, COAL)` based on the miner recipe trace. If
  the spec author intended all five map ores (including silicon),
  the constant is a one-line change.
- **Smelts and intermediates fill-in.** The eight-recipe book
  this slice lands is a working but gapped book. The user owns
  the post-merge work of adding `IRON_PLATE`, `COPPER_PLATE`,
  `TIN_PLATE`, `WAFER`, `FRAME`, `CIRCUIT`, `WIRE`, `MOTOR`, and
  `SENSOR` recipes (and any per-recipe balance tweaks) so the
  chain becomes craftable end-to-end. Not a plan task; tracked
  here for visibility.
