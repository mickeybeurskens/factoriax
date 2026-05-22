# Spec: Easy Rocket Scenario

## Objective


Add a new scenario, `EasyRocketScenario`, alongside the existing
`RocketScenario`. It is a deliberately stripped-down variant aimed at
fast RL iteration: a small map, a short episode, a tight local
observation, and a hand-designed recipe book with at most three tiers
between raw ore and the rocket.

The scenario description that defines the intended agent experience:

> The player needs to build a rocket to escape the planet. The player
> starts without any resources, and needs to complete the rocket
> build. There is a step limit of 2000 steps. Note that the assemblers
> and miners are much more efficient at producing than producing by
> hand.

Why this scenario exists, in terms a reader new to FactoriaX can
understand: the canonical `RocketScenario` runs on a 32x32 map with 38
achievements, 25+ recipes, an 8000-step budget, and forbids hand
crafting outright. That setup is too large and too sparse for early
PPO / DQN iteration cycles — a single training run is slow to compile
and the agent rarely reaches the rocket within a reasonable wall-clock
budget. The "easy rocket" variant compresses every axis:

- The map is 16x16 instead of 32x32, so the spatial search is small
  and procedurally generated layouts still fit in a tight local
  observation.
- The recipe book has three tiers (raw → intermediate → rocket) with
  the dependency-heavy items (avionics, rocket_core, refractory,
  science packs) removed.
- The step budget is 2000, four times shorter than the full rocket
  scenario.
- Observations are vector-based local windows (`radius=5`, 11x11
  spatial patch) instead of either full-map flat arrays or RGB
  frames.
- Hand crafting is allowed. The 2000-step budget is itself the
  forcing function — an agent that hand-crafts every plate and
  intermediate runs out of steps before reaching the rocket. A
  separate spec (`docs/specs/2026_machine_efficiency_multipliers.md`)
  adds per-recipe knobs so machines can be made strictly more
  productive than hand crafting; once that lands, this scenario will
  apply non-neutral multipliers via a `RecipeBalance` overlay to make
  the gradient sharper. Until then the spec ships with neutral
  defaults and relies on the step budget alone.

The intended user is someone running RL baselines who wants a "does
the algorithm learn anything at all" signal in minutes rather than
hours. Achievements are sparse and listed in the scoring contract so
researchers can also use this scenario as a curriculum stepping stone
toward the full rocket.

### Non-goals

- Replacing or modifying `RocketScenario`. The existing scenario keeps
  its current map, recipe book, achievement list, and 8000-step
  budget. Easy rocket is additive.
- Designing a new observation function. We reuse `local_array` from
  `factoriax.observations`.

The achievement list is locked in this spec (see "Achievement contract"
below). It is 13 items, equally weighted, covering raw mining,
crafting, machine placement, and the first non-trivial belt-network
compositions.

## Tech Stack

- Python 3.12, JAX, Flax (for the existing `RecipeTable` struct).
- Project package manager: `uv`.
- Testing: pytest. Type checks: mypy. Lint/format: ruff.
- All scenario code is JIT-compatible — condition functions and reward
  functions are pure functions of `EnvState` / `EnvParams`.

## Commands

```
Install      uv sync
Run tests    uv run pytest tests/scenarios/ -q
Type check   uv run mypy factoriax/scenarios/easy_rocket.py
Lint         uv run ruff check factoriax/scenarios/easy_rocket.py
Format       uv run ruff format factoriax/scenarios/easy_rocket.py
Smoke run    uv run python -m factoriax  # play; pick easy_rocket scenario from menu
```

## Project Structure

New file:

```
factoriax/scenarios/easy_rocket.py    # Scenario class, level builder,
                                       # recipe book template,
                                       # 13-achievement contract
                                       # (with 4 graph-gated stubs),
                                       # reward + scoring.
```

Touched files:

```
factoriax/scenarios/__init__.py       # Export EasyRocketScenario and
                                       # the public helpers.
tests/scenarios/test_easy_rocket.py   # New file (see "Testing").
```

The scenario is fully self-contained in
`factoriax/scenarios/easy_rocket.py`. No cross-cutting changes —
`factoriax/recipes.py` and the JIT'd engine are not touched. The
machine-vs-hand efficiency mechanic is deferred to its own spec
(`docs/specs/2026_machine_efficiency_multipliers.md`); this scenario
ships with the existing balance machinery and reintroduces the knobs
via a `RecipeBalance` overlay once that spec lands.

## Code Style

- Snake_case for functions, PascalCase for classes, UPPER_CASE for
  module constants.
- Constants prefixed with ``_`` are module-private; the public surface
  is whatever `factoriax/scenarios/__init__.py` re-exports.
- No emojis in code, comments, or docstrings.

## Testing Strategy

Location: `tests/scenarios/test_easy_rocket.py`.

Tests must stay fast. Bloat in test time hurts the iteration loop, so
every test in this file runs at the unit level against pure functions
— **no `env.reset` / `env.step` calls anywhere**. Stepping the env
triggers a full JIT compile of the engine (seconds per test), and
none of the contracts below actually require running the engine to
verify. They verify either pure data (recipe table contents, level
geometry) or pure functions (condition functions, reward, score,
`local_array`) called against hand-built `EnvState` fixtures.

The full test list, all expected to run in well under a second total:

1. **Construction** — `EasyRocketScenario()` returns one
   `ScenarioLevel` whose `env_params.max_timesteps == 2000`,
   `map_width == map_height == 16`, and whose `Level` has a furnace
   and an assembler adjacent to spawn. Pure attribute reads.
2. **Procgen determinism** — calling the level builder with the same
   PRNG key twice produces equal `Level` objects (per-field array
   equality); two different keys differ in at least one of the five
   patch positions. Generator runs as plain JAX numpy ops — no env
   compile.
3. **Procgen invariants** — across **5** sampled keys (not 100):
   every level has all six required ore blocks present, no patches
   overlap, no patch overlaps spawn or the furnace/assembler tiles,
   and every patch is at least `forbid_radius` from spawn (Chebyshev).
   Five keys is enough to catch a broken generator with high
   confidence and keeps the test in the millisecond range.
4. **Recipe book shape** — the easy-rocket `RecipeBook` constructs
   without raising (so all `__post_init__` invariants pass), contains
   the eight user-designed outputs (assembler, conveyor_belt,
   splitter, crossing, miner, hull, engine_unit, rocket), and reaches
   all of them from raw ore through the supporting smelts and
   intermediates. Plain set/membership assertions on the recipe
   tuple.
5. **Observation shape** — call
   `local_array(hand_built_state, easy_rocket_params, 0, radius=5)`
   on an `EnvState` constructed directly from a `Level` (no
   `env.reset`). Assert dtype `float32`, 1-D, and the size matches
   the documented `11*11*channels + scalar_block` formula. This
   exercises the observation projection without compiling the env.
7. **Condition functions** — for each of the 13 achievements, build
   one `EnvState` where the condition should be `True` and one where
   it should be `False`, then call the easy-rocket condition function
   directly and assert the corresponding bit. Belt-network
   achievements (#8, #10, #11, #12 in the contract) get the most
   careful fixtures since their helpers are new code.
8. **Reward & scoring** — call the easy-rocket reward function on a
   `(prev_state, new_state)` pair where one achievement bit flips
   from 0 to 1; assert the returned scalar equals the configured
   weight for that achievement. Call `EasyRocketScenario().score()`
   over a synthetic `LevelResult` whose `achievements_unlocked` mask
   has all 13 bits set; assert it returns `MAX_EASY_ROCKET_SCORE`
   (`= 13.0` with equal weights). No env step.

The only contract this strategy does *not* directly verify is
"the engine, when actually stepped, produces the rewards and
observations the unit tests expect" — that integration is covered by
the existing `tests/scenarios/test_rocket.py` and the engine-level
test suite, which exercise the same `FactoriaXEnv` code paths. If a
future change breaks the integration without breaking the unit
contract, those existing tests catch it.


## Boundaries

### Always do

- Keep `RocketScenario` untouched. Easy rocket is a new class.
- Use `local_array` from `factoriax.observations` for the
  observation. Do not write a new observation function.
- Make recipe edits via the `RecipeBalance` overlay or by constructing
  a new `RecipeBook` — never by mutating `BASE_RECIPES` in place.
- Procgen runs once per `reset` and is keyed off the env's PRNG so
  episodes are reproducible from `key`.
- Update `factoriax/scenarios/__init__.py` whenever a new public
  symbol is added.

### Ask first

- Renaming or changing the eight recipe outputs the user wants in the
  template.
- Touching JIT'd engine code (e.g. `crafting.craft_recipe` or
  `machines.run_combiners`) or `factoriax/recipes.py`. This spec is
  fully scenario-local; any engine or recipe-table edit belongs in
  the multipliers spec, not here.
- Adding new `ItemType` or `MachineType` enum values.
- Changing the locked-in achievement list or its equal-weight
  scoring.

### Never do

- Mask hand-craft actions in this scenario. Hand crafting stays
  available; the only forcing function in this spec is the 2000-step
  budget.
- Bake balance numbers into Python globals — they flow through
  `EnvParams.recipe_table` so they are runtime-tunable.

## Scenario design

### Map

- 16x16 square map. Spawn at the centre, `(8, 8)`.
- No pre-placed devices.
- Six 2x2 patches are placed procedurally: iron, copper, tin,
  silicon, coal, limestone. Patch positions are sampled uniformly per
  PRNG key with rejection sampling so:
  - No two patches overlap.
  - No patch overlaps the spawn or the furnace/assembler tiles.
  - Every patch sits at least `forbid_radius=1` tile away from spawn
    (Chebyshev distance), so the agent always has at least one open
    tile to walk into immediately after spawning.
- Per-tile resources: 3000 for ore patches 
- The remainder of the map is dirt, which is walkable.

### Observation

- `local_array(state, params, player_idx, radius=5)`.
- Spatial patch is 11x11 with 10 channels (block, machine, resources,
  three slot type+count pairs, machine direction) — fully inside the
  16x16 map at any spawn position when padded.
- Player scalar block is appended unchanged: position, direction,
  timestep, recipe affordability, facing-machine state, inventory.
- The scenario does *not* override the observation function on the
  env — callers continue to pass `obs="local"` (or however
  `factoriax.make` exposes it) and set `obs_radius=5`. The scenario
  documents the recommended radius and tests assert it.

### Action mask

- Empty. Hand crafting stays available.
  `EasyRocketScenario.blocked_actions = frozenset()`.

### Step budget

- `EnvParams.max_timesteps = 2000`.

### Recipe book

The recipe book is the central piece of the scenario design. It has
three tiers:

- **Tier 1 (raw → plate):** the four smelts (iron, copper, tin,
  silicon → plates). Run on the pre-placed furnace.
- **Tier 2 (intermediates and machines):** frame, wire, circuit,
  motor, sensor as intermediates; conveyor_belt, miner, assembler,
  splitter, crossing as machines. Run on the assembler.
- **Tier 3 (rocket):** hull, engine_unit, rocket. Run on the
  assembler. Note: avionics and rocket_core are dropped; the rocket
  recipe takes hull + engine_unit directly.

The user owns the eight recipe definitions for assembler,
conveyor_belt, splitter, crossing, miner, hull, engine_unit, and
rocket. They will be added by hand with empty input tuples in the
template file. Smelts and intermediates are not in the template —
either they are added by the user manually or the build will fail
validation, which is the intended forcing function for the user to
think through their full chain.

**Recipe book template** that the implementation lands as a starting
point (inputs intentionally empty for the user to fill):

```python
EASY_ROCKET_RECIPES_TEMPLATE: tuple[Recipe, ...] = (
    Recipe(output=int(ItemType.ASSEMBLER),      inputs=(), ticks=0, output_count=1, name="Assembler"),
    Recipe(output=int(ItemType.CONVEYOR_BELT),  inputs=(), ticks=0, output_count=1, name="Conveyor Belt"),
    Recipe(output=int(ItemType.SPLITTER),       inputs=(), ticks=0, output_count=1, name="Splitter"),
    Recipe(output=int(ItemType.CROSSING),       inputs=(), ticks=0, output_count=1, name="Crossing"),
    Recipe(output=int(ItemType.MINER),          inputs=(), ticks=0, output_count=1, name="Miner"),
    Recipe(output=int(ItemType.HULL),           inputs=(), ticks=0, output_count=1, name="Hull"),
    Recipe(output=int(ItemType.ENGINE_UNIT),    inputs=(), ticks=0, output_count=1, name="Engine Unit"),
    Recipe(output=int(ItemType.ROCKET),         inputs=(), ticks=0, output_count=1, name="Rocket"),
)
```

The user fills the inputs by hand. The build step then constructs the
`RecipeBook` and the spec-mandated validation kicks in (unique outputs,
unique input type-set per machine type). If the agent
attempts to hand-craft a Tier 1 plate but the user has left the
template empty, validation fails fast at scenario construction time —
which is the explicit point.

**Machine efficiency knobs — deferred.** This scenario uses the
existing `Recipe` / `RecipeOverride` / `RecipeBalance` machinery
unchanged. There is no hand-vs-machine differentiation in the
balance numbers. The original intent — "machines outproduce hand
crafting by a tunable multiplier" — has been split out into
`docs/specs/2026_machine_efficiency_multipliers.md` because it
requires touching the JIT'd engine (`machines.run_combiners`) and
the spec for that change needs its own review cycle. Once that spec
lands, this scenario will set non-neutral multipliers via a
`RecipeBalance` overlay; until then the 2000-step budget alone is
the forcing function that pushes the agent toward using the
pre-placed machines.

### Achievement contract

Thirteen achievements, all weight 1.0, scored as the unweighted sum
of unlocked bits. The list is **locked** by this spec — do not add,
remove, rename, or reweight without an explicit human decision.

```
weights              = jnp.ones(13, dtype=jnp.float32)
MAX_EASY_ROCKET_SCORE = 13.0
```

The scenario implements the standard
`Scenario.achievement_fn` / `Scenario.score` interface. The
`achievement_fn` returns a `bool` array zero-padded to
`MAX_ACHIEVEMENTS`, matching the rocket scenario's shape contract.
Achievements that depend on belt-network topology are gated on
`docs/specs/2026_entity_connection_graph.md` landing; until then
their condition function returns `False` unconditionally and the
bit cannot unlock. The list and the weighting do not change — a v1
agent simply tops out at 9.0 / 13.0, and the remaining four bits
become reachable once the graph module ships.

The full list, with the condition each achievement checks and a
status flag (**v1** = wired in the first easy-rocket release;
**graph** = wired once the connection-graph spec lands):

1. **Mine 1 Ore** — agent holds ≥1 of any raw ore (iron, copper,
   tin, silicon, or coal). Status: **v1**. Trivial inventory sum.
2. **Mine 1 Ore of each** — agent holds ≥1 of every raw ore type
   present on the map (iron, copper, tin, silicon, coal, limestone).
   Status:
   **v1**. Per-type inventory check.
3. **Mine 10 Ore of each** — agent holds ≥10 of each ore needed to
   craft a miner. Tracing the base recipes, the miner needs
   `IRON_PLATE + WIRE`, which reduces to raw inputs
   `IRON_ORE + COAL + COPPER_ORE + TIN_ORE`, so this checks ≥10 of
   each of those four types. Silicon is excluded because it is not
   on the miner-craft path. Status: **v1**. Per-type inventory
   check.
4. **Craft a miner** — agent holds ≥1 `MINER`. Status: **v1**.
5. **Craft an assembler** — agent holds ≥1 `ASSEMBLER`. Status:
   **v1**.
6. **Craft at least one belt** — agent holds ≥1 `CONVEYOR_BELT`.
   Status: **v1**.
7. **Place miner on ore** — at least one placed `MINER` entity sits
   on a tile whose underlying block is an ore block. Status: **v1**.
   Needs a per-entity world-block lookup
   (`state.world_blocks[ent_x, ent_y]` over active miners). Helper
   required; no graph dependency.
8. **Feed a belt with a miner** — at least one belt entity in the
   miner's output direction whose `buf_count > 0`. Status:
   **graph**. Requires belt-chain identification that the
   connection-graph spec provides.
9. **Have miners on three ore types** — count the distinct ore-block
   types beneath placed `MINER` entities; at least three distinct
   types. Status: **v1**. Per-entity world-block lookup plus a
   distinct-count reduction. Shares the helper from #7.
10. **Feed an assembler with a belt** — at least one `ASSEMBLER`
    entity with a non-empty input slot whose feeding belt chain has
    `buf_count > 0`. Status: **graph**.
11. **Feed an assembler with two belts, producing output** — an
    `ASSEMBLER` with two distinct upstream belt chains each feeding
    a different input slot (both chains carrying items), *and*
    `ent_asm_out_count > 0` on that same assembler. Status:
    **graph**.
12. **Connect two assembler outputs with one other assembler using
    belts** — three assemblers `A`, `B`, `C` such that belt chains
    carry items from `A`'s output port to one of `C`'s input slots
    and from `B`'s output port to `C`'s other input slot. Status:
    **graph**.
13. **Place the rocket on the map** — at least one `ROCKET` machine
    entity placed. Status: **v1**. Trivial:
    `_count_machines(state, MachineType.ROCKET) >= 1`.

Implementation notes for the plan phase:

- v1 achievements (1–7, 9, 13) reuse the `_holds_item` /
  `_count_machines` helpers already living in
  `factoriax/scenarios/rocket.py`. Lift them to a shared module
  (`factoriax/scenarios/_common.py` or similar) rather than
  importing across scenarios, so the easy-rocket module does not
  take a dependency on the rocket module.
- Achievements 7 and 9 need a per-entity world-block lookup. New
  helper, JIT-pure, lives in the same shared module.
- Graph-gated achievements (8, 10, 11, 12) get stub condition
  functions that always return `False` in v1, with a `# TODO`
  comment pointing at the connection-graph spec. The bits remain in
  the unlock mask so adding the wiring later is a code change with
  no contract change.

### Reward and scoring

- Per-step reward: thin wrapper around `achievement_reward` bound to
  the easy-rocket weight vector. Pattern matches `rocket_reward` in
  `factoriax/scenarios/rocket.py`.
- Aggregate scoring: `EasyRocketScenario.score()` returns the mean
  weighted-achievement score across `LevelResult`s. With one level,
  this is just the single level's score, in `[0, MAX_EASY_ROCKET_SCORE]`.

## Success criteria

Each is specific and testable from a unit test, with no env stepping
required.

1. `from factoriax.scenarios import EasyRocketScenario` works and
   the class implements the `Scenario` protocol
   (`isinstance(EasyRocketScenario(), Scenario)` is `True`).
2. `EasyRocketScenario().levels()` returns one `ScenarioLevel` with
   `env_params.max_timesteps == 2000`, `map_width == map_height == 16`,
   and `env_params.recipe_table` derived from the easy-rocket book.
3. Calling the easy-rocket level builder twice with the same PRNG key
   yields equal `Level` objects (per-field array equality); two
   different keys produce different ore positions in at least one of
   the five patches across a small sample of trial keys.
4. `local_array(hand_built_state, easy_rocket_params, 0, radius=5)`
   returns a 1-D `float32` array whose length matches the documented
   `11*11*channels + scalar_block` formula. No `env.reset` / `env.step`
   in the test path.
5. For each of the 13 achievements, calling the easy-rocket condition
   function on a hand-built `EnvState` returns the expected bit. v1
   achievements (1–7, 9, 13) exercise both `True` and `False` cases;
   graph-gated achievements (8, 10, 11, 12) assert their bit is
   currently `False` regardless of state (stub semantics) and carry a
   `# TODO` pointing at the connection-graph spec.
6. `EasyRocketScenario.score()` over a single `LevelResult` whose
   `achievements_unlocked` mask has all 13 bits set returns
   `MAX_EASY_ROCKET_SCORE` (`= 13.0` with equal weights).
7. `uv run ruff check`, `uv run mypy`, and
   `uv run pytest tests/scenarios/test_easy_rocket.py -q` all pass,
   with the test file completing in under a second wall-clock.

## Open questions

These are not blockers for the spec — they get resolved during the
plan/implement phases.

- **Procgen `forbid_radius`.** Currently set to 1 (no patch touches
  spawn). If empirically the agent gets stuck because patches block
  movement off-spawn, bump to 2.
- **Graph-gated achievements wiring.** Tracked in
  `docs/specs/2026_entity_connection_graph.md`. Until that spec
  lands, achievements 8, 10, 11, 12 are stubs that always read
  `False`. Re-open the easy-rocket spec once the graph module is
  available to wire them up; the achievement list and weights stay
  unchanged.
- **Re-introducing the machine efficiency gradient.** Tracked in
  `docs/specs/2026_machine_efficiency_multipliers.md`. Once that
  spec lands, this scenario will apply a `RecipeBalance` overlay so
  pre-placed machines outproduce hand crafting on the
  rocket-critical recipes.
