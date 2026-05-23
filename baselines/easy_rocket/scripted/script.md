# Easy Rocket Scripted Agent — Spec, Plan, Todo

Single document covering all three. Implementation lives next to it
under `baselines/easy_rocket/scripted/`. The existing `runner.py`
plumbing is the entry point; this agent slots into it as a stateful
`ScriptedPolicy`.

---

## Spec

### Objective

Hand-author a scripted agent that solves the easy rocket scenario
end-to-end on any procgen seed of the level builder. "Solves" means
the agent ultimately places a `ROCKET` machine, which unlocks the
`rocket_placed` achievement and is the canonical ceiling check for
the bench. The agent must not assume specific ore patch positions —
it reads the map at startup and plans accordingly.

This agent exists to anchor the PPO baseline's numbers. The current
PPO best is ~4.5/9 mean achievements with high seed variance; a
scripted policy that reliably hits 9/9 (or whatever the true reachable
maximum turns out to be) establishes the gap between learned and
hand-engineered behavior on this bench.

The seven phases the user prescribed are load-bearing: each one is
both a chunk of work and a verification boundary. They are not
suggestions to be reorganized.

### Success criteria

- **Ore-layout robustness.** On any easy_rocket seed
  `s ∈ [0, 100)`, the agent reaches
  `state.achievements_unlocked[rocket_placed] == True` within
  `env_params.max_timesteps = 2000`. Layout-affecting reads come
  only from `state.map`; no `(x, y)` coordinate is ever hardcoded.
- **Recipe-book robustness.** The agent solves the rocket goal
  for any recipe book whose DAG (a) ultimately produces
  `ItemType.ROCKET`, (b) leaves resolve to ore types present on
  the map, and (c) keeps `MINER`, `ASSEMBLER`, and
  `CONVEYOR_BELT` hand-craftable from base ores. The layout
  planner re-derives the assembler set and per-ore miner count
  from the runtime `recipe_table`; the phase drivers index into
  these derived structures, not into hardcoded item names. A
  smoke test exercises this by passing a tweaked recipe book
  (e.g. tick counts doubled, an extra intermediate inserted)
  and confirming the agent still solves.
- **Phase predicates evaluate against `EnvState` only.** The
  agent halts fail-fast on the first phase whose predicate goes
  false past its deadline.
- **Runner reporting.** For every run: per-phase outcome
  (success/failure/timeout), the tick the rocket placed at, the
  final achievement bitmask.
- **No hand-mining after phase 2.** Subsequent ore comes from
  placed miners.
- **Implementation scope.** Code lives only under
  `baselines/easy_rocket/scripted/`. No modifications to
  `factoriax/` or `baselines/easy_rocket/train_ppo.py`.

### Recipe-derived layout (two DAG passes, no throughput math)

The agent's plan is **derived from the recipe book at runtime**
by two passes over the recipe DAG. There is no tick-budget
calculation; phases simply wait for inventory to accumulate
before proceeding.

**Pass 1 — assemblers in the automated chain.**
Walk the recipe graph backwards from the target. Every recipe
encountered becomes one assembler in the automated factory. For
easy_rocket targeting `ROCKET`:

- `ROCKET` recipe → 1 assembler (consumes HULL + ENGINE_UNIT).
- `HULL` recipe → 1 assembler (consumes IRON + LIMESTONE).
- `ENGINE_UNIT` recipe → 1 assembler (consumes COPPER + LIMESTONE).
- Three assemblers total. The other recipes
  (MINER, ASSEMBLER, CONVEYOR_BELT, SPLITTER, CROSSING) are
  hand-crafted by the agent and need no assembler station.

**Pass 2 — miner allocation.**
For each ore type touched anywhere in the recipe DAG:

```
manual_miners(ore)  = 1                                          # always
factory_miners(ore) = number of automated assemblers consuming ore
total(ore)          = manual + factory
```

The manual miner is what the agent personally empties to
hand-craft every machine (more miners, assemblers, belts,
splitters, crossings). The factory miners feed the automated
chain via belts. For easy_rocket:

| ore       | manual | factory consumers      | factory | total |
|-----------|-------:|------------------------|--------:|------:|
| IRON      |      1 | HULL                   |       1 |     2 |
| COPPER    |      1 | ENGINE_UNIT            |       1 |     2 |
| LIMESTONE |      1 | HULL + ENGINE_UNIT     |       2 |     3 |
| COAL      |      1 | —                      |       0 |     1 |
| SILICON   |      1 | —                      |       0 |     1 |

Sum: **9 miners, 0 splitters.** TIN gets a miner only if the
belt layout uses CROSSING entities.

**Patch capacity.** Each 2x2 ore patch is 4 tiles, so it fits up
to 4 miners. The LIMESTONE case (3 miners) sits well within
capacity — no splitter sharing needed.

**No throughput math.**
The planner stops here. No estimate of "how long the rocket
takes," no parallelism sizing, no tick-budget allocation. Each
phase below waits for `EnvState`-readable inventory or
ent-buffer thresholds before proceeding, so the agent
automatically adapts to whatever the real production rate ends
up being.

### Miner role split

The level builder places six 2x2 ore patches, one per ore type. Each
patch fits two miners on its four tiles. The agent uses two distinct
miner roles:

- **Manual miners** — placed on critical ore patches in phase 3
  (LIMESTONE, COPPER at minimum) to bootstrap the factory. The
  agent walks to each one and picks up its buffered output by
  hand. These keep the agent off raw-ore hand-mining once phase 2
  finishes.
- **Factory miners** — placed in phases 4–6 on every ore patch
  needed by that section. Their output side faces an adjacent belt
  that carries the ore into the section's assembler inputs. The
  agent never personally interacts with factory miners after they
  are placed and belted.

Patches large enough for two miners means at most 12 miners total
(6 ores × 2 miners). In practice 8–10 is enough: dual on
LIMESTONE and IRON (the high-volume ores), single on the rest.

### Tech stack

- Python 3.11+, JAX (for env step only; the agent itself is plain
  Python).
- Reuse `baselines.easy_rocket.scripted.runner.run_policy` as the
  driver. Extend the `ScriptedPolicy` signature to also allow a
  stateful callable object (the `__call__(state, params) -> action`
  pattern), since the agent must remember which phase it's in.
- State reader, not observation reader. The agent reads
  `EnvState.map`, `EnvState.ent_*`, `EnvState.player_*` directly.
- No tests are written until the agent works end-to-end. Tests
  are added afterward, scoped tight (one test asserts the agent
  hits rocket_placed on seed 0 within budget; per-phase predicate
  tests added only if a phase becomes flaky).

### Commands

```
Run scripted agent:  uv run python -m baselines.easy_rocket.scripted.runner \
                       --policy scripted --seed 0
Render rollout mp4:  same command (the runner saves video by default).
Multi-seed eval:     for s in 0 1 2 3 4; do \
                       uv run python -m baselines.easy_rocket.scripted.runner \
                         --policy scripted --seed $s; done
Lint:                uv run ruff check baselines/easy_rocket/scripted --fix
Format:              uv run ruff format baselines/easy_rocket/scripted
Typecheck:           uv run mypy baselines/easy_rocket/scripted
```

### Boundaries

- **Always:**
  - Halt fail-fast on the first phase whose predicate fails past
    its deadline. The runner reports which phase, the tick, and
    the current state hash.
  - Plan every miner / assembler / belt position from runtime
    observation of `state.map`. Never hardcode (x, y) coordinates
    that would only work on seed 0.
  - Verify each phase's success predicate against `EnvState` —
    not against the agent's internal model of what it tried to do.
    The world is the truth.
  - Run `pytest`, `ruff`, `mypy` before each commit. Tests are
    optional per the above; the lint/typecheck gate isn't.
- **Ask first:**
  - Any change to `factoriax/scenarios/easy_rocket.py` or
    anything else under `factoriax/`. Note this does **not**
    include the recipe book at runtime — the agent must already
    handle arbitrary recipe DAGs that produce `ROCKET`. This
    boundary is about the engine source tree, not the bench's
    recipe configuration.
  - Adding action masks to the runner. Easy_rocket has
    `blocked_actions = frozenset()` by design.
  - Any deviation from the seven-phase production order.
- **Never:**
  - Hand-mine after phase 2 finishes.
  - Silently advance past a failed phase predicate.
  - Reach into other agents' code (`baselines/rocket/scripted/`,
    `baselines/skills/scripted/`) to share private helpers. Copy
    paste if needed.

---

## Plan

### Architecture

The agent is a stateful callable: an object with `__call__(state,
params) -> action` plus internal fields for the current phase, the
sub-step within the phase, and the layout produced in phase 1.

```
ScriptedAgent
├── layout: FactoryLayout      # produced by phase 1, immutable after
├── phase_idx: int             # 0..6 across the seven phases
├── phase_substep: int         # per-phase progress counter
├── phase_deadline: int        # tick by which current phase must finish
└── __call__(state, params) -> action
        ├── if phase predicate met → advance phase, recurse
        ├── if phase deadline exceeded → halt(reason)
        └── else → emit next action for current phase
```

Each phase has three pieces:

- **driver(state, layout) -> action** — what to do this tick.
- **success(state, layout) -> bool** — true when phase done.
- **deadline** — tick budget; phase halts if exceeded.

The driver is small per phase because the per-tick state machine
inside it is itself broken into sub-skills (navigate, pickup, place,
craft). Skills live in `skills.py`.

### Module layout

```
baselines/easy_rocket/scripted/
  __init__.py        # one-line marker (already exists)
  runner.py          # already exists; gains a "scripted" policy choice
  script.md          # this document
  agent.py           # ScriptedAgent class, phase dispatch
  layout.py          # Phase 1: read state.map, plan all positions
  phases.py          # Per-phase driver + success predicate functions
  skills.py          # Navigate, craft, place, pickup — atomic action emitters
  state_reader.py    # Helpers that read EnvState (find_patches, inv_count, ...)
```

Sizes (estimated):

- `agent.py` — 100–150 lines.
- `layout.py` — 150–250 lines (DAG walk + BFS routing; smaller now
  that there is no throughput math).
- `phases.py` — 300–400 lines (seven phases × predicate + driver).
- `skills.py` — 200–300 lines (movement pathfinding, hand-craft
  sequences).
- `state_reader.py` — 100–150 lines.

Total ≈ 850–1250 lines. Larger than the train script but smaller
than the rocket scripted agent. The wait-and-check phase pattern
keeps each phase short.

### Dependency graph

```
  ┌──────────────────────────────────────────┐
  │ factoriax: EnvState, MachineType,        │
  │           Action, ItemType, RecipeTable  │
  └─────────────────┬────────────────────────┘
                    │
                    ▼
       ┌───────────────────────────┐
       │ state_reader.py           │ ◄── pure functions; no deps inward
       │   find_patches, inv,      │
       │   player_pos, machine_at  │
       └─────────────┬─────────────┘
                     │
                     ▼
           ┌─────────────────┐
           │ skills.py       │ ◄── emits actions; consumes state_reader
           │   navigate_to,  │
           │   craft, place, │
           │   pickup        │
           └────────┬────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │ layout.py            │ ◄── phase 1 only; consumes state_reader
        │   plan_factory()     │
        └────────┬─────────────┘
                 │
                 ▼
       ┌──────────────────────┐
       │ phases.py            │ ◄── consumes layout + skills
       │   PHASES = [Phase1,  │
       │             ...,     │
       │             Phase7]  │
       └──────────┬───────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │ agent.py            │ ◄── consumes phases
        │   ScriptedAgent     │
        └──────────┬──────────┘
                   │
                   ▼
          ┌────────────────┐
          │ runner.py      │ ◄── registers "scripted" policy choice
          └────────────────┘
```

Build bottom-up: state_reader and skills first (testable in
isolation), then layout, then phases, then the agent that wires
them, then the runner switch.

### Phase-by-phase contract

Each entry below pins the driver behaviour, the success predicate,
the failure-mode signature, and the tick-budget intuition. Predicates
are written against `EnvState` only.

**Recipe-DAG generalization.** Phases 4–6 are described below
using easy_rocket's specific intermediates (engine, hull, rocket)
for readability. The code generalizes: for an arbitrary recipe
book, the planner produces `layout.sections: list[Section]`, one
per intermediate recipe in the DAG, plus a `final_section` for
the recipe whose output equals the target item. Phases 4..(K-1)
iterate over `layout.sections[0..K-2]` building each intermediate
section in the order the DAG requires (topologically); the
penultimate phase builds `final_section`; phase K-1 (the last)
waits for the target item and places it. For easy_rocket this
reduces to two intermediate phases (engine + hull, in either
order) and one final-section phase, matching the seven-phase
breakdown shown.

#### Phase 1 — Plan factory layout

**Driver.** Pure-Python at phase start (no env action emitted).

1. Read `state.map` to locate every ore patch by
   `block_resources > 0`. Record each patch's ore type and the
   tile coordinates available for miners.
2. Walk the `recipe_table` graph backwards from `ROCKET` to
   compute the assembler set (Pass 1 above) and the miner counts
   per ore type (Pass 2 above).
3. Place each ore patch's manual miner on a fixed corner of the
   patch (e.g. the top-left tile); place factory miners on the
   remaining patch tiles.
4. Choose tiles for the three automated assemblers (HULL, ENGINE,
   ROCKET) using a "near the consuming miners" heuristic plus
   distance from the spawn zone.
5. BFS-route belts from each factory miner to its consuming
   assembler's input side, and from HULL/ENGINE outputs to ROCKET
   inputs. Tiles consumed by one belt are off-limits for the
   next.
6. Reserve a free tile adjacent to the ROCKET assembler's output
   for the agent to eventually place the `ROCKET` machine.

Phase 1 emits one `NOOP` and the predicate flips true on the
next tick.

**Success predicate.** `agent.layout is not None and
agent.layout.valid`. `valid` is true when:

- Every ore type referenced by the recipe DAG has a patch on the
  map.
- Every planned miner / assembler / belt / rocket position is on
  a currently-free tile.
- BFS found a route for every (factory miner → assembler) pair
  and every (section output → rocket input) pair.

**Deadline.** 1 tick.

**Failure modes.** Map missing an ore type. BFS can't route a
belt within the map (in which case the planner relocates the
assembler closer to its source and retries once; halt if still
unroutable).

#### Phase 2 — Hand-mine the manual miners

**Driver.** Hand-mine the **only** thing this phase produces:
the `layout.manual_miners` set. For each manual miner that means
1 LIMESTONE + 1 COPPER (from the `MINER` recipe). The agent
alternates `navigate_to(ore_patch)` and `Action.MINE` until
inventory holds N × LIMESTONE + N × COPPER, where N is the
number of manual miners, then fires `CRAFT_MINER × N`.

Belts, splitters, and assemblers are **not** crafted here. Each
section (Phase 4 / 5 / 6) crafts the belts and the one assembler
it needs, on demand, from material the manual miners have
delivered by then.

**Success predicate.** Player inventory holds N `ItemType.MINER`,
where N = `len(layout.manual_miners)`. The predicate watches
inventory directly — it does not estimate how long mining will
take.

**Deadline.** 1000-tick safety cap so a broken navigation never
burns the whole episode silently. In practice the phase finishes
in well under 200 ticks.

**Failure modes.** Patch geometrically unreachable (planner bug;
caught by Phase 1 predicate). `CRAFT_MINER` emits but inventory
doesn't update — defensive predicate catches it.

#### Phase 3 — Place manual miners

**Driver.** Walk to each `layout.manual_miners[i].position` and
place a `MachineType.MINER` facing the patch's resource tile so
the miner actively extracts. Order doesn't matter; the agent
picks the nearest unplaced site each tick.

**Success predicate.** For every entry in
`layout.manual_miners`, a `MachineType.MINER` entity exists at
the expected `(x, y)`.

**Deadline.** 200-tick safety cap.

**Failure modes.** Targeted tile not free (re-checked at place
time). Halt with the failing position recorded.

#### Phase 4 — Build engine section

**Driver.** Loop:

1. Check inventory against what the engine section still needs
   to be fully built (the 1 ENGINE assembler + its belts + its
   COPPER and LIMESTONE factory miners).
2. If inventory is short of any item, walk to a manual miner with
   non-empty `ent_buf` for the relevant ore, pick up, then
   hand-craft the missing item. Repeat until inventory holds the
   full engine-section bill of materials.
3. Once inventory is complete: walk and place each engine-section
   entity — factory miners on their patches, belts along the
   planned route, the ENGINE assembler. Face each correctly.

The "wait for resources" step uses no time estimate; it just
polls inventory and the manual miners' buffers each tick.

**Success predicate.**
- Engine factory miners exist at layout positions on COPPER and
  LIMESTONE patches.
- Engine-section belt entities exist at every layout-prescribed
  position.
- ENGINE assembler exists at its layout tile.
- At least one engine-section assembler shows non-zero
  `ent_asm_in_count` for both COPPER and LIMESTONE — confirming
  the belt is actually flowing material in, not just placed.

**Deadline.** 600-tick safety cap.

**Failure modes.** Manual miner sits empty too long (mining
saturation lower than expected; agent walks between miners
emptying each but inventory never reaches target). Halt and
report the binding ore. Tile occupation conflict at place time.

#### Phase 5 — Build hull section

**Driver.** Mirror of phase 4 with IRON + LIMESTONE inputs and
hull-section positions. The agent inventories what's needed,
walks the manual miners until inventory has it, then places
the HULL assembler, its factory miners (IRON and LIMESTONE),
and the connecting belts.

**Success predicate.** Hull factory miners exist on IRON and
LIMESTONE patches at layout positions; hull-section belts
placed; HULL assembler placed; the assembler shows non-zero
`ent_asm_in_count` for both IRON and LIMESTONE.

**Deadline.** 600-tick safety cap.

#### Phase 6 — Build rocket section

**Driver.** Same wait-craft-place pattern, but now the inputs
are the HULL and ENGINE_UNIT items being produced upstream — no
miners are placed in this phase. The agent waits for the engine
and hull sections to start producing (visible via their
assemblers' `ent_asm_out_count` > 0), then crafts the belts
that carry HULL and ENGINE_UNIT from the upstream sections into
the ROCKET assembler. Finally places the ROCKET assembler and
the belts.

**Success predicate.** ROCKET assembler exists at its layout
tile. Belts from the hull and engine sections terminate at the
ROCKET assembler's input sides. The assembler shows non-zero
`ent_asm_in_count` for either HULL or ENGINE_UNIT (whichever
the upstream produced first).

**Deadline.** 400-tick safety cap.

#### Phase 7 — Wait for rocket, then place it

**Driver.** Two sub-modes:

1. **Waiting**: emit `NOOP`. Watch the ROCKET assembler's
   `ent_asm_out_count` (and the player inventory, in case a
   pallet routes the rocket back). The factory runs autonomously
   at this point.
2. **Placing**: once an `ItemType.ROCKET` is reachable (in the
   ROCKET assembler's output buffer or in the player's
   inventory), navigate to it, pick it up, and place it as a
   `MachineType.ROCKET` at the layout's reserved rocket tile.

**Success predicate.** `_count_machines(state,
MachineType.ROCKET) >= 1` (matches the `rocket_placed`
achievement's predicate).

**Deadline.** Whatever remains of the 2000-tick episode budget.

**Failure modes.** Production stalls — `ent_asm_in_count` on
some upstream assembler drops to zero and stays there. The
driver logs the most recent counts every 100 ticks so the
failure trace points to which intermediate dried up.

### Risks

- **A — Belt routing on a 16×16 grid.** With ~9 miners +
  3 assemblers + the rocket + the belts connecting them, the
  BFS routing can fail to fit on adversarial seeds. Mitigate by
  ordering BFS routes longest-first and falling back to splitter
  shares when capacity is tight.
- **B — Hand-craft action timing.** Hand-crafting takes ticks
  (per the recipe `ticks` field). The driver must wait between
  `CRAFT_*` calls and verify the item landed in inventory before
  moving on. The wait-and-check phase pattern handles this
  naturally — every phase already polls inventory each tick.
- **C — Miner output side mechanics.** A miner outputs to its
  facing tile, and only if that tile holds an entity capable of
  receiving (a belt). If the agent places a miner with no belt
  adjacent yet, the miner's `ent_buf` accumulates locally
  (still pickable by hand). Factory miners must have their
  output-side belt placed first.
- **D — Assembler input wiring.** Assemblers have two input slots.
  A belt arriving on an input side deposits items into the slot
  matching that side's role. The planner must know which side is
  which (helper in `factoriax/belts.py`). Get this wrong and the
  assembler never starts producing.
- **E — Single-seed overfit.** The agent must work on more than
  seed 0. Easy to accidentally encode "the LIMESTONE patch is at
  (3, 5)" and have the agent fall over on seed 1. Mitigation: run
  on 5+ seeds before considering any phase done.
- **F — Bootstrap saturation.** Phase 4–6 wait on manual miners
  to produce. If the agent walks slowly between miners and the
  manual miners' `ent_buf` saturates (default cap is small),
  throughput tanks. Mitigate by emptying buffers eagerly even
  before the agent needs the items — pickup is cheap.

---

## Todo

Checklist tracker. Mark each as it lands. Stop and review at every
checkpoint.

Sizes: S (~30–90 min), M (~1–3 hours).

### Phase A — Foundations (no agent yet)

- [ ] A.1 (S) Write `state_reader.py` — `find_patches(state)`,
      `inv_count(state, item)`, `player_pos(state)`,
      `machine_at(state, x, y)`, `tile_free(state, x, y)`.
- [ ] A.2 (M) Write `skills.py` — `navigate_to(state, tx, ty) ->
      action | None`, `pickup_from(state, x, y) -> action`,
      `place_machine(state, mtype, x, y, facing) -> action`,
      `craft_item(state, item) -> action | None`.
- [ ] **Checkpoint A:** state_reader + skills lint+typecheck
      clean. Manual smoke: import in REPL, exercise each function
      on a fresh easy_rocket state. Review with human before B.

### Phase B — Layout planner

- [ ] B.1 (M) Write `layout.py` with `FactoryLayout` dataclass +
      `plan_factory(state, recipe_table) -> FactoryLayout`. BFS
      pathfinder for belt routes. Encodes the miner-role split
      from this spec.
- [ ] B.2 (S) Render the planned layout as ASCII art for visual
      inspection (debug helper, not committed).
- [ ] **Checkpoint B:** `plan_factory` produces a `valid` layout
      for seeds 0–4. ASCII renders look reasonable. Review with
      human.

### Phase C — Phases

- [ ] C.1 (M) Write `phases.py` Phase 1 (layout install) +
      Phase 2 (hand-mine bootstrap) + their predicates.
- [ ] C.2 (M) Write Phase 3 (place bootstrap miners) +
      predicate.
- [ ] C.3 (M) Write Phase 4 (engine section build) +
      predicate. Most code lives here — the section-build pattern
      will be copied for 5 and 6.
- [ ] C.4 (S) Write Phase 5 (hull section) — small once C.3 is
      done.
- [ ] C.5 (S) Write Phase 6 (rocket section).
- [ ] C.6 (S) Write Phase 7 (wait + place rocket).

### Phase D — Agent + runner wiring

- [ ] D.1 (S) Write `agent.py` with `ScriptedAgent` class
      orchestrating the seven phases, fail-fast on predicate
      timeout, structured halt reporting.
- [ ] D.2 (S) Extend `runner.py` to accept `--policy {noop,
      scripted}` and dispatch. Log per-phase outcomes at the end.
- [ ] **Checkpoint D:** agent reaches `rocket_placed` on seed 0
      within budget. Lint + typecheck clean.

### Phase E — Multi-seed and multi-recipe validation

- [ ] E.1 (S) Run on seeds 0–4. Record per-seed outcome (rocket
      placed yes/no, tick of placement, failing phase if any).
- [ ] E.2 (S, conditional) Fix whichever phase fails on whichever
      seed. Repeat E.1.
- [ ] E.3 (S) Once 5/5 succeed, run on seeds 5–19 for a wider
      smoke. Capture the agent's wall-clock and tick-count
      distribution.
- [ ] E.4 (S) **Recipe-book robustness.** Construct a tweaked
      recipe book (e.g. all `ticks` doubled; or an extra
      intermediate inserted; or a recipe input swapped to a
      different existing ore) and pass it to the agent via a
      direct `RecipeTable` instantiation in a smoke test. Verify
      the agent re-derives its plan and still hits
      `rocket_placed`. Don't write a generator for arbitrary
      recipe books — just hand-author 2–3 perturbations that
      cover the cases the planner is supposed to flex on.
- [ ] **Checkpoint E:** report agent's success rate across seeds
      and recipe-book perturbations. Compare to PPO numbers in
      `experiments/easy_rocket_ppo_initial.md`.

### Phase F — Documentation & comparison

- [ ] F.1 (S) Write up the scripted agent's results in
      `experiments/easy_rocket_scripted_initial.md` (parallel to
      the PPO writeup). Include side-by-side metrics: PPO best,
      PPO mean ± std, scripted success rate, scripted tick
      cost.
- [ ] F.2 (S) Optional: render an annotated video showing per-phase
      transitions overlaid on the rollout. Park as v2 unless
      trivially cheap.

### Parked

- [ ] Tests beyond the seed-0 success check. Add only if a phase
      becomes flaky.
- [ ] Adaptive layout planner (handles ore patches in
      adversarial positions). Easy_rocket's patch sampler avoids
      the spawn zone, so the easy cases handle themselves; tackle
      only if E.1 turns up real seed-specific failures.
- [ ] Generalising to the full rocket scenario (different recipe
      book, blocked CRAFT actions, larger map). Different spec.

### Notes / Ideas

(Empty; populate during implementation per
[[feedback_notes_in_todo]].)
