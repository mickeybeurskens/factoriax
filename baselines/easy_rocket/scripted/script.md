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

- On any easy_rocket seed `s ∈ [0, 100)`, the agent reaches
  `state.achievements_unlocked[rocket_placed] == True` within
  `env_params.max_timesteps = 2000`.
- Each of the seven phases has a binary success predicate that
  evaluates against `EnvState` only. The agent halts fail-fast
  on the first phase whose predicate goes false past its deadline.
- The runner reports, for every run: the phase outcomes
  (success/failure/timeout per phase), the tick the rocket placed
  at, and the final achievement bitmask.
- No phase hand-mines after phase 2 finishes — all subsequent ore
  comes from placed miners.
- The agent works without modifying anything under `factoriax/`
  or `baselines/easy_rocket/train_ppo.py`. New code lives only
  under `baselines/easy_rocket/scripted/`.

### Recipe arithmetic (binding constraints)

The recipe book is fixed and small. The agent's plan is derived
from it.

```
MINER          ← LIMESTONE + COPPER         (6 ticks)
ASSEMBLER      ← COAL + SILICON             (8 ticks)
CONVEYOR_BELT  ← COAL + IRON                (4 ticks)
SPLITTER       ← COPPER + IRON              (4 ticks)
CROSSING       ← TIN + IRON                 (4 ticks)
HULL           ← IRON*2 + LIMESTONE*2       (8 ticks)
ENGINE_UNIT    ← COPPER*2 + LIMESTONE*1     (8 ticks)
ROCKET         ← HULL*200 + ENGINE_UNIT*200 (5 ticks)
```

To place one rocket the agent needs 200 hulls and 200 engines.
That works out to:

- **IRON**: 400 (hulls) + N_belts + N_splitters + N_crossings
- **COPPER**: 400 (engines) + N_miners + N_splitters
- **LIMESTONE**: 400 (hulls) + 200 (engines) + N_miners
- **COAL**: N_assemblers + N_belts
- **SILICON**: N_assemblers
- **TIN**: N_crossings (zero if the belt layout avoids them)

With one assembler producing one item per 8 ticks, 200 hulls in a
single assembler takes 1600 ticks — already over 80 % of the
episode budget. The plan therefore needs **multiple assemblers per
section** running in parallel. With four hull assemblers and four
engine assemblers, the per-section production tail is ~400 ticks.

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
  - Any change to the recipe book or
    `factoriax/scenarios/easy_rocket.py`.
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
- `layout.py` — 200–300 lines (the hardest module).
- `phases.py` — 300–400 lines (seven phases × predicate + driver).
- `skills.py` — 200–300 lines (movement pathfinding, hand-craft sequences).
- `state_reader.py` — 100–150 lines.

Total ≈ 1000–1500 lines. Larger than the train script. The layout
planner is the algorithmic centerpiece; the rest is mechanical.

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

#### Phase 1 — Plan factory layout

**Driver.** Pure-Python at phase start (no env action emitted). Reads
`state.map` to locate the six ore patches by `block_resources > 0`.
For each patch, identifies the ore type, the 2x2 footprint, and the
two tile coordinates available for miners. Assigns roles:

- LIMESTONE patch → 1 manual miner + 1 factory miner.
- COPPER patch → 1 manual miner + 1 factory miner.
- IRON patch → 2 factory miners (high volume).
- COAL patch → 1 factory miner.
- SILICON patch → 1 factory miner.
- TIN patch → 1 factory miner (optional; only if crossings used).

Plans assembler positions in three sections:

- **Engine section** — four assemblers, fed by belts from the
  COPPER and LIMESTONE factory miners. Output buffered for the
  rocket section.
- **Hull section** — four assemblers, fed by belts from the IRON
  and LIMESTONE factory miners. Output buffered for the rocket
  section.
- **Rocket section** — one assembler, fed by belts from the engine
  and hull section outputs. Output: one ROCKET item, placed as a
  machine by the agent.

Plans belt routes from each factory miner to each consuming
assembler, and from each section's output to the rocket section's
inputs. Uses a BFS pathfinder on the map's free tiles.

After this is done the first tick, the driver emits `NOOP` and the
phase succeeds immediately on the next predicate evaluation. No
in-world action is needed for phase 1.

**Success predicate.** `agent.layout is not None and
agent.layout.valid` — set by the planner. `valid` is true when:

- Every ore type required by the recipe chain has at least one
  patch present in the map.
- Every planned miner / assembler / belt position is on a free
  tile.
- A path exists from every factory miner to its consuming
  assembler.
- A free tile exists for the eventual ROCKET placement adjacent
  to the rocket-section assembler's output.

**Deadline.** 1 tick. Layout planning is in-Python.

**Failure modes.** Map missing an ore type (shouldn't happen for
easy_rocket but defensible). No path between a miner and its
intended assembler (rare; report which pair and the tiles
considered).

#### Phase 2 — Hand-mine bootstrap resources

**Driver.** Walk to each critical ore patch (LIMESTONE, COPPER) and
hand-mine the per-tile blocks. Uses the navigate + `Action.MINE`
sub-skills. Bootstrap target inventory:

- LIMESTONE: 5 (for 5 miners)
- COPPER: 5 (for 5 miners)
- COAL: 2 (for 1 first assembler + 2 first belts)
- SILICON: 1 (for the first assembler)
- IRON: 2 (for the first 2 belts)

After the inventory threshold is met, hand-craft the bootstrap:

- `CRAFT_MINER × 5` (consumes 5 LIMESTONE + 5 COPPER).
- `CRAFT_ASSEMBLER × 1` (consumes 1 COAL + 1 SILICON).
- `CRAFT_CONVEYOR_BELT × 2` (consumes 2 COAL + 2 IRON).

**Success predicate.** Player inventory holds 5 MINER + 1
ASSEMBLER + 2 CONVEYOR_BELT items. No remaining LIMESTONE/COPPER
required to craft them (i.e. they were actually crafted, not just
mined as raw ore).

**Deadline.** 250 ticks. Hand-mining is slow; this is a generous
budget but should not need anything near it.

**Failure modes.** Patch unreachable (should never happen at 16×16
spawn-centered layouts). Mining yields fewer items than expected
(possible if patch resources depleted by an earlier place — but
this is the first phase to mine).

#### Phase 3 — Place bootstrap miners

**Driver.** Walk to each of the 5 designated manual-miner positions
(2 LIMESTONE + 2 COPPER + 1 IRON or similar, defined in phase 1's
layout). Place a miner at each, facing a direction that puts the
output side over the patch tile so it actively mines.

**Success predicate.** 5 MachineType.MINER entities exist, each
sitting on a tile whose `block_resources > 0`.

**Deadline.** 100 ticks.

**Failure modes.** Targeted tile not free (a tree or another entity
landed there since planning). The planner re-checks free-tile
status at place time and falls back to a contingency tile if
possible; if no fallback, halt.

#### Phase 4 — Build engine section

**Driver.** Loop until the engine section is fully built:

1. Walk to a manual miner with non-empty `ent_buf`. Pick up.
2. Craft (hand) any needed COPPER+LIMESTONE-based items for the
   section's bootstrap: more belts (CONVEYOR_BELT), splitters if
   the layout uses them, the section's four assemblers.
3. Walk to each engine-section position from layout, place the
   appropriate machine, face it correctly.
4. Walk to each engine-section factory miner position, place
   miners on COPPER and LIMESTONE patches there (these miners feed
   the engine section assemblers via belts).

**Success predicate.** All four engine-section assemblers exist at
their layout-prescribed tiles. All engine-section belts exist on
their prescribed paths. Engine factory miners exist on COPPER and
LIMESTONE patches. At least one engine-section assembler has
non-zero `ent_asm_in_count` for both COPPER and LIMESTONE (the
belt is actually flowing).

**Deadline.** 400 ticks.

**Failure modes.** Manual miner runs dry before phase finishes (the
planner under-budgeted the bootstrap; halt and report which
material the agent is short on). Tile occupation conflict.

#### Phase 5 — Build hull section

**Driver.** Mirror of phase 4 with IRON+LIMESTONE inputs and
hull-section positions. Reuses skills and the layout's hull-section
plan.

**Success predicate.** Hull-section assemblers + belts + factory
miners (IRON and LIMESTONE) placed and connected. At least one
hull-section assembler showing non-zero `ent_asm_in_count` for
both IRON and LIMESTONE.

**Deadline.** 400 ticks.

#### Phase 6 — Build rocket section

**Driver.** Build the final assembler that takes HULL+ENGINE_UNIT
inputs (one per recipe call, batched 200 times). Lay belts from
the engine-section output buffer and the hull-section output
buffer to this assembler. The agent does not need to place
additional miners.

**Success predicate.** Rocket-section assembler exists at its
prescribed tile. Belts from engine and hull sections terminate
adjacent to the rocket-section assembler's input sides. The
assembler's `ent_asm_in_count[HULL]` is non-zero or the upstream
sections have visible hull/engine output ready to flow.

**Deadline.** 200 ticks.

#### Phase 7 — Wait for rocket production

**Driver.** Emit `NOOP` every tick. The agent's job in phase 7 is
strictly to wait — production happens autonomously now. The only
exception: once `ItemType.ROCKET` appears in the player's
inventory (or in a pallet adjacent to the rocket-section
assembler), the agent walks to it, picks it up, and places it as
a `MachineType.ROCKET`.

**Success predicate.** `_count_machines(state, MachineType.ROCKET)
>= 1` (the same predicate easy_rocket's `rocket_placed`
achievement uses).

**Deadline.** Whatever remains of the 2000-tick episode budget
after the earlier phases. In practice this is the bulk of the
time.

**Failure modes.** Production stalls (assembler input dries up).
The driver logs the most-recent ent_asm_in counts so the failure
trace points to which intermediate ran out.

### Risks

- **A — Layout planner complexity.** Pathfinding belts on a tight
  16×16 grid with 6 patches + 9 assemblers + several miners is the
  most likely place to spend a day fighting edge cases. Mitigate
  by starting with the simplest layout topology (sections arranged
  in concentric rings or quadrants) and only adding splitter /
  crossing routing if a path doesn't fit.
- **B — Hand-craft action timing.** Hand-crafting takes ticks
  (per the recipe `ticks` field). The driver must wait between
  `CRAFT_*` calls and verify the item landed in inventory before
  moving on. Easy to write a busy-wait loop that emits NOOP until
  inventory updates.
- **C — Miner output side mechanics.** A miner outputs to its
  facing tile, and only if that tile holds an entity capable of
  receiving (a belt). If the agent places a miner with no belt
  adjacent yet, the miner's `ent_buf` accumulates locally and is
  pickable but isn't fed onward. Planner must place the miner's
  output-side belt *first* (or at least before the buffer
  saturates).
- **D — Assembler input wiring.** Assemblers have two input slots.
  A belt arriving on an input side deposits items into the slot
  matching that side's role. The planner must know which side is
  which (helper exists in `factoriax/belts.py`). Get this wrong
  and the assembler never starts producing.
- **E — Tile budget.** 16×16 = 256 tiles. Subtract spawn-centered
  3×3 blocked area (9), 6 ore patches × 4 tiles (24), and you
  have ~223 free tiles. Nine assemblers + ~12 miners + the rocket
  + ~50 belts = ~72 entities. Comfortable, but tight enough that
  the planner can't be lazy.
- **F — Single-seed overfit.** The agent must work on more than
  seed 0. Easy to accidentally encode "the LIMESTONE patch is at
  (3, 5)" and have the agent fall over on seed 1. Mitigation: run
  on 5+ seeds before considering any phase done.

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

### Phase E — Multi-seed validation

- [ ] E.1 (S) Run on seeds 0–4. Record per-seed outcome (rocket
      placed yes/no, tick of placement, failing phase if any).
- [ ] E.2 (S, conditional) Fix whichever phase fails on whichever
      seed. Repeat E.1.
- [ ] E.3 (S) Once 5/5 succeed, run on seeds 5–19 for a wider
      smoke. Capture the agent's wall-clock and tick-count
      distribution.
- [ ] **Checkpoint E:** report agent's success rate and
      tick-cost distribution. Compare to PPO numbers in
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
