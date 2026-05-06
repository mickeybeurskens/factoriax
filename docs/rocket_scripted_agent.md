# Scripted rocket agent — design document

A hand-authored agent that solves the `RocketBenchmark` (and, ideally, any
factoriax level with reachable ore patches for the five mineable types)
end-to-end: scratch → rocket placed on the map. The agent exists for four
reasons:

1. **Ceiling reference** for PPO baselines. If the scripted agent reliably
   reaches 100 / 120 score, PPO progress becomes easier to read.
2. **Reward-correctness oracle.** Running the scripted agent through the
   benchmark should unlock every achievement in a predictable order. A
   missing or out-of-order unlock points at a condition-function bug.
3. **Cheese detector.** After the scripted agent finishes, its achievement
   set should match its actual factory. If an achievement unlocked
   before its "intended" phase, the condition is probably too lenient
   and should be tightened.
4. **Imitation / offline-RL data source.** Every rollout is a usable
   trajectory once saved in `factoriax.analysis.Trajectory` format.

The agent **must** be map-agnostic: no hardcoded coordinates, no "walk
to (7, 7)" style logic. Map information comes from the global
observation vector, not from direct state inspection.

---

## 1. Interface and constraints

### 1.1 Environment

- Uses `FactoriaXEnv(achievement_fn=rocket_conditions)` so the engine
  latches `state.achievements_unlocked` each tick and we can check
  which achievements have fired without recomputing conditions.
- One player. `EnvParams` with `num_players=1`, map size ≥ ~16×16,
  `max_timesteps` ≥ ~3000 (scripted agent doesn't need to be efficient,
  it just needs to finish).

### 1.2 Input

The agent receives a `global_array` observation each tick. Layout:

```
[ block_type    * H*W ]   # tile block ids, normalized by max(BlockType) = 9
[ machine_type  * H*W ]   # tile machine ids, normalized by max(MachineType) = 7
[ block_resources * H*W ] # per-tile ore remaining, normalized by BLOCK_MAX_RESOURCES
[ buffer_type   * H*W ]   # per-tile buffer item id, normalized by NUM_ITEM_TYPES
[ player scalars: pos_x, pos_y, direction, timestep, recipe affordability,
                  facing-machine info, full inventory, research ]
```

All four spatial channels can be reshaped back to `(H, W)` and
de-normalized to ints; that's the scripted agent's world view.

### 1.3 Output

A single discrete action per tick from the `Action` enum (88 actions
total: movement, face-direction, mine, place, craft, withdraw,
deposit, research). One action per env step.

### 1.4 What the agent may NOT use

- Raw `EnvState` fields. The observation vector is the sole input.
- Hard-coded tile coordinates. All positions are discovered at runtime.

### 1.5 What the agent MAY use

- The achievement mask from the wrapper — this is meta-information
  about its own progress and doesn't reveal map structure.
- Static constants: `ItemType`, `MachineType`, `RECIPES`,
  `ROCKET_ACHIEVEMENT_INFO`, etc. These are game rules, not map info.

---

## 2. Architecture

Four layers, top-down:

```
┌───────────────────────────────────────────────────────┐
│ PlanController      — owns the phase FSM              │
│                       chooses the next Goal           │
├───────────────────────────────────────────────────────┤
│ Goal library        — MineOre, CraftItem, PlaceEntity │
│                       ChargeFurnace, ProduceInAsm,    │
│                       AssembleRocket, PlaceRocket     │
├───────────────────────────────────────────────────────┤
│ Skill library       — low-level sequences:            │
│                       NavigateTo(tile), FaceAndMine,  │
│                       FaceAndPlace, WithdrawFrom,     │
│                       DepositInto, CraftFromInventory │
├───────────────────────────────────────────────────────┤
│ WorldModel          — parsed global-obs view of the   │
│                       world; pathfinding; nearest-*   │
└───────────────────────────────────────────────────────┘
```

Each tick the controller:

1. Asks `WorldModel.update(obs)` to re-parse the observation.
2. If the current `Goal` reports done/failed, pops to the planner for
   the next one.
3. Otherwise, asks the active `Skill` for the next action.
4. Emits that action.

Keeping a tiny interpreter instead of one giant switch statement keeps
the agent debuggable (each skill can be unit-tested) and replaceable
(a single goal can be swapped for a learned policy without touching
the rest).

---

## 3. WorldModel

The scripted agent's `(H, W)` model of the world, recomputed each tick
from the observation.

### 3.1 Decoding the obs

```
obs   shape (4*H*W + S,)  float32 in [0, 1]
spatial = obs[:4*H*W].reshape(4, H, W)
block_type     = round(spatial[0] * MAP_NORM).astype(int)
machine_type   = round(spatial[1] * MACHINE_NORM).astype(int)
block_resources= round(spatial[2] * BLOCK_MAX_RESOURCES).astype(int)
buffer_type    = round(spatial[3] * NUM_ITEM_TYPES).astype(int)
scalars = obs[4*H*W:]
# unpack scalars in the same order as _PLAYER_SCALAR_FIELDS
```

Round rather than floor — normalization is divide-float; quantisation
error would push `MachineType.FURNACE=7` to 6 on some floats.

### 3.2 Derived views

- `walkable: (H, W) bool` — `block_type` is DIRT/WATER‑free and
  `machine_type == NONE`. True where the player can stand.
- `ore_patches: dict[int, list[tuple[int, int]]]` — for each
  `BlockType.{IRON, COPPER, TIN, COAL, SILICON}`, the list of tile
  coordinates. Convert block type to item type via `BLOCK_TO_ITEM`.
- `machines: dict[int, list[tuple[int, int]]]` — placed machines by
  `MachineType`.
- `buffered_machines: list[(x, y, item_type)]` — any tile whose
  `buffer_type > 0`, ready for withdrawal.

### 3.3 Nearest-neighbour queries

```
def nearest(target_tiles, from=(px, py)) -> (x, y) | None
```

Breadth-first search on the `walkable` mask from the player's
position; returns the first target tile whose adjacency is reachable.
Manhattan-distance tie-breaks are fine.

### 3.4 Pathfinding

```
def plan_path(goal_tile, allow_target_occupied=True) -> list[Action]
```

A* on walkable tiles. Heuristic = Manhattan distance. If
`allow_target_occupied=True` the goal tile itself may be non-walkable
(that's normal — the player stands *next to* the ore patch and mines
into it, rather than onto it).

Actions emitted: `MOVE_UP / DOWN / LEFT / RIGHT`. Direction changes
are implicit in the move actions themselves; no separate `FACE_*`
needed for movement.

### 3.5 Interaction helpers

The game distinguishes "facing direction" and "position". Most
interact actions target the tile *in front of* the player. So
`FaceAndMine(target_tile)` is:

1. `plan_path(adjacent_tile(target_tile))` and walk there.
2. Emit the `FACE_*` action that orients the player toward
   `target_tile`.
3. Emit `MINE`.

Same template for `FaceAndPlace`, `WithdrawFrom`, `DepositInto`.

---

## 4. Goal library

Each goal is a small state machine. They all expose `step(obs) -> Action | DONE | FAIL`.

### 4.1 `MineOre(item_type, count)`
Find the nearest tile of the matching block type, navigate adjacent,
mine until player inventory has ≥ `count` of the item. If the patch is
depleted mid-way (`block_resources → 0`), switch to the next nearest
patch.

### 4.2 `CraftItem(item_type, count)`
Check player inventory for recipe inputs (via `RECIPES` lookup); emit
the corresponding `CRAFT_*` action until the output count is ≥ target.
Fails (retriable) if ingredients are missing — controller will
schedule more mining.

### 4.3 `PlaceEntity(machine_type, location_predicate)`
Navigate to a tile matching `location_predicate`, face it, emit
`PLACE_<machine>`. Common predicates:

- `on_ore(ore_type)` — for miner placement
- `adjacent_to(machine_type)` — for logistics chains
- `free_walkable_tile_near(player)` — for "just put it somewhere"

The predicate also decides acceptable distance — e.g. `on_ore`
requires the tile itself to hold the ore, while `free_walkable` just
wants any reachable empty tile.

### 4.4 `WithdrawFrom(machine_type, item_type)`
Find any tile holding a machine of type `machine_type` with
`buffer_type == item_type`, navigate adjacent, emit `WITHDRAW_<item>`.

### 4.5 `DepositInto(machine_type, item_type)`
Mirror of withdraw. Needed for pallet filling and assembler feeding.

### 4.6 `ProduceInAssembler(recipe_output)`
1. `PlaceEntity(ASSEMBLER, free_walkable_tile_near(player))`.
2. Inspect `RECIPES` to enumerate input items for `recipe_output`.
3. `CraftItem(input)` for any missing inputs (bootstrap by hand).
4. `DepositInto(ASSEMBLER, input)` for each input.
5. Wait (emit `NOOP`) for `RECIPE_TICKS[recipe]` ticks.
6. `WithdrawFrom(ASSEMBLER, recipe_output)`.

### 4.7 `AssembleRocket`
Composite goal that chains:

```
MineOre(IRON, 40) → MineOre(COPPER, 40) → MineOre(TIN, 20) →
MineOre(COAL, 20) → MineOre(SILICON, 10) →
(handcraft 4 plates each + wafer) →
(handcraft 2× wire, 2× circuit, 2× frame) →
ProduceInAssembler(MOTOR) × 2 →
ProduceInAssembler(SENSOR) × 2 →
ProduceInAssembler(ROCKET) →
PlaceEntity(ROCKET, free_walkable_tile_near(player))
```

Counts are upper bounds; actual minimums come from `RECIPES`. The
whole composite is ~20 goals — a clear chain but not magic.

---

## 5. Phase FSM (the planner)

Phases, each delegating to goals above. Each phase has an explicit
**achievement delta** it expects to unlock; after the phase, the
controller verifies against the live wrapper mask and raises if any
expected achievement is still False.

### Phase A — Reconnaissance (1 tick)
- WorldModel decode.
- Sanity: at least one patch of each of the 5 ore types exists. If
  not, log and continue best-effort.

### Phase B — Bootstrap gather (≈ 30–200 ticks)
Goals:
- `MineOre(IRON, 1)` → unlocks `collect_iron`.
- Repeat for COPPER, TIN, COAL, SILICON → unlocks all 5 Basic ore achievements.

Expected new unlocks: `collect_iron`, `collect_copper`, `collect_tin`,
`collect_coal`, `collect_silicon`.

### Phase C — Handcraft basics (≈ 10 ticks)
- Gather extra ore (~3 of each).
- Handcraft 1 iron plate, 1 copper plate, 1 tin plate, 1 wafer,
  1 wire.

Expected: `smelt_iron`, `smelt_copper`, `smelt_tin`, `smelt_wafer`,
`craft_wire`.

> Depends on whether the engine lets the player smelt plates by hand
> or only via a furnace. The `_make_plate_by_hand?` check is decided
> at runtime by attempting one craft; if it fails, fall back to
> building a furnace first (Phase D, reordered).

### Phase D — Handcraft intermediates (≈ 10 ticks)
Handcraft one each of: circuit, frame, motor, sensor.

Expected: `craft_circuit`, `craft_frame`, `craft_motor`, `craft_sensor`.

### Phase E — First machines (≈ 40 ticks)
1. Handcraft a miner, a furnace, a belt, a pallet, an arm.
2. Place the miner on a reachable ore tile (any type).
3. Place the furnace anywhere walkable nearby.
4. Place the belt next to the miner.

Expected: `craft_miner`, `place_miner`, `craft_furnace`,
`place_furnace`, `craft_belt`, `place_belt`, `automated_mining` (fires
as soon as the miner's buffer fills on its first tick).

### Phase F — Logistics (≈ 60 ticks)
1. Place the pallet adjacent to the miner's arm side.
2. Place the arm between miner and pallet.
3. Wait a few ticks for the arm to move ore into the pallet
   (`pallet_filled` fires).
4. Handcraft + place an assembler.
5. Load an assembler recipe (e.g. WIRE) with handcrafted inputs, wait
   for output (`first_assembly`).
6. Place 4 more belts to reach `belt_network` (total 5).

Expected: `craft_pallet`, `place_pallet`, `pallet_filled`,
`craft_arm`, `place_arm`, `craft_assembler`, `place_assembler`,
`first_assembly`, `belt_network`.

### Phase G — Scale (≈ 30 ticks)
- Place 2 more miners → `scaling_up` (total 3 miners).
- Place belts/pallets/arms until total-machine count ≥ 10 →
  `industrialist`.

### Phase H — Rocket (≈ 200 ticks)
1. Stockpile 2 motors, 2 sensors (use the assembler placed in Phase F
   or build more).
2. Feed the assembler the rocket recipe.
3. Withdraw the rocket.
4. `PlaceEntity(ROCKET, free_walkable_tile_near(player))`.

Expected: `craft_rocket`, `place_rocket`.

### Phase I — Verification (1 tick)
- All 34 achievements unlocked? Log the mask and timestep-of-first-unlock.
- If any achievement is False, report which, which phase should have
  set it, and exit with a non-zero status so this is loud in CI.

Total budget: ≈ 400–700 ticks in the optimistic case. The 2000-step
benchmark ceiling leaves plenty of slack for A* detours and wait
loops.

---

## 6. Expected achievement ordering

The agent should unlock achievements in roughly this order — useful as
a regression oracle:

| Phase | New unlocks (in order) |
|------|------------------------|
| B    | collect_iron, collect_copper, collect_tin, collect_coal, collect_silicon |
| C    | smelt_iron, smelt_copper, smelt_tin, smelt_wafer, craft_wire |
| D    | craft_circuit, craft_frame, craft_motor, craft_sensor |
| E    | craft_miner, place_miner, craft_furnace, place_furnace, automated_mining, craft_belt, place_belt |
| F    | craft_pallet, place_pallet, pallet_filled, craft_arm, place_arm, craft_assembler, place_assembler, first_assembly, belt_network |
| G    | scaling_up, industrialist |
| H    | craft_rocket, place_rocket |

If `automated_mining` unlocks during Phase B (before any miner is
placed), that's a condition bug. If `pallet_filled` unlocks before an
arm is placed, the condition is too lenient (or a game mechanic
we've not modelled). Either is a finding worth fixing.

---

## 7. Files and module layout

```
baselines/
  scripted/
    __init__.py
    world_model.py        # obs decode, nearest, A*
    skills.py             # NavigateTo, FaceAndMine, ...
    goals.py              # MineOre, CraftItem, PlaceEntity, ...
    planner.py            # phase FSM
    agent.py              # ScriptedAgent.act(obs) -> Action
    run_agent.py          # CLI: run N eval episodes, save trajectories

tests/baselines/scripted/
  test_world_model.py     # obs → walkable / ore_patches round-trip
  test_skills.py          # hand-crafted states + single-skill rollouts
  test_agent_rocket.py    # @slow: run full agent, assert all 34 unlocked
```

---

## 8. Data collection for training

### 8.1 Per-episode
For each rollout the runner writes:

- `trajectory.msgpack` — `states_to_trajectory(states, actions, rewards)`
  + achievements mask, shape `(1, T, 34)`.
- `rollout.mp4` — rendered video (same helper as `baselines/rocket`).
- `summary.json` — achievements unlocked, unlock timestep each.

### 8.2 Batching
`run_agent.py --episodes 100 --seeds random` writes 100 trajectory
files. Multiple can be merged into a single Trajectory (concat along
batch dim) for training-style analysis via `factoriax.analysis`.

### 8.3 Use cases
- **BC (behavior cloning).** Train a policy to predict the scripted
  agent's action from the observation. Strong warm-start for PPO.
- **Advantage-weighted regression.** Combine scripted rollouts with
  PPO rollouts; positive-advantage scripted transitions bias the
  policy toward factory-building.
- **Curriculum / shaping reward.** Use the scripted unlock ordering
  as expert knowledge: reward the learner for matching the ordering,
  not just the unlock count.

---

## 9. Robustness edge-cases

1. **Partial recipes in inventory** (leftover from a previous craft) —
   `CraftItem` must treat existing inventory as credit, not re-mine.
2. **Blocked paths** after placements — rebuild walkable mask each
   tick, retry navigation if blocked.
3. **Miner on depleting patch** — `automated_mining` stops when
   `block_resources[miner_tile] == 0`. Replace the miner if other
   goals still need ore.
4. **Assembler stuck** (missing input) — check
   `ent_asm_in_count` via `buffer_type`; deposit more inputs.
5. **Action space gotchas** — the game has direction-specific
   `DEPOSIT_<ITEM>` and `WITHDRAW_<ITEM>` actions, 22 pairs each. Maps
   live in `factoriax.constants.Action`; the skill layer picks the
   right action from the item type.

---

## 10. Integration test plan

The scripted agent earns its keep as a **reward-correctness oracle and
cheese detector** only if an automated test verifies that every
achievement fires in the expected phase. The test layout mirrors the
phase FSM so a single failure points directly at the offending phase.

### 10.1 Test staircase

| Test | Achievements covered | Budget | Mark |
|------|----------------------|--------|------|
| `test_world_model` | 0 (decoder correctness) | <1s | fast |
| `test_skills` | collect_iron (via `MineOre`) | <30s | slow |
| `test_craft_chain_to_wire` | 5 collect_* + 4 smelt_* + craft_wire (10) | <60s | slow |
| `test_intermediate_crafts` | craft_circuit/frame/motor/sensor (4) | <60s | slow |
| `test_miner_on_ore` | craft_miner + place_miner + automated_mining (3) | <60s | slow |
| `test_pallet_and_deposit` | craft_pallet + place_pallet + pallet_filled (3) | <60s | slow |
| `test_assembler_produces` | craft_assembler + place_assembler + first_assembly (3) | <90s | slow |
| `test_scale_to_industrialist` | scaling_up + industrialist + belt_network + craft_belt + place_belt (5) | <90s | slow |
| **`test_full_rocket_agent`** | **all 34 — end-to-end** | <180s | slow |

Each intermediate test constructs a minimal planner with just the
goals needed for its scope, runs the agent in the real env until done
(or timeout), and asserts the exact achievement mask.

### 10.2 Failure signatures

When `test_full_rocket_agent` fails, the intermediate tests disambiguate:

- **`test_craft_chain_to_wire` fails** → the handcraft path doesn't work
  as assumed (plates may require a furnace, not hand-crafting).
  Fix: reorder phase to build a furnace first.
- **`test_miner_on_ore` fails** → `PLACE_MINER` on an ore tile isn't
  accepted, or `automated_mining` doesn't fire after placement. Fix:
  adjust placement predicate or wait-time; flag as condition bug.
- **`test_pallet_and_deposit` fails** → `pallet_filled` requires an arm
  moving items (not direct deposit). Fix: add a miner→arm→pallet
  pipeline to the phase.
- **`test_assembler_produces` fails** → input matching is stricter than
  expected or the recipe-selection mechanic needs a separate action.
- **`test_scale_to_industrialist` fails** → placement predicates
  collide; players can't place three machines in a tiny radius. Fix:
  the predicate should walk further out or loop over free tiles.

Every one of these is a potential finding about the achievement
conditions or the game mechanics — which is exactly the point.

### 10.3 Reliability sweep (out of scope for v1)

Once the single-seed test is green, extend to a 10-seed sweep under
`pytest -n auto` with a target ≥95% full-34 success rate. Flaky
achievements show up as "sometimes fires, sometimes doesn't" — a
strong hint about race conditions in the engine (e.g. `pallet_filled`
that depends on arm-tick ordering).

### 10.4 Budget model

Full rocket rollout cost (back-of-envelope on the 32×32 rocket
benchmark, ~10 tiles between player spawn and nearest ore patch):

| Phase | Cost per op | Ops | Subtotal |
|-------|-------------|-----|----------|
| Gather 5 ore types (~120 iron, ~80 copper, ~35 tin, ~120 coal, ~20 silicon) | 1 step per mine, ~15 steps walking per patch | 375 mines + 5 walks | ~450 |
| Hand-craft all intermediates + machines | 1 step per craft | ~70 | ~70 |
| Place 13+ machines (3 miner, 1 furnace, 5 belt, 1 pallet, 1 arm, 1 assembler, 1 rocket) | ~10-20 steps per place (nav + face + place) | 13+ | ~260 |
| Deposit-to-pallet + deposit-to-assembler + waits | ~30 steps total | — | ~30 |
| Buffer for BFS detours and reselection | — | — | ~200 |
| **Total** | | | **~1000** |

Fits the `max_timesteps=2000` rocket benchmark envelope with ~1000
steps of slack for navigation blocking, patch depletion, etc.

## 11. Milestones / sequencing

1. `world_model.py` + unit tests — decode a known state, verify maps
   match.
2. Skills + goal library — unit tests per skill in isolation.
3. Planner — integration on a tiny 8×8 map to `collect_coal + collect_iron`
   only.
4. Full rocket run on 32×32 `RocketBenchmark` — first-light test that
   all 34 achievements unlock deterministically.
5. Multi-seed reliability sweep (100 seeds, log unlock rates). Target:
   ≥ 95% full-34 success on the stock rocket benchmark.
6. CI integration as a `@pytest.mark.slow` test — guards achievement
   conditions against regressions.

---

## 12. Non-goals

- **Efficiency.** A scripted agent that finishes in 1500 ticks is
  fine. It's not a baseline for wall-clock speed.
- **Learning.** No gradient descent anywhere. Every decision is
  deterministic given the obs + internal FSM state.
- **Multi-agent coordination.** Rocket benchmark is 1-player. If this
  ever extends to multi-agent, the planner would need a new layer;
  the WorldModel and skills transfer as-is.
- **Exploration beyond the 5-ore-plus-rocket progression.** No
  research, no advanced science, no combat (there is no combat in
  this benchmark).
