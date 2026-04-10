# FactoriaX Architecture Redesign

## Problem

The current architecture wastes ~90% of memory bandwidth on oversized data
structures. Every tile carries a 15-element inventory vector even though belts
and arms hold 1 item. Every field uses int32 even when int8 suffices. The
assembler recipe system requires a configuration action that's hard for RL
agents. With biters removed, machine_health and scent_field are dead weight.

Current throughput: 22,251 steps/s at batch=1024.

## Design Principles

1. **Store what you use.** Belts hold 1 item -> store 1 item, not 15.
2. **Match the dtype to the range.** Direction has 4 values -> int8, not int32.
3. **Auto-discover over configure.** Assemblers infer recipe from inputs.
4. **Compute once, not every tick.** Research gating is static between unlocks.

## Part 1: New Item & Recipe System

### Items (23 enum values, 22 non-empty)

| ID | Name | Category | Mined? |
|----|------|----------|--------|
| 0 | EMPTY | -- | -- |
| 1 | COAL | Fuel | Yes |
| 2 | IRON_ORE | Base | Yes |
| 3 | COPPER_ORE | Base | Yes |
| 4 | TIN_ORE | Base | Yes |
| 5 | SILICON | Base | Yes |
| 6 | IRON_PLATE | Plate | -- |
| 7 | COPPER_PLATE | Plate | -- |
| 8 | TIN_PLATE | Plate | -- |
| 9 | WAFER | Plate | -- |
| 10 | STEEL | T1 | -- |
| 11 | CIRCUIT | T1 | -- |
| 12 | WIRE | T1 | -- |
| 13 | MOTOR | T2 | -- |
| 14 | SENSOR | T2 | -- |
| 15 | BELT | Machine | -- |
| 16 | ARM | Machine | -- |
| 17 | MINER | Machine | -- |
| 18 | ASSEMBLER | Machine | -- |
| 19 | PALLET | Machine | -- |
| 20 | BASIC_SCIENCE | Science | -- |
| 21 | ADV_SCIENCE | Science | -- |
| 22 | ROCKET | Goal | -- |

### Auto-recipe system (17 recipes)

Each recipe has a unique input type-set. All inputs are different types.
Amounts vary for production balancing. Used by both assemblers and player
hand-crafting (same table, same logic).

| # | Inputs | Output | Tier |
|---|--------|--------|------|
| 1 | 2 Iron Ore | Iron Plate | Plate |
| 2 | 2 Copper Ore | Copper Plate | Plate |
| 3 | 2 Tin Ore | Tin Plate | Plate |
| 4 | 2 Silicon | Wafer | Plate |
| 5 | 2 Iron Plate + 1 Tin Plate | Steel | T1 |
| 6 | 1 Copper Plate + 1 Wafer | Circuit | T1 |
| 7 | 1 Iron Plate + 2 Copper Plate | Wire | T1 |
| 8 | 1 Steel + 1 Wire | Motor | T2 |
| 9 | 1 Circuit + 1 Wire | Sensor | T2 |
| 10 | 1 Motor + 2 Iron Plate | Belt | Machine |
| 11 | 1 Motor + 1 Circuit | Arm | Machine |
| 12 | 1 Sensor + 2 Steel | Miner | Machine |
| 13 | 1 Sensor + 2 Circuit | Assembler | Machine |
| 14 | 2 Steel + 1 Tin Plate | Pallet | Machine |
| 15 | 1 Motor + 1 Tin Plate | Basic Science | Science |
| 16 | 1 Sensor + 1 Wafer | Adv. Science | Science |
| 17 | 2 Motor + 2 Sensor | Rocket | Goal |

### Player crafting (same recipes, instant)

Players craft with CRAFT_{output} actions. Same recipe table as assemblers,
but instant (no progress timer). The check is 3 indexed reads + 2 comparisons:

```python
type_a, amt_a = R_TYPE_A[recipe_idx], R_AMT_A[recipe_idx]
type_b, amt_b = R_TYPE_B[recipe_idx], R_AMT_B[recipe_idx]
can_craft = (inv[type_a] >= amt_a) & ((type_b == 0) | (inv[type_b] >= amt_b))
# Consume inputs, produce output (branchless with jnp.where)
```

~10 ops per player per craft action. Negligible cost.

### Action space

| Category | Actions | Count |
|----------|---------|-------|
| NOOP | NOOP | 1 |
| Movement | MOVE_UP/DOWN/LEFT/RIGHT | 4 |
| Facing | FACE_UP/DOWN/LEFT/RIGHT | 4 |
| Mining | MINE | 1 |
| Pickup | PICKUP | 1 |
| Place | PLACE_{6 machines} | 6 |
| Craft | CRAFT_{17 outputs} | 17 |
| Research | RESEARCH_{BASIC,ADV} | 2 |
| Deposit | DEPOSIT_{22 item types} | 22 |
| Withdraw | WITHDRAW_{22 item types} | 22 |
| **Total** | | **80** |

Removed: REPAIR, ROTATE (direction set at placement).

## Part 2: Simplified Per-Tile State

### Split inventory by machine role

**Buffer machines** (belt, arm, miner output, pallet): hold 1 item type.

```
buffer_type:  int8[H, W]    -- item type (0 = empty)
buffer_count: int16[H, W]   -- quantity
```

3 bytes/tile. Replaces 30-byte inventory + argmax.

**Assembler slots**: hold up to 2 input types + 1 output.

```
asm_in_type:   int8[H, W, 2]   -- input types (0 = empty)
asm_in_count:  int16[H, W, 2]  -- input quantities
asm_out_type:  int8[H, W]      -- output type (0 = none)
asm_out_count: int16[H, W]     -- output quantity
```

8 bytes/tile. Recipe-matched automatically each tick.

**Miner fuel**: separate from buffer (miners output ore into buffer, consume
coal from fuel).

```
machine_fuel: int16[H, W]  -- coal count (only miners use this)
```

### New EnvState

```python
@struct.dataclass
class EnvState:
    # Map
    world_map:         int8[H, W]
    block_resources:   int16[H, W]

    # Machines
    machine_type:      int8[H, W]
    machine_direction: int8[H, W]
    machine_power:     int16[H, W]
    machine_fuel:      int16[H, W]       # coal for miners

    # Buffer inventory (belt, arm, miner output, pallet)
    buffer_type:       int8[H, W]
    buffer_count:      int16[H, W]

    # Assembler inventory
    asm_in_type:       int8[H, W, 2]
    asm_in_count:      int16[H, W, 2]
    asm_out_type:      int8[H, W]
    asm_out_count:     int16[H, W]

    # Player
    player_positions:  int16[P, 2]
    player_directions: int8[P]
    player_inventory:  int16[P, NUM_ITEMS]
    selected_player:   int32

    # Progress
    timestep:          int32
    items_mined:       int32[NUM_ITEMS]
    research_progress: int16[NUM_TECH]
    research_unlocked: bool[NUM_TECH]
    achievements:      bool[NUM_ACHIEVEMENTS]
```

Removed: machine_inventory[H,W,15], machine_health, machine_selected_recipe,
scent_field, biter_positions, biter_health, crafting_recipe, craft_progress.

### Memory comparison (32x32 map, batch=1024)

| Component | Current | Proposed | Savings |
|-----------|--------:|--------:|---------:|
| Machine inventory | 30,720 B | 0 | -100% |
| Buffer + asm slots | 0 | 11,264 B | new |
| Machine scalars (type, dir, power, fuel) | 16,384 B | 5,120 B | -69% |
| Dead fields (health, recipe, scent, biters) | 12,672 B | 0 | -100% |
| **Total machine state** | **~60 KB** | **~16 KB** | **-73%** |

At batch=1024: **60 MB -> 16 MB**.

## Part 3: Simplified Machine Operations

### Belt transport: roll-based pull

```python
for d in range(4):
    dy, dx = OFFSETS[d]
    up_type = jnp.roll(buffer_type, (dy, dx), axis=(0, 1))
    up_count = jnp.roll(buffer_count, (dy, dx), axis=(0, 1))
    facing_d = (direction == d) & is_belt
    up_is_source = jnp.roll(facing_d, (dy, dx), axis=(0, 1))
    can_pull = facing_d & up_is_source & (buffer_count == 0) & (up_count > 0)
    buffer_type = jnp.where(can_pull, up_type, buffer_type)
    buffer_count = jnp.where(can_pull, up_count, buffer_count)
    was_pulled = jnp.roll(can_pull, (-dy, -dx), axis=(0, 1))
    buffer_type = jnp.where(was_pulled, 0, buffer_type)
    buffer_count = jnp.where(was_pulled, 0, buffer_count)
```

No scatter. No argmax. No contention.

### Arm transfer: directional roll with buffer/assembler awareness

Arms pull from backward neighbor into their buffer, deposit from buffer into
forward neighbor. Uses same roll pattern but distinguishes between:
- Buffer machines (read/write buffer_type/buffer_count)
- Assemblers (read asm_out for pick, write asm_in for deposit)

### Miner: extract to buffer

Miners write ore into their own buffer_type/buffer_count. Coal is consumed
from machine_fuel. Arms pull miner output from the buffer.

### Assembler: auto-recipe match

Each tick, check asm_in_type/asm_in_count against recipe table. On match:
consume inputs, write to asm_out_type/asm_out_count. Arms pull output.

## Part 4: Implementation Stages

Each stage is independently deployable and testable. A/B profiling after
each stage measures whether the change actually helps.

### Stage 1: New constants and recipe table

**What:** Replace ItemType, MachineType, BlockType enums. Define new recipe
table (R_TYPE_A, R_TYPE_B, R_AMT_A, R_AMT_B, R_OUTPUT arrays). Add new
BlockTypes for tin ore and silicon deposits. Update world generation.

**Files:** constants.py, recipes.py, world generation

**Test:** Unit tests for recipe uniqueness, item/machine mappings.
No performance test yet (state structure unchanged).

**Risk:** Low. Additive changes, no logic changes.

### Stage 2: New EnvState structure

**What:** Replace EnvState dataclass with new fields. Remove machine_inventory,
machine_health, machine_selected_recipe, scent_field, biter arrays,
crafting_recipe, craft_progress. Add buffer_type, buffer_count, asm slots,
machine_fuel. Shrink dtypes (int32 -> int8/int16).

**Files:** state.py, all files that create/access EnvState

**Test:** `pytest` -- most tests will break here and need updating.
Profile state memory: verify ~73% reduction.

**A/B:** Measure `step_env` with stub machine functions (NOOP bodies).
This isolates the state-copy overhead from computation.

**Risk:** High. Touches everything. Do in one focused pass.

### Stage 3: Port machine functions to new state

**What:** Rewrite refuel_machines, run_miners, run_assemblers,
push_miner_output, run_conveyor_belts, run_arms to use buffer_type/
buffer_count/asm slots instead of machine_inventory.

**Sub-stages:**
- 3a: Miners (simplest -- write to buffer, consume from fuel)
- 3b: Assemblers (auto-recipe matching from asm_in slots)
- 3c: Belts (roll-based pull)
- 3d: Arms (directional pull/deposit with buffer + asm awareness)
- 3e: Remove push_miner_output (arms pull from miner buffer instead)

**Test per sub-stage:** Machine-specific unit tests. A/B profiling after
each sub-stage against the previous.

**A/B metric:** `--decompose` timing for each machine function.
Key target: belt time should drop significantly (no scatter, no argmax).

### Stage 4: Port player actions

**What:** Update deposit/withdraw to work with buffer + asm slots.
Replace CRAFT actions with new recipe system (instant crafting, same
recipe table). Remove REPAIR, ROTATE. Update action dispatch.

**Files:** game_logic.py, crafting.py (simplify or merge)

**Test:** Action integration tests. Deposit/withdraw/craft unit tests.

**A/B:** `handle_player_action` timing before/after.

### Stage 5: Port observations

**What:** Update spatial channels to use new state fields. Update player
scalars for new item types and recipe affordability. Remove biter_presence
channel, machine_health channel.

**Files:** observations.py, achievements.py

**Test:** Observation shape tests. Verify observation values are sensible.

**A/B:** `global_array` timing before/after.

### Stage 6: Clean up dead code

**What:** Remove biters.py functions (already not called, but code still
exists). Remove inventory.py helpers that are no longer needed
(can_add_to_machine, _distinct_types). Remove deprecated constants
(MACHINE_NUM_SLOTS, MACHINE_SLOT_ROLES). Delete dead test files.

**Files:** biters.py, inventory.py, constants.py, tests/

**Test:** `ruff check`, `pytest`, grep for orphaned imports.

**A/B:** Compile time reduction (smaller code -> faster JIT).

### Stage 7: Final profiling and recording

**What:** Full profiling suite. Compare against baseline.

```bash
python scripts/profile_step.py --all
```

**Measurements:**
- Throughput curve (batch 1 -> 8192)
- Sub-operation decomposition
- HLO kernel count (target: <400, down from 714)
- State memory per env

**Record:** Update IMPROVEMENTS.md, autoresearch.jsonl.

**Target:** >45,000 steps/s at batch=1024 (2x current).

## Stage Summary

| Stage | What | Risk | A/B Metric |
|-------|------|------|------------|
| 1 | Constants + recipes | Low | N/A (no perf change) |
| 2 | New EnvState | High | State memory, stub step time |
| 3a | Miners on new state | Med | run_miners time |
| 3b | Assemblers on new state | Med | run_assemblers time |
| 3c | Belts with roll | Med | run_conveyor_belts time |
| 3d | Arms on new state | Med | run_arms time |
| 3e | Remove push_miner_output | Low | step_env time |
| 4 | Player actions | Med | handle_player_action time |
| 5 | Observations | Low | global_array time |
| 6 | Dead code removal | Low | Compile time |
| 7 | Final profiling | -- | Full throughput curve |

Each stage: measure A (before), implement, run tests, measure B (after),
log to autoresearch.jsonl. Revert if B is worse than A by >5%.
