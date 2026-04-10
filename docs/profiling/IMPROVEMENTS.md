# Performance Improvements Log

Each entry documents a committed optimization with measured impact.

## Baseline

- **Date**: 2026-04-10
- **Hardware**: RTX 3050 6GB Laptop GPU
- **Metric**: 16,608 steps/s at batch=1024, 32x32 map
- **Bottleneck**: Launch-overhead + memory-bandwidth bound
- **Script**: `scripts/profile_step.py --all`

---

## 1. Replace lax.cond with branchless jnp.where (steps 1-9)

**Problem:** 26 `lax.cond` calls in the step path, each a potential fusion
barrier and GPU-CPU sync point.

**Solution:** Converted all `lax.cond` calls in game_logic.py, crafting.py,
placement.py, and biters.py to branchless `jnp.where`/`jax.tree.map` patterns.
Also unrolled 3 `lax.scan` loops with small iteration counts (MAX_RECIPE_INPUTS=2)
to Python for-loops.

**Measured impact:** Negligible at batch>=64 (~0.5%). Under `vmap`, JAX already
converts `lax.cond` to `lax.select`, so the fusion barriers didn't exist in
the batched case. Minor improvement at batch=1 (+4.5%).

**Files:** game_logic.py, crafting.py, placement.py, biters.py

**Lesson learned:** `lax.cond` is only a problem when not vmapped. For batched
RL environments, the conversion is a code quality improvement but not a
performance win.

**References:**
- JAX Issue #7934 (confirmed: vmap converts cond to select)
- Craftax (arXiv:2402.16801) uses this pattern

---

## 2. Remove biter system from step function

**Problem:** `update_biters` (29.3% of step) and `update_scent_field` (3.2%)
dominated step time. The `spawn_biters` scan over 1024 tiles produced 156
while-loop iterations in the HLO, each a kernel launch.

**Solution:** Removed `update_scent_field` and `update_biters` calls from
`factoriax_step`. Biter arrays remain in EnvState (initialized to zeros,
never modified). Deleted tests/test_biters.py.

**Measured impact:**

| Batch | Before | After | Delta |
|------:|-------:|------:|------:|
| 1 | 44 steps/s | 92 steps/s | **+109%** |
| 64 | 2,944 | 7,025 | **+139%** |
| 1024 | 16,608 | 22,264 | **+34.1%** |
| 8192 | 17,990 | 25,861 | **+43.8%** |

HLO kernel count: 866 → 714 (-17.6%). While loops: 156 → 0 (-100%).

**Files:** game_logic.py, tests/test_biters.py (deleted)

---

## 3. Architecture redesign: split inventory, auto-recipe, roll-based belts

**Problem:** The original machine_inventory[H,W,15] stored 15 item types per
tile when belts/arms hold 1 item. Assemblers required manual recipe selection.
Machine functions used argmax and scatter-add with contention. State memory
was ~60 KB per environment.

**Solution:** Complete rewrite across Stages 1-3:
- New item system (22 types, 17 auto-recipes)
- Split inventory: buffer_type/count (3 bytes/tile) for belts/arms/miners +
  asm_in/out slots (8 bytes/tile) for assemblers
- Roll-based belt transport (4 directional jnp.roll passes, no scatter)
- Auto-recipe assemblers (unrolled recipe matching, no recipe selection)
- Instant player crafting (same recipe table, no progress timer)
- Removed: machine_health, scent_field, biters, crafting progress
- Dtype shrinking: int32 -> int8/int16 throughout

**Measured impact:**

| Batch | Original | Redesigned | Speedup |
|------:|--------:|-----------:|--------:|
| 1 | 44 | 2,245 | **51x** |
| 16 | 890 | 39,721 | **45x** |
| 64 | 2,944 | 95,554 | **32x** |
| 1024 | 16,608 | 143,443 | **8.6x** |
| 8192 | 17,990 | 138,847 | **7.7x** |

State memory: 60 KB -> 16 KB per environment (-73%).
HLO: 561 fusions (was 866), 0 while loops (was 156).

**What didn't work:** Roll-based arms (tried, reverted). The 4-direction
loop with 2 phases (deposit + pick) and assembler slot awareness created
~160 roll+where operations, doubling the HLO fusion count and halving
throughput. Scatter-based arms are faster because the arm operation is
complex enough that the scatter contention cost is lower than the roll
overhead.

**Files:** constants.py, recipes.py, state.py, machines.py, game_logic.py,
crafting.py, placement.py, observations.py, levels.py (complete rewrites)

**References:**
- CAX (arXiv:2410.02651): jnp.roll for cellular automata neighbor access
- Craftax (arXiv:2402.16801): branchless game logic patterns
- XLA fusion paper (arXiv:2301.13062): fusion barrier analysis

---

## 4. Remove arms, assemblers self-service

**Problem:** `run_arms` was 42% of step time (3,762 us) due to scatter
operations for bidirectional neighbor access.

**Solution:** Removed arms entirely. Assemblers pull inputs from adjacent
buffers via jnp.roll (4 directions) and push output to their own buffer.
Belts handle all transport.

**Impact:** 143K -> 171K steps/s at batch=1024 (+19%).

---

## 5. Entity-based machine processing

**Problem:** Grid-based operations iterate O(H*W) tiles per tick regardless
of machine count. At 128x128 (16K tiles) with ~50 machines, 99.7% of
computation is wasted.

**Solution:** Machine state moved from grid arrays to fixed-size entity
arrays (MAX_MACHINES slots). Machine functions iterate over entity slots
instead of the full grid. A ``tile_entity[H, W]`` reverse-lookup grid
maps tile positions to entity indices for neighbor access.

``max_machines`` defaults to ``max(64, map_area // 4)`` — 4x less work
than grid while supporting 25% machine density.

**Impact (batch=256):**

| Map | Grid | Entity | Speedup |
|-----|-----:|-------:|--------:|
| 16x16 | 366K | 506K | **+38%** |
| 32x32 | 155K | 243K | **+56%** |
| 64x64 | 43K | 74K | **+73%** |
| 128x128 | 10K | 19K | **+86%** |

Peak: 16x16 at batch=4096: **1,022K steps/s** (58x original baseline).

**Performance tuning:** ``max_machines`` is the key tuning knob. Cost
scales linearly with it regardless of active machine count. Reduce it
for faster stepping when few machines are needed; increase it for complex
factories. The auto default (area//4) balances speed and capacity.

**References:**
- NAVIX (arXiv:2407.19396): entity-based grid game architecture
