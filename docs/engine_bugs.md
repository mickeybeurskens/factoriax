# Engine bugs encountered while developing the scripted rocket agent

Tracked as discovered so they can be reviewed / fixed as standalone
commits per the project's commit hygiene rule.

## 1. `WITHDRAW_*` can't pull output from furnaces or assemblers

**Symptom**: A player who deposited inputs into a pre-placed furnace /
assembler, waited for the recipe timer, and then emitted the correct
`WITHDRAW_<item>` action while standing adjacent and facing the
machine received nothing — inventory didn't change.

**Root cause**: Two code paths get out of sync.

- `factoriax/machines.py::run_combiners` Phase 4 ("Push output to
  buffer") unconditionally moves any completed recipe output from
  `ent_asm_out_*` → `ent_buf_*`, every tick, for both furnaces and
  assemblers. So one tick after the recipe completes, the output
  lives in `ent_buf_count` / `ent_buf_type`.

- `factoriax/game_logic.py::withdraw_from_adjacent` gates its two
  withdrawal paths like this:

  ```python
  is_combiner = (mt == MachineType.ASSEMBLER) | (mt == MachineType.FURNACE)
  # combiner: only look at asm_out
  can_withdraw_asm = ... is_combiner & out_match & out_has ...
  # buffer machines: only look at buf
  is_buf = ~is_combiner & (mt != MachineType.NONE)
  can_withdraw_buf = ... is_buf & buf_match & buf_has ...
  ```

  Because `is_buf = ~is_combiner ...`, combiners can never use the
  buf-based withdraw. And because Phase 4 always empties `asm_out`,
  the combiner's asm-based withdraw never has anything to withdraw.

**Reproduction** (confirmed by hand): deposit 2 iron ore + 1 coal into
the pre-placed furnace of the rocket benchmark, wait 10 ticks, emit
`WITHDRAW_IRON_PLATE` while facing the furnace — nothing happens. A
trace of the furnace entity shows:

```
t=7  out=(IRON_PLATE, 0)  power=2   buf=(EMPTY, 0)
t=8  out=(IRON_PLATE, 0)  power=1   buf=(EMPTY, 0)
t=9  out=(EMPTY, 0)       power=0   buf=(IRON_PLATE, 1)   ← Phase 4 fired
t=10+                                 buf=(IRON_PLATE, 1)   ← still stuck
```

**Impact**: Without this fix the scripted agent cannot produce
anything via machines — plates, intermediates, rocket all become
unreachable unless the agent builds a full arm+pallet extraction
pipeline first, which is itself blocked on acquiring plates.

**Suggested fix**: Drop the `~is_combiner` gate so the `buf` path is
available for every machine type. The two paths are still mutually
exclusive for any given entity because `asm_out` is always empty on
combiners post-Phase-4, so no double-withdrawal risk.

```diff
- is_buf = ~is_combiner & (mt != MachineType.NONE)
+ is_buf = mt != MachineType.NONE
```

**Fix landed** in `is_buf = mt != MachineType.NONE` (`game_logic.py`).
The two withdrawal paths stay mutually exclusive in practice because
`asm_out` is empty post-Phase-4 on combiners, so no double-deduct.

## 2. Assembler picks the wrong recipe when inputs are a superset

**Symptom**: Depositing the canonical WIRE inputs (1 iron plate + 2
copper plate) into a placed assembler produces a FURNACE, not a wire.
Same pattern for any recipe whose inputs are a strict superset of a
later recipe's inputs.

**Root cause**: `factoriax/machines.py::run_assemblers` Phase 3
iterates recipes in list order and uses
``matched = jnp.where(cond, r, matched)`` — so the *last* matching
recipe's index wins. Combined with the ``>=`` input-count check, a
recipe with smaller requirements (e.g. FURNACE at index 14, needing
1 iron + 1 copper) always beats a recipe with larger requirements at
an earlier index (WIRE at index 6, needing 1 iron + 2 copper) when
both are satisfiable.

**Impact**: WIRE is the only affected core recipe on the rocket
path, but since MOTOR / SENSOR / ARM / BELT / MINER / ASSEMBLER /
ROCKET all transitively depend on WIRE, every late-tier recipe is
blocked too.

**Fix landed**: iterate ``reversed(range(NUM_RECIPES))`` in
`run_assemblers`, so the *earliest* (lowest-index) match wins. Aligns
with the intuitive "more-specific recipe wins" expectation at no
extra cost. Agents still need to deposit inputs in
descending-count order so the assembler doesn't fire a sub-recipe
mid-deposit — see ``ProduceInMachine`` in the scripted agent.

## 3. `DEPOSIT_FURNACE` and `WITHDRAW_FURNACE` silently dispatch as movement

**Symptom**: `WITHDRAW_FURNACE` emitted while facing a machine holding
a furnace item has no effect — player inventory stays unchanged.

**Root cause**: `factoriax/game_logic.py::factoriax_step`'s action
dispatch ranges (around lines 533–540) end at `DEPOSIT_ROCKET` /
`WITHDRAW_ROCKET`. But `Action` was extended with
`DEPOSIT_FURNACE = 60` *after* `DEPOSIT_ROCKET = 59`, and
`WITHDRAW_FURNACE = 83` *after* `WITHDRAW_ROCKET = 82`. So
`DEPOSIT_FURNACE` and `WITHDRAW_FURNACE` fall outside the compared
range, don't get `cat=6` / `cat=7`, default to `cat=0` (movement),
and become no-ops.

**Fix landed**: tighten the comparisons to `<= Action.DEPOSIT_FURNACE`
and `<= Action.WITHDRAW_FURNACE`. Two-line change.

## 4. `PLACE_FURNACE` silently dispatches as movement

**Symptom**: `PLACE_FURNACE` emitted while holding a furnace item and
facing an empty walkable tile has no effect.

**Root cause**: Same family of bug as #3. `Action.PLACE_FURNACE = 17`
was added after `Action.PLACE_ROCKET = 16`, but
`factoriax/game_logic.py::factoriax_step`'s dispatch range stops at
`PLACE_ROCKET`. So `PLACE_FURNACE` falls out, defaults to `cat=0`
(movement), no-ops.

**Fix landed**: change the upper bound to `<= Action.PLACE_FURNACE`.
One-line change.

## 5. `first_assembly` condition watches the wrong slot

**Symptom**: the rocket benchmark's ``first_assembly`` achievement
never fires even when assemblers are visibly producing outputs.

**Root cause**: ``_any_assembler_has_output`` in
``factoriax/benchmarks/rocket.py`` checked
``state.ent_asm_out_count > 0``. But ``run_combiners`` Phase 4
immediately moves the completed output from ``ent_asm_out_*`` into
``ent_buf_*`` in the same tick, so by the time the achievement
condition runs at end-of-tick, ``ent_asm_out_count`` has been
cleared. The condition never observed a positive value.

**Fix landed**: OR the check with ``ent_buf_count > 0`` — either slot
having content signals the assembler produced output. Adjacent to an
engine bug but the fix is in the benchmark condition (which we own).

## (reserved for further bugs as they surface)
