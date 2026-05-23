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

**Initial fix (reverted)**: iterating ``reversed(range(NUM_RECIPES))``
in `run_assemblers` made the *earliest* (lowest-index) match win.
That paid off a small cost at every tick and only papered over the
ambiguity — recipe lookup still depended on ordering, and agents
still had to deposit inputs in descending-count order to avoid
firing a sub-recipe mid-deposit.

**Structural fix landed**: recipes now have pairwise-distinct,
non-subset input type-sets. To unblock this, a new ``REFRACTORY``
half-fab (tin plate + coal on the furnace) was introduced and the
``FURNACE`` recipe switched from ``{iron_plate, copper_plate}`` to
``{iron_plate, refractory}``, eliminating the WIRE/FURNACE collision
that motivated the loop-reversal hack. Phase 3 now iterates
``range(NUM_RECIPES)`` normally and deposit order is irrelevant.
The uniqueness invariant is guarded by ``tests/test_recipes.py``.

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

## 6. Hand-crafting (`CRAFT_*`) reads the wrong recipe in scenarios with custom recipe tables

**Symptom**: `CRAFT_MINER`, `CRAFT_ARM`, and other `CRAFT_*` actions
silently NOOP when emitted against a scenario whose recipe table is
smaller than (or differently ordered from) `BASE_RECIPES`. The action
returns success, no exception is raised, no log line is emitted, but
the player's inventory is unchanged.

Reproducible against `EASY_ROCKET_RECIPE_TABLE` (9 recipes in a
different order from `BASE_RECIPES`'s 26 recipes). The
`DEFAULT_RECIPE_TABLE` scenarios (rocket benchmark, skills benchmark)
are unaffected because their tables match `BASE_RECIPES` exactly.

**Minimal reproduction** (clean repo state, no patches applied):

```python
import jax, jax.numpy as jnp
import factoriax
from factoriax.constants import Action, ItemType
from factoriax.levels import build_state
from factoriax.scenarios.easy_rocket import (
    EASY_ROCKET_RECIPE_TABLE,
    build_easy_rocket_level,
    easy_rocket_conditions,
)

level = build_easy_rocket_level(jax.random.PRNGKey(0))
env, params = factoriax.make(level, obs="global", achievement_fn=easy_rocket_conditions)
params = params.replace(
    num_players=1,
    max_timesteps=2000,
    recipe_table=EASY_ROCKET_RECIPE_TABLE,
)
state = build_state(level, params)

# Give the player enough raw ore for one MINER (recipe: LIMESTONE + SILICON).
state = state.replace(
    player_inventory=state.player_inventory
        .at[0, int(ItemType.LIMESTONE)].set(5)
        .at[0, int(ItemType.SILICON)].set(5)
)

_, state, *_ = jax.jit(env.step_env)(
    jax.random.PRNGKey(0),
    state,
    jnp.asarray(int(Action.CRAFT_MINER), dtype=jnp.int32),
    params,
)

# Expected: 1 MINER, 4 LIMESTONE, 4 SILICON.
# Actual:   0 MINER, 5 LIMESTONE, 5 SILICON.
print(f"MINER={int(state.player_inventory[0, int(ItemType.MINER)])}")
print(f"LIMESTONE={int(state.player_inventory[0, int(ItemType.LIMESTONE)])}")
print(f"SILICON={int(state.player_inventory[0, int(ItemType.SILICON)])}")
```

**Root cause**: `factoriax/game_logic.py:532` computes the recipe
index from the action directly:

```python
recipe_idx = jnp.clip(action - CRAFT_BASE, 0, NUM_RECIPES - 1)
```

with `CRAFT_BASE = int(Action.CRAFT_IRON_PLATE) = 21` and
`NUM_RECIPES = len(BASE_RECIPES) = 26`. For
`Action.CRAFT_MINER = 31` this gives `recipe_idx = 10` — correct in
`BASE_RECIPES` (where MINER is at index 10) but **out of bounds** in
`EASY_ROCKET_RECIPE_TABLE` (8 rows; MINER at index 0). JAX silently
clamps the gather to the last valid row, so the dispatcher reads
`table.outputs[7] = ROCKET`, checks ROCKET's affordability (needs 200
HULL + 200 ENGINE_UNIT), fails (the player has neither), and returns
without crafting. When the player's materials *do* coincidentally
match the wrongly-selected recipe's inputs, the wrong item is
crafted — the failure mode is silently incorrect output rather than
silent NOOP.

`RecipeTable` already declares a `craft_action_to_recipe` field whose
docstring explicitly says it exists *"so a future re-ordering can
rewire the mapping cheaply"*. The dispatcher does not read it. The
remapping is half-implemented: the field exists and is correctly
populated for `BASE_RECIPES`-shaped tables (as the identity
`arange(n)`), but the dispatcher hardwires the identity in
`game_logic.py` rather than reading the field.

**Impact**: every `CRAFT_*` action is broken for any scenario whose
recipe set is smaller than or reordered relative to `BASE_RECIPES`.
PPO training on easy_rocket did not surface this because the high
entropy of an untrained policy means CRAFT actions are emitted with
random inputs; the silent NOOP looks no different from "I lacked the
materials." Achievement chains that require hand-crafted items (e.g.
`has_miner_in_inventory`, `has_assembler_in_inventory`,
`has_belt_in_inventory` in easy_rocket) are unreachable from any
external interface as a result, capping the bench's reachable score.

**Fix landed**: crafting now resolves through the output *item*, not a
positional recipe index. `factoriax/game_logic.py` defines
`CRAFT_ACTION_TO_ITEM` (CRAFT_* offset -> output `ItemType`) and the
dispatcher computes
`recipe_idx = params.recipe_table.output_to_recipe[CRAFT_ACTION_TO_ITEM[offset]]`
(`-1` when the active table has no recipe for that item);
`crafting.py::craft_recipe` no-ops on a negative index. This is a cleaner
route than the originally-suggested `craft_action_to_recipe` remap — that
field is left unused (dead plumbing, to be removed when the
constants/recipes re-export is untangled). Recipe-list order is no longer
a dispatch join key, so reordered/subset tables craft correctly.

The `PLACE_*` / `ROTATE_*` / `CRAFT_*` offset-resolution tables also moved
from `constants.py` to `game_logic.py` (they are dispatch wiring, not
environment constants). Guarded by the alternate-book dispatch tests in
`tests/test_recipe_book.py`, which build a reordered/subset recipe book
and assert every `CRAFT_*` action resolves to a recipe whose output is the
action's item (absent items resolve to `-1`).

## (reserved for further bugs as they surface)
