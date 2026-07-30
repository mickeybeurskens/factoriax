# Known Issues

Open defects and design gaps found while working the codebase. Each entry
records the date it was found, the files involved, and what goes wrong.

Line numbers are a starting point and drift with edits. The named symbol is the
reliable anchor.

## Open

- **Science lab contents load into a buffer the lab never reads**
  Found 2026-07-29.
  Files: `factoriax/engine/levels.py` (the `else` branch of the inventory
  block in `build_state`, around line 819), `factoriax/engine/machines.py:378`.
  `build_state` routes any machine that is not a miner, assembler, or furnace
  to `ent_buf`. A science lab consumes from `ent_asm_in`, so a level that ships
  a lab pre-filled with science packs loads those packs into storage the lab
  never looks at. The lab reads as empty and does not run. Reproduced by
  building a level with one `SCIENCE_LAB` holding `TIER1_SCIENCE_PACK`: the
  packs land in `ent_buf_count` and `ent_asm_in_count` stays zero.

- **Crossing contents load into a buffer the crossing never reads**
  Found 2026-07-29.
  Files: `factoriax/engine/levels.py` (same `else` branch),
  `factoriax/engine/tables.py:110-113`.
  Same routing bug as the science lab, but harder to fix. A crossing runs a
  vertical and a horizontal flow and keeps one buffer per axis in the two
  `ent_asm_in` columns (`CROSSING_VERT_SLOT`, `CROSSING_HORIZ_SLOT`). Items
  loaded from a level land in `ent_buf` instead, so a pre-filled crossing
  starts empty on both axes. Fixing it needs a decision first: the level format
  stores machine contents as an item-indexed pouch with no slot, so nothing in
  a saved level says which axis an item belongs to. Either the format grows a
  slot dimension or the loader picks an axis by rule.

- **`build_state` classifies items using the default recipe book**
  Found 2026-07-29.
  Files: `factoriax/engine/levels.py:63` (`_ITEM_PRODUCER`),
  `factoriax/engine/state.py:88` (`EnvParams.recipe_table`).
  Deciding whether a combiner's stored item is a finished output or an input
  requires knowing which machine produces it. `_ITEM_PRODUCER` derives that
  from the engine's default recipes at import. `build_state` takes no
  `EnvParams`, so a scenario shipping its own recipe book is not consulted, and
  an item that is an output only under that book loads into an input slot.
  No shipped scenario changes the output set today, so this is latent. Fixing
  it means adding a recipe table argument to `build_state`.

- **Agent crafting does not have a tick count**
  Found 2026-07-30.
  Files: `factoriax/engine/crafting.py:109` (`craft_recipe`),
  `factoriax/engine/recipes.py` (`Recipe.ticks`).
  `craft_recipe` never reads `ticks`. A player craft checks affordability,
  consumes the inputs, and writes the output in the same step, so a recipe's
  duration applies to combiners only. Hand-crafting is therefore free of the
  cost the same recipe imposes on a machine, and no action or state field
  exists to carry a partially finished player craft.

- **`first_pipeline` unlocks without a pipeline**
  Found 2026-07-30.
  Files: `factoriax/playground/play/achievements.py:106` (`first_pipeline`),
  `factoriax/engine/achievements.py` (`any_buffer_nonempty`),
  `factoriax/engine/step.py:405` (`deposit_to_adjacent`).
  The achievement is hinted "Use an arm to move miner output into a pallet",
  but its condition is only `any_buffer_nonempty(PALLET)`. `deposit_to_adjacent`
  writes `ent_buf` for any machine that is neither a miner nor a combiner, so
  a player standing next to a pallet unlocks it with one `DEPOSIT_` action, no
  arm and no miner involved. Reproduced: place a pallet, face it, deposit one
  coal, and the bit flips False to True in a single step. An agent rewarded
  for this bit learns the shortcut, not the automation.
  `pallet_filled` in `factoriax/engine/envs/rocket.py:243` shares the
  condition but is hinted "Put an item in a pallet", which matches what it
  actually checks. Fixing `first_pipeline` needs a condition that implies
  transport, since no state field records how an item reached a buffer.

- **Machine inventory loading is a hand-written branch per machine kind**
  Found 2026-07-29.
  Files: `factoriax/engine/levels.py:796-825`.
  The loader decides where a machine's items go with an `if` chain naming
  `MINER`, then `ASSEMBLER`/`FURNACE`, then everything else, and hardcodes the
  two `ent_asm_in` columns as a literal. It consults no per-machine
  configuration. The two routing bugs above are both instances of a machine
  kind falling into a branch that does not match how it stores items, and a
  new machine kind will land in the `else` branch by default whether or not
  that is correct.

## Fixed

- **A combiner's finished output loaded into an input slot**
  Found and fixed 2026-07-29, commit `67f45cf`.
  Files: `factoriax/engine/levels.py`, `tests/test_levels.py`.
  `build_state` filled `ent_asm_in` with the first two non-zero items and left
  `ent_asm_out` hardcoded to zeros. A furnace holding `IRON_PLATE` loaded it as
  an input it can never consume, and a furnace holding ore, coal, and a plate
  exceeded the two input columns so the plate was dropped. The loader now asks
  whether the machine's own recipes produce an item and routes it accordingly.
  Covered by `TestCombinerInventoryLoading` in `tests/test_levels.py`.
