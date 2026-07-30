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

- **Machine health has no damage source, so repair is unreachable**
  Found 2026-07-30.
  Files: `factoriax/engine/placement.py:220` (place sets full),
  `:325` (pickup gate), `:411` (pickup clears to 0), `:494` (`apply_repair`),
  `factoriax/engine/constants.py` (`InteractAction.REPAIR`),
  `factoriax/engine/tables.py` (`MACHINE_HEALTH`, `MACHINE_MAX_HEALTH`).
  Every write to `ent_health` either sets it to `MACHINE_MAX_HEALTH` on
  placement, clears it to 0 on pickup, or raises it toward full in
  `apply_repair`. Nothing decrements it, so a placed machine is always at full
  health. Two consequences: `apply_repair` can only ever be a no-op, and the
  `is_full_health` gate on pickup always passes. `REPAIR` is still a member of
  `InteractAction`, so an agent spends part of its action space on an action
  that cannot change the state.
  The field itself is not dead: pickup clearing it to 0 and placement setting
  it make it an occupancy marker alongside `ent_y`. Removing `REPAIR` would
  renumber `Action` and invalidate trained checkpoints, so this needs a
  decision, either add a damage source or drop the action and accept the
  action-space change.

- **Every doctest in `levels.py` calls an API that does not exist**
  Found 2026-07-30.
  Files: `factoriax/engine/levels.py` (`build_state`, `save_level`,
  `load_level`, and the `Level` docstring), `factoriax/__init__.py`.
  The examples call `factoriax.LevelBuilder`, `factoriax.build_state`,
  `factoriax.save_level`, `factoriax.load_level`, `factoriax.make`,
  `factoriax.LEVELS`, `FactoriaxEnv`, and `get_level` as top-level names.
  `factoriax/__init__.py` is deliberately kept free of imports, so none of
  them resolve. They also pass `EnvParams(num_players=...)` to `build_state`,
  which takes `num_players: int` and reads it from an `EnvParams` field that
  does not exist. `build_state`'s example additionally assigns `params` twice,
  discarding the first value.
  Nothing catches this: `pyproject.toml` sets `testpaths = ["tests"]` with no
  `--doctest-modules`, so the examples are inert text. They are wrong in a way
  that is worse than absent, because they read as the supported entry point.
  Either wire up doctest collection so examples are executable, or write them
  against the real import paths and accept that nothing verifies them.

- **Terrain layers overwrite each other, so only water hits its share**
  Found 2026-07-30. Overlap accepted 2026-07-30; not a bug to fix.
  Files: `factoriax/engine/levels.py` (`generate_terrain`).
  The `jnp.where` chain applies silicon, tin, coal, copper, iron, then water,
  and each overwrites the last. The noise draws are independent but the
  outcomes are not, so a layer keeps only the part no later layer claimed.
  Measured on a 200x200 map with default params: water 0.100 against 0.100
  configured, iron 0.109, copper 0.094, coal 0.079, tin 0.055, silicon 0.052
  against 0.100. Silicon is applied first and loses most.
  Recorded so the gap between a configured probability and a realised share is
  not mistaken for a regression later. Removing it would mean partitioning one
  draw into bands, which changes what the probabilities mean.

- **The one built-in level ships ore patches holding 3 units**
  Found 2026-07-30.
  Files: `factoriax/engine/levels.py` (`_15X15_RESOURCES`).
  Each `fill_rect` passes `resources=3` where `BLOCK_MAX_RESOURCES` is 30000,
  so every patch in `15x15_resources` holds 3 units per tile, 48 per ore
  across the whole map. It is the only registered level and the default for
  anything calling `get_level`. Three units may be deliberate for a short
  test, but it is four orders of magnitude off the constant the rest of the
  module treats as a full deposit, and nothing records which was intended.

- **The pixel observation hides state an agent needs to play**
  Found 2026-07-30.
  Files: `factoriax/engine/observations.py:843` (`rgb`), `:258`
  (`_common_scalars`), `:321` (`_facing_scalars`),
  `factoriax/engine/renderer.py` (`render_map`),
  `factoriax/assets/atlas.layout.md` (rows 5 and 7).
  `observations.rgb` returns `render_map` output unchanged, so a pixel agent
  sees terrain, machine kind and facing, and player position and facing, and
  nothing else. The symbolic profiles carry more: `_common_scalars` adds the
  timestep, the player's whole inventory, and a per-item craft-affordability
  flag, and `_facing_scalars` adds the kind and contents of the machine the
  player faces. A pixel agent therefore cannot see what it is carrying or
  whether a recipe is affordable, which makes crafting unlearnable from
  visuals alone. The two observation profiles are not solving the same task.

  The split to draw is HUD against UI. HUD is game state an agent must read to
  act: carried items and counts, what is selected for placement, machine
  contents and craft progress, and time remaining. That belongs in
  `render_map`. UI is interaction chrome, the pause menu, help overlay,
  control hints, and dialogs in `factoriax/playground/play/ui.py`, which an
  agent never needs and which should stay out of the renderer.

  Two pieces of the groundwork are already committed and unused. Atlas row 5
  holds an icon per `ItemType` and row 7 holds digit glyphs baked by
  `build_digit_atlas`; `render_map` reads neither, and `atlas.layout.md`
  records row 5 as reserved for exactly this. Anything added has to stay
  traceable and vmappable, so a readout has to be a gather and a blend over
  fixed-size arrays rather than a text draw, and it changes the observation
  shape, which invalidates checkpoints trained on the current `rgb` output.

- **Player 8 draws in two different colors**
  Found 2026-07-30.
  Files: `factoriax/engine/renderer.py` (`_ATLAS_NUM_PLAYERS`,
  `player_icon_rgba`, `render_map`), `factoriax/assets/build_atlas.py`
  (`NUM_PLAYERS`), `factoriax/playground/ui/icons.py:306` (`PLAYER_COLORS`).
  `PLAYER_COLORS` lists nine palettes, but the atlas bakes eight player
  sprites and both the map renderer and `player_icon_rgba` wrap the index
  modulo 8. `create_player_texture` and the editor inventory panel wrap modulo
  `len(PLAYER_COLORS)`, which is 9. Player 8 therefore appears light green in
  the inventory panel and in player 0's red on the map. Every earlier slot
  agrees, so the split only shows on a nine-player scenario. The fix is to
  pick one count: either drop the ninth entry of `PLAYER_COLORS` or raise
  `NUM_PLAYERS` to 9 and rebuild the atlas, which widens the misc row and
  changes the committed PNG that `tests/test_atlas_fresh.py` pins.

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

- **A recipe repeating an input item crafted into a negative inventory**
  Found and fixed 2026-07-30.
  Files: `factoriax/engine/recipes.py` (`RecipeBook.__post_init__`),
  `factoriax/engine/crafting.py`, `tests/test_recipe_book.py`.
  `can_afford_recipe` checks each of the two input slots on its own, so a
  recipe naming the same item in both slots read as affordable while the
  player held enough for one slot, and `craft_recipe` then subtracted each
  slot in turn. A one-recipe book taking `(IRON_PLATE, 1)` twice left a
  player holding 1 plate on -1 plate plus the output. `RecipeBook` now
  rejects a repeated input item at construction, so no book reaching the
  engine can express it. Covered by `test_repeated_input_item_raises`.

- **A combiner's finished output loaded into an input slot**
  Found and fixed 2026-07-29, commit `67f45cf`.
  Files: `factoriax/engine/levels.py`, `tests/test_levels.py`.
  `build_state` filled `ent_asm_in` with the first two non-zero items and left
  `ent_asm_out` hardcoded to zeros. A furnace holding `IRON_PLATE` loaded it as
  an input it can never consume, and a furnace holding ore, coal, and a plate
  exceeded the two input columns so the plate was dropped. The loader now asks
  whether the machine's own recipes produce an item and routes it accordingly.
  Covered by `TestCombinerInventoryLoading` in `tests/test_levels.py`.
