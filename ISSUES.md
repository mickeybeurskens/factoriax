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

- **A belt fills a science lab's buffer instead of being refused**
  Found 2026-07-31.
  Files: `factoriax/engine/machines.py` (`dn_is_combiner` in
  `run_conveyor_belts`, around line 921), `factoriax/engine/step.py`
  (`run_labs`).
  The belt pass refuses to push into an assembler or a furnace, which take
  items only through an arm or a player deposit. A science lab is not in that
  refusal set, so a belt aimed at one delivers into `ent_buf`, up to a
  `MACHINE_MAX_STACK` of 1000. `run_labs` consumes from `ent_asm_in` and
  nothing else, so a belt-fed lab yields no science however long it runs. The
  items are recoverable by an arm or a player, but nothing signals the
  mistake. Reproduced with a belt facing a lab and holding 3 `COAL`: after one
  `run_conveyor_belts` the lab holds `ent_buf_count == 3` and
  `ent_asm_in_count == [0, 0]`.
  A lab has no belt pull either, so the coherent rule is the one assemblers
  and furnaces already follow: combiners are never belt-fed. Adding
  `SCIENCE_LAB` to the refusal set turns a silent sink into a visible stall.
  `tests/test_belt_arm.py:322` already records arms as the intended route.
  Same symptom as the level-loading entry above, different cause.

- **`miner_output_reward` pays for a miner that nothing drains**
  Found 2026-07-31.
  Files: `factoriax/engine/rewards.py` (`miner_output_reward`, and
  `dense_deploy_reward`, which scales it by 5).
  The reward is the rise in `ent_buf_count` across placed miners, which is a
  level and not a flow. A miner wired into a belt or a pallet ends each step as
  empty as it started, so automating scores nothing, while an unattended miner
  scores the full mining rate. Reproduced on a one-tile coal patch: a miner
  facing a pallet scores 0.0 and the same miner facing nothing scores 3.0.
  Draining a miner scores negative. `miner_throughput_reward` measures the same
  intent correctly, from `block_resources`, and should be preferred.

- **`mining_reward` guides toward ore it does not pay for**
  Found 2026-07-31.
  Files: `factoriax/engine/rewards.py` (`mining_reward`),
  `factoriax/engine/tables.py` (`MINEABLE_BLOCKS`).
  The proximity term targets every block in `MINEABLE_BLOCKS`, which is coal,
  iron, copper, tin, silicon, and limestone. The bonus counts only coal, iron
  ore, and copper ore. An agent drawn to a silicon patch by the shaping term
  and mining it there earns nothing for the ore. The two terms need the same
  set, whichever set that turns out to be.

- **`dense_pickup_reward` pays for a place-and-pickup loop**
  Found 2026-07-31.
  Files: `factoriax/engine/rewards.py` (`dense_pickup_reward`).
  The reward is `max(prev_machine_count - new_machine_count, 0)` scaled by 10.
  Placing a machine raises the count and is clamped to no penalty, so an agent
  can place a machine and pick it straight back up for 10.0 every two steps for
  as long as the episode runs. The level it scores is short enough that the
  loop does not dominate, but the term does not survive a longer episode.

- **`dense_craft_reward` cannot tell crafting from picking things up**
  Found 2026-07-31.
  Files: `factoriax/engine/rewards.py` (`dense_craft_reward`).
  The craft term is the rise in the player's total count of miners, pallets,
  belts, and assemblers, clamped at zero. It never checks that materials were
  spent, so picking a machine up off the map pays the same 10.0 as crafting
  one. `sparse_pallet_crafting_reward` and `sparse_miner_crafting_reward`
  already do this correctly by requiring the inputs to fall in the same step.

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

- **Nothing runs the doctests, so examples rot silently**
  Found 2026-07-30. `levels.py` corrected 2026-07-30; the gap remains.
  Files: `pyproject.toml` (`testpaths = ["tests"]`, no `--doctest-modules`).
  Docstring examples across the tree are inert text. The ones in
  `factoriax/engine/levels.py` had drifted to an API that never existed,
  calling `factoriax.LevelBuilder`, `factoriax.make`, and
  `EnvParams(num_players=...)` against an `__init__.py` deliberately kept free
  of imports. They are now written against real import paths and verified once
  by hand with `uv run python -m doctest factoriax/engine/levels.py`, which
  passes 16 examples. Nothing repeats that check, so they can drift again.
  Enabling `--doctest-modules` would catch it, but it collects the whole tree
  and other modules have not been checked, so turning it on is its own piece
  of work rather than a flag flip.
  Confirmed 2026-07-31 to be wider than `levels.py`. Examples in
  `factoriax/engine/rewards.py`, `factoriax/engine/observations.py`,
  `factoriax/engine/envs/base.py`, and `factoriax/engine/envs/wrappers.py` all
  called `factoriax.make`, `factoriax.mining_reward`, `factoriax.global_x_ray`,
  `factoriax.rgb`, or `from factoriax import FactoriaxEnv`. None of those
  exist: `factoriax/__init__.py` is empty by design. All of them have been
  removed rather than repaired, since nothing would keep a repair honest.

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

- **Fifteen modules have no test that reaches them**
  Found 2026-08-03.
  Files: the modules in the two lists below.
  The test suite restructure gives each source module a mirror entry, so a
  module with no entry is now visible. Fifteen have no entry and no test that
  imports them by any route:
  `analysis/eval.py`, `analysis/video.py`, `engine/crafting.py`,
  `playground/app.py`, `playground/editor/inventory_panel.py`,
  `playground/editor/main.py`, `playground/editor/toolbar.py`,
  `playground/menu/main_menu.py`, `playground/play/main.py`,
  `playground/ui/compositing.py`, `playground/ui/fonts.py`,
  `playground/ui/forms.py`, `playground/ui/labels.py`,
  `playground/ui/panels.py`, `playground/ui/primitives.py`, and
  `playground/ui/window.py`.
  Six more have no entry but a test does reach them: `analysis/categories.py`,
  `engine/envs/common.py`, `engine/envs/miner_curriculum.py`,
  `engine/envs/mining.py`, `engine/envs/science_tiers.py`, `make.py`, and
  `playground/play/play_state.py`. The four `engine/envs` modules have
  scenario tests, which move to `tests/integration/scenarios/` in Task 17.
  Two of the gaps are large. `playground/editor/main.py` is 1680 lines and
  `playground/ui/panels.py` is 761 lines, and nothing tests either. Both draw,
  so a test costs a display fixture and an output-contract assertion, the same
  shape the `tests/playground/ui/` files already use.
  Three are entry points and are fair waivers: `playground/app.py`,
  `playground/editor/main.py`, and `playground/play/main.py`. So is
  `analysis/video.py`, which shells out to ffmpeg.
  `engine/crafting.py` is the one that matters most. It is 182 lines of engine
  core with nothing on it. Task 21 of `tasks/plan.md` covers it. Task 20
  writes the full verdict table to `tasks/coverage-gaps.md`.

- **The craft progress bar in the inventory menu is dead code**
  Found 2026-08-03.
  Files: `factoriax/playground/play/ui.py:1966` (`craft_progress = 0`), `:2058`
  (the `if craft_progress > 0` branch).
  `render_inventory_menu` binds `craft_progress` to the literal 0 and never
  writes it again. The branch that draws the progress bar tests
  `craft_progress > 0`, so it never runs. `EnvState` carried a
  `craft_progress` field before the entity migration, and the field went away
  with it. The binding stayed.
  Found through `tests/play/test_ui.py::test_craft_in_progress`, which passed
  `craft_progress=` to the state factory. The factory absorbed unknown names
  in a `**_kwargs` catch-all and dropped them, so the test rendered a default
  state and asserted the output shape. It duplicated
  `test_inventory_focus`. The catch-all now raises, and the test is deleted.
  Either delete the branch and the binding, or restore per-player craft
  progress to `EnvState` and read it here.

- **The functional `IntEnum` call for `ItemType` blocks mypy across the repo**
  Found 2026-08-03.
  Files: `factoriax/engine/constants.py:186`.
  `ItemType` is built through the functional API, `IntEnum("ItemType", ...)`,
  because its member names compose from `Machine` and the resource blocks.
  Mypy cannot resolve a member of a functional enum, so every
  `ItemType.COAL` in the repo raises `attr-defined`. This one cause produces
  1182 of the 1268 errors that `mypy tests` reports, and it dominates the 527
  errors that `mypy factoriax` reports. `[tool.mypy] strict = true` is
  therefore aspirational: no gate can run until this is settled.
  Two ways out. Declare the members in a `.pyi` stub beside the module, which
  keeps the composition at runtime. Or write the class out in full and assert
  at import time that it agrees with the composed names. The stub is smaller.
  Not urgent. The runtime behaviour is correct, and the composition is
  deliberate. This entry records why `strict = true` reports thousands of
  errors, so that a reader does not mistake it for real type rot.

- **The science tiers oracle costs 58.7 seconds of fixture setup**
  Found 2026-08-03.
  Files: `tests/scenarios/test_science_tiers_economics.py`,
  `factoriax/engine/envs/science_tiers.py`.
  One fixture setup in this file takes 58.7 seconds. The next slowest item in
  the suite is `tests/test_env_contract.py::test_step_is_vmappable` at 13.1
  seconds. The whole rest of the suite is faster than this one fixture, so it
  dominates the wall time of every full run. Measured with
  `uv run pytest -q --durations=25` on the CPU backend that `tests/conftest.py`
  pins.
  The cost is scripted rollouts across several seeds, and each seed pays its
  own XLA compile. A module-scoped env plus one shared `jax.jit` step would
  remove most of it.
  Not urgent. The `ScienceTiers-v1` scenario is not in active use yet, so the
  cost buys nothing today and blocks nobody. Revisit when the scenario carries
  real training work, or when a `slow` marker gives the suite a fast inner
  loop.

## Fixed

- **A science lab's contents appear in no observation channel**
  Found 2026-07-31.
  Files: `factoriax/engine/observations.py:121` (`is_combiner`), `:124`
  (`is_buffer_machine`).
  The symbolic observation writes `ent_asm_in` into slots 0 and 1 for
  combiners and `ent_buf` into slot 2 for buffer machines. `is_combiner` names
  `ASSEMBLER` and `FURNACE`; `is_buffer_machine` names `MINER`, `PALLET`, and
  `CONVEYOR_BELT`. Two kinds are in neither set: `SCIENCE_LAB`, and `CROSSING`,
  which keeps a per-axis buffer in the same two `ent_asm_in` columns a combiner
  uses. Every slot channel reads zero whatever they hold. An agent cannot see whether a lab has packs, is
  running, or is starved, which makes the science loop unobservable from the
  symbolic profile. The fix is to add labs to `is_combiner`, since they use
  the same two input slots; it changes no shape, only which values are
  non-zero.
  Fixed 2026-07-31. Slots 0 and 1 now carry the two `ent_asm_in` columns for
  anything that stores items there, which covers science labs and crossings,
  and slot 2 carries `ent_buf` for every machine with a non-zero
  `MACHINE_MAX_STACK`, which adds splitters and arms. Covered by
  `TestSlotGridsSeeEveryMachine`.

- **"Combiner" has no single definition, so the sets have drifted apart**
  Found 2026-07-31.
  Files: `factoriax/engine/machines.py` (`self_is_combiner`,
  `dst_is_combiner`, `dn_is_combiner`, `is_combiner`),
  `factoriax/engine/step.py` (`deposit_to_adjacent`),
  `factoriax/engine/observations.py:121`.
  Six sites spell out "combiner" as an inline `|` chain over machine types,
  under two different meanings: "has two input slots" and "crafts a recipe".
  Nothing ties them together. Of the five sites meaning two input slots, three
  include `SCIENCE_LAB` (the two arm sites and the player deposit) and two do
  not (the belt refusal and the observation), which is exactly the two entries
  above. The crafting site correctly excludes labs.
  A new machine kind with input slots has to be added to five separate
  expressions, and omitting one fails silently. Naming the predicate once,
  next to `MACHINE_MAX_STACK` in `tables.py`, would make the two defects above
  impossible rather than merely unlikely.
  Fixed 2026-07-31. The word is gone from the codebase. The shared set is now
  `MACHINE_HAS_INPUT_SLOTS` in `factoriax/engine/tables.py`, read by the arm
  delivery, the player deposit, and the observation projection. The crafting
  set is spelled `is_assembler_or_furnace` at its one use site, since a
  science lab shares the slots but runs no recipe.

- **Picking up a machine destroys everything in its crafting slots**
  Found 2026-07-31.
  Files: `factoriax/engine/placement.py` (`pickup_machine`).
  The pickup pays the machine item back and adds `ent_buf` to the player's
  inventory, then zeroes `ent_asm_in` and `ent_asm_out` without paying either
  out. Picking up an assembler therefore discards both what was loaded into it
  and what it had finished. Reproduced with an assembler holding 5 `IRON_ORE`
  in input slot 0 and 4 `IRON_PLATE` in its output: after `pickup_machine` the
  player holds 1 assembler, 0 iron ore, and 0 iron plate.
  Fixing it means paying out all three slots, which needs a decision about what
  happens when the player has no room, since the gate does not check that
  today. See the entry below.
  Fixed 2026-07-31. `pickup_machine` now totals a per-item payout across the
  machine item, `ent_buf`, both `ent_asm_in` slots, and `ent_asm_out`, and pays
  the whole thing out. Covered by `TestPickupReturnsEveryStoredItem` in
  `tests/test_engine_contracts.py`.

- **Pickup pays buffer contents past `PLAYER_MAX_STACK`**
  Found 2026-07-31.
  Files: `factoriax/engine/placement.py` (`pickup_machine`).
  The `fits` gate checks room for the machine item only. Buffer contents are
  added afterwards with no cap, so emptying a full pallet can push a stack
  above its per-item limit. Reproduced with the player holding 1023 coal
  (`PLAYER_MAX_STACK` is 1024) and a pallet holding 250: the inventory ends at
  1273, 249 over. Every other transfer path in the engine clamps to the cap.
  Fixed 2026-07-31. The gate now checks the whole payout against
  `PLAYER_MAX_STACK` per item and refuses the pickup outright if any stack
  would overflow, rather than paying out what fits and destroying the rest
  with the machine. A full inventory can therefore block a pickup; the player
  withdraws first. Covered by `TestPickupRespectsPlayerStack`.

- **A player deposit ignores `MACHINE_MAX_STACK`**
  Found 2026-07-31.
  Files: `factoriax/engine/step.py` (`deposit_to_adjacent`, the `buf_space`
  term), `factoriax/engine/tables.py` (`MACHINE_MAX_STACK`).
  The buffer branch tests `ent_buf_count < 64`, a literal, rather than the
  machine's own capacity. Every machine-driven transfer in
  `factoriax/engine/machines.py` respects the table and this one does not.
  Reproduced with 10 successive deposits: a `CONVEYOR_BELT` that holds 3 ends
  at 10, a `CROSSING` that holds 2 ends at 10, an `ARM` that holds 1 ends at
  10, and a `ROCKET` that holds nothing ends at 10. Items pushed into a rocket
  this way are unreachable by every machine pass.
  Fixed 2026-07-31. Both deposit routes read `MACHINE_MAX_STACK` for the target
  machine, and a machine whose capacity is 0 refuses every item. Covered by
  `TestDepositRespectsMachineCapacity`.

- **`dense_withdraw_reward` cannot see a target in the bottom-right corner**
  Found 2026-07-31.
  Files: `factoriax/engine/rewards.py` (`dense_withdraw_reward`).
  The proximity mask is built with
  `has_output_grid.at[ent_y, ent_x].set(has_output)` over every entity slot.
  A free slot holds `ent_y == ent_x == -1`, which Python indexing wraps to the
  last row and column rather than clipping, so every free slot writes False
  onto the bottom-right tile. Duplicate scatter indices resolve in an
  unspecified order and the real write loses. Reproduced on a 3x3 map with the
  player at (0, 0), a miner holding 3 coal at (2, 2), and 63 free slots: the
  reward is 1/7, the no-target floor, where the Manhattan distance of 4 gives
  1/5. Same shape as the entity-array scatters in `machines.py`.
  Fixed 2026-07-31. Positions are clipped before the scatter, and the mask is
  built with `.at[].max(...)` so a free slot's False cannot overwrite a real
  machine's True. Covered by `TestWithdrawRewardSeesTheCorner`.

- **A machine on tile (0, 0) is invisible in the x_ray slot channels**
  Found 2026-07-31.
  Files: `factoriax/engine/observations.py` (`_reconstruct_slot_grids`,
  `_reconstruct_machine_direction_grid`).
  Both scatter with `.at[ey, ex].set(...)` at clipped positions. Every free
  entity slot clips onto (0, 0) carrying a zero, so that tile collects one real
  write and many stale ones, and duplicate scatter indices resolve in an
  unspecified order. Reproduced on a 2x2 map with a pallet holding 50 coal at
  (0, 0) and 63 free slots: `slot2_type` and `slot2_count` both read 0, and the
  direction channel reads 0. Same shape as the two entries above.
  Fixed 2026-07-31. Both grids accumulate with `.at[].add(...)` instead of
  assigning, so the zeros every free slot contributes to tile (0, 0) leave a
  real machine there intact. Covered by `TestSlotGridsSeeEveryMachine`.

- **Facing an empty tile reports entity 0's contents**
  Found 2026-07-31.
  Files: `factoriax/engine/observations.py` (`_facing_scalars`).
  The readouts index the `ent_*` arrays at
  `clip(tile_entity[sy, sx], 0, max_e - 1)`, and `tile_entity` is -1 on a tile
  with no machine, so the clip turns that into entity 0. The only mask applied
  is `in_bounds`, which does not cover it. Reproduced with a player facing an
  empty in-bounds tile while entity 0 held 50 coal: `facing_buffer_type` reads
  0.0294 and `facing_buffer_count` reads 0.781, both of which should be 0.
  `facing_machine_type` correctly reads 0, so the nine-float block contradicts
  itself. An x_ray agent sees phantom contents on every empty tile it faces.
  Fixed 2026-07-31. The mask is now `in_bounds & (tile_entity >= 0)`, so an
  empty tile reads zero on every field. Covered by
  `TestFacingScalarsNeedAMachine`.

- **The declared observation bounds do not hold for the x_ray profile**
  Found 2026-07-31.
  Files: `factoriax/engine/envs/base.py` (`observation_space`),
  `factoriax/engine/observations.py` (`_facing_scalars`).
  `observation_space` returns a `Box` with `low=0.0, high=1.0`. The facing
  readouts divide counts by 64 while the slots they read hold far more: a
  pallet holds 256, so facing a full one puts that field at 4.0, and a combiner
  input slot has no cap at all. Training code that trusts the declared bounds,
  for normalisation or for clipping, is working from a wrong number.
  Fixed 2026-07-31. The facing readouts divide by `_SLOT_COUNT_NORM` rather
  than 64, which is at or above every machine's `MACHINE_MAX_STACK`, so the
  values stay inside the declared `Box`. This changes the numeric observation
  and invalidates checkpoints trained on the old scaling. Covered by
  `TestObservationsStayInDeclaredBounds`.

- **A combiner input slot has no capacity**
  Found 2026-07-31.
  Files: `factoriax/engine/machines.py` (`dst_combiner_accepts` in `run_arms`),
  `factoriax/engine/step.py` (`can_deposit_asm` in `deposit_to_adjacent`).
  Both routes into `ent_asm_in` test the item type and never the count, and
  `MACHINE_MAX_STACK` is read on neither path. An arm feeding an assembler that
  is not consuming piles items into one slot until int16 overflows; measured at
  8 after 8 arm steps with no sign of a limit. Whether a cap is wanted is a
  design decision, but its absence was undocumented until this pass.
  Fixed 2026-07-31. Both routes into `ent_asm_in`, the arm delivery in
  `run_arms` and the player deposit, now test the slot count against the
  machine's `MACHINE_MAX_STACK`. Covered by `TestInputSlotCapacity`.

- **`state_factory` builds machines at zero health, so pickup tests pass vacuously**
  Found 2026-07-31.
  Files: `tests/conftest.py` (`ent_health` seeding in `state_factory`),
  `factoriax/engine/placement.py` (`pickup_machine` health gate).
  The fixture leaves `ent_health` at 0 for every machine it places, while
  `place_machine` in the real engine always sets `MACHINE_MAX_HEALTH`. Pickup
  is gated on full health, so a machine built by the fixture can never be
  picked up, and a test asserting on a pickup silently exercises the refused
  path instead. Found while probing `pickup_machine`: the first two probes
  returned 0 for every field, including the machine item. Same shape as the
  `max_machines=1` fixture already recorded under **Correctness review** in
  `DOCUMENTATION_PLAN.md`. Seeding full health will change what some existing
  tests assert.
  Fixed 2026-07-31. The fixture seeds `MACHINE_MAX_HEALTH` for every machine it
  places, matching `place_machine`, and leaves free slots at 0. Covered by
  `TestFixtureSeedsFullHealth`.
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
