# Plan: make the test suite mirror the package

## Context

The `factoriax/` package has four subpackages: `engine/`, `playground/`, `analysis/`, and `assets/`. The `tests/` tree does not mirror this structure. Today `tests/` holds 42 loose files at its root, plus four subdirectories that grew one at a time.

The two halves overlap. `tests/test_analysis.py` holds 78 tests in 977 lines. It covers the same modules as the `tests/analysis/` package beside it. `tests/test_editor_roundtrip.py` sits at the root, but `tests/editor/` exists. The name `tests/test_mining.py` points at `engine/envs/mining.py`, but the file tests `engine/step.py::mine_block`.

All 1448 tests pass today. Nothing is broken. But a reader cannot answer two questions without a grep. Where are the tests for `machines.py`? Where does a new test go?

The suite also holds rot:

- The root `conftest.py` docstring names three fixtures and one spec file. None of them exist.
- An autouse pygame display session runs for all 1448 tests. Only 11 files import pygame.
- Two directories are empty residue of the deleted `baselines/` tree.
- A benchmark script sits in `tests/`, but it holds no test.
- Six ruff errors sit in `tests/`. Ruff never reads that directory.
- The ruff per-file-ignore pattern `tests/*.py` misses every nested directory.

This plan gives one result. The path `tests/<area>/test_<module>.py` maps to `factoriax/<area>/<module>.py` by construction. Three named siblings hold the tests that belong to no single module. Task 3 settles the count at 1447. It then stays there through the move phases. A failure is then a bad move, never a lost test.

## Decisions

**Mirror at the top level.** The directories `tests/engine/`, `tests/playground/`, `tests/analysis/`, and `tests/assets/` mirror the package one to one. The directories `integration/`, `contracts/`, and `benchmarks/` are siblings of the mirror, not children of it.

**A mirror entry is a file or a package.** The module `factoriax/engine/machines.py` maps to `tests/engine/test_machines.py` or to `tests/engine/machines/`. The threshold is 600 lines. Above it, the mirror entry becomes a package of topic-named files. This rule keeps the 2600 lines of tests for `machines.py` readable.

**Move first. Deduplicate last.** Phases 1 to 4 only move and split files. Phase 5 deletes duplicates. Count conservation is the safety net for phases 1 to 4.

**Three buckets, each defined by what it is not:**

- `integration/` holds a test that needs two or more subpackages to have meaning. One example is the round trip from editor state to `Level` to engine state.
- `contracts/` holds a test that asserts a rule no single module owns. Two examples are gymnax API conformance and the AST guard that stops the play UI from writing `EnvState`.
- `benchmarks/` holds scripts. Pytest does not collect this directory.

**Lint and typecheck the tests.** Ruff runs against `factoriax/` only today. This plan adds `tests/`.

Three guardrails are out of scope by request: a layout guard test, pytest markers, and a CI workflow. The last section lists them as follow-up work.

## Target tree

```
tests/
  conftest.py              # SDL and JAX pins, state_factory, pygame_display,
                           # canonical_env_8x8_1p
  helpers/                 # support code, never collected
    states.py              # the factory body, plus the grid builders
    trajectories.py        # trajectory builders for the analysis tests
    observations.py        # map size, radius, channel slicer
    oracles.py             # was tests/scenarios/oracle_utils.py (Task 17)
  engine/                  <- factoriax/engine/
    test_achievements.py  test_actions.py  test_constants.py
    test_crafting.py  test_placement.py  test_renderer.py
    test_rewards.py  test_state.py  test_tables.py
    levels/  machines/  observations/  recipes/  step/
    envs/                  <- factoriax/engine/envs/
      conftest.py          # level8, params
      test_base.py  test_easy_rocket.py  test_registry.py
      test_rocket.py  test_wrappers.py
  playground/              <- factoriax/playground/
    conftest.py            # pygame display and font, autouse here only
    config/  editor/  menu/  play/  ui/
  analysis/                <- factoriax/analysis/
    conftest.py            # matplotlib Agg backend, unchanged
    test_actions.py  test_build_progression.py  test_categories.py
    test_curriculum_strip.py  test_graph_layout.py  test_inventory.py
    test_milestones.py  test_recipe_graph.py  test_recorder.py
    test_state.py  test_trajectory.py  test_utils.py
  assets/                  <- factoriax/assets/
    test_build_atlas.py
  integration/
    scenarios/
  contracts/
  benchmarks/              # bench_rollout.py
```

`engine/envs/` has no entry for `common.py`, `miner_curriculum.py`,
`mining.py`, or `science_tiers.py`. Their tests are end-to-end scenario
rollouts, so they live under `integration/scenarios/`. `ISSUES.md` records
that, with the rest of the coverage gaps.

## How to verify a move

Each move task uses a node ID multiset that does not depend on the file path. Record the baseline one time, before Task 1:

```bash
uv run pytest --collect-only 2>/dev/null | grep -E '\.py::' \
  | sed -E 's|^.*\.py::||' | sort > /tmp/factoriax-test-baseline.txt
wc -l < /tmp/factoriax-test-baseline.txt   # must print 1447 after Task 3
```

After each move task, run the same pipeline. Then run `diff` against the baseline. The output must be empty.

This method finds three faults that a green run hides: a dropped file, a lost class, and a test that pytest stopped to collect.

The second gate is a green suite. The command `uv run pytest` must exit 0 with 1447 tests passed.

Use `git mv` for a whole-file move, so that `git blame` survives. To split a file, first `git mv` it to the destination of its largest part. Then carve out the remainder in a second commit.

---

## Phase 0: foundation

### Task 1: delete the dead trees and correct the conftest docstring

**Goal:** Two directories are residue of commit `edfd195 "Remove the baselines tree"`. The `tests/conftest.py` docstring names a fixture catalogue that is mostly fiction.

**Steps:**

1. Delete `baselines/`. It holds 30 files, all `.pyc`, and git tracks none of them.
2. Delete `tests/baselines/`. It holds only `__pycache__`.
3. Correct the `tests/conftest.py` docstring. It points at `tests/scenarios/conftest.py`, at `tests/scenarios/skills/test_skills_scripted_solves.py`, and at `SPEC_TEST_SUITE.md`. None of these exist.

**Done when:**

- [x] `baselines/` and `tests/baselines/` are gone.
- [x] The `tests/conftest.py` docstring names only fixtures that exist.
- [x] `.gitignore` covers `__pycache__/`.

**How to verify:** `uv run pytest` collects 1448 tests and exits 0. The command `find . -name '*.pyc' -path '*baselines*'` prints nothing.

**Depends on:** nothing. **Files:** 3. **Size:** XS.

### Task 2: make the tooling cover the tests

**Goal:** Ruff reads `factoriax/` only. Its per-file-ignore pattern `tests/*.py` does not match `tests/play/*.py` or any other nested directory. Six ruff errors sit in `tests/` today. Mypy is strict and reports 1400 errors for `tests/`.

**Steps:**

1. Change the ruff per-file-ignore pattern to `tests/**/*.py`.
2. Correct the six ruff errors. They are one `F401`, four `F841`, and one `I001`.
3. Add a mypy override that relaxes `disallow_untyped_defs` for `tests.*`.
4. Change the lint line in `README.md` to `uv run ruff check factoriax tests`.

The six errors are at `tests/scenarios/test_rocket_scenario.py:13`, `tests/test_coal_capacity.py:28`, `tests/test_coal_capacity.py:61`, `tests/test_editor_inventory_init.py:55`, `tests/test_placement.py:355`, and `tests/test_inventory.py:3`.

**Done when:**

- [x] `uv run ruff check factoriax tests` exits 0.
- [x] The override removes the test-shaped mypy noise.
- [x] `README.md` names both directories in the lint command.

Mypy cannot become a gate here. `ItemType` is a functional `IntEnum`, so mypy resolves none of its members. That one cause raises 1182 of the 1268 errors that `mypy tests` reports. It affects `factoriax/` the same way. `ISSUES.md` records it. After the override, `tests/` holds 54 real errors, down from 121.

**How to verify:** Run the two commands above. Then run `uv run pytest`. It stays green at 1448.

**Depends on:** nothing. Task 1 and Task 2 can run in parallel. **Files:** 7. **Size:** S.

### Task 3: build `tests/helpers/` and move each fixture to its correct scope

**Goal:** The root `conftest.py` is 425 lines. The `state_factory` body is about 250 of them. The file imports `jax` and `pygame` at module scope. Its autouse session fixture calls `pygame.display.init()` and `set_mode((800,600))` for all 1448 tests, but only 11 files touch pygame. The file `tests/play/conftest.py` repeats that same init.

**Steps:**

1. Move the factory body to `tests/helpers/states.py`.
2. Keep a thin `state_factory` fixture in the root conftest. 33 files consume it.
3. Keep the `os.environ.setdefault` calls for SDL and `JAX_PLATFORMS` at root conftest module scope. They must run before any pygame import.
4. Delete the `pygame` import from the root conftest.
5. Define a session fixture `pygame_display` at the root. Make it autouse under `tests/playground/` only.
6. Replace the duplicate init in `tests/play/conftest.py` with an autouse
   fixture that requests `pygame_display`. Add the same file to
   `tests/editor/`. Make `tests/test_scaling.py` request the fixture by name.
7. Make the `**_kwargs` catch-all in the factory raise instead of drop. It
   swallows retired `EnvState` field names in silence today. Run the suite and
   correct what fails.

`canonical_env_8x8_1p` stays in the root conftest for now. Its three consumers
are still at the `tests/` root, so it moves to `tests/engine/conftest.py` in
Task 9, with them.

**Done when:**

- [x] `tests/helpers/states.py` holds the factory.
- [x] The root `conftest.py` holds no factory body and no module-scope pygame
      import.
- [x] `pygame_display` runs under `tests/play/` and `tests/editor/` only.

**How to verify:** `uv run pytest` is green. The command
`uv run pytest --setup-show tests/analysis/test_utils.py` names no
`pygame_display` line, and the same command against `tests/play/test_ui.py`
does.

Step 7 changes the node ID set. The catch-all hid two retired field names. Six
tests that passed `machine_inventory=` move to the entity model, and one of
them takes a truer name. One test that passed `craft_progress=` is deleted:
the field is gone from `EnvState`, and the branch it named is dead code in
`ui.py`. `ISSUES.md` records both. **Re-baseline at 1447 after this task.**

**Depends on:** Task 1. **Files:** 8. **Size:** M.

### Checkpoint: foundation

- [x] 1447 tests, exit 0, empty node ID diff.
- [x] `uv run ruff check factoriax tests` exits 0.
- [x] No dead directory remains.
- [x] A human reviews the result before the moves start.

---

## Phase 1: `analysis/` and `assets/`

This area is the smallest and the most self-contained. Its split brain is also the clearest. Phase 1 therefore proves the method at low cost.

### Task 4: dissolve `tests/test_analysis.py` into `tests/analysis/`

**Goal:** `tests/test_analysis.py` holds 78 tests in 977 lines and 6 classes. It covers `analysis.trajectory`, `analysis.actions`, `analysis.state`, `analysis.milestones`, and `analysis.recorder`. The `tests/analysis/` package already holds files for the same modules.

**Steps:**

1. Split `tests/test_analysis.py` one class at a time into the mirror. Merge each class with the file that is already there.
2. Fold `tests/analysis/test_state_shapes.py` into `tests/analysis/test_state.py`.
3. Split `tests/analysis/test_palettes.py` by the module that owns each
   constant. `ACTION_LABELS` and `ACTION_COLORS` live in `analysis/actions.py`,
   so those classes go to `test_actions.py`. `DEFAULT_ITEM_LABELS` and
   `DEFAULT_ITEM_COLORS` live in `analysis/state.py`, so those go to
   `test_state.py`. Then delete the file.
4. Move `tests/test_inventory_panel.py` to `tests/analysis/test_inventory.py`. It tests `analysis/inventory.py`, and the old name hides this.
5. Move the three trajectory builders to `tests/helpers/trajectories.py`. Four
   destination files share them. Keep `FakeRollout` in `test_recorder.py`,
   which is its only consumer.

`analysis/categories.py` has no test file of its own. Only
`test_recipe_graph.py` reaches it, and only through `item_palette`. Task 20
records the gap. Do not write new tests here.

**Done when:**

- [x] The four source files listed above are gone.
- [x] Each file name in `tests/analysis/` matches a module in `factoriax/analysis/`.

**How to verify:** `uv run pytest tests/analysis -q`. The node ID diff is empty.

**Depends on:** Task 3. **Files:** 8. **Size:** M.

### Task 5: create `tests/assets/`

**Goal:** `tests/test_atlas_fresh.py` tests `factoriax/assets/build_atlas.py`. It rebuilds the atlas and compares the bytes against the committed PNG and JSON files.

**Steps:**

1. Move the file with `git mv` to `tests/assets/test_build_atlas.py`.

**Done when:**

- [x] `tests/assets/test_build_atlas.py` exists and `tests/test_atlas_fresh.py` is gone.

**How to verify:** `uv run pytest tests/assets -q`. The node ID diff is empty.

**Depends on:** Task 3. **Files:** 1. **Size:** XS.

### Checkpoint: analysis and assets

- [x] 1447 tests, exit 0, empty node ID diff.
- [x] The method holds. The remaining phases repeat it.

---

## Phase 2: `playground/`

### Task 6: build `tests/playground/ui/`, `test_config.py`, and `menu/`

**Steps:**

1. Move `tests/test_scaling.py` to `tests/playground/ui/test_scaling.py`. Move its `apply_scale` half to `test_theme.py`.
2. Move `tests/test_machine_icon_coverage.py` to `tests/playground/ui/test_icons.py`.
3. Build `tests/playground/config/` as a package from `tests/play/test_config.py`
   and `tests/play/test_controller_config.py`, plus three classes out of
   `test_rebinding.py`. All three files test `playground/config.py`, and
   together they run to about 710 lines, which is over the threshold. The
   package files are `test_keyboard.py`, `test_controller.py`,
   `test_persistence.py`, and `test_env_params.py`.
4. Split `tests/play/test_rebinding.py` three ways, by what each class reaches
   for. `TestBindingFormatting` and `TestRebindActions` are the only two that
   touch `menu/controls_menu.py`, so they go to
   `tests/playground/menu/test_controls_menu.py`. Three classes test binding
   round trips in `config.py` and join the config package.
   `TestGameUIWithReboundKey` builds a `GameUI` and asserts an engine
   `Action`, so it goes to `tests/integration/`.
5. Create `tests/playground/conftest.py` with the autouse pygame session from
   Task 3.

**Done when:**

- [x] Each new file name matches a module in `factoriax/playground/`.
- [x] `tests/playground/conftest.py` holds the autouse pygame session.

**How to verify:** `uv run pytest tests/playground -q`. The node ID diff is empty.

**Depends on:** Task 3. **Files:** 7. **Size:** M.

### Task 7: build `tests/playground/editor/`

**Steps:**

1. Split `tests/editor/test_editor_state.py` into a `state/` package. It holds 77 tests in 743 lines, which is over the threshold. Split it by topic.
2. Rename `test_editor_canvas.py` to `test_canvas.py`.
3. Rename `test_editor_save_load.py` to `test_dialogs.py`. It tests `FileDialog`.
4. Leave `test_level_round_trip.py` in place. Task 16 moves it to `integration/`.

**Done when:**

- [x] Each file name in `tests/playground/editor/` matches a module in `factoriax/playground/editor/`.
- [x] No file in the `state/` package is longer than 600 lines.

**How to verify:** `uv run pytest tests/playground/editor -q`. The node ID diff is empty.

**Depends on:** Task 6. **Files:** 6. **Size:** M.

### Task 8: build `tests/playground/play/`

**Goal:** The module `playground/play/ui.py` is 2203 lines. Three test files cover it, with 67 tests between them.

**Steps:**

1. Split `test_ui.py`, `test_scroll.py`, and `test_machine_ui.py` into a `tests/playground/play/ui/` package.
2. Merge `test_game_ui.py` and `test_action_dispatch.py` into `test_game_ui.py`.
3. Rename `test_free_play_achievements.py` to `test_achievements.py`.
4. Keep the name `test_launch_screen.py`.
5. Move `test_game_ui_integration.py` and `test_smoke.py` to `tests/integration/`.
6. Move `test_engine_agreement.py` and `test_no_direct_state_edits.py` to `tests/contracts/`.

Steps 5 and 6 belong to Phase 4 by topic. Do them now, so that this directory needs one pass and not two.

**Done when:**

- [x] Each file name in `tests/playground/play/` matches a module in `factoriax/playground/play/`.
- [x] `tests/play/` is gone.

**How to verify:** `uv run pytest tests/playground tests/integration tests/contracts -q`. The node ID diff is empty.

**Depends on:** Task 7. **Files:** 10. **Size:** L. If the task runs long, split it at the `play/ui/` package.

### Checkpoint: playground

- [x] 1447 tests, exit 0, empty node ID diff.
- [x] `tests/play/` and `tests/editor/` are gone.

---

## Phase 3: `engine/`

This phase is the bulk of the work. 26 root-level files land here. The order puts the mechanical renames first and the difficult splits last.

### Task 9: move the engine leaf modules

**Steps:**

1. Move `test_achievements.py`, `test_actions.py`, and `test_placement.py` into `tests/engine/` under the same names.
2. Merge `test_item_categories.py` and `test_action_categories.py` into `test_constants.py`. Both test enum partitions in `engine/constants.py`.
3. Merge `test_belt_helpers.py` and `test_machine_config.py` into `test_tables.py`. Both assert golden values in `engine/tables.py`.
4. Merge `test_rewards.py` and `test_rewards_inventory.py` into `test_rewards.py`.
5. Move `test_placement.py::TestWrapperContract` to `envs/test_wrappers.py`.
6. Move `test_placement.py::TestSolidBlocksIsTheSingleSource` to `test_tables.py`.
7. Dissolve `test_factoriax.py`. Send `TestConstants` to `test_constants.py`, `TestEnvStateSchema` to `test_state.py`, `TestRenderer` to `test_renderer.py`, `TestWorldGeneration` and `TestEnvConstructorLevel` to `levels/`, `TestGameLogic` to `step/`, and `TestEnvironment` to `contracts/test_env_api.py`.

**Done when:**

- [x] `tests/test_factoriax.py` is gone, and each of its 7 classes has a recorded destination.
- [x] Each file name in `tests/engine/` matches a module in `factoriax/engine/`.

**How to verify:** `uv run pytest tests/engine -q`. The node ID diff is empty.

**Depends on:** the Phase 2 checkpoint. **Files:** 12. **Size:** L. If the task runs long, split it at `test_factoriax.py`.

### Task 10: build `tests/engine/machines/`

**Goal:** Seven files test `engine/machines.py`. Together they hold 117 tests in about 2600 lines.

**Steps:**

1. Split the files into a package by machine kind: `test_miners.py`,
   `test_belts.py`, `test_arms.py`, `test_splitters.py`, `test_crossings.py`,
   `test_assemblers.py`, `test_pallets.py`, and `test_update_all.py`.
2. Split `test_belt_arm.py` across `test_belts.py` and `test_arms.py`.
3. Send each conservation class in `test_machines.py` to the file for its
   machine kind.
4. Move `test_science_lab.py::test_lab_slot_roles` to
   `tests/playground/editor/test_slot_display.py`. It asserts
   `MACHINE_SLOT_ROLES`, which that file owns.

The six source files are `test_machines.py`, `test_belt_arm.py`,
`test_crossing.py`, `test_splitter.py`, `test_assembler.py`, and
`test_machine_inventory.py`.

`test_science_lab.py` is not one of them. It imports ``run_labs`` from
`engine/step.py`, not from `engine/machines.py`, so it belongs to the step
package that Task 11 builds. The lab is the one machine the step loop drives
directly.

**Done when:**

- [x] All seven source files are gone.
- [x] No file in the package is longer than 600 lines.

**How to verify:** `uv run pytest tests/engine/machines -q`. The node ID diff is empty.

**Depends on:** Task 9. **Files:** 9. **Size:** L.

### Task 11: build `tests/engine/step/`

**Goal:** Three root files and one class from `test_factoriax.py` test `engine/step.py`. Together they hold about 65 tests in about 1350 lines.

**Steps:**

1. Split `test_step.py`, `test_mining.py`, and `test_deposit_withdraw.py` into a package.
2. Name the package files for what they test: `test_player_actions.py` for move, face, and noop, `test_mining.py` for `mine_block`, `test_deposit_withdraw.py`, and `test_step_loop.py` for `factoriax_step`.
3. Add `test_factoriax.py::TestGameLogic` to `test_player_actions.py`.

Note: the root `test_mining.py` does not test `engine/envs/mining.py`. Task 17 handles the real mining scenario test.

**Done when:**

- [x] The three root files are gone.
- [x] Each package file name states what it tests, not where it came from.

**How to verify:** `uv run pytest tests/engine/step -q`. The node ID diff is empty.

**Depends on:** Task 10. **Files:** 5. **Size:** M.

### Task 12: build `tests/engine/observations/`

**Steps:**

1. Split `test_observations.py` (39 tests, 760 lines) and `test_observations_superficial.py` (16 tests) into a package.
2. Name the package files for the observation profiles in `engine/observations.py`: `test_player_scalars.py`, `test_global_x_ray.py`, `test_local_x_ray.py`, `test_rgb.py`, `test_slot_projection.py`, `test_superficial.py`, and `test_equivalence.py`.

**Done when:**

- [x] Both source files are gone.
- [x] Each package file name matches an observation profile.

**How to verify:** `uv run pytest tests/engine/observations -q`. The node ID diff is empty.

**Depends on:** Task 11. **Files:** 7. **Size:** M.

### Task 13: build `tests/engine/levels/` and `tests/engine/recipes/`

**Steps:**

1. Split `test_levels.py` (85 tests, 835 lines, 19 classes) into a `levels/` package: `test_validation.py`, `test_builder.py`, `test_build_state.py`, `test_serialization.py`, `test_registry.py`, `test_generation.py`, and `test_builtin_levels.py`.
2. Split `test_recipes.py`, `test_recipe_book.py`, and `test_recipe_balance.py` (89 tests, 684 lines) into a `recipes/` package: `test_book.py`, `test_table.py`, `test_balance.py`, and `test_shipped_recipes.py`.
3. Put the craft-dispatch tests from `test_recipe_book.py` in `recipes/test_table.py` or in `tests/engine/step/`. These tests import `engine.step`. Pick the file that reads better.

**Done when:**

- [x] Both packages exist.
- [x] No file in either package is longer than 600 lines.

**How to verify:** `uv run pytest tests/engine/levels tests/engine/recipes -q`. The node ID diff is empty.

**Depends on:** Task 12. **Files:** 12. **Size:** L.

### Task 14: build `tests/engine/envs/`

**Steps:**

1. Split `test_env_hooks.py`. Send the `terrain_fn`, step, reset, reward, done, and achievement hooks to `test_base.py`. Send `AutoResetWrapper(resample=True)` to `test_wrappers.py`.
2. Move `test_action_mask_wrapper.py` into `test_wrappers.py`.
3. Move `test_coal_capacity.py` into `test_rocket.py`. It asserts the coal column balance of the rocket level.
4. Move `tests/scenarios/test_scenario_registry.py` to `test_registry.py`.
5. Move the unit-shaped tests out of the `tests/scenarios/` files. One example is `test_easy_rocket_generator.py`, which goes to `test_easy_rocket.py`.
6. Leave the scripted-oracle and solvability tests in place. Task 17 handles them.
7. Leave `test_env_contract.py` in place. Task 15 moves it to `contracts/`, because it is a cross-module API contract.

**Done when:**

- [x] Each file name in `tests/engine/envs/` matches a module in `factoriax/engine/envs/`.
- [x] `test_env_contract.py` is not in this directory.

**How to verify:** `uv run pytest tests/engine -q`. The node ID diff is empty.

**Depends on:** Task 13. **Files:** 10. **Size:** L.

### Checkpoint: engine

- [x] 1447 tests, exit 0, empty node ID diff.
- [x] The command `ls tests/*.py` prints `conftest.py` and nothing else.

---

## Phase 4: `contracts/`, `integration/`, and `benchmarks/`

### Task 15: build `tests/contracts/`

**Goal:** Six files assert rules that no single module owns.

**Steps:**

1. Move `test_env_contract.py` to `test_env_api.py`. Add `test_factoriax.py::TestEnvironment` from Task 9. This file pins gymnax conformance: the declared spaces match, `step_env` returns the 5-tuple, the env is vmappable, and the state dtypes hold.
2. Move `test_engine_contracts.py` to `test_transfer_and_observation.py`. It has the widest import set in the suite and pins defects recorded in `ISSUES.md`.
3. Move `test_invariants.py` under the same name. It asserts properties over random episodes.
4. Move `tests/play/test_engine_agreement.py` to `test_engine_agreement.py`. It asserts that the play UI constants equal the engine enums and tables.
5. Move `tests/play/test_no_direct_state_edits.py` under the same name. It is an AST guard against `state.replace(...)` and `.at[...].set(...)` in the play sources.
6. Move `tests/scenarios/test_achievement_contract.py` to `test_achievement_sets.py`. It runs over every declared achievement set, because the bit order is a wire format.

**Done when:**

- [ ] Each module docstring states the rule that the file guards, and why the file is not in the mirror.
- [ ] `test_no_direct_state_edits.py` runs without an import of JAX or pygame. It is a pure AST scan, and Task 3 makes this possible.

**How to verify:** `uv run pytest tests/contracts -q`. The node ID diff is empty.

**Depends on:** the Phase 3 checkpoint. **Files:** 6. **Size:** M.

### Task 16: build `tests/integration/`

**Goal:** These tests need two or more subpackages to have meaning.

**Steps:**

1. Merge `test_editor_roundtrip.py`, `tests/editor/test_level_round_trip.py`, and `test_editor_inventory_init.py` into `test_editor_level_round_trip.py`.
2. Move `tests/play/test_game_ui_integration.py` to `test_play_ui_engine.py`. It drives every `GameUI` handler against a real `EnvState`.
3. Move `tests/play/test_smoke.py` to `test_playground_smoke.py`.
4. Move `test_achievement_engine.py` to `test_achievement_evaluation.py`. It drives the engine achievement pass with a `playground.play.achievements` set.
5. Move `test_inventory.py` to `test_player_inventory.py`. It spans `state`, `observations`, `renderer`, and `ui.theme`.

**Done when:**

- [ ] Each module docstring names the subpackages that the file spans.
- [ ] Each file that needs a display requests `pygame_display` by name. Task 3 deleted the root autouse fixture.

**How to verify:** `uv run pytest tests/integration -q`. The node ID diff is empty.

**Depends on:** Task 15. **Files:** 8. **Size:** M.

### Task 17: build `tests/integration/scenarios/` and `tests/helpers/oracles.py`

**Goal:** The files in `tests/scenarios/` are end-to-end scripted rollouts, not unit tests of `envs/`.

**Steps:**

1. Move `tests/scenarios/` to `tests/integration/scenarios/`.
2. Move `oracle_utils.py` to `tests/helpers/oracles.py`. It holds navigation primitives that two files share.
3. Change the imports from `tests.scenarios.oracle_utils` to `tests.helpers.oracles`.
4. Drop the `_scenario` suffix from the file names, because the directory now states it. Rename `test_mining_scenario.py` to `test_mining.py` and `test_rocket_scenario.py` to `test_rocket.py`.
5. Fold `test_rocket_level_v2.py` into `test_rocket.py`.
6. Keep the name `test_science_tiers_economics.py`. It is a separate economics oracle.

**Done when:**

- [ ] `tests/scenarios/` is gone.
- [ ] The tests that Task 14 moved to `tests/engine/envs/` do not appear here a second time.

**How to verify:** `uv run pytest tests/integration -q`. The node ID diff is empty.

**Depends on:** Task 16. **Files:** 11. **Size:** L.

### Task 18: build `tests/benchmarks/` and sweep the tree

**Goal:** `tests/bench_rollout.py` holds `bench_single_env` and `bench_batched_env`. It holds no test function, so pytest collects nothing from it.

**Steps:**

1. Move the file to `tests/benchmarks/bench_rollout.py`.
2. Keep the directory out of collection. Use `norecursedirs`, or keep `testpaths` precise.
3. Sweep the tree. Delete every stray file and every empty directory.
4. Add each `__init__.py` that the `tests.helpers` imports need.

**Done when:**

- [ ] The command `ls tests/*.py` prints `conftest.py` and nothing else.
- [ ] Pytest does not collect `tests/benchmarks/`, and `python tests/benchmarks/bench_rollout.py` still runs.
- [ ] The tree matches the target tree above.

**How to verify:** `uv run pytest` is green at 1447. The node ID diff is empty. `uv run ruff check factoriax tests` exits 0.

**Depends on:** Task 17. **Files:** 3. **Size:** S.

### Checkpoint: structure complete

- [ ] 1447 tests and exit 0.
- [ ] The node ID diff against the original baseline is empty. 15 tasks moved every test and lost none.
- [ ] `tests/` matches the target tree.
- [ ] A human reviews the result before any deletion starts.

---

## Phase 5: deduplicate

Phase 5 runs last, when the structure is settled and each duplicate sits next to its twin. The test count drops here. That is the purpose of the phase. Each drop needs a reason in the commit message.

### Task 19: delete the recorded duplicates

**Goal:** The audit found the overlaps in the table below. Each row is a claim that one file is a strict subset of another.

**Steps:**

1. Verify each claim against the moved code before you delete anything.
2. If a duplicate asserts something that the survivor does not, move that assertion to the survivor. Do not drop it.
3. Delete the duplicate.
4. Record the new node ID baseline for later work.

| Duplicate | Survivor |
|---|---|
| `test_step.py::TestCompoundDeposit` and `TestCompoundWithdraw` | `step/test_deposit_withdraw.py` |
| `test_factoriax.py::TestEnvironment` | `contracts/test_env_api.py` |
| `test_factoriax.py::TestGameLogic` movement and facing | `step/test_player_actions.py` |
| `test_recipes.py` unique-input-set checks | `recipes/test_book.py` |
| `test_editor_inventory_init.py`, 1 test | `levels/test_build_state.py` |
| `test_machine_inventory.py::test_output_full_blocks_mining` | `machines/test_miners.py` |
| autoreset tests in `test_env_contract.py` | `envs/test_wrappers.py` |
| inventory in observation, tested 3 times | `observations/test_player_scalars.py` |
| mining, tested 3 times | `step/test_mining.py`, plus the invariant in `contracts/` |

**Done when:**

- [ ] The commit message lists each deletion with its survivor.
- [ ] No assertion is lost.

**How to verify:** `uv run pytest` exits 0. Then run `uv run pytest --cov`. The line coverage does not drop. Coverage is the real gate here, because a deleted duplicate cannot move it.

**Depends on:** the Phase 4 checkpoint. **Files:** 12. **Size:** L.

### Task 20: record the layout and report the gaps

**Steps:**

1. Add a "Tests" section to `README.md`. State the path rule, the 600-line package threshold, and the admission rule for each of the three buckets.
2. Run `uv run pytest --cov`. List each module that has no direct test in `tasks/coverage-gaps.md`.
3. Mark each module in that list as a gap or as waived, with a reason. An entry point and ffmpeg-bound code are fair waivers.
4. Set `fail_under` in `pyproject.toml` to the measured figure. It reads 64 today, which is stale.

The modules with no direct test today are `engine/crafting.py`, `analysis/eval.py`, `analysis/video.py`, `playground/app.py`, `playground/editor/main.py`, `playground/editor/toolbar.py`, `playground/editor/inventory_panel.py`, `playground/play/main.py`, `playground/menu/main_menu.py`, and six modules under `playground/ui/`.

**Done when:**

- [ ] `README.md` states the path rule and the bucket rules.
- [ ] `tasks/coverage-gaps.md` gives a verdict for every module in `factoriax/`.
- [ ] `fail_under` holds the measured figure.

**How to verify:** `uv run pytest --cov` reports a figure at or above the new `fail_under`.

**Depends on:** Task 19. **Files:** 3. **Size:** S.

### Task 21: test `engine/crafting.py`

This task is separable. If you want the work to stay structural, drop it.

**Goal:** `factoriax/engine/crafting.py` is 182 lines of engine core with no direct test. It is the one hole in the mirror that matters.

**Steps:**

1. Write `tests/engine/test_crafting.py`.
2. Cover the public functions of the module.

**Done when:**

- [ ] `tests/engine/test_crafting.py` exists.
- [ ] Coverage of `crafting.py` is above 80 percent.

**How to verify:** `uv run pytest tests/engine/test_crafting.py --cov=factoriax.engine.crafting`.

**Depends on:** Task 20. **Files:** 1. **Size:** S.

### Checkpoint: complete

- [ ] The suite is green.
- [ ] Coverage is at or above the recorded gate.
- [ ] The layout is recorded and the gaps are listed.

---

## Risks

| Risk | Impact | What prevents it |
|---|---|---|
| A split loses a test in silence | High | The node ID diff runs after each task. 1447 is the invariant from Task 3 through Phase 4. |
| A move loses `git blame` | Medium | Use `git mv` for a whole-file move. For a split, `git mv` the largest part first, then carve out the rest in a second commit. |
| The fixture change breaks a test that relied on the root autouse pygame | Medium | Task 3 runs before any move. A failure then appears against the old layout, where its cause is clear. |
| The bucket boundaries drift under judgement calls | Medium | Each file in `integration/` and `contracts/` must state in its docstring why it is not in the mirror. A file that cannot state a reason belongs in the mirror. |
| Five tasks in Phase 3 are L-sized | Medium | Tasks 9, 10, 13, 14, and 17 each name a split point. |

## Notes

The suite holds 1448 tests today. All of them pass. There is no custom marker, no `skipif`, and no CI.

The slowest item by far is `tests/scenarios/test_science_tiers_economics.py`. Its fixture setup takes 58.7 seconds. This plan does not address it. `ISSUES.md` records it, because the `ScienceTiers-v1` scenario is not in active use yet.

Three guardrails are out of scope by request. Each one is a small and independent follow-up:

- A layout guard test. It makes the path rule self-enforcing instead of documented.
- Pytest markers with `--strict-markers`. They give a fast inner loop through `-m "not slow"`.
- A CI workflow. There is no `.github/` directory today, so nothing enforces ruff, mypy, or the coverage gate.
