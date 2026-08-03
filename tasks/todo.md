# Todo: make the test suite mirror the package

Full detail is in `tasks/plan.md`. Each task there gives its steps, its
done-when list, and its verify command.

## State

Phase 0 is complete, in commit `c670a22`. Phase 1 is complete, in commits
`89e5785` and `9910499`.

- The suite holds **1447 tests**. All pass. A full run takes about 220 seconds.
- 38 loose test files remain at the `tests/` root. `tests/bench_rollout.py`
  sits beside them and holds no test.
- `tests/analysis/` and `tests/assets/` mirror the package. Every file in
  them names a module.
- Three directories do not mirror it yet: `tests/editor/`, `tests/play/`,
  `tests/scenarios/`.
- `tests/helpers/` holds `states.py` and `trajectories.py`.
- `uv run ruff check factoriax tests` exits 0.

## Verify each move

Record the baseline once per session. The count is 1447:

```bash
uv run pytest --collect-only 2>/dev/null | grep -E '\.py::' \
  | sed -E 's|^.*\.py::||' | sort > /tmp/factoriax-test-baseline.txt
wc -l < /tmp/factoriax-test-baseline.txt   # must print 1447
```

After each task in phases 1 to 4, run the same pipeline. Then `diff` it
against the baseline. The output must be empty. Also run `uv run pytest`. It
must exit 0 with 1447 tests passed.

Use `git mv` for a whole-file move. To split a file, `git mv` it to the
destination of its largest part first. Then carve out the rest.

## Phase 0: foundation (done)

- [x] **Task 1** Delete `baselines/` and `tests/baselines/`. Correct the stale
      `tests/conftest.py` docstring.
- [x] **Task 2** Change the ruff ignore pattern to `tests/**/*.py`. Correct the
      six ruff errors behind it. Add the mypy override for `tests.*`. Update
      `README.md`.
- [x] **Task 3** Create `tests/helpers/states.py`. Cut the factory body out of
      the root conftest. Move the pygame session out of suite-wide autouse and
      into `tests/play/conftest.py` and `tests/editor/conftest.py`. Make the
      factory's `**_kwargs` catch-all raise.
- [x] **Checkpoint** 1447 tests, exit 0. Ruff exits 0. No dead directory
      remains.

Task 3 changed the count from 1448 to 1447. The catch-all had hidden two
retired `EnvState` field names. Six tests move to the entity model and one
test is deleted, because the field and the branch it named are both gone.
`ISSUES.md` records that defect, the dead `ItemType` typing, and the slow
science tiers fixture.

## Phase 1: analysis and assets

- [x] **Task 4** Dissolve `tests/test_analysis.py`, which held 78 tests in 977
      lines, into `tests/analysis/`. Fold `test_state_shapes.py` into
      `test_state.py`. Split `test_palettes.py` by the module that owns each
      constant. Move `tests/test_inventory_panel.py` to
      `tests/analysis/test_inventory.py`.
- [x] **Task 5** Move `tests/test_atlas_fresh.py` to
      `tests/assets/test_build_atlas.py`.
- [x] **Checkpoint** 1447 tests, exit 0, empty diff.

`test_palettes.py` did not test `analysis/categories.py`, whatever its name
suggested. Its classes cover palette constants that `actions.py` and
`state.py` own, so they follow those two modules. `analysis/categories.py`,
`analysis/eval.py`, and `analysis/video.py` have no test file. Task 20 records
the three gaps.

## Phase 2: playground

- [ ] **Task 6** (M) Build `tests/playground/`. Move `test_scaling.py` and
      `test_machine_icon_coverage.py` into `ui/`. Merge `play/test_config.py`
      and `play/test_controller_config.py` into `test_config.py`. Move
      `play/test_rebinding.py` to `menu/test_controls_menu.py`. Create
      `tests/playground/conftest.py` with the autouse display fixture that
      `tests/play/conftest.py` and `tests/editor/conftest.py` hold today.
- [ ] **Task 7** (M) Build `tests/playground/editor/`. Split
      `test_editor_state.py`, which holds 77 tests in 743 lines, into a
      `state/` package. Rename `test_editor_canvas.py` to `test_canvas.py` and
      `test_editor_save_load.py` to `test_dialogs.py`. Delete
      `tests/editor/conftest.py`, because Task 6 replaced it.
- [ ] **Task 8** (L) Build `tests/playground/play/`. Split `test_ui.py`,
      `test_scroll.py`, and `test_machine_ui.py` into a `ui/` package. Merge
      `test_action_dispatch.py` into `test_game_ui.py`. Rename
      `test_free_play_achievements.py` to `test_achievements.py`. Send
      `test_game_ui_integration.py` and `test_smoke.py` to `integration/`, and
      `test_engine_agreement.py` and `test_no_direct_state_edits.py` to
      `contracts/`. Delete `tests/play/conftest.py`. If it runs long, split at
      the `ui/` package.
- [ ] **Checkpoint** 1447 tests, exit 0, empty diff. `tests/play/` and
      `tests/editor/` are gone.

## Phase 3: engine

- [ ] **Task 9** (L) Move the engine leaf modules. Merge the two category files
      into `test_constants.py`, and `test_belt_helpers.py` with
      `test_machine_config.py` into `test_tables.py`. Merge the two reward
      files. Dissolve `test_factoriax.py` across 6 destinations. Move
      `canonical_env_8x8_1p` from the root conftest to
      `tests/engine/conftest.py`, with its three consumers. If it runs long,
      split at `test_factoriax.py`.
- [ ] **Task 10** (L) Split seven files into `tests/engine/machines/`, by
      machine kind. They hold 117 tests in about 2600 lines.
- [ ] **Task 11** (M) Split `test_step.py`, `test_mining.py`, and
      `test_deposit_withdraw.py` into `tests/engine/step/`.
- [ ] **Task 12** (M) Split `test_observations.py` and
      `test_observations_superficial.py` into `tests/engine/observations/`, by
      observation profile.
- [ ] **Task 13** (L) Build `tests/engine/levels/` from `test_levels.py`, which
      holds 85 tests in 19 classes. Build `tests/engine/recipes/` from the
      three recipe files.
- [ ] **Task 14** (L) Build `tests/engine/envs/`. Split `test_env_hooks.py`
      across `test_base.py` and `test_wrappers.py`. Fold in
      `test_action_mask_wrapper.py` and `test_coal_capacity.py`.
- [ ] **Checkpoint** 1447 tests, exit 0, empty diff. `ls tests/*.py` prints
      `conftest.py` and nothing else.

## Phase 4: contracts, integration, benchmarks

- [ ] **Task 15** (M) Build `tests/contracts/` from six files. Each module
      docstring states the rule the file guards, and why the file is not in the
      mirror.
- [ ] **Task 16** (M) Build `tests/integration/` from five files. Each module
      docstring names the subpackages the file spans. Each file that renders
      requests `pygame_display` by name.
- [ ] **Task 17** (L) Move `tests/scenarios/` to `tests/integration/scenarios/`.
      Move `oracle_utils.py` to `tests/helpers/oracles.py` and update the two
      importers.
- [ ] **Task 18** (S) Move `bench_rollout.py` to `tests/benchmarks/` and keep
      it out of collection. Sweep the tree for stray files and empty
      directories.
- [ ] **Checkpoint** 1447 tests, exit 0. The diff against the Phase 0 baseline
      is empty: 15 tasks moved every test and lost none. A human reviews the
      result before any deletion starts.

## Phase 5: deduplicate

The test count drops from here. That is the purpose of the phase.

- [ ] **Task 19** (L) Delete the nine recorded duplicates. Verify each
      subset claim against the moved code first. Move any unique assertion to
      the survivor. Coverage must not drop.
- [ ] **Task 20** (S) Add a "Tests" section to `README.md` with the path rule,
      the 600-line package threshold, and the three bucket rules. Write
      `tasks/coverage-gaps.md`. Set `fail_under` to the measured figure, which
      reads a stale 64 today.
- [ ] **Task 21** (S, separable) Write `tests/engine/test_crafting.py`.
      `engine/crafting.py` is 182 lines of engine core with no direct test.
      This is the one task that writes new tests. Drop it to keep the work
      structural.
- [ ] **Checkpoint** The suite is green. Coverage is at or above the gate. The
      layout is recorded and the gaps are listed.

## Follow-up, out of scope by request

- [ ] A layout guard test, to make the path rule self-enforcing.
- [ ] Pytest markers with `--strict-markers`, for a fast `-m "not slow"` loop.
- [ ] A CI workflow. There is no `.github/` directory today.
- [ ] The three open entries in `ISSUES.md` that this work surfaced: the dead
      craft progress bar, the functional `IntEnum` that blocks mypy, and the
      58.7-second science tiers fixture. None blocks the restructure.
