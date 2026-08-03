# Todo: make the test suite mirror the package

Full detail is in `tasks/plan.md`. Each task there gives its steps, its
done-when list, and its verify command.

## State

Phases 0 to 4 are complete. Only Phase 5 remains.

- Phase 0: `c670a22`
- Phase 1: `89e5785`, `9910499`
- Phase 2: `9dd25dc`, `2cbfd06`, `124c6dd`
- Phase 3: `6e9ed51`, `ee51be0`, `31090e9`, `0316324`, `d2750c1`, `6d1c8f3`
- Phase 4: `3228927`, `4543607`, `fe4aa12`

- The suite holds **1447 tests**. All pass. A full run takes about 220 seconds.
- `ls tests/*.py` prints `conftest.py` and `__init__.py`. Nothing else is
  loose.
- `tests/engine/`, `tests/playground/`, `tests/analysis/`, and `tests/assets/`
  mirror the package. All 77 files in them map to a module. The longest is
  553 lines.
- `tests/integration/` holds 7 files plus `scenarios/` with 6.
  `tests/contracts/` holds 6. `tests/benchmarks/` is not collected.
- `tests/helpers/` holds `states.py`, `trajectories.py`, `observations.py`,
  and `oracles.py`.
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

- [x] **Task 6** Build `tests/playground/`. `config.py` needed a package: its
      three test files ran to 710 lines together. `test_rebinding.py` split
      three ways, by what each class reaches for.
- [x] **Task 7** Build `tests/playground/editor/`. `test_editor_state.py`
      became a `state/` package of five topic files. The save-load round trip
      inside `test_editor_save_load.py` went to `tests/integration/`.
- [x] **Task 8** Build `tests/playground/play/`. Three files against `ui.py`
      became a `ui/` package. Four files left the mirror for `integration/`
      and `contracts/`.
- [x] **Checkpoint** 1447 tests, exit 0, empty diff. `tests/play/` and
      `tests/editor/` are gone.

Moving a file out from under `tests/playground/` takes it out from under the
autouse display. Two files then passed in a full run and failed alone,
because they had been riding on a display that an earlier directory
initialised. **Run each moved file on its own.** Neither the node ID diff nor
a full-suite run finds this.

## Phase 3: engine

- [x] **Task 9** Move the engine leaf modules and dissolve `test_factoriax.py`
      across 6 destinations. `test_machine_config.py` needed four destinations,
      not two.
- [x] **Task 10** Split six files into `tests/engine/machines/`, by machine
      kind.
- [x] **Task 11** Split four files into `tests/engine/step/`. Deposit and
      withdraw came in as one 646-line file and split in two.
- [x] **Task 12** Split two files into `tests/engine/observations/`, by profile
      and encoder.
- [x] **Task 13** Build `tests/engine/levels/` and `tests/engine/recipes/`.
- [x] **Task 14** Build `tests/engine/envs/`.
- [x] **Checkpoint** 1447 tests, exit 0, empty diff.

Two plan errors surfaced here. `test_science_lab.py` imports `run_labs` from
`engine/step.py`, not from `engine/machines.py`, so it joined the step package.
`test_machine_config.py` split four ways: golden values in `tables.py`, the
editor slot view, a cross-subpackage agreement, and a guard on a deleted
module.

`canonical_env_8x8_1p` stays in the root conftest. Its consumers ended up
spread across `tests/engine/` rather than in one file, so a package conftest
buys nothing.

The checkpoint is not `ls tests/*.py` yet. Seven root files remain and Phase 4
places them.

## Phase 4: contracts, integration, benchmarks

- [x] **Task 15** Build `tests/contracts/` from six files.
- [x] **Task 16** Build `tests/integration/` from five files.
- [x] **Task 17** Move `tests/scenarios/` to `tests/integration/scenarios/` and
      `oracle_utils.py` to `tests/helpers/oracles.py`.
- [x] **Task 18** Move `bench_rollout.py` to `tests/benchmarks/` and sweep.
- [x] **Checkpoint** 1447 tests, exit 0, empty diff. Every test survived the
      move phases. A human reviews before any deletion starts.

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
