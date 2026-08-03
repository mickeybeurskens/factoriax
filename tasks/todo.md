# Todo: make the test suite mirror the package

Full detail is in `tasks/plan.md`. Each task there gives its steps, its done-when
list, and its verify command.

Before Task 1, record the baseline:

```bash
uv run pytest --collect-only 2>/dev/null | grep -E '\.py::' \
  | sed -E 's|^.*\.py::||' | sort > /tmp/factoriax-test-baseline.txt
wc -l < /tmp/factoriax-test-baseline.txt   # 1448 now, 1447 after Task 3
```

After each task in phases 0 to 4, run the same pipeline and `diff` it against the
baseline. The output must be empty.

## Phase 0: foundation

- [ ] **Task 1** (XS) Delete `baselines/` and `tests/baselines/`. Correct the stale
      `tests/conftest.py` docstring. Depends on nothing.
- [ ] **Task 2** (S) Change the ruff ignore pattern to `tests/**/*.py`. Correct the
      six ruff errors. Add the mypy override for `tests.*`. Update `README.md`.
      Depends on nothing. Runs in parallel with Task 1.
- [ ] **Task 3** (M) Create `tests/helpers/states.py`. Cut the factory body out
      of the root conftest. Move the pygame session to `tests/play/` and
      `tests/editor/`. Make the factory's `**_kwargs` catch-all raise.
      Re-baseline at 1447. Depends on Task 1.
- [ ] **Checkpoint** 1447 tests, exit 0, empty diff. Ruff exits 0. A human reviews
      the result before the moves start.

## Phase 1: analysis and assets

- [ ] **Task 4** (M) Dissolve `tests/test_analysis.py` into `tests/analysis/`. Fold
      in `test_state_shapes.py`. Rename `test_palettes.py` to `test_categories.py`.
      Move `test_inventory_panel.py` to `test_inventory.py`. Depends on Task 3.
- [ ] **Task 5** (XS) Move `test_atlas_fresh.py` to
      `tests/assets/test_build_atlas.py`. Depends on Task 3.
- [ ] **Checkpoint** 1447 tests, exit 0, empty diff.

## Phase 2: playground

- [ ] **Task 6** (M) Build `tests/playground/ui/`, `tests/playground/test_config.py`,
      and `tests/playground/menu/`. Add `tests/playground/conftest.py`. Depends on
      Task 3.
- [ ] **Task 7** (M) Build `tests/playground/editor/`. Split `test_editor_state.py`
      into a `state/` package. Depends on Task 6.
- [ ] **Task 8** (L) Build `tests/playground/play/`. Split the three `ui.py` test
      files into a `ui/` package. Send four files to `integration/` and
      `contracts/`. Depends on Task 7. If it runs long, split at the `ui/` package.
- [ ] **Checkpoint** 1447 tests, exit 0, empty diff. `tests/play/` and
      `tests/editor/` are gone.

## Phase 3: engine

- [ ] **Task 9** (L) Move the engine leaf modules. Dissolve `test_factoriax.py`
      across 6 destinations. Depends on the Phase 2 checkpoint. Split at
      `test_factoriax.py` if it runs long.
- [ ] **Task 10** (L) Split seven files into `tests/engine/machines/`, by machine
      kind. Depends on Task 9.
- [ ] **Task 11** (M) Split three files into `tests/engine/step/`. Depends on
      Task 10.
- [ ] **Task 12** (M) Split two files into `tests/engine/observations/`, by
      observation profile. Depends on Task 11.
- [ ] **Task 13** (L) Build `tests/engine/levels/` and `tests/engine/recipes/`.
      Depends on Task 12.
- [ ] **Task 14** (L) Build `tests/engine/envs/`. Split `test_env_hooks.py` across
      `test_base.py` and `test_wrappers.py`. Depends on Task 13.
- [ ] **Checkpoint** 1447 tests, exit 0, empty diff. `ls tests/*.py` prints
      `conftest.py` and nothing else.

## Phase 4: contracts, integration, benchmarks

- [ ] **Task 15** (M) Build `tests/contracts/` from six files. Each docstring states
      the rule it guards. Depends on the Phase 3 checkpoint.
- [ ] **Task 16** (M) Build `tests/integration/` from five files. Each docstring
      names the subpackages it spans. Depends on Task 15.
- [ ] **Task 17** (L) Move `tests/scenarios/` to `tests/integration/scenarios/`.
      Move `oracle_utils.py` to `tests/helpers/oracles.py`. Depends on Task 16.
- [ ] **Task 18** (S) Move `bench_rollout.py` to `tests/benchmarks/` and keep it out
      of collection. Sweep the tree. Depends on Task 17.
- [ ] **Checkpoint** 1447 tests, exit 0. The diff against the original baseline is
      empty: 15 tasks moved every test and lost none. A human reviews the result
      before any deletion starts.

## Phase 5: deduplicate

The test count drops from here. That is the purpose of the phase.

- [ ] **Task 19** (L) Delete the nine recorded duplicates. Verify each subset claim
      first. Move any unique assertion to the survivor. Depends on the Phase 4
      checkpoint.
- [ ] **Task 20** (S) Add a "Tests" section to `README.md`. Write
      `tasks/coverage-gaps.md`. Set `fail_under` to the measured figure. Depends on
      Task 19.
- [ ] **Task 21** (S, separable) Write `tests/engine/test_crafting.py`. This is the
      one task that writes new tests. Drop it to keep the work structural. Depends
      on Task 20.
- [ ] **Checkpoint** The suite is green. Coverage is at or above the gate. The
      layout is recorded and the gaps are listed.

## Follow-up, out of scope by request

- [ ] A layout guard test, to make the path rule self-enforcing.
- [ ] Pytest markers with `--strict-markers`, for a fast `-m "not slow"` loop.
- [ ] A CI workflow. There is no `.github/` directory today.
- [ ] The 58.7-second fixture setup in `test_science_tiers_economics.py`. Recorded
      in `ISSUES.md`. The scenario is not in active use, so this waits.
