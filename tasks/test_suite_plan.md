# Implementation Plan: Test-Suite Speed-Up

Spec: `SPEC_TEST_SUITE.md`. Read it first — this plan assumes
familiarity with the analysis, the baseline numbers, the hypothesis,
and the Phase 1 / Phase 2 split.

## Overview

The work is gated. Phase 1 stands up a single dummy A/B experiment
that proves (or kills) the spec's hypothesis: that hoisting
`(env, params, jit_step_fn)` to session scope plus forcing a CPU
backend cuts the full suite's wall time by at least 50%. Phase 2 only
runs if Phase 1 clears its gate. The plan stops cold otherwise — no
production code is touched, no real test is rewritten, no
`pyproject.toml` knobs move until the gate is in.

Phase 1 is four small tasks producing one new file under
`performance_experiments/jit_share/`, one new shell script, and four short timing
runs. Phase 2 is six to eight slices, each a one-module vertical
migration that lands independently with the suite green between
commits. Every Phase 2 task either reduces wall time or sets up the
next reduction; nothing is "scaffolding for later".

## Architecture Decisions

- **Vertical slices, not horizontal layers.** Each migration task
  takes one test module from per-test JIT to shared session-scope
  fixture and lands as a single commit with the suite green. We do
  not "build all the fixtures, then migrate all the tests."
- **Canonical fixture lives in `tests/conftest.py`.** The 8x8
  single-player env + JITted step + reset state is the load-bearing
  shape across the suite. It is hoisted once, reused everywhere.
  Modules that genuinely need a different shape keep building their
  own — the fixture is opt-in.
- ~~**CPU backend pinned in tests, overridable.**~~ Dropped after
  Phase 1: B2-vs-B1 delta was only 4.3%, so the CPU backend is not
  load-bearing for the speedup. The session-scoped fixture is the
  whole mechanism; the suite inherits the developer's default
  backend.
- **`test_jit_retrace.py` stays untouched.** Its job is to *catch*
  retrace bugs by JITting from scratch in every test. We verify it
  still passes after Phase 2 but do not migrate it to the canonical
  fixture.
- **One-module-per-commit migrations.** Smaller diffs surface
  hidden shape-specific assertions early and keep the bisect graph
  clean if a regression slips in.
- **No new pytest plugins.** No `pytest-xdist`, no
  `pytest-benchmark`. The experiment proves the speedup with
  fixture scope + backend alone. Parallelization stays a separate
  future investigation.

## Task List

### Phase 1: Dummy A/B Experiment

The four tasks below produce a written results table at the bottom of
`SPEC_TEST_SUITE.md` and either clear or kill the gate. Total work is
small — roughly half a day end-to-end on the user's box. No
production code is modified.

#### Task 1.1: Stand up the experimental package and Class A

**Description:** Create the `performance_experiments/jit_share/` package, an empty
`conftest.py` placeholder, and `test_jit_share_experiment.py`
containing only `TestBaselineFunctionScope` (the A class). Each of
its ten tests builds its own `FactoriaXEnv`, wraps `env.step_env`
with `jax.jit` locally, and asserts the cheap shape / cache-size /
timestep / reward type checks described in the spec.

**Acceptance criteria:**
- [ ] `performance_experiments/jit_share/__init__.py`,
      `performance_experiments/jit_share/conftest.py`,
      `performance_experiments/jit_share/test_jit_share_experiment.py` exist.
- [ ] `TestBaselineFunctionScope` contains exactly ten test methods,
      all asserting structurally identical things.
- [ ] Every test in the class passes against the engine as-is.
- [ ] No imports from `factoriax/` other than `Action`, `FactoriaXEnv`,
      `EnvParams` (and `random` from `jax`).
- [ ] The file's module docstring states this is an experimental
      harness for `SPEC_TEST_SUITE.md` Phase 1.

**Verification:**
- [ ] `JAX_PLATFORMS=cpu uv run pytest performance_experiments/jit_share/test_jit_share_experiment.py::TestBaselineFunctionScope -q`
      reports 10 passed.
- [ ] `uv run pytest --collect-only -q performance_experiments/jit_share` lists
      ten collected items only.

**Dependencies:** None.

**Files likely touched:**
- `performance_experiments/jit_share/__init__.py` (new)
- `performance_experiments/jit_share/conftest.py` (new, empty placeholder)
- `performance_experiments/jit_share/test_jit_share_experiment.py` (new)

**Estimated scope:** S (3 files, single class).

---

#### Task 1.2: Add the session-scoped fixture and Class B

**Description:** Populate `performance_experiments/jit_share/conftest.py` with a
single session-scoped fixture `canonical_env_8x8_1p` returning the
tuple `(env, params, jit_step_fn, initial_state)`. Add
`TestSharedSessionScope` to `test_jit_share_experiment.py` containing
ten test methods that assert *the same things* as
`TestBaselineFunctionScope` but consume the fixture instead of
building their own env.

**Acceptance criteria:**
- [ ] `canonical_env_8x8_1p` is session-scoped and built with
      `EnvParams(map_width=8, map_height=8, num_players=1)`.
- [ ] The fixture's `jit_step_fn` is a single `jax.jit(env.step_env)`
      shared across all ten tests in `TestSharedSessionScope`.
- [ ] Class B's assertions are line-for-line equivalent to Class A's
      so a side-by-side reviewer can confirm the only delta is
      *where* the env is built.
- [ ] After Class B runs, `jit_step_fn._cache_size() == 1` (proves
      sharing actually worked).

**Verification:**
- [ ] `JAX_PLATFORMS=cpu uv run pytest performance_experiments/jit_share -q` reports
      20 passed.
- [ ] Wall time of `TestSharedSessionScope` alone (via
      `--durations=0`) is markedly less than
      `TestBaselineFunctionScope` alone.

**Dependencies:** 1.1.

**Files likely touched:**
- `performance_experiments/jit_share/conftest.py` (fixture)
- `performance_experiments/jit_share/test_jit_share_experiment.py` (new class)

**Estimated scope:** S (2 files).

---

#### Task 1.3: Build the four-variant runner script

**Description:** Write `scripts/test_speedup_experiment.sh` that runs
the harness four times — CUDA/function, CPU/function, CUDA/session,
CPU/session — and prints a markdown-ready summary line per variant:
wall time, time per test, peak RSS (from `/usr/bin/time -v`), JIT
cache size at end-of-run, pass/fail. The script exits non-zero if any
variant has a failing test.

**Acceptance criteria:**
- [ ] Script lives at `scripts/test_speedup_experiment.sh` with a
      shebang and execute bit.
- [ ] Each variant runs `pytest performance_experiments/jit_share
      -p no:cacheprovider --no-header --no-cov --durations=0
      -o addopts=` with the right env-var prefix and the right
      `-k` filter to select just A or just B.
- [ ] Output is a four-row markdown table the user can paste into
      `SPEC_TEST_SUITE.md` without editing.
- [ ] The script does not write to anywhere other than
      `/tmp/test_speedup_<timestamp>/` and stdout.
- [ ] Script accepts a `--dry-run` flag that prints the four commands
      it would run, without running them.

**Verification:**
- [ ] `bash scripts/test_speedup_experiment.sh --dry-run` prints
      exactly four pytest invocations.
- [ ] `bash scripts/test_speedup_experiment.sh` exits 0 on the user's
      box and emits a four-row table with non-zero seconds in each
      cell.

**Dependencies:** 1.2.

**Files likely touched:**
- `scripts/test_speedup_experiment.sh` (new)

**Estimated scope:** S (1 file).

---

#### Task 1.4: Run the experiment and update the spec

**Description:** Execute the runner, paste the four-row table into
`SPEC_TEST_SUITE.md` under "Phase 1 Results", and write a one-paragraph
verdict that explicitly compares B2 to A1 (must be ≥50% faster) and
B2 to B1 (must be ≥30% faster). If the gate fails, stop and surface
the result to the user — do not silently roll into Phase 2.

**Acceptance criteria:**
- [ ] `SPEC_TEST_SUITE.md` "Phase 1 Results" section is filled in
      with the four-row table, the JIT cache sizes per variant, and
      a yes/no verdict on each gate inequality.
- [ ] The verdict paragraph names the winning variant and quantifies
      the speedup as a percentage.
- [ ] If gate failed, the section ends with "Phase 2 is NOT
      authorised; further investigation needed in directions X / Y."

**Verification:**
- [ ] `git diff SPEC_TEST_SUITE.md` shows only the results section
      filled in — no other edits to the spec.
- [ ] Verdict is unambiguous (no hedging like "roughly" or "close
      to").

**Dependencies:** 1.3.

**Files likely touched:**
- `SPEC_TEST_SUITE.md`

**Estimated scope:** XS (1 file, no code).

---

### Checkpoint: Phase 1 Complete

- [ ] Twenty experimental tests pass under every variant.
- [ ] Spec's "Phase 1 Results" section reflects real numbers from
      the user's hardware.
- [ ] Gate inequalities are evaluated explicitly: B2 vs A1 ≥ 50%,
      B2 vs B1 ≥ 30%.
- [ ] **Human review.** If the gate fails, Phase 2 does not start.
      If it passes, the user explicitly authorises Phase 2.

---

### Phase 2: Repo-Wide Rework (gated on Phase 1)

Each task below is a single vertical slice: foundation + one or more
modules migrated to it, suite green at the end of the commit. Tasks
are ordered so that the gains are visible early — the pilot in 2.2
hits the highest-cost cluster from the top-30 durations table first.

#### Task 2.1: Canonical fixture + centralised pygame init in root conftest

**Description:** Add a session-scoped `canonical_env_8x8_1p` fixture
to `tests/conftest.py` mirroring the one proved in Phase 1.
Centralise pygame display + font init at session scope in the root
conftest; remove the duplicate session fixture in
`tests/test_scaling.py`. Verify no existing test changes behaviour.

*Note: this task no longer pins `JAX_PLATFORMS=cpu`. Phase 1's
B2-vs-B1 result (4.3% delta) showed the CPU backend isn't load-bearing
for the speedup, only the fixture scope is. Dropping the pin keeps
the rollout attributable to a single mechanism and lets developers
use whichever backend they prefer.*

**Acceptance criteria:**
- [ ] `canonical_env_8x8_1p` fixture available to every test in
      `tests/` without per-module conftest copies.
- [ ] `tests/test_scaling.py` no longer defines `_init_pygame`; it
      consumes the root conftest's session fixture.
- [ ] No other test changes its assertions; the full suite still
      passes 1329/1329.
- [ ] Coverage floor `fail_under = 37` not lowered.

**Verification:**
- [ ] `uv run pytest -q` (full suite) reports 1329 passed.
- [ ] `uv run pytest --collect-only -q` collected count unchanged
      (1355).
- [ ] Baseline timing captured for comparison: full suite wall time
      logged in this task's commit message.

**Dependencies:** Phase 1 gate cleared.

**Files likely touched:**
- `tests/conftest.py`
- `tests/test_scaling.py`

**Estimated scope:** S (2 files).

---

#### Task 2.2: Pilot — migrate `test_science_lab.py` and `analysis/test_trajectory.py`

**Description:** These are the highest-signal targets outside of the
already-slow modules: each spends ~6-7s on a `jax.jit(env.step_env)`
that the canonical fixture can serve for free. Migrate the two
integration tests in `TestLabInEnvStep` and both tests in
`TestStatesToTrajectoryWithParams`. Confirm wall time of these
modules drops by at least half.

**Acceptance criteria:**
- [ ] Both `TestLabInEnvStep` integration tests consume
      `canonical_env_8x8_1p` instead of building their own
      `step_fn = jax.jit(env.step_env)`.
- [ ] Both `TestStatesToTrajectoryWithParams` tests share a single
      JITted step path (via the canonical fixture or a
      module-scoped derivative thereof).
- [ ] Pure-kernel `TestRunLabsDelta` tests untouched (they do not
      JIT).
- [ ] All four migrated tests assert the same conditions as before.

**Verification:**
- [ ] `uv run pytest tests/test_science_lab.py tests/analysis/test_trajectory.py -q --durations=0`
      shows total wall time at most half of baseline (baseline:
      ~26s for these four tests).
- [ ] Full suite passes 1329/1329 still.
- [ ] Commit message records before/after seconds for the four
      tests.

**Dependencies:** 2.1.

**Files likely touched:**
- `tests/test_science_lab.py`
- `tests/analysis/test_trajectory.py`

**Estimated scope:** S (2 files).

---

#### Task 2.3: Migrate `test_achievement_engine.py`

**Description:** Two of the slowest tests on the board live here
(`test_core_game_conditions_vmaps` at ~15s,
`test_core_game_conditions_jits_via_step` at ~10s). They each build
a fresh `FactoriaXEnv(achievement_fn=...)` to vary the achievement
fn. Add a small fixture factory that returns
`(env_with_fn, jit_step_fn, initial_state)` for an arbitrary
`achievement_fn`, scoped to the module so each `achievement_fn`
compiles once. Migrate every test in the file.

**Acceptance criteria:**
- [ ] A `make_env_with_achievement_fn` fixture or helper lives in
      `tests/conftest.py` (or
      `tests/test_achievement_engine.py` if module-local makes more
      sense) and returns a JITted step path.
- [ ] Each unique `achievement_fn` compiles its step path exactly
      once across the file.
- [ ] All assertions unchanged.

**Verification:**
- [ ] `uv run pytest tests/test_achievement_engine.py -q --durations=0`
      shows total wall time at most half of baseline (~25s baseline).
- [ ] The two named-and-shamed tests each drop below 3s after the
      first compile is amortised.
- [ ] Full suite passes 1329/1329 still.

**Dependencies:** 2.1.

**Files likely touched:**
- `tests/test_achievement_engine.py`
- `tests/conftest.py` (if helper hoisted to root)

**Estimated scope:** S (1-2 files).

---

### Checkpoint: Pilot speedups validated

- [ ] Full suite wall time has dropped on the user's box by a
      visible margin (target: ≥ 60s saved relative to baseline).
- [ ] No assertion behaviour changed (1329 passing, coverage
      unchanged).
- [ ] **Human review.** Confirm direction is right before committing
      to the wider rollout. If the per-module gains are smaller than
      the spec predicted, revisit assumptions.

---

#### Task 2.4: Migrate `benchmarks/test_runner.py` and `benchmarks/test_rocket_benchmark.py`

**Description:** `benchmarks/test_runner.py` already has a
session-scoped `runner` fixture, but the four
`TestPerLevelBlockedActions` tests and
`test_correct_count_does_not_raise` each build their own
`BenchmarkRunner(seed=0)` to exercise error paths. Move the
non-validation-path tests to consume the shared runner; the
validation tests that *need* a fresh runner stay as-is, but they
share a JITted step path via a session fixture in the benchmarks
conftest. Apply the same pattern to the end-to-end test in
`benchmarks/test_rocket_benchmark.py`.

**Acceptance criteria:**
- [ ] `tests/benchmarks/conftest.py` gains a session-scoped fixture
      that owns the JITted runner step path for the 10x10
      stub-level shape used across the file.
- [ ] Each `TestPerLevelBlockedActions` test goes through the shared
      step path; validation-only tests in `TestRunnerValidation`
      still build their own runner for the error path they exercise.
- [ ] `test_runner_populates_achievements_end_to_end` in
      `test_rocket_benchmark.py` reuses the shared runner via the
      benchmarks conftest.

**Verification:**
- [ ] `uv run pytest tests/benchmarks/test_runner.py tests/benchmarks/test_rocket_benchmark.py -q --durations=0`
      shows total wall time at most half of baseline (~85s baseline
      across these two files).
- [ ] Full suite passes 1329/1329 still.

**Dependencies:** 2.1.

**Files likely touched:**
- `tests/benchmarks/conftest.py`
- `tests/benchmarks/test_runner.py`
- `tests/benchmarks/test_rocket_benchmark.py`

**Estimated scope:** M (3 files).

---

#### Task 2.5: Migrate `test_invariants.py` and `play/test_smoke.py::TestEnvStepSmoke`

**Description:** `test_invariants.py` runs four 100-step random
rollouts via `lax.scan` over `factoriax_step`. Each builds a fresh
env and pays a fresh compile. Move them to a module-scoped fixture
that builds the env once and exposes a JITted scan function.
`play/test_smoke.py::TestEnvStepSmoke::test_vmapped_step` is the
last 14-second outlier — migrate it to a module-scoped vmap fixture.

**Acceptance criteria:**
- [ ] `test_invariants.py` builds its env once at module scope and
      reuses it across the four property tests.
- [ ] `TestEnvStepSmoke::test_vmapped_step` consumes a
      module-scoped vmap step path.
- [ ] The other `TestEnvStepSmoke::test_reset_and_step` parametrize
      cases at 16/32/64 keep compiling per-shape (they assert
      multi-shape behaviour) — no change.

**Verification:**
- [ ] `uv run pytest tests/test_invariants.py tests/play/test_smoke.py -q --durations=0`
      shows total wall time at most half of baseline (~50s baseline
      across these two files).
- [ ] Full suite passes 1329/1329 still.

**Dependencies:** 2.1.

**Files likely touched:**
- `tests/test_invariants.py`
- `tests/play/test_smoke.py`

**Estimated scope:** S (2 files).

---

#### Task 2.6: Migrate `benchmarks/skills/test_skills_scripted_solves.py` and `examples/test_examples_run.py`

**Description:** The skills scripted-solves cluster is the
single largest cluster in the top-30 (~60s across seven tests, each
running its own scripted solve through a freshly-JITted env). Each
parametrized seed currently rebuilds. Hoist the env + JITted step
to a module-scoped fixture parameterised on the level. The
examples-run tests build wrapper stacks per example; share where
the stack is identical.

**Acceptance criteria:**
- [ ] Each skill level compiles its scripted-solve env exactly once
      across the file (assert via `step_fn._cache_size()` at session
      teardown).
- [ ] `examples/test_examples_run.py`'s shared wrapper stack
      (the four examples that use the canonical stack) compiles
      once for the file.
- [ ] All assertions preserved.

**Verification:**
- [ ] `uv run pytest tests/benchmarks/skills/test_skills_scripted_solves.py tests/examples/test_examples_run.py -q --durations=0`
      shows total wall time at most half of baseline (~90s baseline
      across these two files).
- [ ] Full suite passes 1329/1329 still.

**Dependencies:** 2.1.

**Files likely touched:**
- `tests/benchmarks/skills/test_skills_scripted_solves.py`
- `tests/examples/test_examples_run.py`
- `tests/benchmarks/skills/conftest.py` (new, possibly)

**Estimated scope:** M (3 files).

---

### Checkpoint: Bulk migrations done

- [ ] Full suite wall time on the user's box at or below 367s
      (the 50% target).
- [ ] All 1329 tests still pass.
- [ ] Coverage at or above 37%.
- [ ] `test_jit_retrace.py` still passes with `step_fn._cache_size()`
      assertions intact (independently verified — the retrace
      regression guard cannot be broken).
- [ ] **Human review.** If the 50% target is hit, proceed to 2.7
      and 2.8. If not, audit which modules still dominate and add a
      task before continuing.

---

#### Task 2.7: Drop default `-x`, add explicit fail-fast target

**Description:** Remove `-x` from `pyproject.toml`'s `addopts` (it
currently masks the rest of the suite when a single test fails).
Add an explicit fail-fast pre-commit target — e.g. a `Makefile`
entry — so developers who want stop-at-first-failure still have it
one keystroke away.

**Acceptance criteria:**
- [ ] `pyproject.toml` `addopts` no longer contains `-x`.
- [ ] A `make test-fast-fail` (or equivalent) target exists and
      runs `pytest -x -q`.
- [ ] CI runs the full suite without `-x` (already does, but verify).

**Verification:**
- [ ] `uv run pytest -q` runs through all failures, not the first
      one.
- [ ] `make test-fast-fail` exits on the first failure as before.

**Dependencies:** 2.6 (don't change defaults until the suite is
fast).

**Files likely touched:**
- `pyproject.toml`
- `Makefile`

**Estimated scope:** XS (2 files, one-line edits).

---

#### Task 2.8: Add pre-push regression guard and finalise the spec

**Description:** Add a new `scripts/hooks/pre-push` hook that runs
the full no-cov suite and fails the push if wall time exceeds 400s
(the floor that protects the gain we just won). Wire it through
the existing `make install-hooks` symlink pattern. Pre-commit stays
unchanged — it already runs the fast suite with coverage on every
commit, and we don't want to make that pass slower. Update
`SPEC_TEST_SUITE.md` with the final numbers under a "Phase 2
Results" section and document the threshold next to it.

**Acceptance criteria:**
- [ ] `scripts/hooks/pre-push` exists, is executable, and runs
      `uv run pytest -q --no-cov -o addopts=` with a wall-time
      check that exits non-zero on > 400s.
- [ ] `Makefile`'s `install-hooks` target symlinks the new
      pre-push hook alongside the existing pre-commit / post-commit.
- [ ] The hook honours a `FACTORIAX_SKIP_PERF`-style escape hatch
      (e.g. `FACTORIAX_SKIP_PRE_PUSH=1`) so developers can override
      it ad hoc when they know what they're doing.
- [ ] `SPEC_TEST_SUITE.md` "Phase 2 Results" section records the
      before/after numbers, the 400s threshold, and the hook
      location.

**Verification:**
- [ ] `make install-hooks && git push --dry-run` triggers the new
      hook and exits 0 on a healthy suite.
- [ ] Adding a `time.sleep(60)` to a test and re-running the hook
      trips the guard with a clear error message.
- [ ] `FACTORIAX_SKIP_PRE_PUSH=1 git push --dry-run` skips the
      check.

**Dependencies:** 2.7.

**Files likely touched:**
- `scripts/hooks/pre-push` (new)
- `Makefile` (one extra symlink in `install-hooks`)
- `SPEC_TEST_SUITE.md`

**Estimated scope:** S (3 files).

---

### Checkpoint: Phase 2 complete

- [ ] Full suite < 367s on the user's hardware (50% of 734s baseline).
- [ ] All 1329 tests pass; coverage ≥ 37%.
- [ ] CI guard active and verified.
- [ ] Spec finalised with Phase 2 results.
- [ ] `test_jit_retrace.py` retrace guard intact and verified.
- [ ] **Final human review.** Sign off.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Phase 1 fails the gate | High — kills the whole effort | Phase 1 is cheap (half a day); if it fails we surface the result, do not silently continue, and the spec already lists three follow-up directions (state_factory cost, shape proliferation, xdist) |
| Shared session-scoped state gets mutated by a test | High — silent cross-test corruption | Fixture returns frozen pytrees from JAX (immutable by construction); a teardown assertion re-derives the canonical state and compares; we migrate one module per commit so any pollution is bisectable |
| A test "happened to work" only because it used a specific `EnvParams` shape | Medium — fix-time during migration | Migrate one module per commit. A failure on a migrated module is investigated before moving on, not papered over by reverting the canonical fixture |
| CPU-backend numerics differ from CUDA on some kernel (e.g. reductions) | Medium — could fail invariants | Keep `test_invariants.py` runnable on either backend; verify bit-identical results for the canonical fixture under both backends once during 2.1 |
| `test_jit_retrace.py` gets accidentally migrated | High — kills the retrace regression guard | The plan calls it out explicitly; checkpoints verify its `step_fn._cache_size()` assertions still pass after every migration phase |
| Coverage drops below the 37% floor when collapsing shapes | Medium — pre-commit hook would block | Migrations are *not allowed* to delete assertions; if a shape is collapsed, the test that asserted shape-specific behaviour stays on the original shape; coverage delta logged per commit |
| Run-to-run variance hides real regressions | Low-medium | Capture before/after on the *same* machine in the *same* session; the AC-power rerun proved variance is ~5%, so look for changes that exceed that |

## Resolved Decisions

1. **Phase 1 results land in `SPEC_TEST_SUITE.md` "Phase 1 Results"**
   (confirmed). Tasks 1.4 and 2.8 edit that file directly; nothing
   spills into a separate document.
2. **Measurement protocol: full suite, `--no-cov`** (matches the
   734s / 766s baselines). The spec's 50% target tracks this number.
   Coverage instrumentation slows the suite ~20-40% on a JAX-heavy
   tree, and the pre-commit hook already enforces the 37% floor
   on its own pass, so we don't conflate the two. Phase 2 commits
   *also* report the fast-suite-with-coverage delta in their commit
   messages as a secondary signal — that is what developers feel
   on every `git commit` — but the spec's gate is the no-cov full
   suite.
3. **Single canonical fixture at 8x8 single-player.** No parallel
   15x15 fixture up front. If a module breaks during migration, we
   either keep that module on its own shape or add a second
   fixture then — see where it breaks, don't pre-build.
4. **Regression guard lives in `scripts/hooks/pre-push`** (new hook),
   not in `.github/workflows/`. The repo already wires hooks via
   `make install-hooks` symlinking `scripts/hooks/*` into
   `.git/hooks/`. Pre-commit stays as-is (lint + fast suite with
   coverage); the new pre-push runs the full no-cov suite once per
   push with the wall-time threshold.
5. **No CPU pin in `tests/conftest.py`** (resolved by Phase 1
   results). B2-vs-B1 came in at 4.3% — the CPU backend isn't
   load-bearing. The session-scoped canonical fixture is the whole
   mechanism. Task 2.1 was rewritten to drop the pin; the suite
   inherits the developer's default JAX backend.

## Parallelization

Phase 1 is strictly sequential — each task feeds the next.

Phase 2 has some parallel headroom: 2.2 / 2.3 / 2.4 / 2.5 / 2.6 are
independent of each other once 2.1 has landed. If multiple agents or
sessions are available, they can pick from this set in any order.
2.7 and 2.8 wait for the bulk migrations to finish, since they
depend on the wall-time target being met.
