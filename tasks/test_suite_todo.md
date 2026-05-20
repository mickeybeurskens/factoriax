# Todo: Test-Suite Speed-Up

Tracking checklist for `tasks/test_suite_plan.md`. Mark each sub-task
when its acceptance criteria are met. Stop and review at every
checkpoint. Phase 2 does not start until Phase 1's gate is cleared in
writing.

Sizes: XS (1 file, ~10-30 min), S (1-2 files, ~30-90 min),
M (3-5 files, ~1-3 hours).

## Phase 1 — Dummy A/B Experiment

- [ ] 1.1 (S) Stand up `performance_experiments/jit_share/` + `TestBaselineFunctionScope`
- [ ] 1.2 (S) Add session-scoped fixture + `TestSharedSessionScope`
- [ ] 1.3 (S) Build `scripts/test_speedup_experiment.sh` (four variants)
- [ ] 1.4 (XS) Run experiment, fill in `SPEC_TEST_SUITE.md` "Phase 1 Results"

### Checkpoint — Phase 1 Complete
- [ ] All 20 experimental tests pass under every variant
- [ ] Results table in spec reflects real numbers from user's box
- [ ] Gate inequalities explicit: B2 vs A1 ≥ 50%, B2 vs B1 ≥ 30%
- [ ] **Human review.** Gate pass/fail recorded; Phase 2 authorised only if pass.

---

## Phase 2 — Repo-Wide Rework (gated on Phase 1)

- [x] 2.1 (S) Canonical fixture + centralise pygame init in `tests/conftest.py` (CPU pin dropped per Phase 1)
- [x] 2.2 (S) Pilot — migrate `test_science_lab.py` + `analysis/test_trajectory.py`
- [ ] 2.1.5 (XS) Unify `state_factory` scalar dtypes (`selected_player`/`timestep`) with `env.reset_env`
- [ ] 2.3 (S) Migrate `test_achievement_engine.py`

### Checkpoint — Pilot speedups validated
- [ ] Full suite ≥ 60s faster than baseline on user's box
- [ ] 1329 tests still pass; coverage unchanged
- [ ] **Human review.** Direction confirmed before wider rollout.

- [ ] 2.4 (M) Migrate `benchmarks/test_runner.py` + `benchmarks/test_rocket_benchmark.py`
- [ ] 2.5 (S) Migrate `test_invariants.py` + `play/test_smoke.py::TestEnvStepSmoke`
- [ ] 2.6 (M) Migrate `benchmarks/skills/test_skills_scripted_solves.py` + `examples/test_examples_run.py`

### Checkpoint — Bulk migrations done
- [ ] Full suite ≤ 367s on user's hardware (50% of 734s baseline)
- [ ] 1329 tests still pass; coverage ≥ 37%
- [ ] `test_jit_retrace.py` retrace guard still passes (independently verified)
- [ ] **Human review.** Target met before finalising.

- [ ] 2.7 (XS) Drop `-x` from default `addopts`; add `make test-fast-fail` opt-in
- [ ] 2.8 (S) Add `scripts/hooks/pre-push` wall-time guard (>400s fails); finalise spec

### Checkpoint — Phase 2 Complete
- [ ] Full suite < 367s on user's hardware
- [ ] All 1329 tests pass; coverage ≥ 37%
- [ ] CI guard active and verified (sleep-60s test trips it locally)
- [ ] Spec finalised with Phase 2 results
- [ ] `test_jit_retrace.py` intact
- [ ] **Final human review.** Sign off.

---

## Cleanup (after Phase 2 sign-off)

- [ ] Delete `performance_experiments/jit_share/` — the gate question
      is answered; the harness has done its job.
- [ ] If `performance_experiments/` is otherwise empty, delete the
      directory and its README too.
- [ ] Remove the `jit_share/` entry from
      `performance_experiments/README.md` if the directory survives.

---

## Working rules added during execution

- **Try small follow-up tweaks to beat 50% on each task, but don't
  hunt forever.** First pass: the prescribed migration. If it falls
  short of 50%, try one or two small tweaks (action-arg type
  unification, pre-warm a cache, parameterise a fixture, etc.). If
  those don't close the gap, ship the partial win and move to the
  next task. Document why in the commit message.

## Notes / Ideas (for end-of-sprint discussion)

Scratchpad. Not load-bearing tasks — observations and possible
follow-ups to surface to the user when the immediate work pauses.

- **Task 2.3 fell short of the 50% target on `test_achievement_engine.py`
  (35.17s → 33.99s, ~3%).** The file's cost is dominated by tests that
  *intrinsically* exercise JIT/vmap tracing: `env.step` (@jit'd) and
  `jax.vmap(env.step_env)` each have their own cache entry, and every
  unique `achievement_fn` forces a fresh `factoriax_step` compile.
  Sharing the env amortises construction but not the compiles.
  Question to discuss: do we want to keep this migration anyway for
  code-pattern consistency, or revert and document why this file is
  structurally resistant?
- **The "50% per file" target in the plan was uniform, but the gain
  is actually highly file-dependent.** Files with N tests sharing the
  same `(env, achievement_fn, shape)` triple → big wins (2.2, 2.1.5).
  Files where every test exercises a different JIT/vmap entry point
  → tiny wins (2.3). Worth revising per-file targets after measuring,
  rather than pretending one number fits all.
- **The `state_factory` dtype fix (2.1.5) was a much bigger force
  multiplier than any single migration.** ~42s saved across the
  whole suite from one 2-line conftest change. There may be other
  similar foundation tweaks worth looking for before continuing to
  migrate one file at a time — e.g. whether other fixtures emit
  Python-int or weak-typed scalars that retrace silently.
- **`env.step_env` transparently triggers an internal JIT compile**
  via `factoriax_step` with `achievement_fn` baked in as a closure
  variable. Tests that look like "eager calls" actually pay a JIT
  cost. Worth a one-line comment in `factoriax/envs/factoriax_env.py`
  so future readers don't assume `step_env` is the eager path.
- **The pre-commit hook runs the fast suite with coverage every
  commit (~70s).** Task 2.8's pre-push wall-time threshold will be
  on the full no-cov suite. Two different measurement modes — worth
  thinking about whether the spec should publish *both* baselines
  (fast+cov for the dev loop, full no-cov for the gate) so we can
  spot regressions in either.
