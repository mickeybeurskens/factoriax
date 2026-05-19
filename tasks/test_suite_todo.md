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

- [ ] 2.1 (S) Pin CPU backend default + canonical fixture in `tests/conftest.py`; centralise pygame init
- [ ] 2.2 (S) Pilot — migrate `test_science_lab.py` + `analysis/test_trajectory.py`
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
