# Spec: Test-Suite Speed-Up

This spec scopes a focused investigation into the pytest suite's wall
time. The work splits into two gated phases. Phase 1 is analysis plus a
dummy-test experiment that either validates or invalidates the
hypothesis behind the proposed rework. Phase 2 only runs if Phase 1
clears the gate. No production code is touched in Phase 1.

The user's belief is that the suite can run at least 50% faster. The
target is concrete: cut total wall time of the full suite from 734s to
under 367s, while keeping every assertion that exists today.

## Assumptions

These are the assumptions baked into the analysis below. Correct any
that look wrong before Phase 1 starts.

1. The suite runs on a single machine, one pytest process, no
   `pytest-xdist`. Baseline numbers below were collected this way.
2. Pre-commit runs the fast suite (`-m "not slow"`); CI runs both
   fast and slow. The 50% target applies to the full suite because
   that is what gates merges.
3. Tests are the source of friction the user wants to reduce. Engine
   correctness is not in scope — no `factoriax/` code changes in
   Phase 1.
4. The `jax[cuda13]` dependency is for training. Tests do not need
   GPU. Forcing a CPU backend during pytest is acceptable if the
   experiment shows it helps.
5. `pytest -x` (stop at first failure, set in `pyproject.toml`) is a
   developer-loop choice, not a CI invariant. The 50% target measures
   the full run, so the experiment overrides `-x`.
6. The one failing test today
   (`tests/benchmarks/skills/test_action_masks.py::TestMineMask::test_allows_movement_mine_noop`)
   is unrelated to this work and stays out of scope.

If any of these are wrong, the experiment plan below has to change
before Phase 1.

## Objective

Make `uv run pytest` complete in under half its current wall time on
the same hardware, without dropping assertions or weakening the
existing regression guards. The first job is to *prove the hypothesis*
that the dominant cost is JIT compilation that gets repeated across
tests with shareable inputs. The second job is to roll out the pattern
across the suite — but only if the proof is in.

## Baseline Measurements

Captured 2026-05-19 on the user's local box, JAX default backend (CUDA
13). Numbers are wall time, single process, no xdist.

| Run                    | Tests collected | Tests run | Wall time | Marker          |
| ---------------------- | --------------- | --------- | --------- | --------------- |
| Fast suite             | 1355            | 1220      | **115s**  | `-m "not slow"` |
| Full suite             | 1355            | 1329      | **734s**  | (no filter)     |
| Full suite (AC rerun)  | 1355            | 1329      | **766s**  | (no filter)     |
| Slow portion (derived) | —               | 109       | **619s**  | `-m slow`       |

The AC-power rerun came in 4% *slower*, not faster, which is run-to-run
variance dominated by XLA compile timings rather than CPU clock speed.
That is useful evidence: if CPU throttling were the bottleneck, plugging
in would have moved the number. Instead it stayed flat, which reinforces
the hypothesis that repeated `jax.jit` tracing — not raw throughput — is
where time is going.

Top-30 durations from the full run consume ~286s — roughly 39% of
total time and concentrated in a small number of patterns:

| Cluster                                                       | Count | Approx. time | Pattern                                            |
| ------------------------------------------------------------- | ----- | ------------ | -------------------------------------------------- |
| `benchmarks/skills/test_skills_scripted_solves.py`            | 7     | ~52s         | Scripted runs through 5x5 skills levels, each JITs |
| `benchmarks/test_runner.py::TestPerLevelBlockedActions`       | 4     | ~46s         | Runner builds 10x10 env per test                   |
| `test_jit_retrace.py`                                         | 4     | ~30s         | Each test JITs an 8x8 env from scratch             |
| `test_invariants.py`                                          | 4     | ~30s         | 100-step rollouts on a fresh 8x8 env per test      |
| `test_achievement_engine.py` (`vmaps`, `jits_via_step`)       | 2     | ~25s         | Vmap and JIT of `env.step_env` per test            |
| `play/test_smoke.py::TestEnvStepSmoke::test_vmapped_step`     | 1     | ~14s         | Vmap reset+step on 16x16                           |
| `agentdebugger/test_replay.py` (env-params reconstruction)    | 1     | ~13s         | Module-scope env then per-test JIT of replay path  |
| `examples/test_examples_run.py::test_batched_evaluation_runs` | 1     | ~16s         | Builds the full canonical wrapper stack from `make` |

The pattern is uniform: every line above pays for one or more
`jax.jit`-traced compilations of `env.step_env`, `env.reset_env`, or
`factoriax_step`. Tracing a step function for an 8x8 single-player env
takes ~7s on CPU and a similar amount on GPU (the bottleneck is XLA
compile time, not device dispatch).

## Test-Module Inventory

A walk through every file in `tests/`, grouped by what it actually
proves. Notes flag the modules that JIT-compile the engine versus
those that only touch pure helpers.

### Core engine kernels (mostly pure functions, no env-level JIT)

- **`test_machines.py`** (12 tests) — `run_miners`, machine init.
  Builds 1x1 / 16x24 states via `state_factory`. Pure-function calls
  on small grids; no `step_env`.
- **`test_assembler.py`** (~370 lines) — `run_assemblers` on 3x3
  states. Pure kernel, fast.
- **`test_belt_arm.py`** — `run_conveyor_belts` and `run_arms` on
  small entity grids. Pure kernel.
- **`test_crossing.py`** — `run_conveyor_belts` exercising crossing
  semantics on 3x3 grids. Pure kernel.
- **`test_splitter.py`** — splitter routing on 3x3 grids. Pure kernel.
- **`test_mining.py`** — `mine_block` and resource generation. Pure
  helpers.
- **`test_placement.py`** (37 tests) — `place_machine` / `pickup_machine`
  on hand-built states. Uses many distinct `EnvParams` (1x1, 4x4, 8x8,
  16x24). Most go through the dispatcher, not the full step.
- **`test_deposit_withdraw.py`** — compound `deposit_to_adjacent` /
  `withdraw_from_adjacent`. Pure helpers. One subclass triggers full
  `step_env` for end-to-end coverage.
- **`test_inventory.py`** / `test_machine_inventory.py` — inventory
  packing/unpacking. Pure helpers.
- **`test_recipes.py`** / `test_recipe_balance.py` /
  `test_recipes_io.py` / `test_recipe_book.py` — recipe data, balance
  checks, serialization. No JAX path.
- **`test_machine_config.py`** / `test_coal_capacity.py` —
  configuration constants. No JAX.
- **`test_belt_helpers.py`** — pure encoding/decoding of belt state.

### Engine integration (pays JIT compile)

- **`test_game_logic.py`** — mostly direct `_handle_player_action`
  calls. One marker test (`test_movement_sets_facing`, `@slow`) traces
  the full dispatcher.
- **`test_factoriax.py`** (457 lines) — generation, gymnax interface,
  one `@jax.jit` end-to-end test. Distinct shapes: default, 15x15,
  8x8, 1x1.
- **`test_levels.py`** (568 lines) — level builder, registry, build
  state. Heavy use of `EnvParams(8x8, 1p / 2p)` constants. Most
  assertions are structural; a handful build states that get passed to
  engine functions.
- **`test_invariants.py`** — `@slow`, the most expensive module per
  test. Each test JITs a fresh `factoriax_step` and runs 100 random
  steps via `lax.scan` on an 8x8 env. Shareable across tests.
- **`test_observations.py`** (721 lines, 37 tests) — `global_array`,
  `local_array`, `rgb`. Builds 8x8 states. One test vmaps over
  player_idx (~2s).
- **`test_achievements.py`** / **`test_achievement_engine.py`** —
  `test_achievement_engine.py` runs the full env with a custom
  `achievement_fn`. Two tests are top-5 slowest: `test_core_game_conditions_vmaps`
  (~15s) and `test_core_game_conditions_jits_via_step` (~10s). They
  build their own env per test.
- **`test_jit_retrace.py`** — `@slow`. Regression guard: must not
  retrace after warmup. Each test calls `_setup()` which builds a
  fresh `FactoriaXEnv` and JITs it. Four tests × ~7s.
- **`test_science_lab.py`** — `run_labs` direct kernel tests (fast)
  plus two integration tests (`TestLabInEnvStep`) that each call
  `jax.jit(env.step_env)` independently. The two integration tests
  alone cost ~13s.
- **`test_science_tally_wrapper.py`** — wrapper exercises ~2s of
  `env.step_env`.
- **`test_constraints.py`** / **`test_rewards.py`** /
  **`test_rewards_inventory.py`** — reward fns and constraints, mostly
  pure but some run through the env.
- **`test_make.py`** / **`test_public_api.py`** — gymnax-style
  `factoriax.make()` factory. Several wrapper-stack permutations, each
  involving a small step.

### Documentation freshness

- **`test_atlas_fresh.py`** — checks the bundled icon atlas.
- **`test_api_reference_fresh.py`** — generated docs stay in sync.
- **`test_machine_icon_coverage.py`** — pure asset check.

### Editor (pure-Python state machine)

- **`tests/editor/test_editor_state.py`** (822 lines) — pure numpy
  state, no JAX.
- **`tests/editor/test_editor_canvas.py`** — canvas geometry.
- **`tests/editor/test_editor_save_load.py`** — orjson round-trips.
- **`test_editor_roundtrip.py`** / **`test_editor_inventory_init.py`**
  — bridge tests between editor and engine `EnvState`.

### Play UI (headless pygame)

- **`tests/play/conftest.py`** — initialises pygame display + font
  once per session.
- **`play/test_smoke.py`** — module-scoped env at 16x16 for renderer
  tests, plus a `@slow` `TestEnvStepSmoke` class that JITs at 16, 32,
  and 64 (separate compile each).
- **`play/test_game_ui.py`** / `test_game_ui_integration.py` /
  `test_machine_ui.py` / `test_action_dispatch.py` — function-scoped
  fresh `GameUI` and state per test. Pygame work is cheap; the
  bottleneck is each test that calls `env.step_env`.
- **`play/test_config.py`** / `test_controller_config.py` /
  `test_rebinding.py` / `test_launch_screen.py` — pure config logic.
- **`play/test_scroll.py`** / `test_transfer.py` / `test_ui.py` —
  pygame-only UI rendering.

### Agent debugger

- **`tests/agentdebugger/conftest.py`** — module-scoped env at 8x8 +
  a function-scoped Debugger. Good fixture discipline.
- **`agentdebugger/test_stepping.py`** / `test_layout.py` /
  `test_charts.py` / `test_action_strip_seek.py` — work through the
  module-scoped env. Mostly fast.
- **`agentdebugger/test_replay.py`** — `@slow`. Builds replay caches
  and re-runs episodes through `jax.jit(env.step_env)`.

### Benchmarks (the runner harness)

- **`tests/benchmarks/conftest.py`** — empty; inherits root
  `state_factory`.
- **`benchmarks/test_core.py`** — protocol conformance, no JIT.
- **`benchmarks/test_runner.py`** (288 lines) — `@slow`, session-scoped
  runner intended to amortize JIT once. Yet `TestPerLevelBlockedActions`
  and `TestRunnerValidation::test_correct_count_does_not_raise` each
  build a *fresh* `BenchmarkRunner` to exercise error paths; the
  session-scoped fixture isn't reused across those tests, so each test
  pays a full compile.
- **`benchmarks/test_rocket_benchmark.py`** — catalogue + per-condition
  tests on `state_factory`-built states. One `@slow` end-to-end
  runner test (~9s).
- **`benchmarks/test_rocket_level_v2.py`** — pure geometry checks.
- **`benchmarks/skills/test_skills.py`** / `test_skills_benchmark.py` /
  `test_skills_levels.py` — fast structural tests.
- **`benchmarks/skills/test_skills_scripted_solves.py`** — top
  cluster: ~7 parametrized tests, each scripted-solves one skill
  level. Each pays its own JIT compile.
- **`benchmarks/skills/test_action_masks.py`** — has the one
  pre-existing failure noted above.

### Analysis & examples

- **`tests/analysis/test_trajectory.py`** — two
  `TestStatesToTrajectoryWithParams` tests run a 3-step episode
  through `jax.jit(env.step_env)` each (~7s × 2).
- **`test_analysis.py`** (1130 lines) — pure numpy + matplotlib. Slow
  import, but per-test cost is small.
- **`tests/examples/test_examples_run.py`** — `@slow`. Five examples,
  each instantiates its own wrapper stack and JITs.
- **`tests/test_scaling.py`** — UI scaling. Pure pygame, no JAX.

### Top-level helpers

- **`tests/conftest.py`** — the single big shared piece. `state_factory`
  is function-scoped and assembles a fresh `EnvState` per test, packing
  entity arrays from python loops over `H × W`. Cheap for small grids
  but multiplies across the 1300+ test calls.
- **`tests/bench_rollout.py`** — not a test; benchmark CLI. Ignored.

## Analysis of the Current Testing Approach

The suite reads as a careful, well-targeted set of assertions. Almost
every module has clear intent, sensible helpers, and isolation between
tests. The cost structure, however, is dominated by repeated JAX
tracing — and the conftest patterns we have in place do not amortize
that tracing as much as the file-level docstrings imply they should.

The single root cause: **every test that touches `env.step_env`,
`env.reset_env`, or `factoriax_step` triggers its own JIT compilation
because the JIT cache is keyed on the callable identity**. When a test
builds a fresh `FactoriaXEnv` or wraps a method with its own
`jax.jit(...)`, no prior compile is reused. Three patterns drive most
of the cost:

1. **Per-test `jax.jit(env.step_env)`.** `test_science_lab.py`,
   `analysis/test_trajectory.py`, `agentdebugger/test_replay.py`,
   `test_jit_retrace.py`, and `examples/test_examples_run.py` each
   build a fresh `step_fn = jax.jit(env.step_env)` *inside* each test.
   Even when the underlying env is shared, the wrapped function is not,
   so the trace is repeated. Each compile costs ~7s for an 8x8 single
   player env.
2. **Per-test `FactoriaXEnv(...)` construction.** `test_invariants.py`,
   `test_achievement_engine.py`, `test_jit_retrace.py`, and several
   benchmark-runner tests build a fresh env per test. Even without an
   explicit `jax.jit` wrap, `env.step` (decorated with `@jit` on the
   class) caches on the bound-method identity — fresh env, fresh cache.
3. **Shape proliferation.** Tests sample ~15 distinct `EnvParams`
   shapes. Many of those shapes are arbitrary (`16x24`, `3x1`, `48`)
   and exist only to make a specific assertion pop. Each unseen shape
   buys a new compile that is never reused. The suite would lose
   nothing structurally by collapsing most of these to a single
   canonical small shape (e.g., 8x8 single-player) when the test isn't
   *about* the dimension.

Secondary contributors:

- **Function-scoped `state_factory`.** Building `EnvState` from a
  Python loop is cheap for one test but accumulates across ~1300 calls.
  The state-building loop runs in pure-Python (numpy) and could be
  precomputed for the common canonical shape.
- **CUDA backend by default.** With `jax[cuda13]` in dependencies, the
  default backend at test time is CUDA. For sub-millisecond
  computations on 1x1 to 32x32 grids, CUDA pays the kernel-launch cost
  (~5-50μs each) without amortizing it across batch dimensions. Tests
  also pay the host↔device transfer cost on every `jnp.array(...)` call
  in `state_factory`. CPU is plausibly faster for the tiny shapes
  tests use, but we should measure rather than assume.
- **No xdist parallelization.** A 4- or 8-core box could overlap the
  fixed compile cost across files. JIT caches don't carry across
  workers, so this only helps tests that don't pay the compile, but
  those are the majority by count.
- **`-x` in default addopts** stops at first failure. The user has
  one pre-existing failure today; the `-x` masks the rest of the slow
  modules behind it on a full run.
- **Pre-existing test fixture leaks**. `tests/test_scaling.py` defines
  its own `_init_pygame` session-scoped fixture even though
  `tests/play/conftest.py` already initialises pygame at session
  scope. Tiny cost, easy cleanup, but a sign that fixture ownership is
  not centralised.

The `tests/agentdebugger/conftest.py` is a good counter-example: it
uses `scope="module"` for the env and a function-scoped `Debugger`
that reuses the module's env. The fixture pattern works when applied;
it just isn't applied widely enough.

The `test_jit_retrace.py` module is worth calling out for the
opposite reason: every test is a regression guard for the *opposite*
problem (retracing inside a single test). It deliberately builds and
JITs from scratch and uses `step_fn._cache_size()` to assert cache
reuse. We must keep its behaviour intact even as we change everything
around it.

## Hypothesis

If we hoist `(env, params, jitted_step_fn)` for the canonical 8x8
single-player shape to `scope="session"` and force a CPU backend at
test time, the suite's wall time drops by at least 50% on the full
run. The bulk of the saving comes from compiling once instead of N
times for the dominant shape.

The hypothesis is testable in isolation. The experiment below either
clears the gate or it doesn't, and the rest of the plan only happens
if it does.

## Phase 1: Dummy-Test Experiment

Goal: prove or disprove the hypothesis with a minimal, additive
change. No production code is modified. No real test is rewritten.
The experiment lives in a new file that pytest can pick up or skip.

### Layout

The harness lives **outside** `tests/` on purpose — it is a
throwaway A/B that gets deleted once the gate is recorded. Putting
it under `tests/` (even behind `@pytest.mark.slow`) would muddle the
regression net with exploratory timing code.

```
performance_experiments/
  README.md                                # what this directory is for
  jit_share/
    __init__.py
    conftest.py                            # session-scoped env + jitted step (Task 1.2)
    test_jit_share_experiment.py           # the A/B harness
```

The directory is removed in Phase 2's final cleanup step once the
question is answered.

Both classes execute ten short, structurally identical assertions
against `env.step_env` for an 8x8 single-player env. The assertions
themselves are cheap (cache size, output shape, timestep increments,
reward type) — the point is to expose the *setup cost*, not to
benchmark the engine.

- `TestBaselineFunctionScope` (A) — each test builds its own
  `FactoriaXEnv` and wraps `env.step_env` with `jax.jit` locally.
- `TestSharedSessionScope` (B) — every test consumes a session-scoped
  fixture providing `(env, params, jit_step_fn)`. Assertions identical
  to A.

### Variants to measure

The harness is run four times. The same 20 tests, the same
assertions, four configurations:

| Variant | Backend | Fixture scope | Worker count |
| ------- | ------- | ------------- | ------------ |
| A1      | CUDA    | function      | 1            |
| A2      | CPU     | function      | 1            |
| B1      | CUDA    | session       | 1            |
| B2      | CPU     | session       | 1            |

The CUDA vs CPU split also answers the user's open question on whether
forcing `JAX_PLATFORMS=cpu` would actually help tests. CPU should
win because tests are tiny-shape — but the experiment validates that
rather than assuming.

### How the experiment is run

A single shell script `scripts/test_speedup_experiment.sh` invokes
pytest four times and emits a one-line summary per variant
(seconds total, seconds per test, JIT cache sizes captured via
`step_fn._cache_size()` at the end of the run). Output is plain text
the user can paste back here. Nothing publishes anywhere.

The script also collects:

- `pytest --durations=0` per variant.
- The peak resident memory per variant (via `/usr/bin/time -v`).
- A sanity check that all 20 tests pass in each variant.

### Gate criterion

Phase 1 passes if **B2 is at least 50% faster than A1** *and* B2 is
at least 30% faster than B1. The first comparison validates the core
mechanism (session-scoped sharing + CPU backend). The second
validates that CPU is part of the recipe.

If only one of those holds, the spec needs revision before Phase 2.
If neither holds, we abandon the hypothesis and look elsewhere
(state_factory cost, shape proliferation, or xdist parallelism).

### Out of scope for Phase 1

- Rewriting `tests/conftest.py`.
- Changing any production code in `factoriax/`.
- Touching any existing test file.
- Adjusting `pyproject.toml` pytest options.
- Adding new pytest plugins (xdist, etc.).

Phase 1 deliverable is a written result (numbers + verdict) attached
to this spec under a "Phase 1 Results" section. Nothing else.

## Phase 2: Repo-Wide Rework (gated on Phase 1 passing)

A sketch only. We do not commit to specifics until Phase 1 lands.

If Phase 1 clears the gate, the rollout looks roughly like:

1. **Hoist canonical fixtures to `tests/conftest.py`.** A
   session-scoped `canonical_env_8x8_1p` provides `(env, params,
   jit_step_fn, initial_state)`. Tests that currently build their own
   8x8 1-player env switch to consuming the fixture.
2. **Collapse arbitrary shapes.** Audit every `EnvParams(map_width=…,
   map_height=…)` call. Where the test does not actually assert
   anything dimension-specific, switch to the canonical shape. Where
   it does, document why.
3. **Force `JAX_PLATFORMS=cpu` in `tests/conftest.py`** (before any
   JAX import) — environment-overridable so a developer can flip back
   to CUDA when they want to.
4. **Centralize pygame init** in `tests/conftest.py`; remove the
   duplicate fixtures in `tests/play/conftest.py` and
   `tests/test_scaling.py`.
5. **Replace per-test `jax.jit(env.step_env)`** patterns with calls
   to the shared fixture. The exception is `test_jit_retrace.py`,
   which must keep its from-scratch JIT to do its job.
6. **Drop `-x` from default addopts.** Make it opt-in via a separate
   make target.
7. **Confirm and ratchet.** Re-run the baseline. The full suite must
   land under 367s. If it does, ratchet a regression guard into CI
   (a `pytest --durations=1` check that fails if total time exceeds
   400s).

Phase 2 itself breaks into ~6-8 small commits, each independently
testable. The full plan goes in `tasks/plan.md` once Phase 1 passes.

### Risks Phase 2 has to manage

- **Cross-test mutation of session fixtures.** A test that mutates
  the shared `EnvState` corrupts every later test. Mitigation:
  fixture returns immutable PyTree leaves; documentation explicitly
  forbids in-place updates; a final session-teardown assertion
  re-derives the canonical state and compares.
- **Hidden shape-specific assertions.** A test that "just happens" to
  rely on `map_width=15` may break when collapsed to 8x8.
  Mitigation: change one module at a time, run the whole suite, diff
  assertion failures.
- **CPU-only correctness drift.** Some kernels (e.g. reductions over
  large entity arrays) might exhibit different numerical ordering on
  CPU vs CUDA. Mitigation: keep `test_invariants.py` on whichever
  backend the user runs by default for that module; verify with a
  one-off CUDA pass that bit-identical results hold for the
  canonical fixture.
- **Pre-commit hook coverage floor.** `pyproject.toml` sets
  `fail_under = 37` for coverage. Speeding up the suite must not
  drop any tests; if a test is collapsed or merged with another, the
  coverage floor stays where it is.

## Commands

```
Baseline (full suite):   uv run pytest -p no:cacheprovider --durations=30 -o addopts=
Fast suite:              uv run pytest -m "not slow"
Slow suite:              uv run pytest -m slow
Phase 1 experiment:      bash scripts/test_speedup_experiment.sh
Specific module:         uv run pytest tests/test_<name>.py -q
```

## Project Structure (for Phase 1)

```
performance_experiments/
  README.md                              # NEW: directory purpose
  jit_share/
    __init__.py                          # NEW
    conftest.py                          # NEW: session-scoped fixtures
    test_jit_share_experiment.py         # NEW: A/B harness
scripts/
  test_speedup_experiment.sh             # NEW: run the four variants
SPEC_TEST_SUITE.md                       # this file
```

Nothing in `tests/` or `factoriax/` changes during Phase 1. The
`performance_experiments/jit_share/` directory is deleted once the
gate verdict is in.

## Code Style

The experiment file follows the project conventions in `CLAUDE.md`:
ruff, type hints, docstrings on public helpers, no emoji, snake_case,
`uv` for any package work. The harness uses `pytest.fixture(scope=…)`
explicitly so the scope intent is reviewable at a glance. Because
`performance_experiments/` is outside `pyproject.toml`'s `testpaths`,
pytest only picks it up when invoked with an explicit path — the
runner script is the only caller.

A representative test from the harness:

```python
class TestSharedSessionScope:
    """Ten assertions that share a session-scoped env + jit_step_fn."""

    def test_cache_size_stays_one_after_one_step(
        self, canonical_env_8x8_1p
    ) -> None:
        """Calling the shared jit_step_fn must not retrace."""
        env, params, jit_step_fn, state = canonical_env_8x8_1p
        rng = random.PRNGKey(0)
        _, _, _, _, _ = jit_step_fn(rng, state, int(Action.NOOP), params)
        assert jit_step_fn._cache_size() == 1
```

The baseline class produces the same assertions, but builds env +
jit_step_fn locally in each method body. Side-by-side diff readability
is what makes the experiment convincing.

## Testing Strategy

Phase 1 is itself a test of the suite. The experiment's pass/fail is
determined by the four-variant timing table. The assertions inside
the experiment are deliberately weak (shape checks, cache-size
checks) because their job is to provoke the JIT — not to validate
engine behaviour. Engine behaviour is already validated by the
existing suite.

For Phase 2 (if it runs), every refactor commit must satisfy three
checks:

1. `uv run pytest` passes the same set of tests as before (collected
   count is unchanged).
2. Coverage stays at or above 37% (current `fail_under`).
3. The full-suite wall time monotonically decreases. We log it
   per-commit in the PR description.

## Boundaries

**Always:**

- Capture before-and-after wall times for every change.
- Keep `test_jit_retrace.py` semantics intact — those tests *must*
  see a fresh JIT cache.
- Run the full suite (not just fast) when timing the result.
- Use the same hardware for baseline and re-measurement.

**Ask first:**

- Touching `pyproject.toml` (pytest options, dependencies, coverage
  floor).
- Adding new pytest plugins (`xdist`, `cov`, `benchmark`, etc.).
- Anything that changes default backend visibility for non-test
  invocations.
- Reducing the assertion count or merging tests during Phase 2.

**Never:**

- Modify production code in `factoriax/` to make tests faster.
- Mark a test `@pytest.mark.skip` to win wall time.
- Lower the coverage `fail_under` floor.
- Land Phase 2 work before Phase 1's gate has been cleared in
  writing.

## Success Criteria

Phase 1:

- The experiment file runs all four variants without errors.
- A written results table lands at the bottom of this spec
  ("Phase 1 Results") with seconds-per-variant, JIT cache sizes,
  and a yes/no on the gate.
- B2 is at least 50% faster than A1 and at least 30% faster than B1.

Phase 2 (only if Phase 1 passes):

- Full suite wall time drops below 367s on the same hardware (50%
  of 734s baseline).
- All 1329 tests still pass (modulo the pre-existing
  `test_action_masks.py` failure, which is out of scope).
- Coverage stays at or above 37%.
- A CI guard fires if total suite time regresses above 400s.

## Open Questions

1. **Where does the experiment record its results?** A "Phase 1
   Results" section appended to this file is the default. Confirm
   before running.
2. **Hardware variability.** The 734s baseline is the user's local
   box. If CI hardware is different, the absolute target may need to
   shift. Should we also capture a baseline on CI before Phase 2?
3. **Coverage tooling.** The current `pyproject.toml` has
   `pytest-cov` configured. Running coverage instrumentation also
   slows the suite. The baseline above was captured with `--no-cov`.
   Is the 50% target measured with or without coverage?
4. **Backend escape hatch shape.** If Phase 2 pins
   `JAX_PLATFORMS=cpu` in `tests/conftest.py`, what's the override
   mechanism? An env var? A pytest CLI flag? A separate marker for
   the few tests (if any) that genuinely need GPU?

## Phase 1 Results

*To be filled in after the experiment runs.*
