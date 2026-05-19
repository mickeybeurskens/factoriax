# performance_experiments/

Throwaway A/B harnesses that exist to answer a specific hypothesis
before committing to a larger refactor. Each subdirectory is tied to
a spec or plan and gets **deleted** once the question is answered —
nothing here is part of the codebase's regression net.

This is the *not* the home for kept-forever benchmarks. Those live in
`scripts/` (see `scripts/bench_rotate.py`, `scripts/bench_rocket_wrapper.py`).

## Current experiments

- `jit_share/` — Phase 1 of `SPEC_TEST_SUITE.md`. Validates whether
  hoisting `(env, params, jit_step_fn)` to session scope plus a CPU
  backend cuts JIT-dominated test wall time by ≥50%. Delete after the
  gate is recorded in `SPEC_TEST_SUITE.md` "Phase 1 Results".
