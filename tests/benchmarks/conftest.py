"""Shared fixtures for the benchmark test package.

The ``state_factory`` fixture is inherited from the root
``tests/conftest.py``. The two fixtures below are session-scoped so
the BenchmarkRunner's internal JIT cache for the 10x10 stub-level
shape is built exactly once across every benchmark test that consumes
them — including tests in :mod:`tests.benchmarks.test_runner` and
:mod:`tests.benchmarks.test_rocket_benchmark`.

Tests that exercise BenchmarkRunner *error paths* (mis-counted
policies, etc.) deliberately keep building their own runner so the
shared fixture's JIT cache isn't polluted by an aborted run.
"""

from __future__ import annotations

import pytest

from factoriax.benchmarks.runner import BenchmarkRunner


@pytest.fixture(scope="session")
def runner() -> BenchmarkRunner:
    """Shared BenchmarkRunner — its ``_build_env`` JIT compiles once.

    The constructor builds the default (no-mask, no-achievement-fn)
    env and JIT-compiles ``step_env`` for the canonical 10x10 shape;
    every test that consumes this fixture skips that cost. Tests that
    need a *different* config (per-level masks, custom achievement_fn)
    feed it through ``runner.run(...)`` and pay only the per-config
    rebuild cost in ``_ensure_env``, not the constructor compile.
    """
    return BenchmarkRunner(seed=0)
