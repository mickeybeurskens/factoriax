"""Factoriax benchmarks.

A benchmark is a pre-configured set of challenges that researchers use to
compare agents on the factoriax simulation environment. This package is a
consumer of factoriax — factoriax has no knowledge of it. The simulation
engine stays unchanged; benchmarks layer on top.

All benchmarks follow the same entry point: construct a ``Benchmark``,
construct a ``BenchmarkRunner``, call ``runner.run(benchmark, policies)``.
"""

from factoriax.benchmarks.core import (
    Benchmark,
    BenchmarkLevel,
    BenchmarkResult,
    LevelResult,
    Policy,
)
from factoriax.benchmarks.runner import BenchmarkRunner

__all__ = [
    "Benchmark",
    "BenchmarkLevel",
    "BenchmarkResult",
    "BenchmarkRunner",
    "LevelResult",
    "Policy",
]
