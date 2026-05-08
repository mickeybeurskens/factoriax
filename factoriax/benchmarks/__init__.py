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
from factoriax.benchmarks.rocket import (
    MAX_ROCKET_SCORE,
    ROCKET_ACHIEVEMENT_INFO,
    ROCKET_ACHIEVEMENT_WEIGHTS,
    ROCKET_RECIPE_BALANCE,
    ROCKET_RECIPE_BOOK,
    ROCKET_RECIPE_TABLE,
    RocketBenchmark,
    build_rocket_level,
    rocket_conditions,
    rocket_reward,
)
from factoriax.benchmarks.runner import BenchmarkRunner
from factoriax.benchmarks.skills import (
    SKILLS_ACHIEVEMENT_INFO,
    SkillsBenchmark,
    skills_conditions,
    skills_reward,
)

__all__ = [
    "MAX_ROCKET_SCORE",
    "ROCKET_ACHIEVEMENT_INFO",
    "ROCKET_ACHIEVEMENT_WEIGHTS",
    "ROCKET_RECIPE_BALANCE",
    "ROCKET_RECIPE_BOOK",
    "ROCKET_RECIPE_TABLE",
    "SKILLS_ACHIEVEMENT_INFO",
    "Benchmark",
    "BenchmarkLevel",
    "BenchmarkResult",
    "BenchmarkRunner",
    "LevelResult",
    "Policy",
    "RocketBenchmark",
    "SkillsBenchmark",
    "build_rocket_level",
    "rocket_conditions",
    "rocket_reward",
    "skills_conditions",
    "skills_reward",
]
