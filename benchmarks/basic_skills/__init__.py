"""Basic skills benchmark.

Three levels testing foundational agent abilities: mining resources,
crafting items, and depositing items into machines. Each level isolates
a core skill so that performance can be attributed to specific capabilities.
"""

from benchmarks.basic_skills.benchmark import (
    BasicSkillsBenchmark,
)

__all__ = ["BasicSkillsBenchmark"]
