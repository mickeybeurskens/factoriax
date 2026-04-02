"""Basic skills benchmark.

Five levels testing foundational agent abilities: mining resources,
crafting items, fueling miners, deploying automated mining, and
producing science packs. Each level defines custom achievement
milestones so that scoring reflects actual game-state progress.
"""

from factoriax.benchmarks.basic_skills.benchmark import (
    BasicSkillsBenchmark,
)

__all__ = ["BasicSkillsBenchmark"]
