"""Factoriax scenarios.

A scenario is a pre-configured set of challenges that researchers use to
compare agents on the factoriax simulation environment. This package is a
consumer of factoriax — factoriax has no knowledge of it. The simulation
engine stays unchanged; scenarios layer on top.

All scenarios follow the same entry point: construct a ``Scenario``,
construct a ``ScenarioRunner``, call ``runner.run(scenario, policies)``.
"""

from factoriax.scenarios.core import (
    LevelResult,
    Policy,
    Scenario,
    ScenarioLevel,
    ScenarioResult,
)
from factoriax.scenarios.rocket import (
    MAX_ROCKET_SCORE,
    ROCKET_ACHIEVEMENT_INFO,
    ROCKET_ACHIEVEMENT_WEIGHTS,
    ROCKET_RECIPE_BALANCE,
    ROCKET_RECIPE_BOOK,
    ROCKET_RECIPE_TABLE,
    RocketScenario,
    build_rocket_level,
    rocket_conditions,
    rocket_reward,
)
from factoriax.scenarios.runner import ScenarioRunner
from factoriax.scenarios.skills import (
    SKILLS_ACHIEVEMENT_INFO,
    SkillsBenchmark,
    build_craft_miner_level,
    build_mine_level,
    build_navigate_level,
    build_place_miner_level,
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
    "Scenario",
    "ScenarioLevel",
    "ScenarioResult",
    "ScenarioRunner",
    "LevelResult",
    "Policy",
    "RocketScenario",
    "SkillsBenchmark",
    "build_craft_miner_level",
    "build_mine_level",
    "build_navigate_level",
    "build_place_miner_level",
    "build_rocket_level",
    "rocket_conditions",
    "rocket_reward",
    "skills_conditions",
    "skills_reward",
]
