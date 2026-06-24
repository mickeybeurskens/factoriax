"""Built-in FactoriaX scenarios as gymnax-style env factories.

Each scenario is a factory ``() -> (env, params)`` registered by version id in
:mod:`~factoriax.engine.scenarios.registry`. Load one with
``factoriax.make("EasyRocket-v1")``; iterate :func:`list_scenarios` for the
catalog (used by the playground picker).
"""

from __future__ import annotations

from factoriax.engine.scenarios.registry import (
    SCENARIOS,
    ScenarioSpec,
    list_scenarios,
    make,
)

__all__ = [
    "SCENARIOS",
    "ScenarioSpec",
    "list_scenarios",
    "make",
]
