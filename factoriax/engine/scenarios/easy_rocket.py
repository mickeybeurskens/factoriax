"""Factory for the ``EasyRocket-v1`` scenario.

Binds the easy-rocket variation points onto a generic
:class:`~factoriax.engine.envs.factoriax_env.FactoriaXEnv`: the keyed procgen
generator as ``reset_fn`` (a fresh layout per reset), the production
achievement conditions as a step hook, and the achievement reward.

Transitional: the conditions, reward, recipe table, and generator still live in
:mod:`factoriax.scenarios.easy_rocket`; they relocate into this module when the
old package is removed.
"""

from __future__ import annotations

from typing import Any

from factoriax.engine.envs.factoriax_env import FactoriaXEnv
from factoriax.engine.envs.hooks import achievement_hook
from factoriax.engine.state import EnvParams
from factoriax.scenarios.easy_rocket import (
    EASY_ROCKET_RECIPE_TABLE,
    easy_rocket_conditions,
    easy_rocket_reward,
    generate_easy_rocket_state,
)

_MAP_SIZE: int = 16
_MAX_TIMESTEPS: int = 2000
_MAX_MACHINES: int = 100
_ORE_RESOURCES: int = 3000


def easy_rocket() -> tuple[Any, EnvParams]:
    """Return the easy-rocket env (keyed procgen reset) and its params."""
    env = FactoriaXEnv(
        reset_fn=generate_easy_rocket_state,
        step_hooks=(achievement_hook(easy_rocket_conditions),),
        reward_fn=easy_rocket_reward,
    )
    params = EnvParams(
        map_width=_MAP_SIZE,
        map_height=_MAP_SIZE,
        num_players=1,
        max_timesteps=_MAX_TIMESTEPS,
        max_machines=_MAX_MACHINES,
        recipe_table=EASY_ROCKET_RECIPE_TABLE,
        base_resources=_ORE_RESOURCES,
    )
    return env, params
