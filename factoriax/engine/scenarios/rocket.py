"""Factory for the ``Rocket-v1`` scenario.

Builds the full 32x32 rocket env: a fixed pre-placed level as the reset target
(no keyed procgen — one canonical layout), the rocket achievement conditions as
a step hook, the achievement reward, and the hand-craft action mask so
production must flow through machines.

Transitional: the conditions, reward, recipe table, level, and blocked-action
set still live in :mod:`factoriax.scenarios.rocket`; they relocate here when the
old package is removed.
"""

from __future__ import annotations

from typing import Any

from factoriax.engine.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.engine.envs.factoriax_env import FactoriaXEnv
from factoriax.engine.envs.hooks import achievement_hook
from factoriax.engine.state import EnvParams
from factoriax.scenarios.rocket import (
    ROCKET_BLOCKED_ACTIONS,
    ROCKET_RECIPE_TABLE,
    build_rocket_level,
    rocket_conditions,
    rocket_reward,
)

_MAP_SIZE: int = 32
_MAX_TIMESTEPS: int = 8000


def rocket() -> tuple[Any, EnvParams]:
    """Return the rocket env (fixed level, hand-craft masked) and its params."""
    env: Any = FactoriaXEnv(
        level=build_rocket_level(),
        step_hooks=(achievement_hook(rocket_conditions),),
        reward_fn=rocket_reward,
    )
    env = ActionMaskWrapper(env, tuple(ROCKET_BLOCKED_ACTIONS))
    params = EnvParams(
        map_width=_MAP_SIZE,
        map_height=_MAP_SIZE,
        num_players=1,
        max_timesteps=_MAX_TIMESTEPS,
        recipe_table=ROCKET_RECIPE_TABLE,
    )
    return env, params
