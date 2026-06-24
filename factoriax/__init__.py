"""FactoriaX: A JAX-based grid environment.

Top-level public API. :func:`make` mirrors ``gymnax.make()``: pass a
registered scenario id and get back a fully-configured ``(env, params)``
pair. Direct construction of :class:`FactoriaXEnv` (with its own ``obs``
and ``obs_radius`` args, plus optional ``level=`` / ``achievement_fn=``
/ ``reset_fn=`` / ``step_hooks=`` / ``reward_fn=``) remains available
for callers that need a non-scenario env.
"""

from typing import Any

from factoriax.engine.constants import (
    NUM_ITEM_TYPES,
    Action,
    BlockType,
    Direction,
    ItemType,
)
from factoriax.engine.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.engine.envs.auto_reset_wrapper import AutoResetWrapper
from factoriax.engine.envs.factoriax_env import FactoriaXEnv
from factoriax.engine.envs.science_tally_wrapper import ScienceTallyWrapper
from factoriax.engine.levels import (
    LEVELS,
    Level,
    LevelBuilder,
    build_state,
    generate_state,
    get_level,
    load_level,
    save_level,
)
from factoriax.engine.observations import (
    OBSERVATIONS,
    global_superficial,
    global_x_ray,
    local_superficial,
    local_x_ray,
    rgb,
)
from factoriax.engine.rewards import mining_reward
from factoriax.engine.scenarios import registry as _scenario_registry
from factoriax.engine.state import EnvParams, EnvState


def make(
    env_id: str,
    *,
    obs: str | None = None,
    obs_radius: int | None = None,
    auto_reset: bool = False,
    resample: bool | None = None,
) -> tuple[Any, EnvParams]:
    """Build a registered scenario env.
    
    Mirrors :func:`gymnax.make`. The returned env carries its own
    observation function and observation space; the wrappers stack
    only when auto-reset is requested.

    Parameters
    ----------
    env_id :
        Registered scenario id (e.g. ``"EasyRocket-v1"``).
    obs :
        Observation variant name. ``None`` uses the scenario's
        opinionated default. Valid keys live in
        :data:`~factoriax.engine.observations.OBSERVATIONS`.
    obs_radius :
        Local-window half-width. ``None`` uses the scenario
        default; ignored for ``_global`` obs variants.
    auto_reset :
        Wrap in :class:`AutoResetWrapper` for ``lax.scan``
        training loops.
    resample :
        Auto-reset mode. ``None`` uses the scenario's
        ``resample`` setting.
    env_id : str :
        
    * :
        
    obs : str | None :
        (Default value = None)
    obs_radius : int | None :
        (Default value = None)
    auto_reset : bool :
        (Default value = False)
    resample : bool | None :
        (Default value = None)
    env_id: str :
        
    obs: str | None :
         (Default value = None)
    obs_radius: int | None :
         (Default value = None)
    auto_reset: bool :
         (Default value = False)
    resample: bool | None :
         (Default value = None)

    Returns
    -------

    
    >>> env, params = make("EasyRocket-v1")
        >>> env, params = make("Rocket-v1", obs="superficial_local", obs_radius=5)
    """
    return _scenario_registry.make(
        env_id,
        obs=obs,
        obs_radius=obs_radius,
        auto_reset=auto_reset,
        resample=resample,
    )


__all__ = [
    "Action",
    "ActionMaskWrapper",
    "AutoResetWrapper",
    "BlockType",
    "Direction",
    "EnvParams",
    "EnvState",
    "FactoriaXEnv",
    "ItemType",
    "LEVELS",
    "Level",
    "LevelBuilder",
    "NUM_ITEM_TYPES",
    "OBSERVATIONS",
    "ScienceTallyWrapper",
    "build_state",
    "generate_state",
    "get_level",
    "global_superficial",
    "global_x_ray",
    "load_level",
    "local_superficial",
    "local_x_ray",
    "make",
    "mining_reward",
    "rgb",
    "save_level",
]
