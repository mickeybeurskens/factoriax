"""FactoriaX: A JAX-based grid environment.

Top-level public API. The recommended entry point for researchers is
:func:`make`, which mirrors ``gymnax.make()`` and assembles the
canonical wrapper stack around :class:`FactoriaXEnv`. Power users can
construct :class:`FactoriaXEnv` directly and compose wrappers by hand.
"""

from collections.abc import Callable, Iterable
from typing import Any, Literal

import jax

from factoriax.constants import (
    MAX_STACK_SIZE,
    NUM_ITEM_TYPES,
    Action,
    BlockType,
    Direction,
    ItemType,
)
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.envs.auto_reset_wrapper import AutoResetWrapper
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.envs.local_observation_wrapper import LocalObservationWrapper
from factoriax.envs.science_tally_wrapper import ScienceTallyWrapper
from factoriax.levels import (
    LEVELS,
    Level,
    LevelBuilder,
    build_state,
    generate_state,
    get_level,
    load_level,
    save_level,
)
from factoriax.observations import global_array, local_array, rgb
from factoriax.rewards import mining_reward
from factoriax.state import EnvParams, EnvState

AchievementFn = Callable[[EnvState], jax.Array]


def make(
    level: Level | str | None = None,
    *,
    obs: Literal["global", "local"] = "global",
    obs_radius: int = 7,
    achievement_fn: AchievementFn | None = None,
    auto_reset: bool = False,
    blocked_actions: Iterable[int] = (),
) -> tuple[Any, EnvParams]:
    """Build a FactoriaX environment with the canonical wrapper stack.

    Mirrors ``gymnax.make()``. Composition order, outermost to
    innermost::

        AutoReset -> ActionMask -> LocalObservation -> FactoriaXEnv

    A given wrapper is only included when its corresponding option is
    set (``obs='local'`` adds the local-observation wrapper,
    ``blocked_actions`` non-empty adds the action mask, ``auto_reset``
    adds auto-reset). The bare :class:`FactoriaXEnv` is returned when
    no wrappers are requested.

    Args:
        level: Level instance, registry name, or ``None`` for
            procedural generation.
        obs: ``'global'`` (default) for full-map observations; ``'local'``
            for a windowed view centered on the selected player.
        obs_radius: Half-width of the local observation window. Ignored
            when ``obs='global'``.
        achievement_fn: Optional condition function bound to the inner
            env's constructor. See :class:`FactoriaXEnv`.
        auto_reset: Wrap the outermost env in
            :class:`AutoResetWrapper` for ``lax.scan`` training loops.
        blocked_actions: Iterable of integer action ids to mask to
            ``NOOP`` via :class:`ActionMaskWrapper`.

    Returns:
        Tuple of ``(env, params)``. ``params`` is the inner env's
        :attr:`default_params`; callers can mutate freely.

    Raises:
        KeyError: When ``level`` is a string and the registry has no
            such entry.

    Example:
        >>> env, params = factoriax.make("15x15_resources", obs="local")
        >>> obs, state = env.reset_env(jax.random.PRNGKey(0), params)
    """
    if isinstance(level, str):
        level = get_level(level)

    env: Any = FactoriaXEnv(achievement_fn=achievement_fn, level=level)
    params: EnvParams = env.default_params
    if level is not None:
        params = params.replace(
            map_width=level.map_width,
            map_height=level.map_height,
        )

    if obs == "local":
        env = LocalObservationWrapper(env, radius=obs_radius)

    blocked_tuple = tuple(int(a) for a in blocked_actions)
    if blocked_tuple:
        env = ActionMaskWrapper(env, blocked_tuple)

    if auto_reset:
        env = AutoResetWrapper(env)

    return env, params


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
    "LocalObservationWrapper",
    "MAX_STACK_SIZE",
    "NUM_ITEM_TYPES",
    "ScienceTallyWrapper",
    "build_state",
    "generate_state",
    "get_level",
    "global_array",
    "load_level",
    "local_array",
    "make",
    "mining_reward",
    "rgb",
    "save_level",
]
