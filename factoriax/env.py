from typing import Any

from factoriax.engine.scenarios.registry import make as _registry_make
from factoriax.engine.state import EnvParams


def make_factoriax_env_from_name(
    env_id: str,
    *,
    obs: str | None = None,
    obs_radius: int | None = None,
    auto_reset: bool = False,
    resample: bool | None = None,
) -> tuple[Any, EnvParams]:
    """Build a registered scenario env by id.

    Parameters
    ----------
    env_id :
        Registered scenario id (e.g. ``"EasyRocket-v1"``).
    obs :
        Observation variant name (key into
        :data:`~factoriax.engine.observations.OBSERVATIONS`).
        ``None`` uses the scenario default.
    obs_radius :
        Local-window half-width. ``None`` uses the scenario default;
        ignored for ``_global`` obs variants.
    auto_reset :
        Wrap in :class:`~factoriax.engine.envs.AutoResetWrapper`.
    resample :
        Auto-reset mode. ``None`` uses the scenario's ``resample``
        setting.

    Returns
    -------
    tuple[env, EnvParams]

    Raises
    ------
    KeyError
        If ``env_id`` is not registered.
    """
    return _registry_make(
        env_id,
        obs=obs,
        obs_radius=obs_radius,
        auto_reset=auto_reset,
        resample=resample,
    )
