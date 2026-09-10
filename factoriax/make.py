"""The public entry point that builds a registered environment.

A scenario is a named environment, with a level, a reward, and a set of
achievements. :func:`env_from_name` takes the id of a scenario and returns the
environment together with its parameters.

These three scenarios carry an id:

- ``"MinerBootstrap-v1"``
- ``"EasyRocket-v1"``
- ``"Rocket-v1"``

:func:`factoriax.engine.envs.registry.list_scenarios` returns each id with its
spec, so a caller can read the list instead of writing an id by hand.

Example
-------
>>> from factoriax.make import env_from_name
>>> env, params = env_from_name("EasyRocket-v1")
"""

from typing import Any

from factoriax.engine.envs.registry import make as _registry_make
from factoriax.engine.state import EnvParams


def env_from_name(
    env_id: str,
    *,
    obs: str | None = None,
    obs_radius: int | None = None,
    auto_reset: bool = True,
    resample: bool | None = None,
) -> tuple[Any, EnvParams]:
    """Build the environment of one scenario.

    Each keyword argument overrides one default of the scenario. ``None``
    keeps that default.

    This function passes its arguments to
    :func:`factoriax.engine.envs.registry.make`, which holds the full rules.

    Parameters
    ----------
    env_id
        Id of the scenario, for example ``"EasyRocket-v1"``.
    obs
        Name of an observation variant, which must be a key of
        :data:`~factoriax.engine.observations.OBSERVATIONS`. There are four:
        ``"superficial_local"``, ``"superficial_global"``, ``"x_ray_local"``,
        and ``"x_ray_global"``.
    obs_radius
        Half-width of the local window, in tiles. A ``_global`` observation
        variant reads the whole map, and ignores this argument.
    auto_reset
        Whether to wrap the environment in
        :class:`~factoriax.engine.envs.wrappers.AutoResetWrapper`. The wrapper
        starts a new episode on ``done``, which a rollout of a fixed length
        needs. Pass ``False`` to control the episodes yourself.
    resample
        Whether a reset builds a new world, or restores the first one. The
        restore is cheaper. This argument has an effect only while
        ``auto_reset`` is ``True``.

    Returns
    -------
    tuple[Any, EnvParams]
        The environment, and the parameters to step it with.

    Raises
    ------
    KeyError
        If no scenario carries the id ``env_id``.

    Examples
    --------
    Build a scenario with its own defaults:

    >>> env, params = env_from_name("EasyRocket-v1")

    Read the whole map, and control the episodes in your own loop:

    >>> env, params = env_from_name(
    ...     "Rocket-v1",
    ...     obs="x_ray_global",
    ...     auto_reset=False,
    ... )
    """
    return _registry_make(
        env_id,
        obs=obs,
        obs_radius=obs_radius,
        auto_reset=auto_reset,
        resample=resample,
    )
