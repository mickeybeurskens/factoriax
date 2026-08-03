"""Shared scaffolding for the observation tests.

Every file under ``tests/engine/observations/`` builds its world at the same
size and reads the same radius, so the constants live here rather than in each
file. :func:`slice_channel` cuts one named spatial channel out of the flat
observation vector that the encoder returns.
"""

from __future__ import annotations

import jax
import numpy as np

from factoriax.engine.observations import _X_RAY_SPATIAL_CHANNEL_NAMES
from factoriax.engine.state import EnvParams

#: Map size every observation test builds. Small enough to assert per tile.
MAP_W: int = 8
MAP_H: int = 8

#: Radius of the local window, and the window width it implies.
RADIUS: int = 3
WINDOW: int = 2 * RADIUS + 1

#: Parameters every observation test runs against.
DEFAULT_PARAMS = EnvParams(max_timesteps=100)


def slice_channel(obs: jax.Array, params: EnvParams, name: str) -> np.ndarray:
    """Return the ``(H, W)`` float view of one named spatial channel.

    Parameters
    ----------
    obs
        Flat observation vector from the x-ray encoder.
    params
        Environment parameters. Present for call-site symmetry.
    name
        Channel name, from ``_X_RAY_SPATIAL_CHANNEL_NAMES``.

    Returns
    -------
    np.ndarray
        The channel, reshaped to ``(MAP_H, MAP_W)``.
    """
    idx = _X_RAY_SPATIAL_CHANNEL_NAMES.index(name)
    tile_count = MAP_W * MAP_H
    start = idx * tile_count
    return np.asarray(obs[start : start + tile_count]).reshape(MAP_H, MAP_W)
