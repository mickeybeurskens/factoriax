"""Observation type definitions and fog-of-war rendering.

The ``ObservationType`` enum categorizes how an agent observes the
environment. The enum value is stored as an int in
``Trajectory.observation_scheme["type"]`` and interpreted by the
debugger to choose the right visualization.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class ObservationType(IntEnum):
    """How the agent observes the environment."""

    UNDETERMINED = 0  # Not recorded or unknown.
    GLOBAL = 1  # Full map (global_array).
    LOCAL = 2  # Windowed patch (local_array).
    PLAYER = 3  # Human player (no obs function).


def apply_fog_of_war(
    frame: np.ndarray,
    player_x: int,
    player_y: int,
    radius: int,
    tile_size: int,
    map_w: int,
    map_h: int,
    fog_brightness: float = 0.25,
) -> np.ndarray:
    """Apply a fog-of-war mask to a rendered game frame.

    Darkens all pixels outside the agent's ``(2r+1) x (2r+1)``
    observation window. The window is centered on the player's tile
    position.

    Args:
        frame: RGB uint8 array of shape ``(H, W, 3)``. Not modified.
        player_x: Player tile x coordinate.
        player_y: Player tile y coordinate.
        radius: Observation radius in tiles.
        tile_size: Pixel size per tile used when rendering the frame.
        map_w: Map width in tiles.
        map_h: Map height in tiles.
        fog_brightness: Brightness multiplier for fogged areas (0-1).

    Returns:
        New RGB uint8 array with fog applied.
    """
    result = frame.copy()
    fh, fw = frame.shape[:2]

    # Compute the pixel region of the observation window.
    tile_x0 = max(0, player_x - radius)
    tile_y0 = max(0, player_y - radius)
    tile_x1 = min(map_w, player_x + radius + 1)
    tile_y1 = min(map_h, player_y + radius + 1)

    px0 = tile_x0 * tile_size
    py0 = tile_y0 * tile_size
    px1 = min(tile_x1 * tile_size, fw)
    py1 = min(tile_y1 * tile_size, fh)

    # Darken the entire frame first.
    result = (result.astype(np.float32) * fog_brightness).astype(np.uint8)

    # Restore the visible window to full brightness.
    result[py0:py1, px0:px1] = frame[py0:py1, px0:px1]

    return result
