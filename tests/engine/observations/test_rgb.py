"""Tests for the RGB observation.

``rgb`` renders the map to pixels for a vision model. These tests cover the
output shape, the dtype, and the value range.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.engine.constants import (
    BlockType,
)
from factoriax.engine.observations import (
    NUM_PLAYER_SCALARS,
    NUM_SPATIAL_CHANNELS,
    rgb,
)
from tests.helpers.observations import (
    MAP_H as _MAP_H,
)
from tests.helpers.observations import (
    MAP_W as _MAP_W,
)
from tests.helpers.observations import (
    WINDOW as _WINDOW,
)

_GLOBAL_OBS_SIZE = (
    NUM_SPATIAL_CHANNELS["x_ray"] * _MAP_W * _MAP_H + NUM_PLAYER_SCALARS["x_ray"]
)
_LOCAL_OBS_SIZE = (
    NUM_SPATIAL_CHANNELS["x_ray"] * _WINDOW * _WINDOW + NUM_PLAYER_SCALARS["x_ray"]
)


class TestRgb:
    """Tests for the rgb observation function."""

    def test_shape_and_dtype(self, state_factory) -> None:
        """Output must be uint8 RGB with correct pixel dimensions."""
        bps = 4
        state = state_factory(
            world_map=jnp.ones((8, 8), dtype=jnp.int32) * int(BlockType.DIRT),
        )
        out = rgb(state, block_pixel_size=bps)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.uint8
        assert out.ndim == 3
        assert out.shape == (8 * bps, 8 * bps, 3)

    def test_different_maps_produce_different_images(self, state_factory) -> None:
        """A DIRT map and a WATER map must render visually differently."""
        bps = 4
        state_dirt = state_factory(
            world_map=jnp.ones((4, 4), dtype=jnp.int32) * int(BlockType.DIRT),
        )
        state_water = state_factory(
            world_map=jnp.ones((4, 4), dtype=jnp.int32) * int(BlockType.WATER),
        )
        img_dirt = rgb(state_dirt, block_pixel_size=bps)
        img_water = rgb(state_water, block_pixel_size=bps)
        assert not np.array_equal(img_dirt, img_water)

    @pytest.mark.parametrize("bps", [4, 8, 16])
    def test_block_pixel_size_scales_output(self, state_factory, bps: int) -> None:
        """Output dimensions must scale linearly with block_pixel_size.

        Parameters
        ----------
        bps
            Block pixel size to test.
        """
        state = state_factory(
            world_map=jnp.ones((4, 4), dtype=jnp.int32) * int(BlockType.DIRT),
        )
        out = rgb(state, block_pixel_size=bps)
        assert out.shape == (4 * bps, 4 * bps, 3)
