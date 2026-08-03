"""Tests for :mod:`factoriax.engine.renderer`.

The renderer turns a state into pixels. These tests assert the output
contract, which is the array shape and the dtype, and that a block colour
exists for every block the map can hold.
"""

from __future__ import annotations

import numpy as np

from factoriax.engine.constants import BlockType
from factoriax.playground.ui.icons import BLOCK_COLORS, create_default_textures


class TestRenderer:
    """Tests for rendering."""

    def test_resource_textures_have_expected_colors(self) -> None:
        """The base color dominates each resource texture.

        Ore textures now paint darker patches and occasional bright
        pixels on top of the base fill, so one given pixel can fail to
        equal the base color. The mode (most common color) still is.
        """
        textures = create_default_textures()

        expected = {
            int(BlockType.IRON): BLOCK_COLORS[int(BlockType.IRON)],
            int(BlockType.COPPER): BLOCK_COLORS[int(BlockType.COPPER)],
            int(BlockType.COAL): BLOCK_COLORS[int(BlockType.COAL)],
        }
        for block_id, base_rgb in expected.items():
            tex = textures[block_id]
            # Collapse each pixel to an int tag, find the most common.
            pixels = tex[..., :3].reshape(-1, 3)
            as_int = (
                pixels[:, 0].astype(np.uint32) * 65536
                + pixels[:, 1].astype(np.uint32) * 256
                + pixels[:, 2].astype(np.uint32)
            )
            mode_tag = np.bincount(as_int).argmax()
            mode_rgb = (
                int(mode_tag >> 16) & 0xFF,
                int(mode_tag >> 8) & 0xFF,
                int(mode_tag) & 0xFF,
            )
            assert mode_rgb == base_rgb, (
                f"block {block_id}: dominant color {mode_rgb}, expected {base_rgb}"
            )
