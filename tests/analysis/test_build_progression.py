"""Tests for :mod:`factoriax.analysis.build_progression`.

The module draws what a run built, and when. Its input is a placement
phase grid. The grid holds one integer for each map tile. That integer
is the index of the snapshot that first put a machine on the tile. A
tile with no machine holds ``-1``. Snapshot 0 is the spawn state.

These tests cover two things. They assert on the outline geometry
directly, because it carries the main claim of the module. The built
region gets one continuous perimeter, and not a rectangle around each
tile. The two render entry points get a smoke test only. They write an
image. An assertion on an image needs a committed reference file, and
every deliberate style change makes that file invalid. The assertions
therefore stop at "a non-empty file appeared".

Coordinates are pixels in the rendered frame, offset by half a pixel. A
line therefore falls on the boundary between two pixels, and not
through the middle of one. The package ``conftest`` selects the Agg
backend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from factoriax.analysis.build_progression import (
    _region_outline_segments,
    render_heatmap,
    render_strip,
)

#: Pixels for each map tile. This value keeps the test images small.
#: It is also large enough to keep the half-pixel offsets distinct.
_TILE_PX = 4

#: Side length of the square test map, in tiles.
_GRID = 4


def _frames(n: int) -> np.ndarray:
    """Return ``n`` blank RGB snapshots sized to the tile grid.

    Parameters
    ----------
    n :
        The number of snapshots. The strip draws one panel for each
        snapshot.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, _GRID * _TILE_PX, _GRID * _TILE_PX, 3)`` of uint8
        zeros. The renderer only shows these frames. No code reads the
        pixel values, so black serves as well as a real render.
    """
    return np.zeros((n, _GRID * _TILE_PX, _GRID * _TILE_PX, 3), dtype=np.uint8)


def _placement_phase() -> np.ndarray:
    """Return a phase grid that holds phases 0, 1, and 2.

    The three built tiles sit on the diagonal, so no two of them touch.
    Each tile is therefore its own region with its own four-sided
    outline. The rest of the grid stays at ``-1``, which gives the dim
    overlay something to cover.
    """
    pp = np.full((_GRID, _GRID), -1, dtype=np.int32)
    pp[0, 0] = 0
    pp[1, 1] = 1
    pp[2, 2] = 2
    return pp


class TestRegionOutlineSegments:
    """Perimeter extraction from a boolean tile mask."""

    def test_empty_mask_yields_no_segments(self) -> None:
        """Draw nothing for a mask with no True tile.

        The caller uses the empty result to skip the panel. It does not
        add an empty line collection to the axes.
        """
        mask = np.zeros((3, 3), dtype=bool)
        assert _region_outline_segments(mask, _TILE_PX) == []

    def test_shared_edge_between_neighbours_is_dropped(self) -> None:
        """Give two touching tiles one perimeter, and not two boxes.

        A rectangle for each tile gives eight segments for this mask,
        and it draws a line along the seam. The perimeter gives six
        segments and leaves the seam clear.
        """
        mask = np.array([[True, True]], dtype=bool)
        segments = _region_outline_segments(mask, _TILE_PX)

        assert len(segments) == 6
        seam_x = 1 * _TILE_PX - 0.5
        vertical_xs = {a[0] for a, b in segments if a[0] == b[0]}
        assert seam_x not in vertical_xs

    def test_isolated_tile_is_fully_enclosed(self) -> None:
        """Give a tile with no neighbor all four of its edges."""
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True
        segments = _region_outline_segments(mask, _TILE_PX)
        assert len(segments) == 4


class TestRenderStrip:
    """The multi-panel progression strip."""

    def test_writes_png(self, tmp_path: Path) -> None:
        """Render the strip and return the path of the new file."""
        out = tmp_path / "strip.png"

        # A 2x3 grid holds three panels and has three spare cells.
        # This runs the branch that hides the unused axes.
        result = render_strip(
            _frames(3),
            _placement_phase(),
            ["spawn", "phase 1", "phase 2"],
            _TILE_PX,
            out,
            highlight_color="#ff0000",
            dim_color="#000000",
            dim_alpha=0.7,
            nrows=2,
            ncols=3,
        )

        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_rejects_too_small_grid(self, tmp_path: Path) -> None:
        """When the grid has too few cells, raise before drawing.

        Without this guard, the extra panels disappear without a word
        and the figure looks complete.
        """
        with pytest.raises(ValueError, match="cannot hold"):
            render_strip(
                _frames(3),
                _placement_phase(),
                ["a", "b", "c"],
                _TILE_PX,
                tmp_path / "x.png",
                highlight_color="#ffffff",
                dim_color="#000000",
                dim_alpha=0.5,
                nrows=1,
                ncols=2,
            )


class TestRenderHeatmap:
    """The single-panel final-state view."""

    def test_writes_png(self, tmp_path: Path) -> None:
        """Render the heatmap and return the path of the new file."""
        out = tmp_path / "heatmap.png"
        final = np.zeros((_GRID * _TILE_PX, _GRID * _TILE_PX, 3), dtype=np.uint8)

        result = render_heatmap(
            final,
            _placement_phase(),
            _TILE_PX,
            out,
            highlight_color="#00ff00",
            dim_color="#101010",
            dim_alpha=0.8,
        )

        assert result == out
        assert out.exists()
