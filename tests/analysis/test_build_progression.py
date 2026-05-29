"""Tests for :mod:`factoriax.analysis.build_progression`.

Synthetic frames and placement-phase grids are rendered to a temporary PNG;
the assertions check the output is written and the grid-size guard raises.
The package conftest forces the Agg backend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from factoriax.analysis.build_progression import render_heatmap, render_strip

_TILE_PX = 4
_GRID = 4


def _frames(n: int) -> np.ndarray:
    """Return ``n`` blank RGB snapshots sized to the tile grid."""
    return np.zeros((n, _GRID * _TILE_PX, _GRID * _TILE_PX, 3), dtype=np.uint8)


def _placement_phase() -> np.ndarray:
    """A phase grid touching phases 0, 1, 2 plus never-built tiles (-1)."""
    pp = np.full((_GRID, _GRID), -1, dtype=np.int32)
    pp[0, 0] = 0
    pp[1, 1] = 1
    pp[2, 2] = 2
    return pp


def test_render_strip_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "strip.png"

    # 2x3 grid for 3 panels exercises the hidden-axes path for the spare cells.
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


def test_render_strip_rejects_too_small_grid(tmp_path: Path) -> None:
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


def test_render_heatmap_writes_png(tmp_path: Path) -> None:
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
