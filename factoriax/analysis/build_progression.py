"""Render engine-rendered build snapshots as progression visualisations.

Two public renderers:

* :func:`render_strip` — Form C: an ``nrows x ncols`` grid of snapshot
  panels showing the cumulative factory at each phase. Tiles placed in
  the current phase are surrounded by a single continuous outline
  tracing the perimeter of the region — not per-tile rectangles.
* :func:`render_heatmap` — Form D: a single panel of the final state
  with non-placed tiles covered by a near-opaque dim layer and the
  built region surrounded by a continuous outline. The inverse of the
  earlier phase-tinted heatmap; emphasises the silhouette of what the
  scripted agent constructed.

Both forms accept their highlight colour and (for the heatmap) the dim
overlay parameters from the caller, so paper-side styling lives in
``paper/scripts/style.py`` rather than here.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle


def _bare_axes(ax: plt.Axes) -> None:
    """Strip ticks and spines so the map fills the panel cleanly."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _region_outline_segments(
    mask: np.ndarray,
    tile_px: int,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Return the line segments that trace the perimeter of ``mask``.

    For every True tile, an edge segment is emitted on each side whose
    neighbour is False or out of bounds. The resulting list, fed to
    :class:`matplotlib.collections.LineCollection`, draws a single
    continuous outline around each connected region.
    """
    h, w = mask.shape
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            top = y * tile_px - 0.5
            bot = (y + 1) * tile_px - 0.5
            left = x * tile_px - 0.5
            right = (x + 1) * tile_px - 0.5
            if y == 0 or not mask[y - 1, x]:
                segments.append(((left, top), (right, top)))
            if y == h - 1 or not mask[y + 1, x]:
                segments.append(((left, bot), (right, bot)))
            if x == 0 or not mask[y, x - 1]:
                segments.append(((left, top), (left, bot)))
            if x == w - 1 or not mask[y, x + 1]:
                segments.append(((right, top), (right, bot)))
    return segments


def _draw_region_outline(
    ax: plt.Axes,
    mask: np.ndarray,
    tile_px: int,
    color: str,
    linewidth: float,
) -> None:
    """Draw the continuous perimeter of ``mask`` on ``ax``."""
    segments = _region_outline_segments(mask, tile_px)
    if not segments:
        return
    ax.add_collection(
        LineCollection(segments, colors=color, linewidths=linewidth, capstyle="round")
    )


def _apply_dim_overlay(
    ax: plt.Axes,
    mask: np.ndarray,
    tile_px: int,
    color: str,
    alpha: float,
) -> None:
    """Cover every True tile in ``mask`` with a dim overlay rectangle."""
    ys, xs = np.where(mask)
    for ty, tx in zip(ys, xs, strict=True):
        ax.add_patch(
            Rectangle(
                (int(tx) * tile_px - 0.5, int(ty) * tile_px - 0.5),
                tile_px,
                tile_px,
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
            )
        )


def render_strip(
    frames: np.ndarray,
    placement_phase: np.ndarray,
    snapshot_labels: Sequence[str],
    tile_px: int,
    out_path: Path | str,
    *,
    highlight_color: str,
    dim_color: str,
    dim_alpha: float,
    nrows: int = 2,
    ncols: int = 3,
    outline_linewidth: float = 3.0,
) -> Path:
    """Render the build-progression strip (Form C).

    Each panel after the spawn dims every tile that is not part of the
    cumulative built region at that snapshot, matching the visual
    language of :func:`render_heatmap`. The current phase's additions
    are wrapped in a continuous outline in ``highlight_color``.

    Args:
        frames: shape ``(N, H_px, W_px, 3)`` uint8 — one snapshot per panel.
        placement_phase: shape ``(H, W)`` int. Tile (y, x) carries the
            snapshot index that first placed a machine on it; ``-1``
            for tiles never built on.
        snapshot_labels: per-panel title strings.
        tile_px: pixels per map tile (square).
        out_path: PNG (or SVG) destination.
        highlight_color: outline colour for the this-panel additions.
        dim_color: fill for the non-built overlay.
        dim_alpha: opacity of the dim overlay.
        nrows: rows in the panel grid (defaults to 2).
        ncols: columns in the panel grid (defaults to 3).
        outline_linewidth: matplotlib linewidth for the perimeter.
    """
    out_path = Path(out_path)
    n_panels = frames.shape[0]
    if nrows * ncols < n_panels:
        raise ValueError(f"grid {nrows}x{ncols} cannot hold {n_panels} panels")
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.4 * ncols, 4.4 * nrows),
        constrained_layout=True,
    )
    axes_flat = np.asarray(axes).reshape(-1)
    for k in range(n_panels):
        ax = axes_flat[k]
        ax.imshow(frames[k], interpolation="nearest")
        ax.set_title(snapshot_labels[k], fontsize=33)
        _bare_axes(ax)
        if k > 0:
            built_so_far = (placement_phase >= 1) & (placement_phase <= k)
            _apply_dim_overlay(ax, ~built_so_far, tile_px, dim_color, dim_alpha)
        new_mask = placement_phase == k
        _draw_region_outline(ax, new_mask, tile_px, highlight_color, outline_linewidth)
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def render_heatmap(
    final_frame: np.ndarray,
    placement_phase: np.ndarray,
    tile_px: int,
    out_path: Path | str,
    *,
    highlight_color: str,
    dim_color: str,
    dim_alpha: float,
    outline_linewidth: float = 3.5,
) -> Path:
    """Render the inverted build-heatmap (Form D).

    Non-placed tiles get a translucent dim overlay so the eye drops
    them; the placed region keeps its full-colour render and is
    surrounded by a continuous outline in ``highlight_color``.

    Args:
        final_frame: shape ``(H_px, W_px, 3)`` uint8 — final state.
        placement_phase: shape ``(H, W)`` int as in :func:`render_strip`.
        tile_px: pixels per map tile.
        out_path: PNG (or SVG) destination.
        highlight_color: outline colour for the built region.
        dim_color: fill for the non-placed overlay.
        dim_alpha: opacity of the dim overlay.
        outline_linewidth: matplotlib linewidth for the perimeter.
    """
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(7.0, 7.0), constrained_layout=True)
    ax.imshow(final_frame, interpolation="nearest")
    _bare_axes(ax)

    non_placed = placement_phase < 0
    _apply_dim_overlay(ax, non_placed, tile_px, dim_color, dim_alpha)

    placed = placement_phase >= 1
    _draw_region_outline(ax, placed, tile_px, highlight_color, outline_linewidth)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
