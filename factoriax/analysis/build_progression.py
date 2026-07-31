"""Draw build snapshots as progression figures.

Both renderers take frames that the engine has already rendered, plus
one integer for each map tile that says when a machine first landed
there. Neither renderer runs the engine.

Two public renderers:

* :func:`render_strip` draws an ``nrows x ncols`` grid of panels. Each
  panel shows the factory at one snapshot. One continuous outline
  traces the perimeter of the tiles added in that panel. There is no
  box around each tile.
* :func:`render_heatmap` draws one panel of the final state. A dim
  layer covers the tiles that nothing was built on, and a continuous
  outline surrounds the built region. This shows the silhouette of
  what the run constructed.

Neither form owns a color. The highlight color, the dim color, and
the dim opacity are all necessary arguments. This module draws the
shapes, and the caller decides how they look.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle


def _bare_axes(ax: plt.Axes) -> None:
    """Remove the ticks and spines so the map fills the panel.

    Parameters
    ----------
    ax :
        The axes to change. The function modifies it in place and
        returns nothing.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _region_outline_segments(
    mask: np.ndarray,
    tile_px: int,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Return the perimeter of the True region of ``mask`` as segments.

    A tile gets a segment on each of its four sides where the neighbor
    is False or outside the grid. A side between two True tiles gets no
    segment. The result therefore traces one continuous outline around
    each connected region, and not a box around each tile.

    Parameters
    ----------
    mask :
        Boolean array of shape ``(H, W)``, indexed as ``mask[y, x]``.
        True marks a tile inside the region.
    tile_px :
        The width of one tile in pixels. Tiles are square.

    Returns
    -------
    list
        Segments as ``((x0, y0), (x1, y1))`` pairs, in pixels, ready
        for :class:`matplotlib.collections.LineCollection`. Every
        coordinate is offset by -0.5 so a line falls on the boundary
        between two pixels. The list is empty when no tile is True.

    Notes
    -----
    The scan visits every cell of the mask. The cost therefore grows
    with the area of the map, and not with the size of the region.
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
    """Draw the perimeter of the True region of ``mask`` on ``ax``.

    An empty mask adds nothing. The axes then keeps no empty line
    collection.

    Parameters
    ----------
    ax :
        The axes to draw on. The function modifies it in place and
        returns nothing.
    mask :
        Boolean array of shape ``(H, W)``. True marks a tile inside the
        region.
    tile_px :
        The width of one tile in pixels.
    color :
        Any color that matplotlib accepts, for the whole perimeter.
    linewidth :
        The matplotlib line width of the perimeter.
    """
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
    """Cover every True tile of ``mask`` with one translucent rectangle.

    Parameters
    ----------
    ax :
        The axes to draw on. The function modifies it in place and
        returns nothing.
    mask :
        Boolean array of shape ``(H, W)``. True marks a tile to cover.
    tile_px :
        The width of one tile in pixels.
    color :
        The fill color of the overlay.
    alpha :
        The opacity of the overlay, from 0.0 for invisible to 1.0 for
        opaque.

    Notes
    -----
    The function adds one patch for each True tile. A large map with
    few built tiles therefore adds thousands of patches, which is the
    slowest part of a render.
    """
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

    Panel ``k`` shows snapshot ``k``. Every panel after the first dims
    the tiles outside the region built so far, and outlines the tiles
    added in that panel. Panel 0 is the spawn state. It gets an outline
    but no dim overlay, because nothing is built yet.

    The built region covers phases 1 and higher. A phase-0 tile
    therefore counts as spawn furniture, not as something the run
    built, and panels after the first dim it. :func:`render_heatmap`
    draws the same distinction.

    Parameters
    ----------
    frames :
        Snapshots as uint8 of shape ``(N, H_px, W_px, 3)``, one for
        each panel. ``N`` sets the number of panels.
    placement_phase :
        Integer array of shape ``(H, W)`` in tiles, indexed as
        ``placement_phase[y, x]``. The value is the index of the
        snapshot that first placed a machine on that tile. A tile with
        no machine holds ``-1``.
    snapshot_labels :
        The title of each panel. It must hold at least ``N`` entries.
    tile_px :
        The width of one map tile in pixels. Tiles are square, and
        ``frames`` must be ``tile_px`` times larger than
        ``placement_phase`` on both axes.
    out_path :
        Where to write the image. The extension sets the format, and
        matplotlib accepts PNG and SVG among others. Missing parent
        directories are created.
    highlight_color :
        The outline color for the tiles added in the current panel.
    dim_color :
        The fill color of the overlay on the tiles outside the built
        region.
    dim_alpha :
        The opacity of that overlay, from 0.0 to 1.0.
    nrows :
        Rows in the panel grid.
    ncols :
        Columns in the panel grid.
    outline_linewidth :
        The matplotlib line width of the outline.

    Returns
    -------
    pathlib.Path
        ``out_path`` as a :class:`~pathlib.Path`, after the write.

    Raises
    ------
    ValueError
        When ``nrows * ncols`` is less than the number of panels. The
        check runs before the figure exists, so a failed call leaves
        nothing open and writes no file.

    Notes
    -----
    The function writes a file and closes its own figure. Cells of the
    grid beyond the last panel are hidden, not left blank.
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

    A tile that nothing was built on gets a translucent overlay, so the
    eye passes over it. The built region keeps its full-color render
    and gets a continuous outline in ``highlight_color``.

    The three groups of tiles are not a partition. The overlay covers
    phase ``-1``, and the outline covers phases 1 and higher. A phase-0
    tile therefore gets neither: it stays at full color with no
    outline, because it is spawn furniture and not something the run
    built.

    Parameters
    ----------
    final_frame :
        The final snapshot as uint8 of shape ``(H_px, W_px, 3)``.
    placement_phase :
        Integer array of shape ``(H, W)`` in tiles, as in
        :func:`render_strip`.
    tile_px :
        The width of one map tile in pixels.
    out_path :
        Where to write the image. The extension sets the format.
        Missing parent directories are created.
    highlight_color :
        The outline color for the built region.
    dim_color :
        The fill color of the overlay on the tiles that were never
        built on.
    dim_alpha :
        The opacity of that overlay, from 0.0 to 1.0.
    outline_linewidth :
        The matplotlib line width of the outline.

    Returns
    -------
    pathlib.Path
        ``out_path`` as a :class:`~pathlib.Path`, after the write.

    Notes
    -----
    The function writes a file and closes its own figure.
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
