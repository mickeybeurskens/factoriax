"""Render a graded-achievement curriculum as a horizontal phase strip.

Public surface:

* :class:`AchievementSpec`: plain-Python achievement-bit record.
* :class:`StripLayout`: geometry constants.
* :func:`render`: top-level entry. It consumes a sequence of specs plus
  a phase->hex palette, and writes a PNG.

This module owns no color scheme. ``phase_palette`` is a necessary
argument. The caller therefore sets the look of each phase, and two
reports can style the same curriculum differently.

Internally the renderer is broken into focused helpers: cell-position
computation, per-cell drawing, phase-label drawing, and the dashed
hand-vs-automation boundary. Same decomposition discipline as
:mod:`factoriax.analysis.recipe_graph`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


@dataclass(frozen=True)
class AchievementSpec:
    """One bit in the curriculum.

    This mirrors the per-bit metadata of the engine, but holds plain
    Python types. A caller can therefore draw the strip from a JSON
    dump without an import of the JAX-backed engine module.

    Attributes
    ----------
    bit :
        The index of this achievement in the achievement mask. It is
        also the large number drawn in the cell.
    name :
        The short label under the number. Long names overflow the
        cell, because nothing truncates them.
    phase :
        The name of the curriculum phase. Cells with the same phase
        must be next to each other in the sequence given to
        :func:`render`. The phase label spans from the first cell of a
        phase to the last.
    hand_craftable :
        True when a player can reach this achievement by hand. The
        switch from True to False draws the dashed boundary.
    """

    bit: int
    name: str
    phase: str
    hand_craftable: bool


@dataclass(frozen=True)
class StripLayout:
    """Geometry constants for the curriculum strip render."""

    cell_w: float = 1.6
    cell_h: float = 1.1
    gap: float = 0.05
    phase_gap: float = 0.45
    boundary_dash_extension: float = 0.35
    fallback_color: str = "#cccccc"
    stroke_darken: float = 0.45


def _darken(hex_color: str, factor: float) -> str:
    """Multiply each RGB channel by ``factor`` and return a hex string.

    This is not the same as :func:`factoriax.analysis.recipe_graph._darken`,
    which scales lightness in HLS and keeps the hue. A plain multiply
    moves a saturated color toward black faster, which suits a cell
    border but not a node fill.

    Parameters
    ----------
    hex_color :
        Any color string that matplotlib accepts.
    factor :
        The multiplier. A value below 1.0 darkens. The result is
        clamped at 0.0 from below, but not from above, so a factor
        above 1.0 can raise inside matplotlib.

    Returns
    -------
    str
        The new color as a ``#rrggbb`` string.
    """
    r, g, b = mcolors.to_rgb(hex_color)
    return mcolors.to_hex(
        (max(0.0, r * factor), max(0.0, g * factor), max(0.0, b * factor))
    )


def _text_color(fill_hex: str) -> str:
    """Return a near-black or near-white label color for a fill.

    The choice uses relative luminance with the sRGB weights, and
    switches at 0.65. That threshold is higher than the 0.55 used by
    :func:`factoriax.analysis.recipe_graph._text_palette`, so the two
    figures can disagree on a mid-tone fill.

    Parameters
    ----------
    fill_hex :
        The cell fill color.

    Returns
    -------
    str
        ``"#1A1A1A"`` on a light fill, ``"#FFFFFF"`` on a dark one.
    """
    r, g, b = mcolors.to_rgb(fill_hex)
    if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55:
        return "#1A1A1A"
    return "#FFFFFF"


def _cell_x_positions(
    achievements: Sequence[AchievementSpec],
    layout: StripLayout,
) -> list[float]:
    """Return the left x of each cell, in sequence order.

    Cells advance by ``cell_w + gap``. A change of phase between two
    cells widens that step to ``cell_w + phase_gap``. The wider space
    is the only separator between one phase group and the next, so the
    caller must keep the cells of a phase together.

    Parameters
    ----------
    achievements :
        The bit specs, in the order they will be drawn.
    layout :
        The geometry constants that set the widths.

    Returns
    -------
    list
        One x for each spec, in matplotlib data units. The first is
        always 0.0. An empty input gives an empty list, which
        :func:`render` cannot handle.
    """
    xs: list[float] = []
    cursor = 0.0
    prev_phase: str | None = None
    for spec in achievements:
        if prev_phase is not None and spec.phase != prev_phase:
            cursor += layout.phase_gap - layout.gap
        xs.append(cursor)
        cursor += layout.cell_w + layout.gap
        prev_phase = spec.phase
    return xs


def _draw_cell(
    ax: plt.Axes,
    spec: AchievementSpec,
    x: float,
    fill: str,
    stroke: str,
    layout: StripLayout,
) -> None:
    """Draw one cell: the bit number in bold, the name under it.

    Parameters
    ----------
    ax :
        The axes to draw on. The function modifies it in place and
        returns nothing.
    spec :
        The bit to draw. Only ``bit`` and ``name`` are read.
    x :
        The left edge of the cell, in data units. The bottom edge is
        always 0.0.
    fill :
        The cell fill color, which also sets the label color.
    stroke :
        The cell border color.
    layout :
        The geometry constants that set the cell size.
    """
    ax.add_patch(
        FancyBboxPatch(
            (x, 0.0),
            layout.cell_w,
            layout.cell_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fill,
            edgecolor=stroke,
            linewidth=0.9,
        )
    )
    label = _text_color(fill)
    centre_x = x + layout.cell_w / 2
    ax.text(
        centre_x,
        layout.cell_h * 0.62,
        str(spec.bit),
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
        color=label,
    )
    ax.text(
        centre_x,
        layout.cell_h * 0.22,
        spec.name,
        ha="center",
        va="center",
        fontsize=8,
        color=label,
    )


def _draw_phase_labels(
    ax: plt.Axes,
    achievements: Sequence[AchievementSpec],
    xs: Sequence[float],
    layout: StripLayout,
) -> None:
    """Write each phase name over the middle of its group of cells.

    The label spans from the first cell of a phase to the last. Take a
    phase that appears in two separate runs of the sequence. It gets
    one label, stretched over everything between those runs, including
    the cells of other phases. :func:`render` therefore states that
    the cells of a phase must be next to each other.

    Parameters
    ----------
    ax :
        The axes to draw on. The function modifies it in place and
        returns nothing.
    achievements :
        The bit specs, in draw order.
    xs :
        The left x of each cell, from :func:`_cell_x_positions`.
    layout :
        The geometry constants that set the label height.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for index, spec in enumerate(achievements):
        groups[spec.phase].append(index)
    label_y = layout.cell_h + 0.32
    for phase, indices in groups.items():
        left = xs[indices[0]]
        right = xs[indices[-1]] + layout.cell_w
        ax.text(
            (left + right) / 2,
            label_y,
            phase,
            ha="center",
            va="bottom",
            fontsize=10,
            weight="semibold",
            color="#1A1A1A",
        )


def _draw_boundary(
    ax: plt.Axes,
    achievements: Sequence[AchievementSpec],
    xs: Sequence[float],
    layout: StripLayout,
) -> None:
    """Draw the dashed line marking the hand-craftable → automation transition.

    The line sits in the gap after the last hand-craftable cell and
    before the first automation cell. Two short labels under it name
    the two sides.

    The function draws nothing when every bit is hand craftable, or
    when none is.

    Parameters
    ----------
    ax :
        The axes to draw on. The function modifies it in place and
        returns nothing.
    achievements :
        The bit specs, in draw order. The function takes the last
        hand-craftable index and the first automation index. Take a
        mixed sequence, such as hand, automation, hand. The first of
        those indexes then comes after the second, and the line lands
        in the wrong place. The caller must therefore put every hand
        craftable bit before every automation bit.
    xs :
        The left x of each cell, from :func:`_cell_x_positions`.
    layout :
        The geometry constants that set how far the line extends.
    """
    last_hand = None
    first_auto = None
    for index, spec in enumerate(achievements):
        if spec.hand_craftable:
            last_hand = index
        elif first_auto is None:
            first_auto = index
    if last_hand is None or first_auto is None:
        return
    line_x = (
        xs[last_hand]
        + layout.cell_w
        + (xs[first_auto] - (xs[last_hand] + layout.cell_w)) / 2
    )
    ax.plot(
        [line_x, line_x],
        [
            -layout.boundary_dash_extension,
            layout.cell_h + layout.boundary_dash_extension,
        ],
        linestyle=(0, (3, 2)),
        color="#555555",
        linewidth=1.0,
    )
    ax.text(
        line_x - 0.12,
        -layout.boundary_dash_extension - 0.05,
        "by hand",
        ha="right",
        va="top",
        fontsize=8,
        color="#555555",
        style="italic",
    )
    ax.text(
        line_x + 0.12,
        -layout.boundary_dash_extension - 0.05,
        "automation required",
        ha="left",
        va="top",
        fontsize=8,
        color="#555555",
        style="italic",
    )


def render(
    achievements: Sequence[AchievementSpec],
    out_path: Path | str,
    *,
    phase_palette: Mapping[str, str],
    layout: StripLayout | None = None,
) -> Path:
    """Draw the curriculum strip and return the path of the new file.

    Parameters
    ----------
    achievements :
        The bit specs, in draw order. The strip keeps that order. Two
        rules apply to it. Cells of one phase must be next to each
        other, and every hand-craftable bit must come before every
        automation bit. An empty sequence raises ``IndexError``.
    out_path :
        Where to write the image. The extension sets the format, and
        matplotlib accepts PNG and SVG among others. Missing parent
        directories are created.
    phase_palette :
        ``{phase_name: hex}``. A phase absent from this map gets
        :attr:`StripLayout.fallback_color` and no warning.
    layout :
        Geometry constants, or ``None`` for the defaults.

    Returns
    -------
    pathlib.Path
        ``out_path`` as a :class:`~pathlib.Path`, after the write.

    Notes
    -----
    The function writes a file and closes its own figure.
    """
    layout = layout or StripLayout()
    out_path = Path(out_path)

    xs = _cell_x_positions(achievements, layout)
    total_w = xs[-1] + layout.cell_w
    fig_w = max(total_w * 0.55, 7.0)
    fig_h = 2.3
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.4, total_w + 0.4)
    ax.set_ylim(-0.8, layout.cell_h + 0.95)
    ax.set_aspect("equal")
    ax.axis("off")

    for spec, x in zip(achievements, xs, strict=True):
        fill = phase_palette.get(spec.phase, layout.fallback_color)
        stroke = _darken(fill, layout.stroke_darken)
        _draw_cell(ax, spec, x, fill, stroke, layout)

    _draw_phase_labels(ax, achievements, xs, layout)
    _draw_boundary(ax, achievements, xs, layout)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
