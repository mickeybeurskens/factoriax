"""Render a graded-achievement curriculum as a horizontal phase strip.

Public surface:

* :class:`AchievementSpec` — plain-Python achievement-bit record.
* :class:`StripLayout` — geometry constants.
* :func:`render` — top-level entry; consumes a sequence of specs plus a
  phase->hex palette and writes a PNG.

The render is intentionally palette-agnostic: callers (e.g. the paper
figure adapter) inject the colour scheme through ``phase_palette``.
This keeps the analysis module reusable across reports that style
their figures differently.

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

    Mirrors the engine's per-bit metadata but lives outside the JAX
    module so callers can render the strip from a JSON dump.

    Parameters
    ----------

    Returns
    -------

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
    """Multiply RGB by ``factor`` clamped to [0, 1] and return as hex.

    Parameters
    ----------
    hex_color: str :

    factor: float :


    Returns
    -------

    """
    r, g, b = mcolors.to_rgb(hex_color)
    return mcolors.to_hex(
        (max(0.0, r * factor), max(0.0, g * factor), max(0.0, b * factor))
    )


def _text_color(fill_hex: str) -> str:
    """Pick a near-black or near-white label colour by sRGB luminance.

    Parameters
    ----------
    fill_hex: str :


    Returns
    -------

    """
    r, g, b = mcolors.to_rgb(fill_hex)
    if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55:
        return "#1A1A1A"
    return "#FFFFFF"


def _cell_x_positions(
    achievements: Sequence[AchievementSpec],
    layout: StripLayout,
) -> list[float]:
    """

    Parameters
    ----------
    achievements: Sequence[AchievementSpec] :

    layout: StripLayout :


    Returns
    -------
    type


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
    """Draw one number-primary cell: bold bit integer, short name below.

    Parameters
    ----------
    ax: plt.Axes :

    spec: AchievementSpec :

    x: float :

    fill: str :

    stroke: str :

    layout: StripLayout :


    Returns
    -------

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
    """Write each phase name above the centre of its cell group.

    Parameters
    ----------
    ax: plt.Axes :

    achievements: Sequence[AchievementSpec] :

    xs: Sequence[float] :

    layout: StripLayout :


    Returns
    -------

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

    The line sits in the gap between the last hand-craftable cell and
    the first automation cell. Two short labels above and below name
    the regimes on either side.

    Parameters
    ----------
    ax: plt.Axes :

    achievements: Sequence[AchievementSpec] :

    xs: Sequence[float] :

    layout: StripLayout :


    Returns
    -------

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
    """Render the curriculum strip to ``out_path`` and return the resolved path.

    Parameters
    ----------
    achievements :
        Ordered bit specs. The strip honours the given
        order; phases must appear contiguously (e.g. all Bootstrap
        bits before any Miners-up bits).
    out_path :
        Destination PNG (or SVG by extension).
    phase_palette :
        ``{phase_name: hex}`` lookup. Phases missing from
        the palette get :attr:`StripLayout.fallback_color`.
    layout :
        Optional geometry override.
    achievements: Sequence[AchievementSpec] :

    out_path: Path | str :

    * :

    phase_palette: Mapping[str :

    str] :

    layout: StripLayout | None :
         (Default value = None)

    Returns
    -------

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
