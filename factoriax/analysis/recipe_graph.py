"""Render a recipe DAG as a layered crafting diagram.

Public surface:

* :class:`RecipeSpec` — plain-Python recipe, decoupled from the
  JAX-backed engine so callers can render diagrams from a JSON dump.
* :class:`Layout` — geometry constants.
* :func:`render` — top-level entry that consumes recipes and writes a
  PNG (or SVG, by file extension).

Internally the renderer is broken into focused helpers: adjacency
construction, position computation, node colouring through
:func:`factoriax.analysis.categories.item_palette`, port assignment
for two-input recipes, per-edge channel allocation, edge routing
(adjacent tiers, single-tier skips through inter-row lanes, longer
skips through a skyway above the top row), and final box and edge
drawing. Each helper takes the inputs it needs and returns plain data;
matplotlib calls live in the ``_draw_*`` helpers.
"""

from __future__ import annotations

import colorsys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, PathPatch
from matplotlib.path import Path as MplPath

from factoriax.analysis.categories import (
    category_palette,
    item_palette,
    item_to_category,
)
from factoriax.analysis.graph_layout import assign_tiers, order_within_tiers


@dataclass(frozen=True)
class RecipeSpec:
    """Plain-Python recipe used by the renderer.
    
    Mirrors :class:`factoriax.engine.recipes.Recipe` but stores item
    names as strings so callers can render diagrams from a JSON dump
    without importing the JAX-backed engine module directly.

    Parameters
    ----------

    Returns
    -------

    """

    output: str
    inputs: tuple[tuple[str, int], ...]
    ticks: int
    output_count: int = 1
    name: str = ""


@dataclass(frozen=True)
class Layout:
    """Geometry constants for the recipe DAG render.
    
    Tuned against the easy-rocket recipe set. Override fields when the
    figure needs different proportions; the renderer accepts any
    instance through the ``layout=`` argument.

    Parameters
    ----------

    Returns
    -------

    """

    box_w: float = 4.0
    box_h: float = 0.7
    tier_x_step: float = 6.0
    row_y_step: float = 1.5
    port_offset_frac: float = 0.30
    left_inset: float = 0.06
    right_inset: float = 0.06
    input_name_max_chars: int = 14
    hub_threshold: int = 4
    hub_alpha: float = 0.55
    normal_alpha: float = 1.0
    fallback_color: str = "#cccccc"
    stroke_darken: float = 0.55

    @property
    def port_offset(self) -> float:
        """Distance from a box's centre to its top or bottom port."""
        return self.box_h * self.port_offset_frac


# ---------------------------------------------------------------------------
# Adjacency, positions, colours
# ---------------------------------------------------------------------------


def _adjacency(recipes: Iterable[RecipeSpec]) -> dict[str, list[str]]:
    """Build the ``{output: [input_names]}`` map used for layering.

    Parameters
    ----------
    recipes: Iterable[RecipeSpec] :
        

    Returns
    -------

    """
    return {recipe.output: [inp for inp, _ in recipe.inputs] for recipe in recipes}


def _positions(
    tiers: Mapping[str, int],
    rows: Mapping[str, int],
    layout: Layout,
) -> dict[str, tuple[float, float]]:
    """

    Parameters
    ----------
    tiers: Mapping[str :
        
    int] :
        
    rows: Mapping[str :
        
    layout: Layout :
        

    Returns
    -------
    type
        

    """
    by_tier: dict[int, list[str]] = defaultdict(list)
    for item, tier in tiers.items():
        by_tier[tier].append(item)
    max_rows = max(len(items) for items in by_tier.values())
    positions: dict[str, tuple[float, float]] = {}
    for item, tier in tiers.items():
        tier_count = len(by_tier[tier])
        offset = (max_rows - tier_count) / 2
        x = tier * layout.tier_x_step + layout.box_w / 2
        y = -(rows[item] + offset) * layout.row_y_step
        positions[item] = (x, y)
    return positions


def _darken(hex_color: str, factor: float) -> str:
    """

    Parameters
    ----------
    hex_color: str :
        
    factor: float :
        

    Returns
    -------
    type
        Drops lightness via the HLS model rather than multiplying RGB so
        hues stay recognisable even when the original is very light.

    """
    r, g, b = mcolors.to_rgb(hex_color)
    hue, lightness, sat = colorsys.rgb_to_hls(r, g, b)
    return mcolors.to_hex(
        colorsys.hls_to_rgb(hue, max(0.0, min(1.0, lightness * factor)), sat)
    )


def _node_colors(
    items: Iterable[str],
    layout: Layout,
) -> tuple[dict[str, str], dict[str, str]]:
    """

    Parameters
    ----------
    items: Iterable[str] :
        
    layout: Layout :
        

    Returns
    -------
    type
        Fills come from :func:`factoriax.analysis.categories.item_palette`,
        which shades each item by its index within its category. Strokes
        are darker variants of the fill so edges leaving a node read as
        the same colour family as the node itself.

    """
    palette = item_palette()
    fill: dict[str, str] = {}
    stroke: dict[str, str] = {}
    for item in items:
        base = palette.get(item, layout.fallback_color)
        fill[item] = base
        stroke[item] = _darken(base, layout.stroke_darken)
    return fill, stroke


def _text_palette(fill_hex: str) -> tuple[str, str, str]:
    """

    Parameters
    ----------
    fill_hex: str :
        

    Returns
    -------
    type
        Selects dark or light label colours based on the fill's relative
        luminance (sRGB coefficients). The threshold sits above the saturated
        orange used for Rocket Parts so that warm fills still read with
        light labels rather than dark ones.

    """
    r, g, b = mcolors.to_rgb(fill_hex)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if luminance > 0.65:
        return ("#1A1A1A", "#555555", "#666666")
    return ("#FFFFFF", "#FFFFFF", "#F0F0F0")


# ---------------------------------------------------------------------------
# Ports, channels, edge routing
# ---------------------------------------------------------------------------


def _pretty(name: str) -> str:
    """Display form of an item name: title case, no underscores.

    Parameters
    ----------
    name: str :
        

    Returns
    -------

    """
    return name.replace("_", " ").title()


def _assign_ports(
    recipes: Iterable[RecipeSpec],
    positions: Mapping[str, tuple[float, float]],
    layout: Layout,
) -> tuple[
    dict[tuple[str, str], float],
    dict[str, dict[str, tuple[str, int]]],
]:
    """Assign each recipe input a top, middle, or bottom port.
    
    For two-input recipes the input whose source row is visually higher
    enters the top port; the other enters the bottom. This minimises
    arrow crossings near the target. Single-input recipes use a middle
    port.

    Parameters
    ----------
    recipes: Iterable[RecipeSpec] :
        
    positions: Mapping[str :
        
    tuple[float :
        
    float]] :
        
    layout: Layout :
        

    Returns
    -------
    ``port_info[output]`` — ``{"top"/"bot"/"mid"
        (input_name, count)}``
    ``port_info[output]`` — ``{"top"/"bot"/"mid"
        (input_name, count)}``
        for rendering the per-port labels inside the box.

    """
    port_y: dict[tuple[str, str], float] = {}
    port_info: dict[str, dict[str, tuple[str, int]]] = {}
    for recipe in recipes:
        _, ty = positions[recipe.output]
        sorted_inputs = sorted(recipe.inputs, key=lambda ic: -positions[ic[0]][1])
        if len(sorted_inputs) == 1:
            inp, count = sorted_inputs[0]
            port_y[(inp, recipe.output)] = ty
            port_info[recipe.output] = {"mid": (_pretty(inp), count)}
            continue
        top_inp, top_count = sorted_inputs[0]
        bot_inp, bot_count = sorted_inputs[1]
        port_y[(top_inp, recipe.output)] = ty + layout.port_offset
        port_y[(bot_inp, recipe.output)] = ty - layout.port_offset
        port_info[recipe.output] = {
            "top": (_pretty(top_inp), top_count),
            "bot": (_pretty(bot_inp), bot_count),
        }
    return port_y, port_info


def _gaps_for_edge(source_tier: int, target_tier: int) -> tuple[int, ...]:
    """

    Parameters
    ----------
    source_tier: int :
        
    target_tier: int :
        

    Returns
    -------
    type
        Adjacent tiers use the single gap between them. Tier skips use the
        gap on the source side and the gap on the target side; the
        horizontal traverse between them happens in an inter-row lane or
        the skyway, neither of which counts as a "gap" for channelling.

    """
    if target_tier - source_tier == 1:
        return (source_tier,)
    return (source_tier, target_tier - 1)


def _allocate_edge_channels(
    recipes: Iterable[RecipeSpec],
    tiers: Mapping[str, int],
    rows: Mapping[str, int],
    layout: Layout,
) -> dict[tuple[str, str, int], float]:
    """Give every edge a unique vertical x in every gap it traverses.
    
    Edges within a gap are sorted by ``(source_row, target_row)`` so
    edges that fan out from the same source land in adjacent channels,
    keeping crossings rare.

    Parameters
    ----------
    recipes: Iterable[RecipeSpec] :
        
    tiers: Mapping[str :
        
    int] :
        
    rows: Mapping[str :
        
    layout: Layout :
        

    Returns
    -------

    """
    edges_in_gap: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for recipe in recipes:
        for inp, _ in recipe.inputs:
            for gap in _gaps_for_edge(tiers[inp], tiers[recipe.output]):
                edges_in_gap[gap].append((inp, recipe.output))

    channel_x: dict[tuple[str, str, int], float] = {}
    for gap, edges in edges_in_gap.items():
        unique = list(dict.fromkeys(edges))
        unique.sort(key=lambda e: (rows[e[0]], rows[e[1]]))
        gap_left = gap * layout.tier_x_step + layout.box_w
        gap_right = (gap + 1) * layout.tier_x_step
        usable_l = gap_left + 0.12 * (gap_right - gap_left)
        usable_r = gap_right - 0.12 * (gap_right - gap_left)
        n = len(unique)
        for index, (src, tgt) in enumerate(unique):
            frac = (index + 1) / (n + 1) if n > 1 else 0.5
            channel_x[(src, tgt, gap)] = usable_l + frac * (usable_r - usable_l)
    return channel_x


def _route_edge(
    source_pos: tuple[float, float],
    target_pos: tuple[float, float],
    port_y: float,
    source_tier: int,
    target_tier: int,
    src_channel: float,
    tgt_channel: float | None,
    inter_row_ys: Sequence[float],
    skyway_lane: float,
    layout: Layout,
) -> list[tuple[float, float]]:
    """

    Parameters
    ----------
    source_pos: tuple[float :
        
    float] :
        
    target_pos: tuple[float :
        
    port_y: float :
        
    source_tier: int :
        
    target_tier: int :
        
    src_channel: float :
        
    tgt_channel: float | None :
        
    inter_row_ys: Sequence[float] :
        
    skyway_lane: float :
        
    layout: Layout :
        

    Returns
    -------
    type
        Three routing regimes:
        
        * Adjacent tiers: classic L-bend through the source-side gap to the
        target's port.
        * Single-tier skip: source-gap drop, inter-row lane traverse,
        target-gap drop.
        * Two-or-more-tier skip: source-gap rise into the skyway above the
        top row, horizontal traverse, target-gap drop.

    """
    sx, sy = source_pos
    tx, _ = target_pos
    source_right = (sx + layout.box_w / 2, sy)
    target_left = (tx - layout.box_w / 2, port_y)
    span = target_tier - source_tier
    if span == 1:
        return [
            source_right,
            (src_channel, sy),
            (src_channel, port_y),
            target_left,
        ]
    assert tgt_channel is not None
    if span == 2:
        target_avg = (sy + port_y) / 2
        lane_y = min(inter_row_ys, key=lambda y: abs(y - target_avg))
        return [
            source_right,
            (src_channel, sy),
            (src_channel, lane_y),
            (tgt_channel, lane_y),
            (tgt_channel, port_y),
            target_left,
        ]
    return [
        source_right,
        (src_channel, sy),
        (src_channel, skyway_lane),
        (tgt_channel, skyway_lane),
        (tgt_channel, port_y),
        target_left,
    ]


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------


def _draw_tier_headers(
    ax: plt.Axes,
    max_tier: int,
    max_rows: int,
    skyway_headroom: float,
    layout: Layout,
) -> None:
    """Draw the ``Tier N`` labels and dotted dividers between tiers.

    Parameters
    ----------
    ax: plt.Axes :
        
    max_tier: int :
        
    max_rows: int :
        
    skyway_headroom: float :
        
    layout: Layout :
        

    Returns
    -------

    """
    header_y = 0.8 + skyway_headroom
    for tier in range(max_tier + 1):
        x_center = tier * layout.tier_x_step + layout.box_w / 2
        ax.text(
            x_center,
            header_y,
            f"Tier {tier}",
            ha="center",
            va="center",
            fontsize=10,
            color="#555555",
            weight="bold",
        )
        if tier < max_tier:
            x_div = (tier + 1) * layout.tier_x_step - (
                layout.tier_x_step - layout.box_w
            ) / 2
            ax.plot(
                [x_div, x_div],
                [-max_rows * layout.row_y_step - 0.3, header_y - 0.4],
                linestyle=(0, (2, 3)),
                color="#999999",
                linewidth=0.6,
                alpha=0.5,
            )


def _truncate(name: str, limit: int) -> str:
    """Cap ``name`` at ``limit`` characters with an ellipsis suffix.

    Parameters
    ----------
    name: str :
        
    limit: int :
        

    Returns
    -------

    """
    return name if len(name) <= limit else name[: limit - 1].rstrip() + "…"


def _draw_box(
    ax: plt.Axes,
    item: str,
    position: tuple[float, float],
    recipe: RecipeSpec | None,
    port_info: Mapping[str, tuple[str, int]] | None,
    fill: str,
    stroke: str,
    layout: Layout,
) -> None:
    """Draw one node box.
    
    Raw materials get a single centred label. Recipe outputs get a
    recipe card: result name and ``×N · Tt`` subtitle on the right;
    ``×N InputName`` rows on the left at each used port.

    Parameters
    ----------
    ax: plt.Axes :
        
    item: str :
        
    position: tuple[float :
        
    float] :
        
    recipe: RecipeSpec | None :
        
    port_info: Mapping[str :
        
    tuple[str :
        
    int]] | None :
        
    fill: str :
        
    stroke: str :
        
    layout: Layout :
        

    Returns
    -------

    """
    x, y = position
    ax.add_patch(
        FancyBboxPatch(
            (x - layout.box_w / 2, y - layout.box_h / 2),
            layout.box_w,
            layout.box_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fill,
            edgecolor=stroke,
            linewidth=0.8,
        )
    )
    strong, subtle, muted = _text_palette(fill)
    display_name = (recipe.name if recipe and recipe.name else item).replace("_", " ")
    if recipe is None:
        ax.text(
            x,
            y,
            display_name,
            ha="center",
            va="center",
            fontsize=8,
            color=strong,
            weight="bold",
        )
        return

    result_x = x + layout.box_w / 2 - layout.right_inset
    ax.text(
        result_x,
        y + 0.10,
        display_name,
        ha="right",
        va="center",
        fontsize=8.5,
        color=strong,
        weight="bold",
    )
    ax.text(
        result_x,
        y - 0.18,
        f"×{recipe.output_count} · {recipe.ticks}t",
        ha="right",
        va="center",
        fontsize=6.5,
        color=muted,
        weight="semibold",
    )
    if port_info is None:
        return
    port_label_x = x - layout.box_w / 2 + layout.left_inset

    def write(port_y: float, data: tuple[str, int]) -> None:
        """

        Parameters
        ----------
        port_y: float :
            
        data: tuple[str :
            
        int] :
            

        Returns
        -------

        """
        name, count = data
        ax.text(
            port_label_x,
            port_y,
            f"×{count} {_truncate(name, layout.input_name_max_chars)}",
            ha="left",
            va="center",
            fontsize=7,
            color=subtle,
            weight="semibold",
        )

    if "top" in port_info:
        write(y + layout.port_offset, port_info["top"])
    if "bot" in port_info:
        write(y - layout.port_offset, port_info["bot"])
    if "mid" in port_info:
        write(y, port_info["mid"])


def _draw_legend(
    ax: plt.Axes,
    entries: Sequence[tuple[str, str]],
    layout: Layout,
) -> None:
    """Attach a horizontal category legend below the axes.
    
    ``entries`` is an ordered ``[(category_name, hex_color), ...]`` list.
    The legend renders one swatch per entry, in the order given, so
    callers control the visual ordering by sorting the list.

    Parameters
    ----------
    ax: plt.Axes :
        
    entries: Sequence[tuple[str :
        
    str]] :
        
    layout: Layout :
        

    Returns
    -------

    """
    if not entries:
        return
    handles = [
        Patch(
            facecolor=color,
            edgecolor=_darken(color, layout.stroke_darken),
            linewidth=0.8,
            label=name,
        )
        for name, color in entries
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        handlelength=1.4,
        handleheight=1.1,
        columnspacing=1.6,
    )


def _draw_edge(
    ax: plt.Axes,
    vertices: Sequence[tuple[float, float]],
    color: str,
    alpha: float,
) -> None:
    """Draw the edge polyline plus a small arrowhead at the target end.

    Parameters
    ----------
    ax: plt.Axes :
        
    vertices: Sequence[tuple[float :
        
    float]] :
        
    color: str :
        
    alpha: float :
        

    Returns
    -------

    """
    path = MplPath(
        list(vertices),
        [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices) - 1),
    )
    ax.add_patch(
        PathPatch(
            path,
            fill=False,
            edgecolor=color,
            linewidth=1.0,
            alpha=alpha,
            joinstyle="miter",
            capstyle="butt",
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            vertices[-2],
            vertices[-1],
            arrowstyle="-|>",
            mutation_scale=8,
            color=color,
            linewidth=1.0,
            alpha=alpha,
            shrinkA=0,
            shrinkB=2,
        )
    )


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def render(
    recipes: Sequence[RecipeSpec],
    out_path: Path | str,
    *,
    title: str = "",
    layout: Layout | None = None,
) -> Path:
    """Render the recipe DAG to ``out_path`` and return the resolved path.

    Parameters
    ----------
    recipes: Sequence[RecipeSpec] :
        
    out_path: Path | str :
        
    * :
        
    title: str :
         (Default value = "")
    layout: Layout | None :
         (Default value = None)

    Returns
    -------

    """
    layout = layout or Layout()
    out_path = Path(out_path)

    adjacency = _adjacency(recipes)
    tiers = assign_tiers(adjacency)
    rows = order_within_tiers(adjacency, tiers)

    by_output: dict[str, RecipeSpec] = {r.output: r for r in recipes}
    by_tier: dict[int, list[str]] = defaultdict(list)
    for item, tier in tiers.items():
        by_tier[tier].append(item)
    max_tier = max(tiers.values())
    max_rows = max(len(items) for items in by_tier.values())

    positions = _positions(tiers, rows, layout)
    fill_color, stroke_color = _node_colors(tiers, layout)

    long_skip_count = sum(
        1
        for recipe in recipes
        for inp, _ in recipe.inputs
        if tiers[recipe.output] - tiers[inp] >= 2
    )
    skyway_headroom = 0.5 + 0.18 * long_skip_count

    out_degree: dict[str, int] = defaultdict(int)
    for recipe in recipes:
        for inp, _ in recipe.inputs:
            out_degree[inp] += 1

    def alpha_for(source: str) -> float:
        """

        Parameters
        ----------
        source: str :
            

        Returns
        -------

        """
        return (
            layout.hub_alpha
            if out_degree[source] >= layout.hub_threshold
            else layout.normal_alpha
        )

    fig_w = (max_tier + 1) * layout.tier_x_step / 1.6
    fig_h = max_rows * layout.row_y_step / 1.4 + 1.5 + skyway_headroom * 0.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.5, (max_tier + 1) * layout.tier_x_step)
    ax.set_ylim(-(max_rows + 1) * layout.row_y_step, 1.5 + skyway_headroom)
    ax.set_aspect("equal")
    ax.axis("off")

    port_y_map, port_info_map = _assign_ports(recipes, positions, layout)
    channels = _allocate_edge_channels(recipes, tiers, rows, layout)

    top_y = max(p[1] for p in positions.values()) + layout.box_h / 2
    skyway_base = top_y + 0.4
    row_ys = sorted({-i * layout.row_y_step for i in range(max_rows)}, reverse=True)
    inter_row_ys = [(row_ys[i] + row_ys[i + 1]) / 2 for i in range(len(row_ys) - 1)]
    skyway_seen: dict[tuple[str, str], int] = {}

    _draw_tier_headers(ax, max_tier, max_rows, skyway_headroom, layout)

    for recipe in recipes:
        target_pos = positions[recipe.output]
        target_tier = tiers[recipe.output]
        for inp, _ in recipe.inputs:
            source_tier = tiers[inp]
            source_pos = positions[inp]
            port_y = port_y_map[(inp, recipe.output)]
            src_channel = channels[(inp, recipe.output, source_tier)]
            tgt_channel = (
                channels[(inp, recipe.output, target_tier - 1)]
                if target_tier - source_tier >= 2
                else None
            )
            if target_tier - source_tier >= 3:
                key = (inp, recipe.output)
                if key not in skyway_seen:
                    skyway_seen[key] = len(skyway_seen)
                skyway_lane = skyway_base + 0.18 * skyway_seen[key]
            else:
                skyway_lane = skyway_base
            vertices = _route_edge(
                source_pos,
                target_pos,
                port_y,
                source_tier,
                target_tier,
                src_channel,
                tgt_channel,
                inter_row_ys,
                skyway_lane,
                layout,
            )
            _draw_edge(ax, vertices, stroke_color[inp], alpha_for(inp))

    for item, position in positions.items():
        _draw_box(
            ax,
            item,
            position,
            by_output.get(item),
            port_info_map.get(item),
            fill_color[item],
            stroke_color[item],
            layout,
        )

    item_category = item_to_category()
    category_colors = category_palette()
    categories_present = {
        item_category[item] for item in tiers if item in item_category
    }
    legend_entries = [
        (name, category_colors[name])
        for name in category_colors
        if name in categories_present
    ]
    _draw_legend(ax, legend_entries, layout)

    if title:
        ax.set_title(title, fontsize=12, color="#1A1A1A", pad=14)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
