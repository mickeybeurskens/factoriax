"""Render a recipe DAG as a layered crafting diagram.

Public surface:

* :class:`RecipeSpec`: plain-Python recipe, decoupled from the
  JAX-backed engine so callers can render diagrams from a JSON dump.
* :class:`Layout`: geometry constants.
* :func:`render`: top-level entry that consumes recipes and writes a
  PNG (or SVG, by file extension).

The renderer runs as a chain of helpers. Each one takes the values it
needs and returns plain data. Only the ``_draw_*`` helpers call
matplotlib.

The chain builds the adjacency map, then the positions, then the node
colors through :func:`factoriax.analysis.categories.item_palette`. It
then assigns the ports of each two-input recipe, gives every edge a
channel, and routes each edge. A route takes one of three forms:
between neighboring tiers, across a lane between rows, or over the
skyway above the top row. The boxes and edges are drawn last.
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

    This mirrors :class:`factoriax.engine.recipes.Recipe`, but names
    items with strings instead of enum values. A caller can therefore
    draw a diagram from a JSON dump with no import of the JAX-backed
    engine module.

    Attributes
    ----------
    output :
        The item this recipe makes. It is also the key under which the
        layout code finds the recipe.
    inputs :
        ``((item_name, count), ...)``. The order is kept, because two
        inputs at the same height fall back to it when they choose
        ports. A recipe must have one or two inputs, which
        :func:`_assign_ports` checks.
    ticks :
        How long one craft takes, in engine ticks. It is shown in the
        box subtitle and used nowhere else.
    output_count :
        How many items one craft makes.
    name :
        A display name for the box, or ``""`` to use ``output``.
    """

    output: str
    inputs: tuple[tuple[str, int], ...]
    ticks: int
    output_count: int = 1
    name: str = ""


@dataclass(frozen=True)
class Layout:
    """Geometry constants for the recipe DAG render.

    The defaults are tuned against the easy-rocket recipe set. Pass a
    different instance through the ``layout`` argument of
    :func:`render` when a figure needs other proportions.

    Distances are matplotlib data units. The aspect ratio is locked to
    1, so one unit along x equals one unit along y.

    Attributes
    ----------
    box_w, box_h :
        The size of one node box.
    tier_x_step :
        The distance from one tier column to the next. It must be
        larger than ``box_w``, or the gap between columns closes and
        the edge channels have nowhere to go.
    row_y_step :
        The distance from one row to the next.
    port_offset_frac :
        The offset of the two ports from the middle of a box, as a
        fraction of ``box_h``. See :attr:`port_offset`.
    left_inset, right_inset :
        The padding inside a box, before the input labels on the left
        and the result label on the right.
    input_name_max_chars :
        Where to cut an input label. A longer name is truncated with
        an ellipsis.
    hub_threshold :
        An item that feeds this many recipes counts as a hub. Its
        outgoing edges are drawn with ``hub_alpha``.
    hub_alpha, normal_alpha :
        The opacity of an edge from a hub, and from any other node.
        Fading the hub edges keeps a busy node such as a plate from
        hiding the rest of the graph.
    fallback_color :
        The fill for an item the palette does not know.
    stroke_darken :
        The factor passed to :func:`_darken` to turn a fill into its
        stroke.
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
    """Reduce recipes to the ``{output: [input_names]}`` map for layout.

    Parameters
    ----------
    recipes :
        The recipes to read. Two recipes with the same output collapse
        to one entry, and the last one wins.

    Returns
    -------
    dict
        ``{output_name: [input_name, ...]}`` with the counts dropped.
        The input order of each recipe is kept, because port
        assignment falls back to it.
    """
    return {recipe.output: [inp for inp, _ in recipe.inputs] for recipe in recipes}


def _positions(
    tiers: Mapping[str, int],
    rows: Mapping[str, int],
    layout: Layout,
) -> dict[str, tuple[float, float]]:
    """Turn a tier and a row into a point for each item.

    Columns are centered against each other. A tier with fewer items
    than the tallest one is pushed down by half the difference. A
    short column therefore sits in the middle of the figure, and not
    at the top.

    Parameters
    ----------
    tiers :
        ``{item: tier}`` from
        :func:`factoriax.analysis.graph_layout.assign_tiers`.
    rows :
        ``{item: row}`` from
        :func:`factoriax.analysis.graph_layout.order_within_tiers`.
    layout :
        The geometry constants that set the steps.

    Returns
    -------
    dict
        ``{item: (x, y)}`` in data units, at the middle of the box.
        x grows with the tier. y is 0 at row 0 and falls from there,
        so later rows have more negative y.

    Raises
    ------
    ValueError
        When ``tiers`` is empty, from the ``max`` over no tiers.
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
    """Scale the lightness of a color and return a hex string.

    The scaling happens in HLS and not in RGB, so the hue survives. A
    pale red therefore darkens to a deep red and does not move toward
    gray. That is what lets an edge keep the color of the node it
    left.

    Parameters
    ----------
    hex_color :
        Any color string that matplotlib accepts.
    factor :
        The multiplier on lightness. Below 1.0 darkens, above 1.0
        lightens.

    Returns
    -------
    str
        The new color as a ``#rrggbb`` string. Lightness is clamped to
        the range 0.0 to 1.0, so a factor of 0 gives black and a large
        factor gives white instead of raising.
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
    """Return a fill color and a stroke color for each item.

    Fills come from :func:`factoriax.analysis.categories.item_palette`.
    Every item in a category shares one fill, so the color gives the
    category and the label gives the item.

    Parameters
    ----------
    items :
        The item names to color. A mapping works too, and its keys are
        used.
    layout :
        Supplies ``fallback_color`` and ``stroke_darken``.

    Returns
    -------
    tuple
        ``(fill, stroke)``, two dictionaries with the same keys. Each
        stroke is its fill run through :func:`_darken`, so an edge
        keeps the color of the node it left.

    Notes
    -----
    The palette key is the upper-case ``ItemType`` member name, such as
    ``IRON_PLATE``. An item spelled any other way gets
    ``layout.fallback_color`` and no warning, so a caller that uses
    lower case gets a diagram that is uniformly gray.
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
    """Return three label colors that read against a fill.

    The choice uses relative luminance with the sRGB weights, and
    switches at 0.65. That threshold sits above every fill the palette
    produces, the brightest being the orange for half fabricates at
    0.573. A warm mid-tone fill therefore keeps light labels. Dark labels have
    too little contrast there.

    The threshold differs from the 0.55 used by
    :func:`factoriax.analysis.curriculum_strip._text_color`, so the two
    figures can disagree on a fill between the two values.

    Parameters
    ----------
    fill_hex :
        The box fill color.

    Returns
    -------
    tuple
        ``(strong, subtle, muted)``, for the result name, the input
        rows, and the subtitle. On a light fill they are three shades
        of near-black. On a dark fill the first two are white and the
        third is close to it.
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
    """Return the display form of an item name.

    Parameters
    ----------
    name :
        A snake case engine name, such as ``iron_plate``.

    Returns
    -------
    str
        The name in title case with the underscores replaced by
        spaces, such as ``Iron Plate``. Nothing is truncated here.
    """
    return name.replace("_", " ").title()


#: Ports on the left edge of a box. This value limits how many inputs
#: a recipe drawn here can have. It equals ``MAX_RECIPE_INPUTS`` in the
#: engine, but is repeated to keep the JAX-backed import out of this
#: module.
MAX_PORTS: int = 2


def _assign_ports(
    recipes: Iterable[RecipeSpec],
    positions: Mapping[str, tuple[float, float]],
    layout: Layout,
) -> tuple[
    dict[tuple[str, str], float],
    dict[str, dict[str, tuple[str, int]]],
]:
    """Choose where each input of a recipe meets its target box.

    A one-input recipe uses a middle port. A two-input recipe splits:
    the input drawn higher takes the top port, and the other takes the
    bottom. The higher source sent to the lower port makes the two
    arrows cross just before they arrive.

    Parameters
    ----------
    recipes :
        The recipes to place ports for.
    positions :
        ``{item: (x, y)}`` from :func:`_positions`. Only the y of each
        item is read, to decide which input is higher.
    layout :
        Supplies ``port_offset``, the distance from the middle of a
        box to its top and bottom port.

    Returns
    -------
    tuple
        ``(port_y, port_info)``. ``port_y`` maps
        ``(input, output)`` to the height at which that edge meets the
        box. ``port_info`` maps an output to
        ``{"top"/"bot"/"mid": (input_label, count)}``, which
        :func:`_draw_box` writes inside the box.

    Raises
    ------
    ValueError
        When a recipe has no inputs, or more than :data:`MAX_PORTS`
        inputs. Every input must get a port. The drawing loop reads all
        of them, and a missing port fails there instead, with no
        mention of the recipe that caused it.
    """
    port_y: dict[tuple[str, str], float] = {}
    port_info: dict[str, dict[str, tuple[str, int]]] = {}
    for recipe in recipes:
        arity = len(recipe.inputs)
        if not 1 <= arity <= MAX_PORTS:
            raise ValueError(
                f"Recipe for {recipe.output!r} has {arity} inputs. A box has "
                f"{MAX_PORTS} ports, so a recipe drawn here must have 1 to "
                f"{MAX_PORTS} inputs."
            )
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
    """Return the gaps in which an edge needs a channel.

    A gap is the empty band between two tier columns, and gap ``n``
    lies after tier ``n``.

    Parameters
    ----------
    source_tier :
        The tier of the item the edge leaves.
    target_tier :
        The tier of the recipe the edge enters. It must be greater
        than ``source_tier``.

    Returns
    -------
    tuple
        One gap for an edge between neighboring tiers. Two for a skip:
        the gap after the source and the gap before the target. The
        horizontal run between those two follows a lane or the skyway,
        neither of which needs a channel.
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

    Two edges that cross the same gap get different x, so their
    vertical runs do not sit on top of each other.

    Edges in a gap are sorted by ``(source_row, target_row)``. Edges
    that leave one source therefore land in neighboring channels, which
    keeps crossings rare.

    Parameters
    ----------
    recipes :
        The recipes to route.
    tiers :
        ``{item: tier}`` for every item named in ``recipes``.
    rows :
        ``{item: row}``, which sets the channel order inside a gap.
    layout :
        Supplies ``tier_x_step`` and ``box_w``, which bound each gap.

    Returns
    -------
    dict
        ``{(source, target, gap): x}`` in data units. Every x lies
        inside its gap, in the middle 76 percent of it, so no channel
        touches a tier column. One edge that crosses two gaps gets two
        entries.
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
    """Return the corner points of one edge.

    The route runs from the right side of the source box to the left
    side of the target box, in three regimes.

    A span of 1, between neighboring tiers, gives an L-bend through
    the source-side channel: four points.

    A span of 2 drops into the source-side channel, crosses along a
    lane between two rows, then drops into the target-side channel:
    six points. With no lane available it crosses on the skyway
    instead, which happens when every tier holds one node.

    A span of 3 or more climbs to the skyway over the top row, crosses
    there, then drops into the target-side channel: six points. A long
    edge between the rows cuts through the boxes of several tiers.

    Parameters
    ----------
    source_pos :
        The middle of the source box, as ``(x, y)``.
    target_pos :
        The middle of the target box. Only its x is read. The y comes
        from ``port_y``.
    port_y :
        The height at which the edge meets the target box, from
        :func:`_assign_ports`.
    source_tier, target_tier :
        The tiers at each end. Their difference selects the regime.
    src_channel :
        The x of the source-side vertical run, from
        :func:`_allocate_edge_channels`.
    tgt_channel :
        The x of the target-side vertical run, or ``None`` for a span
        of 1. A span of 2 or more asserts that it is not ``None``.
    inter_row_ys :
        The y of each lane between two rows. This can be empty.
    skyway_lane :
        The y of the horizontal run over the top row.
    layout :
        Supplies ``box_w``, which puts the two ends on the box edges.

    Returns
    -------
    list
        The corner points as ``(x, y)``, source end first. Four points
        for a span of 1, six for any larger span.
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
        # A diagram with one node per tier has no space between rows,
        # so there is no lane to take. The edge must still clear the
        # box between its two ends, and the skyway is the last route.
        lane_y = (
            min(inter_row_ys, key=lambda y: abs(y - target_avg))
            if inter_row_ys
            else skyway_lane
        )
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
    """Draw the ``Tier N`` labels and the dotted lines between tiers.

    Parameters
    ----------
    ax :
        The axes to draw on. The function modifies it in place and
        returns nothing.
    max_tier :
        The highest tier index. Labels run from 0 to this value, and a
        divider is drawn after each tier except the last.
    max_rows :
        The height of the tallest column, in rows. It sets how far
        down each divider reaches.
    skyway_headroom :
        The space kept over the top row for long edges. The labels sit
        above it.
    layout :
        The geometry constants that set the column positions.
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
    name :
        The label to cut.
    limit :
        The greatest length to allow, in characters.

    Returns
    -------
    str
        The name unchanged when it fits. Otherwise the first
        ``limit - 1`` characters, with trailing spaces removed and an
        ellipsis added, which is at most ``limit`` characters.
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

    A raw material gets one label in the middle of the box. An item
    with a recipe gets a card instead: the result name and an
    ``xN . Tt`` subtitle on the right, and one ``xN InputName`` row on
    the left at each port in use.

    Parameters
    ----------
    ax :
        The axes to draw on. The function modifies it in place and
        returns nothing.
    item :
        The item name, used as the label when the recipe has none.
    position :
        The middle of the box, as ``(x, y)``.
    recipe :
        The recipe that makes ``item``, or ``None`` for a raw
        material. ``None`` selects the single-label form.
    port_info :
        ``{"top"/"bot"/"mid": (input_label, count)}`` from
        :func:`_assign_ports`, or ``None`` to draw no input rows even
        for a recipe.
    fill :
        The box fill color. It also selects the label colors through
        :func:`_text_palette`.
    stroke :
        The box border color.
    layout :
        The geometry constants that set the box size and the insets.
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
        """Write one input row inside the box, at one port height.

        Parameters
        ----------
        port_y :
            The y at which to write, in data units.
        data :
            ``(input_label, count)`` for that port. The label is cut
            to ``layout.input_name_max_chars``.
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

    The legend draws one swatch for each entry, in the order given, so
    the caller sets that order by sorting the list.

    Parameters
    ----------
    ax :
        The axes to draw on. The function modifies it in place and
        returns nothing.
    entries :
        ``(category_name, hex)`` pairs, in the order to show them. An
        empty sequence draws no legend at all, rather than an empty
        box.
    layout :
        Supplies ``stroke_darken`` for the swatch borders.
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
    """Draw one edge as a polyline, with an arrowhead at the target.

    Parameters
    ----------
    ax :
        The axes to draw on. The function modifies it in place and
        returns nothing.
    vertices :
        The corner points from :func:`_route_edge`, source end first.
        At least two are needed, because the arrowhead is drawn along
        the last pair.
    color :
        The line color, normally the stroke color of the source node.
    alpha :
        The opacity, which fades the edges of a hub node.
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
    """Draw the recipe graph and return the path of the new file.

    The pipeline runs in stages. Tiers and rows come from
    :mod:`factoriax.analysis.graph_layout`. Positions come from
    :func:`_positions`, ports from :func:`_assign_ports`, and channels
    from :func:`_allocate_edge_channels`. :func:`_route_edge` gives
    each edge its corners. The ``_draw_*`` helpers then put those
    numbers on the axes.

    Parameters
    ----------
    recipes :
        The recipes to draw. Each must have one or two inputs. Every
        item named as an input gets a box, whether or not it has a
        recipe of its own.
    out_path :
        Where to write the image. The extension sets the format, and
        matplotlib accepts PNG and SVG among others. Missing parent
        directories are created.
    title :
        A title over the figure, or ``""`` for none.
    layout :
        Geometry constants, or ``None`` for the defaults.

    Returns
    -------
    pathlib.Path
        ``out_path`` as a :class:`~pathlib.Path`, after the write.

    Raises
    ------
    ValueError
        When a recipe has no inputs or more than
        :data:`MAX_PORTS`, or when the recipes hold a cycle, or when
        ``recipes`` is empty.

    Notes
    -----
    The function writes a file and closes its own figure. The figure
    size grows with the tier count and the tallest column. A large
    recipe set therefore gives a large image, and not a crowded
    one.
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
        """Return the opacity for the edges that leave one item.

        Parameters
        ----------
        source :
            The item the edges leave.

        Returns
        -------
        float
            ``layout.hub_alpha`` when the item feeds at least
            ``layout.hub_threshold`` recipes, and
            ``layout.normal_alpha`` otherwise. Fading a busy node such
            as a plate keeps its edges from hiding the rest.
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
