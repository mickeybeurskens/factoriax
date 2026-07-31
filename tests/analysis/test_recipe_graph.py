"""Tests for the pure helpers in :mod:`factoriax.analysis.recipe_graph`.

The renderer has two halves. One half turns recipes into plain
numbers: tiers, positions, colors, port heights, channel offsets, and
edge polylines. The other half gives those numbers to matplotlib. These
tests assert on the first half. The ``_draw_*`` helpers make an image.
An assertion on an image needs a committed reference file, and every
deliberate style change makes that file invalid.

:func:`render` appears once, in :class:`TestRenderShapes`. It runs there
only to show that a diagram appears at all. Two of its crashes came from
helper arguments that no unit test built. The whole pipeline therefore
runs over the recipe shapes that caused them.

Coordinates are matplotlib data units, not pixels. The x axis grows to
the right with the tier. The y axis grows upward, but the rows descend.
A node in row 0 therefore has y of 0, and later rows have more negative
y. "Higher" in these tests means greater y, which is nearer the top of
the figure.

A gap is the empty band between two tier columns. Every edge that
crosses a gap gets its own vertical line inside that gap, called a
channel. Two edges that cross the same band therefore do not overlap.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import matplotlib.colors as mcolors
import pytest

from factoriax.analysis.categories import item_palette
from factoriax.analysis.graph_layout import assign_tiers, order_within_tiers
from factoriax.analysis.recipe_graph import (
    Layout,
    RecipeSpec,
    _adjacency,
    _allocate_edge_channels,
    _assign_ports,
    _darken,
    _gaps_for_edge,
    _node_colors,
    _positions,
    _pretty,
    _route_edge,
    _text_palette,
    render,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

#: A minimal recipe chain: ore -> plate -> frame, plus coal that feeds
#: frame. Three tiers and one two-input recipe. This is the smallest
#: shape that runs port assignment and channel allocation together.
SIMPLE_RECIPES: tuple[RecipeSpec, ...] = (
    RecipeSpec(output="plate", inputs=(("ore", 1),), ticks=10),
    RecipeSpec(output="frame", inputs=(("plate", 2), ("coal", 1)), ticks=20),
)


@pytest.fixture
def simple_geometry():
    """Run the full layout pipeline over :data:`SIMPLE_RECIPES`.

    Returns
    -------
    tuple
        ``(adjacency, tiers, rows, positions, layout)``. The stages
        run in that order, and each stage reads the result of the one
        before it. A test that needs positions therefore cannot skip
        tier assignment or row assignment.
    """
    adjacency = _adjacency(SIMPLE_RECIPES)
    tiers = assign_tiers(adjacency)
    rows = order_within_tiers(adjacency, tiers)
    layout = Layout()
    positions = _positions(tiers, rows, layout)
    return adjacency, tiers, rows, positions, layout


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class TestRecipeSpec:
    """The plain-Python recipe view."""

    def test_default_output_count_and_name(self) -> None:
        """A recipe that names neither field yields one item and no label."""
        r = RecipeSpec(output="plate", inputs=(("ore", 1),), ticks=10)
        assert r.output_count == 1
        assert r.name == ""

    def test_is_frozen(self) -> None:
        """Reject assignment, so a spec cannot change after layout runs.

        The layout code reads the specs to make tiers, positions, and
        channels, and it caches those results. A change to a spec
        between two stages leaves the cached numbers on a recipe that
        no longer exists.
        """
        r = RecipeSpec(output="plate", inputs=(("ore", 1),), ticks=10)
        with pytest.raises(FrozenInstanceError):
            r.output = "wire"  # type: ignore[misc]


class TestLayout:
    """Layout geometry constants."""

    def test_port_offset_uses_fraction_of_height(self) -> None:
        """``port_offset`` scales with the box, not with absolute units.

        A taller box moves its two ports further apart, by the same
        proportion. The arrowheads therefore keep their spacing inside
        the box edge at every size.
        """
        layout = Layout(box_h=2.0, port_offset_frac=0.25)
        assert layout.port_offset == pytest.approx(0.5)

    def test_defaults_are_positive(self) -> None:
        """Give every default distance a value of more than zero.

        A step of zero stacks every tier, or every row, on one
        coordinate. This fault gives an unreadable image and not an
        exception, so no test catches it later.
        """
        layout = Layout()
        assert layout.box_w > 0 and layout.box_h > 0
        assert layout.tier_x_step > 0 and layout.row_y_step > 0


# ---------------------------------------------------------------------------
# Adjacency and positions
# ---------------------------------------------------------------------------


class TestAdjacency:
    """``_adjacency`` drops the counts and keeps the input names."""

    def test_returns_output_to_input_names(self) -> None:
        """Each recipe becomes one entry keyed by its output."""
        adjacency = _adjacency(SIMPLE_RECIPES)
        assert adjacency == {"plate": ["ore"], "frame": ["plate", "coal"]}

    def test_preserves_input_order(self) -> None:
        """Input order survives, because port assignment reads it.

        Two inputs at the same height use recipe order to decide
        which one takes the top port. A sort or a de-duplication here
        makes that choice arbitrary.
        """
        r = RecipeSpec(
            output="x",
            inputs=(("a", 1), ("b", 2), ("c", 3)),
            ticks=1,
        )
        assert _adjacency([r]) == {"x": ["a", "b", "c"]}


class TestPositions:
    """``_positions`` places items on the tier and row grid."""

    def test_every_item_has_a_position(self, simple_geometry) -> None:
        """No tiered item is left without coordinates."""
        _, tiers, _, positions, _ = simple_geometry
        assert set(positions) == set(tiers)

    def test_x_increases_with_tier(self, simple_geometry) -> None:
        """The chain reads left to right, from raw material to product."""
        _, _, _, positions, _ = simple_geometry
        assert positions["ore"][0] < positions["plate"][0] < positions["frame"][0]

    def test_x_uses_tier_step(self, simple_geometry) -> None:
        """x is ``tier * tier_x_step + box_w / 2``.

        The half-width term makes x the center of the box rather than
        its left edge, which is what the drawing code expects.
        """
        _, tiers, _, positions, layout = simple_geometry
        for item, tier in tiers.items():
            expected = tier * layout.tier_x_step + layout.box_w / 2
            assert positions[item][0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------


class TestDarken:
    """Lightness scaling through the HLS model."""

    def test_factor_one_is_a_no_op(self) -> None:
        """Factor 1.0 returns the input color.

        The comparison allows 1e-3 because the value makes a round
        trip through RGB and HLS. Each step of that trip rounds.
        """
        original = "#7f7f7f"
        out = _darken(original, 1.0)
        assert mcolors.to_rgb(out) == pytest.approx(mcolors.to_rgb(original), abs=1e-3)

    def test_factor_below_one_lowers_luminance(self) -> None:
        """A factor under 1 gives a strictly darker color.

        A change of lightness in HLS keeps the hue. A pale red
        therefore darkens to a deep red and does not move toward gray.
        The outline of a node keeps the color of its fill.
        """
        bright = "#ff8080"
        darker = _darken(bright, 0.5)

        def lum(hx: str) -> float:
            """Relative luminance of a hex color, using sRGB weights."""
            r, g, b = mcolors.to_rgb(hx)
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        assert lum(darker) < lum(bright)

    def test_factor_zero_gives_black(self) -> None:
        """Factor 0 drives lightness to 0, which is black at any hue."""
        out = _darken("#ff0000", 0.0)
        assert mcolors.to_rgb(out) == pytest.approx((0.0, 0.0, 0.0), abs=1e-3)

    def test_factor_above_one_clamps_at_white(self) -> None:
        """Clamp lightness to 1.0, so a large factor cannot overflow.

        ``colorsys.hls_to_rgb`` accepts a lightness of more than 1. It
        then returns channel values of more than 1, which matplotlib
        rejects. Without the clamp, a call with a large factor raises.
        """
        out = _darken("#808080", 10.0)
        assert mcolors.to_rgb(out) == pytest.approx((1.0, 1.0, 1.0), abs=1e-3)


class TestNodeColors:
    """``_node_colors`` returns matched fill and stroke maps."""

    def test_returns_color_per_item(self) -> None:
        """Both maps cover exactly the items passed in."""
        fill, stroke = _node_colors(["ore", "plate"], Layout())
        assert set(fill) == set(stroke) == {"ore", "plate"}

    def test_known_item_uses_the_palette_not_the_fallback(self) -> None:
        """A real engine item name resolves through the palette.

        The palette key is the upper-case :class:`ItemType` member
        name, such as ``IRON_PLATE``. An item with a different spelling
        gets the fallback color. A caller that spells every item in
        lower case therefore gets a diagram that is uniformly gray.

        """
        layout = Layout(fallback_color="#abcdef")
        fill, _ = _node_colors(["IRON_PLATE"], layout)
        assert fill["IRON_PLATE"] == item_palette()["IRON_PLATE"]
        assert fill["IRON_PLATE"] != layout.fallback_color

    def test_unknown_items_use_fallback_fill(self) -> None:
        """An item the palette does not know falls back instead of raising.

        The renderer accepts recipes from a JSON dump, so it can meet
        item names that the engine's palette has never seen.
        """
        layout = Layout(fallback_color="#abcdef")
        fill, _ = _node_colors(["totally_made_up_item"], layout)
        assert fill["totally_made_up_item"] == "#abcdef"

    def test_stroke_is_darker_than_fill(self) -> None:
        """The stroke is the fill run through :func:`_darken`."""
        layout = Layout(fallback_color="#888888", stroke_darken=0.5)
        fill, stroke = _node_colors(["mystery"], layout)
        # Gray, so the red channel alone identifies the whole color.
        assert int(stroke["mystery"][1:3], 16) < int(fill["mystery"][1:3], 16)


class TestTextPalette:
    """Label color chosen from the fill's luminance."""

    def test_dark_fill_yields_light_labels(self) -> None:
        """A black box takes white text."""
        strong, _, _ = _text_palette("#000000")
        assert strong == "#FFFFFF"

    def test_light_fill_yields_dark_labels(self) -> None:
        """A white box takes near-black text."""
        strong, _, _ = _text_palette("#ffffff")
        assert strong == "#1A1A1A"

    def test_threshold_sits_above_every_palette_fill(self) -> None:
        """Every real item keeps light labels.

        The switch is at luminance 0.65. The brightest fill in the
        palette is the orange for half fabricates, at 0.573. A switch
        at the more usual 0.5 puts dark text on that orange, where the
        contrast is too low to read.
        """
        for item, fill in item_palette().items():
            r, g, b = mcolors.to_rgb(fill)
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            assert luminance < 0.65, f"{item} crosses the label threshold"
            strong, _, _ = _text_palette(fill)
            assert strong == "#FFFFFF"


# ---------------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------------


class TestPretty:
    """Item name to label."""

    def test_replaces_underscores_and_title_cases(self) -> None:
        """Convert a snake case engine name to title case words."""
        assert _pretty("iron_plate") == "Iron Plate"

    def test_single_word(self) -> None:
        """A name with no underscore is title cased and nothing else."""
        assert _pretty("ore") == "Ore"


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------


class TestAssignPorts:
    """Where an incoming arrow meets its target box.

    A box takes two arrows at most, because the engine limits a
    recipe to two inputs. One input enters at the middle of the left
    edge. Two inputs enter over and under the middle, offset by
    ``Layout.port_offset``.
    """

    def test_single_input_uses_mid_port(self, simple_geometry) -> None:
        """A one-input recipe puts its arrow at the box center height."""
        _, _, _, positions, layout = simple_geometry
        single = (RecipeSpec(output="plate", inputs=(("ore", 1),), ticks=10),)
        port_y, port_info = _assign_ports(single, positions, layout)
        plate_y = positions["plate"][1]
        assert port_y[("ore", "plate")] == pytest.approx(plate_y)
        assert "mid" in port_info["plate"]

    def test_two_inputs_split_top_and_bottom(self, simple_geometry) -> None:
        """A two-input recipe puts one arrow above center and one below."""
        _, _, _, positions, layout = simple_geometry
        port_y, port_info = _assign_ports(SIMPLE_RECIPES, positions, layout)
        frame_y = positions["frame"][1]
        ys = sorted(
            [port_y[("plate", "frame")], port_y[("coal", "frame")]],
        )
        assert ys[0] < frame_y < ys[1]
        assert set(port_info["frame"]) == {"top", "bot"}

    def test_higher_source_takes_top_port(self, simple_geometry) -> None:
        """The input drawn higher up feeds the top port.

        The higher source in the lower port makes the two arrows
        cross immediately before they arrive.
        """
        _, _, _, positions, layout = simple_geometry
        _, port_info = _assign_ports(SIMPLE_RECIPES, positions, layout)
        plate_y, coal_y = positions["plate"][1], positions["coal"][1]
        # The sort is stable, so a tie keeps recipe order, which puts
        # plate first.
        higher = "plate" if plate_y >= coal_y else "coal"
        assert port_info["frame"]["top"][0] == _pretty(higher)

    def test_three_inputs_raise(self, simple_geometry) -> None:
        """A recipe with too many inputs is rejected here, not later.

        A box has two ports. A pass that gives ports to the first two
        inputs leaves the third input with no entry in the port map.
        The fault then appears as a ``KeyError`` in the drawing loop,
        far from the recipe that caused it.
        """
        _, _, _, positions, layout = simple_geometry
        wide = (
            RecipeSpec(
                output="frame",
                inputs=(("plate", 1), ("coal", 1), ("ore", 1)),
                ticks=1,
            ),
        )
        with pytest.raises(ValueError, match="frame"):
            _assign_ports(wide, positions, layout)

    def test_zero_inputs_raise(self, simple_geometry) -> None:
        """A recipe with no inputs is rejected too.

        Such a node is a raw material. The layout already gives a raw
        material no incoming edges. As a recipe, it raised
        ``IndexError`` on an empty list.
        """
        _, _, _, positions, layout = simple_geometry
        empty = (RecipeSpec(output="frame", inputs=(), ticks=1),)
        with pytest.raises(ValueError, match="frame"):
            _assign_ports(empty, positions, layout)


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------


class TestGapsForEdge:
    """Which gaps an edge needs a channel in."""

    def test_adjacent_tier_uses_single_gap(self) -> None:
        """An edge between neighbouring tiers crosses one gap."""
        assert _gaps_for_edge(0, 1) == (0,)

    def test_skip_uses_source_and_target_gap(self) -> None:
        """A tier skip needs a channel at each end and none between.

        The edge leaves through the gap after its source. It arrives
        through the gap before its target. Between the two, it runs
        horizontally along a lane, over or between the rows. This part
        needs no vertical channel.
        """
        assert _gaps_for_edge(0, 3) == (0, 2)


class TestAllocateEdgeChannels:
    """Every edge gets its own x inside every gap it crosses."""

    def test_returns_x_for_each_edge_gap(self, simple_geometry) -> None:
        """Keys are ``(source, target, gap)``, one per crossing."""
        _, tiers, rows, _, layout = simple_geometry
        channels = _allocate_edge_channels(SIMPLE_RECIPES, tiers, rows, layout)
        # Every edge here joins neighbouring tiers, so each has exactly
        # one entry, keyed by the source tier.
        assert ("plate", "frame", tiers["plate"]) in channels
        assert ("coal", "frame", tiers["coal"]) in channels
        assert ("ore", "plate", tiers["ore"]) in channels

    def test_channels_are_inside_their_gap(self, simple_geometry) -> None:
        """A channel x falls between the two tier columns it separates.

        A channel outside the band draws a vertical line through the
        boxes of a tier, and not through the empty space beside them.
        """
        _, tiers, rows, _, layout = simple_geometry
        channels = _allocate_edge_channels(SIMPLE_RECIPES, tiers, rows, layout)
        for (_src, _tgt, gap), x in channels.items():
            gap_left = gap * layout.tier_x_step + layout.box_w
            gap_right = (gap + 1) * layout.tier_x_step
            assert gap_left < x < gap_right


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class TestRouteEdge:
    """Polyline shape in the three routing regimes.

    ``_route_edge`` returns the corner points of one edge, from the
    right side of the source box to the left side of the target box.
    The count of points identifies the regime. A single bend between
    neighboring tiers gives four points. Each kind of skip gives six.
    """

    def test_adjacent_tiers_use_four_point_l_bend(self) -> None:
        """A one-tier step leaves, turns at the channel, and arrives."""
        layout = Layout()
        pts = _route_edge(
            source_pos=(0.0, 0.0),
            target_pos=(6.0, 0.0),
            port_y=0.0,
            source_tier=0,
            target_tier=1,
            src_channel=4.0,
            tgt_channel=None,
            inter_row_ys=(),
            skyway_lane=0.0,
            layout=layout,
        )
        assert len(pts) == 4

    def test_single_tier_skip_uses_inter_row_lane(self) -> None:
        """A two-tier span crosses on a lane between two rows.

        The function takes the lane nearest the midpoint of the
        source and the port. The edge therefore passes between boxes
        and not over the whole diagram.
        """
        layout = Layout()
        pts = _route_edge(
            source_pos=(0.0, 0.0),
            target_pos=(12.0, -3.0),
            port_y=-3.0,
            source_tier=0,
            target_tier=2,
            src_channel=4.0,
            tgt_channel=10.0,
            inter_row_ys=(-1.5,),
            skyway_lane=5.0,
            layout=layout,
        )
        assert len(pts) == 6
        # The middle two corners ride the lane.
        assert pts[2][1] == pytest.approx(-1.5)
        assert pts[3][1] == pytest.approx(-1.5)

    def test_long_skip_uses_skyway(self) -> None:
        """A span of three or more crosses above the top row.

        Between the rows, a long edge cuts through the boxes of
        several tiers. It climbs over them instead.
        """
        layout = Layout()
        pts = _route_edge(
            source_pos=(0.0, 0.0),
            target_pos=(18.0, -3.0),
            port_y=-3.0,
            source_tier=0,
            target_tier=3,
            src_channel=4.0,
            tgt_channel=16.0,
            inter_row_ys=(-1.5,),
            skyway_lane=5.0,
            layout=layout,
        )
        assert len(pts) == 6
        # The middle two corners ride the skyway, not the lane.
        assert pts[2][1] == pytest.approx(5.0)
        assert pts[3][1] == pytest.approx(5.0)

    def test_two_tier_skip_without_lanes_uses_the_skyway(self) -> None:
        """With no lane available the two-tier span climbs over the top.

        A diagram with one node per tier has no space between rows.
        The lane list is therefore empty. The edge must still clear the
        box between its two ends, and the skyway is the last route.
        """
        layout = Layout()
        pts = _route_edge(
            source_pos=(0.0, 0.0),
            target_pos=(12.0, 0.0),
            port_y=0.0,
            source_tier=0,
            target_tier=2,
            src_channel=4.0,
            tgt_channel=10.0,
            inter_row_ys=(),
            skyway_lane=5.0,
            layout=layout,
        )
        assert len(pts) == 6
        assert pts[2][1] == pytest.approx(5.0)
        assert pts[3][1] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Whole pipeline
# ---------------------------------------------------------------------------


class TestRenderShapes:
    """Recipe shapes that reach the helpers through :func:`render`.

    These assert only that a file appears. Their value is the route
    to that file. Each shape gives the helpers arguments that no unit
    test builds by hand.
    """

    def test_single_row_graph_with_a_tier_skip(self, tmp_path: Path) -> None:
        """One node per tier plus a skip edge renders.

        Every tier holds one node. There is therefore no lane between
        the rows, and the skip edge must use the skyway.
        """
        recipes = (
            RecipeSpec(output="b", inputs=(("a", 1),), ticks=1),
            RecipeSpec(output="c", inputs=(("b", 1), ("a", 1)), ticks=1),
        )
        out = tmp_path / "single_row.png"
        assert render(recipes, out) == out
        assert out.stat().st_size > 0

    def test_simple_chain(self, tmp_path: Path) -> None:
        """The three-tier fixture chain renders end to end."""
        out = tmp_path / "chain.png"
        assert render(SIMPLE_RECIPES, out) == out
        assert out.stat().st_size > 0
