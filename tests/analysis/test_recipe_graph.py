"""Tests for the pure helpers in :mod:`factoriax.analysis.recipe_graph`.

Covers data classes, adjacency / position math, colour helpers, port and
channel allocation, and edge routing. The matplotlib ``_draw_*`` helpers
and the top-level ``render`` entry point are excluded — they're rendering
code that's hard to assert on without snapshot images.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import pytest

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
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

#: A minimal recipe chain: ore -> plate -> frame, plus coal that feeds frame.
SIMPLE_RECIPES: tuple[RecipeSpec, ...] = (
    RecipeSpec(output="plate", inputs=(("ore", 1),), ticks=10),
    RecipeSpec(output="frame", inputs=(("plate", 2), ("coal", 1)), ticks=20),
)


@pytest.fixture
def simple_geometry():
    """Tiers, rows, positions, layout for the SIMPLE_RECIPES chain."""
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
        """Optional fields keep their defaults."""
        r = RecipeSpec(output="plate", inputs=(("ore", 1),), ticks=10)
        assert r.output_count == 1
        assert r.name == ""

    def test_is_frozen(self) -> None:
        """RecipeSpec instances are immutable."""
        r = RecipeSpec(output="plate", inputs=(("ore", 1),), ticks=10)
        with pytest.raises(Exception):  # FrozenInstanceError
            r.output = "wire"  # type: ignore[misc]


class TestLayout:
    """Layout geometry constants."""

    def test_port_offset_uses_fraction_of_height(self) -> None:
        """port_offset is ``box_h * port_offset_frac``."""
        layout = Layout(box_h=2.0, port_offset_frac=0.25)
        assert layout.port_offset == pytest.approx(0.5)

    def test_defaults_are_positive(self) -> None:
        """All geometry defaults are positive — degenerate layouts crash later."""
        layout = Layout()
        assert layout.box_w > 0 and layout.box_h > 0
        assert layout.tier_x_step > 0 and layout.row_y_step > 0


# ---------------------------------------------------------------------------
# Adjacency + positions
# ---------------------------------------------------------------------------


class TestAdjacency:
    """``_adjacency`` strips counts from each recipe's inputs."""

    def test_returns_output_to_input_names(self) -> None:
        adjacency = _adjacency(SIMPLE_RECIPES)
        assert adjacency == {"plate": ["ore"], "frame": ["plate", "coal"]}

    def test_preserves_input_order(self) -> None:
        """Input ordering matches the recipe; the renderer treats the
        first input as the top port for two-input recipes."""
        r = RecipeSpec(
            output="x",
            inputs=(("a", 1), ("b", 2), ("c", 3)),
            ticks=1,
        )
        assert _adjacency([r]) == {"x": ["a", "b", "c"]}


class TestPositions:
    """``_positions`` lays items out on the (tier, row) grid."""

    def test_every_item_has_a_position(self, simple_geometry) -> None:
        _, tiers, _, positions, _ = simple_geometry
        assert set(positions) == set(tiers)

    def test_x_increases_with_tier(self, simple_geometry) -> None:
        """Tier-0 items sit to the left of tier-2 items in x."""
        _, _, _, positions, _ = simple_geometry
        assert positions["ore"][0] < positions["plate"][0] < positions["frame"][0]

    def test_x_uses_tier_step(self, simple_geometry) -> None:
        """x-coordinate is ``tier * tier_x_step + box_w / 2``."""
        _, tiers, _, positions, layout = simple_geometry
        for item, tier in tiers.items():
            expected = tier * layout.tier_x_step + layout.box_w / 2
            assert positions[item][0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------


class TestDarken:
    """HLS-based colour darkening."""

    def test_factor_one_is_a_no_op(self) -> None:
        """Factor 1.0 leaves the colour unchanged (modulo float round-trip)."""
        original = "#7f7f7f"
        out = _darken(original, 1.0)
        assert mcolors.to_rgb(out) == pytest.approx(mcolors.to_rgb(original), abs=1e-3)

    def test_factor_below_one_lowers_luminance(self) -> None:
        """A factor < 1 produces a strictly darker colour."""
        bright = "#ff8080"
        darker = _darken(bright, 0.5)

        def lum(hx: str) -> float:
            r, g, b = mcolors.to_rgb(hx)
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        assert lum(darker) < lum(bright)

    def test_clamps_at_zero(self) -> None:
        """A factor of zero produces black (lightness clipped at 0)."""
        out = _darken("#ff0000", 0.0)
        assert mcolors.to_rgb(out) == pytest.approx((0.0, 0.0, 0.0), abs=1e-3)


class TestNodeColors:
    """``_node_colors`` yields aligned fill + stroke maps."""

    def test_returns_color_per_item(self) -> None:
        fill, stroke = _node_colors(["ore", "plate"], Layout())
        assert set(fill) == set(stroke) == {"ore", "plate"}

    def test_unknown_items_use_fallback_fill(self) -> None:
        """Items the palette doesn't recognise get the layout fallback."""
        layout = Layout(fallback_color="#abcdef")
        fill, _ = _node_colors(["totally_made_up_item"], layout)
        assert fill["totally_made_up_item"] == "#abcdef"

    def test_stroke_is_darker_than_fill(self) -> None:
        """Stroke is a darkened variant of the fill."""
        layout = Layout(fallback_color="#888888", stroke_darken=0.5)
        fill, stroke = _node_colors(["mystery"], layout)
        # Stroke should be a darker hex than the fill.
        assert int(stroke["mystery"][1:3], 16) < int(fill["mystery"][1:3], 16)


class TestTextPalette:
    """Label-colour selection based on fill luminance."""

    def test_dark_fill_yields_light_labels(self) -> None:
        strong, _, _ = _text_palette("#000000")
        assert strong == "#FFFFFF"

    def test_light_fill_yields_dark_labels(self) -> None:
        strong, _, _ = _text_palette("#ffffff")
        assert strong == "#1A1A1A"


# ---------------------------------------------------------------------------
# Pretty
# ---------------------------------------------------------------------------


class TestPretty:
    """Display-name formatting."""

    def test_replaces_underscores_and_title_cases(self) -> None:
        assert _pretty("iron_plate") == "Iron Plate"

    def test_single_word(self) -> None:
        assert _pretty("ore") == "Ore"


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------


class TestAssignPorts:
    """Top/bottom/mid port allocation."""

    def test_single_input_uses_mid_port(self, simple_geometry) -> None:
        _, _, _, positions, layout = simple_geometry
        single = (RecipeSpec(output="plate", inputs=(("ore", 1),), ticks=10),)
        port_y, port_info = _assign_ports(single, positions, layout)
        plate_y = positions["plate"][1]
        assert port_y[("ore", "plate")] == pytest.approx(plate_y)
        assert "mid" in port_info["plate"]

    def test_two_inputs_split_top_and_bottom(self, simple_geometry) -> None:
        _, _, _, positions, layout = simple_geometry
        port_y, port_info = _assign_ports(SIMPLE_RECIPES, positions, layout)
        frame_y = positions["frame"][1]
        # One port lands above frame's centre, the other below.
        ys = sorted(
            [port_y[("plate", "frame")], port_y[("coal", "frame")]],
        )
        assert ys[0] < frame_y < ys[1]
        assert set(port_info["frame"]) == {"top", "bot"}

    def test_higher_source_takes_top_port(self, simple_geometry) -> None:
        """The visually higher input (greater y) feeds the top port."""
        _, _, _, positions, layout = simple_geometry
        _, port_info = _assign_ports(SIMPLE_RECIPES, positions, layout)
        plate_y, coal_y = positions["plate"][1], positions["coal"][1]
        higher = "plate" if plate_y >= coal_y else "coal"
        assert port_info["frame"]["top"][0] == _pretty(higher)


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------


class TestGapsForEdge:
    """``_gaps_for_edge`` derives which gaps an edge traverses."""

    def test_adjacent_tier_uses_single_gap(self) -> None:
        assert _gaps_for_edge(0, 1) == (0,)

    def test_skip_uses_source_and_target_gap(self) -> None:
        """A skip from tier 0 to tier 3 uses gaps 0 (source) and 2 (target-1)."""
        assert _gaps_for_edge(0, 3) == (0, 2)


class TestAllocateEdgeChannels:
    """Every edge gets a unique x in every gap it traverses."""

    def test_returns_x_for_each_edge_gap(self, simple_geometry) -> None:
        _, tiers, rows, _, layout = simple_geometry
        channels = _allocate_edge_channels(SIMPLE_RECIPES, tiers, rows, layout)
        # frame has two inputs (plate, coal) both adjacent-tier, so each
        # has a single gap entry.
        assert ("plate", "frame", tiers["plate"]) in channels
        assert ("coal", "frame", tiers["coal"]) in channels
        assert ("ore", "plate", tiers["ore"]) in channels

    def test_channels_are_inside_their_gap(self, simple_geometry) -> None:
        """Each channel x lies strictly between the gap's tier endpoints."""
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
    """Edge polyline shape across the three routing regimes."""

    def test_adjacent_tiers_use_four_point_l_bend(self) -> None:
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
        """A span of 2 routes through an inter-row lane (6 vertices)."""
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
        # Middle two vertices ride the inter-row lane.
        assert pts[2][1] == pytest.approx(-1.5)
        assert pts[3][1] == pytest.approx(-1.5)

    def test_long_skip_uses_skyway(self) -> None:
        """A span of 3+ routes through the skyway lane (6 vertices)."""
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
        # Middle two vertices ride the skyway, not the inter-row lane.
        assert pts[2][1] == pytest.approx(5.0)
        assert pts[3][1] == pytest.approx(5.0)
