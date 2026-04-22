"""Tests for agentdebugger chart rendering."""

import numpy as np

from factoriax.agentdebugger.charts import (
    _INVENTORY_ITEMS,
    _inventory_slot_positions,
    build_partial_trajectory,
    render_cost_chart,
    render_inventory_panel,
)
from factoriax.constants import ItemType


class TestBuildPartialTrajectory:
    """Tests for build_partial_trajectory."""

    def test_basic_shape(self) -> None:
        """Trajectory has batch=1 and correct time dimension."""
        traj = build_partial_trajectory(
            rewards=[1.0, 2.0, 3.0],
            actions=[0, 1, 2],
        )
        assert traj.actions.shape == (1, 3)
        assert traj.rewards is not None
        assert traj.rewards.shape == (1, 3)

    def test_dtypes(self) -> None:
        """Actions are int32, rewards are float32."""
        traj = build_partial_trajectory(
            rewards=[0.5],
            actions=[7],
        )
        assert traj.actions.dtype == np.int32
        assert traj.rewards is not None
        assert traj.rewards.dtype == np.float32

    def test_empty_lists(self) -> None:
        """Empty inputs produce a trajectory with time=1."""
        traj = build_partial_trajectory(rewards=[], actions=[])
        assert traj.actions.shape == (1, 1)
        assert traj.rewards is not None
        assert traj.rewards.shape == (1, 1)

    def test_mismatched_lengths_padded(self) -> None:
        """Shorter array is zero-padded to match the longer one."""
        traj = build_partial_trajectory(
            rewards=[1.0, 2.0, 3.0],
            actions=[0],
        )
        assert traj.actions.shape == (1, 3)
        assert traj.rewards is not None
        assert traj.rewards.shape == (1, 3)

    def test_values_preserved(self) -> None:
        """Input values appear in the trajectory arrays."""
        traj = build_partial_trajectory(
            rewards=[10.0, 20.0],
            actions=[3, 7],
        )
        np.testing.assert_array_equal(traj.actions[0], [3, 7])
        assert traj.rewards is not None
        np.testing.assert_array_almost_equal(
            traj.rewards[0],
            [10.0, 20.0],
        )


class TestRenderCostChart:
    """Tests for render_cost_chart."""

    def test_output_shape_and_dtype(self) -> None:
        """Returns RGB uint8 array of requested dimensions."""
        costs = [np.array([0.5, 0.1]), np.array([0.3, 0.2])]
        img = render_cost_chart(costs, ["a", "b"], 200, 100)
        assert img.shape == (100, 200, 3)
        assert img.dtype == np.uint8

    def test_empty_costs(self) -> None:
        """Empty cost list produces a valid placeholder image."""
        img = render_cost_chart([], [], 200, 100)
        assert img.shape == (100, 200, 3)
        assert img.dtype == np.uint8

    def test_not_all_black(self) -> None:
        """Chart with data should have non-zero pixels."""
        costs = [np.array([1.0]) for _ in range(10)]
        img = render_cost_chart(costs, ["x"], 200, 100)
        assert img.sum() > 0

    def test_plot_bounds_embedded(self) -> None:
        """Plot area fractions are encoded in the top-left pixels."""
        costs = [np.array([0.5]) for _ in range(5)]
        img = render_cost_chart(costs, ["x"], 300, 150)
        frac_x0 = int(img[0, 0, 0])
        frac_x1 = int(img[0, 1, 0])
        assert frac_x0 > 0
        assert frac_x1 > frac_x0


class TestInventorySlotPositions:
    """Layout helper must place every item within bounds at realistic sizes.

    Regression: at the default 320x240 quadrant the old single-column
    layout clipped the last 7 items (rocket, furnace, refractory, hull,
    engine_unit, avionics, rocket_core) because the rows couldn't fit
    vertically. The helper now falls back to two columns when needed.
    """

    def test_single_column_when_tall(self) -> None:
        """A tall panel uses one column per item."""
        positions = _inventory_slot_positions(
            width=320,
            height=800,
            num_items=28,
            row_top=25,
        )
        assert len(positions) == 28
        xs = {x for (x, _y, _w, _h) in positions}
        assert len(xs) == 1, "expected all slots to share a single column x"

    def test_two_columns_at_default_quadrant_320x240(self) -> None:
        """At the default 320x240 quadrant, 28 items fit via two columns."""
        height = 240
        positions = _inventory_slot_positions(
            width=320,
            height=height,
            num_items=28,
            row_top=25,
        )
        assert len(positions) == 28
        # Two distinct column x-origins.
        xs = sorted({x for (x, _y, _w, _h) in positions})
        assert len(xs) == 2, f"expected two columns, got x-origins {xs}"
        # Every slot lands inside the panel.
        for x, y, w, h in positions:
            assert 0 <= x
            assert 0 <= y
            assert x + w <= 320
            assert y + h <= height, f"slot at y={y} h={h} exceeds panel height {height}"

    def test_all_items_including_rocket_intermediates_are_placed(self) -> None:
        """Every rocket-intermediate item has a slot at the default size."""
        positions = _inventory_slot_positions(
            width=320,
            height=240,
            num_items=len(_INVENTORY_ITEMS),
            row_top=25,
        )
        # Build a map from item to its slot.
        by_item = dict(zip(_INVENTORY_ITEMS, positions, strict=True))
        for item in (
            ItemType.ROCKET,
            ItemType.FURNACE,
            ItemType.REFRACTORY,
            ItemType.HULL,
            ItemType.ENGINE_UNIT,
            ItemType.AVIONICS,
            ItemType.ROCKET_CORE,
        ):
            assert item in by_item, f"{item.name} missing from layout"
            x, y, w, h = by_item[item]
            assert y + h <= 240, f"{item.name} slot clipped (y={y}, h={h}, panel_h=240)"

    def test_empty_input(self) -> None:
        """Zero items yields an empty list, not a crash."""
        positions = _inventory_slot_positions(
            width=320,
            height=240,
            num_items=0,
            row_top=25,
        )
        assert positions == []


class TestRenderInventoryPanel:
    """Rendering integration: all item color swatches visible at 320x240."""

    def test_every_rocket_intermediate_renders_a_color_swatch(self) -> None:
        """At the default quadrant size, HULL/ENGINE_UNIT/AVIONICS/
        ROCKET_CORE each paint their color swatch somewhere on the panel.

        We give every item a non-zero count so active rendering (full
        brightness swatch) is used, then scan the panel for each item's
        signature RGB.
        """
        from factoriax.constants import ITEM_COLORS, NUM_ITEM_TYPES

        inv = np.ones(NUM_ITEM_TYPES, dtype=np.int32)
        img = render_inventory_panel(inv, width=320, height=240)
        assert img.shape == (240, 320, 3)

        missing: list[str] = []
        for item in (
            ItemType.ROCKET,
            ItemType.FURNACE,
            ItemType.REFRACTORY,
            ItemType.HULL,
            ItemType.ENGINE_UNIT,
            ItemType.AVIONICS,
            ItemType.ROCKET_CORE,
        ):
            rgb = ITEM_COLORS.get(int(item), (120, 120, 120))
            match = np.all(img == np.asarray(rgb, dtype=np.uint8), axis=-1)
            if not bool(match.any()):
                missing.append(item.name)
        assert not missing, f"items missing from rendered panel: {missing}"
