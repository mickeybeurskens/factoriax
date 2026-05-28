"""Tests for the generic layered-DAG layout primitives.

Covers tier assignment (raw materials at tier 0, longest-path
semantics, cycle detection) and Sugiyama barycenter ordering
(deterministic on a known DAG).
"""

from __future__ import annotations

import pytest

from factoriax.analysis.graph_layout import assign_tiers, order_within_tiers


class TestAssignTiers:
    """Tier assignment by longest-path layering."""

    def test_raw_material_is_tier_zero(self) -> None:
        """A node with no inputs lands at tier 0."""
        tiers = assign_tiers({"plate": ["ore"]})
        assert tiers["ore"] == 0

    def test_single_recipe_climbs_one_tier(self) -> None:
        """An output sits one tier above its single input."""
        tiers = assign_tiers({"plate": ["ore"]})
        assert tiers["plate"] == 1

    def test_uses_longest_path_not_shortest(self) -> None:
        """An item's tier is determined by its deepest input chain."""
        # frame ← plate ← ore  (depth 2)
        # frame ← coal         (depth 1)
        # Longest path wins: frame should be tier 2.
        adjacency = {"plate": ["ore"], "frame": ["plate", "coal"]}
        tiers = assign_tiers(adjacency)
        assert tiers["plate"] == 1
        assert tiers["coal"] == 0
        assert tiers["frame"] == 2

    def test_discovers_pure_input_nodes(self) -> None:
        """Items appearing only as inputs are still tiered."""
        tiers = assign_tiers({"plate": ["ore", "fuel"]})
        assert tiers["ore"] == 0
        assert tiers["fuel"] == 0

    def test_cycle_raises(self) -> None:
        """A cycle is reported, not silently iterated."""
        with pytest.raises(ValueError, match="Cycle"):
            assign_tiers({"a": ["b"], "b": ["a"]})


class TestOrderWithinTiers:
    """Sugiyama barycenter sweep."""

    def test_returns_row_per_node(self) -> None:
        """Every tiered node appears in the row map exactly once."""
        adjacency = {"plate": ["ore"], "frame": ["plate", "coal"]}
        tiers = assign_tiers(adjacency)
        rows = order_within_tiers(adjacency, tiers)
        assert set(rows) == set(tiers)

    def test_rows_are_zero_indexed_per_tier(self) -> None:
        """Within each tier the row indices form a contiguous 0..n-1 range."""
        adjacency = {
            "a1": ["x"],
            "a2": ["x"],
            "a3": ["x"],
        }
        tiers = assign_tiers(adjacency)
        rows = order_within_tiers(adjacency, tiers)
        tier_one_rows = sorted(rows[node] for node in ("a1", "a2", "a3"))
        assert tier_one_rows == [0, 1, 2]

    def test_deterministic_for_same_input(self) -> None:
        """The same adjacency and tier map yield the same row map."""
        adjacency = {
            "plate": ["ore"],
            "wire": ["ore", "tin"],
            "frame": ["plate", "wire"],
        }
        tiers = assign_tiers(adjacency)
        rows_a = order_within_tiers(adjacency, tiers)
        rows_b = order_within_tiers(adjacency, tiers)
        assert rows_a == rows_b
