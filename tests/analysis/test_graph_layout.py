"""Tests for :mod:`factoriax.analysis.graph_layout`.

The module turns a ``{output: inputs}`` map into two numbers for each
node. The tier becomes the x column. The row becomes the y slot inside
that column. The recipe diagram is the only caller today, but neither
function knows about recipes.

Two properties carry the visual result. These tests assert both.

Tier assignment uses the longest path, not the shortest. An item at the
end of a two-step chain and a one-step chain sits at tier 2. The
shortest path puts that item at tier 1. An arrow into it then runs
backwards, from a higher tier to a lower tier.

Row assignment is a barycenter sweep, which is a heuristic. There is no
optimal order to compare it against. These tests therefore assert the
properties that hold for every order. Each node gets one row. The rows
inside a tier count up from 0 with no gaps. The same input gives the
same output.
"""

from __future__ import annotations

import pytest

from factoriax.analysis.graph_layout import assign_tiers, order_within_tiers


class TestAssignTiers:
    """Longest-path tier assignment."""

    def test_raw_material_is_tier_zero(self) -> None:
        """A node with no inputs of its own lands at tier 0."""
        tiers = assign_tiers({"plate": ["ore"]})
        assert tiers["ore"] == 0

    def test_single_recipe_climbs_one_tier(self) -> None:
        """An output sits one tier above its only input."""
        tiers = assign_tiers({"plate": ["ore"]})
        assert tiers["plate"] == 1

    def test_uses_longest_path_not_shortest(self) -> None:
        """The deepest input chain sets the tier, not the shallowest."""
        # frame <- plate <- ore is two steps. frame <- coal is one step.
        # The shortest path puts frame at tier 1, level with plate. The
        # plate -> frame arrow then has nowhere to go.
        adjacency = {"plate": ["ore"], "frame": ["plate", "coal"]}
        tiers = assign_tiers(adjacency)
        assert tiers["plate"] == 1
        assert tiers["coal"] == 0
        assert tiers["frame"] == 2

    def test_discovers_pure_input_nodes(self) -> None:
        """Give a tier to nodes that appear only as inputs.

        The caller passes one key for each recipe. A raw material has
        no recipe, so it never appears as a key. The function finds it
        in the input lists instead.
        """
        tiers = assign_tiers({"plate": ["ore", "fuel"]})
        assert tiers["ore"] == 0
        assert tiers["fuel"] == 0

    def test_cycle_raises(self) -> None:
        """Raise on a cycle, and do not recurse to a stack overflow."""
        with pytest.raises(ValueError, match="Cycle"):
            assign_tiers({"a": ["b"], "b": ["a"]})


class TestOrderWithinTiers:
    """Barycenter row assignment."""

    def test_returns_row_per_node(self) -> None:
        """Every tiered node gets a row, including raw materials."""
        adjacency = {"plate": ["ore"], "frame": ["plate", "coal"]}
        tiers = assign_tiers(adjacency)
        rows = order_within_tiers(adjacency, tiers)
        assert set(rows) == set(tiers)

    def test_rows_are_zero_indexed_per_tier(self) -> None:
        """Rows restart at 0 in each tier and leave no gaps.

        Position code multiplies the row by a fixed y step. A gap in
        the sequence therefore leaves an empty band across the figure.
        """
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
        """The same adjacency and tiers give the same rows.

        The sweep starts from a sorted node list, and not from
        dictionary order. Two runs in one process therefore agree, and
        so do two runs in different processes. Without this order, a
        committed figure changes at every rebuild.
        """
        adjacency = {
            "plate": ["ore"],
            "wire": ["ore", "tin"],
            "frame": ["plate", "wire"],
        }
        tiers = assign_tiers(adjacency)
        rows_a = order_within_tiers(adjacency, tiers)
        rows_b = order_within_tiers(adjacency, tiers)
        assert rows_a == rows_b
