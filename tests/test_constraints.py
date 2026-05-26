"""Tests for factoriax.engine.constraints — constraint cost functions."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from factoriax.engine.constants import NUM_ITEM_TYPES, ItemType
from factoriax.engine.constraints import (
    balance_cost,
    balance_cost_names,
    diversity_cost,
    diversity_cost_names,
)
from factoriax.engine.state import EnvParams

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_items_mined(
    coal: int = 0,
    iron: int = 0,
    copper: int = 0,
) -> jnp.ndarray:
    """Build an items_mined array with specified counts."""
    arr = jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32)
    arr = arr.at[ItemType.COAL].set(coal)
    arr = arr.at[ItemType.IRON_ORE].set(iron)
    arr = arr.at[ItemType.COPPER_ORE].set(copper)
    return arr


class _FakeState:
    """Minimal stand-in for EnvState with just items_mined."""

    def __init__(self, items_mined: jnp.ndarray) -> None:
        self.items_mined = items_mined


_PARAMS = EnvParams()


# ---------------------------------------------------------------------------
# balance_cost
# ---------------------------------------------------------------------------


class TestBalanceCost:
    """Tests for balance_cost."""

    def test_all_zero_no_violation(self) -> None:
        s = _FakeState(_make_items_mined(0, 0, 0))
        cost = balance_cost(s, s, _PARAMS, threshold=3.0)
        assert cost.shape == (3,)
        np.testing.assert_array_equal(cost, 0.0)

    def test_equal_counts_no_violation(self) -> None:
        s = _FakeState(_make_items_mined(5, 5, 5))
        cost = balance_cost(s, s, _PARAMS, threshold=3.0)
        np.testing.assert_allclose(cost, 0.0, atol=1e-6)

    def test_within_threshold_no_violation(self) -> None:
        s = _FakeState(_make_items_mined(3, 1, 2))
        cost = balance_cost(s, s, _PARAMS, threshold=3.0)
        # coal/iron: 3/(1+1)=1.5, coal/copper: 3/(2+1)=1.0, iron/copper: 2/(1+1)=1.0
        np.testing.assert_allclose(cost, 0.0, atol=1e-6)

    def test_exceeds_threshold_positive_cost(self) -> None:
        s = _FakeState(_make_items_mined(10, 1, 1))
        cost = balance_cost(s, s, _PARAMS, threshold=3.0)
        # coal/iron: 10/(1+1)=5.0, cost=5.0-3.0=2.0
        # coal/copper: 10/(1+1)=5.0, cost=5.0-3.0=2.0
        # iron/copper: 1/(1+1)=0.5, cost=0.0
        assert cost.shape == (3,)
        assert float(cost[0]) > 0.0
        assert float(cost[1]) > 0.0
        assert float(cost[2]) == 0.0

    def test_one_type_mined_others_zero(self) -> None:
        s = _FakeState(_make_items_mined(5, 0, 0))
        cost = balance_cost(s, s, _PARAMS, threshold=3.0)
        # coal/iron: 5/(0+1)=5.0, cost=2.0
        # coal/copper: 5/(0+1)=5.0, cost=2.0
        # iron/copper: 0/(0+1)=0.0, cost=0.0
        assert float(cost[0]) > 0.0
        assert float(cost[1]) > 0.0
        assert float(cost[2]) == 0.0

    def test_names_match_dimensions(self) -> None:
        names = balance_cost_names()
        s = _FakeState(_make_items_mined(1, 1, 1))
        cost = balance_cost(s, s, _PARAMS)
        assert len(names) == cost.shape[0]

    def test_custom_threshold(self) -> None:
        s = _FakeState(_make_items_mined(5, 1, 1))
        strict = balance_cost(s, s, _PARAMS, threshold=1.0)
        lenient = balance_cost(s, s, _PARAMS, threshold=10.0)
        assert float(strict[0]) > float(lenient[0])
        np.testing.assert_allclose(lenient, 0.0, atol=1e-6)

    def test_symmetric(self) -> None:
        s1 = _FakeState(_make_items_mined(10, 1, 5))
        s2 = _FakeState(_make_items_mined(1, 10, 5))
        c1 = balance_cost(s1, s1, _PARAMS)
        c2 = balance_cost(s2, s2, _PARAMS)
        # coal/iron pair should give same cost regardless of which is higher
        np.testing.assert_allclose(float(c1[0]), float(c2[0]), atol=1e-6)


# ---------------------------------------------------------------------------
# diversity_cost
# ---------------------------------------------------------------------------


class TestDiversityCost:
    """Tests for diversity_cost."""

    def test_all_mined_no_cost(self) -> None:
        s = _FakeState(_make_items_mined(1, 1, 1))
        cost = diversity_cost(s, s, _PARAMS)
        assert cost.shape == (1,)
        assert float(cost[0]) == 0.0

    def test_none_mined_cost_three(self) -> None:
        s = _FakeState(_make_items_mined(0, 0, 0))
        cost = diversity_cost(s, s, _PARAMS)
        assert float(cost[0]) == 3.0

    def test_one_mined_cost_two(self) -> None:
        s = _FakeState(_make_items_mined(5, 0, 0))
        cost = diversity_cost(s, s, _PARAMS)
        assert float(cost[0]) == 2.0

    def test_two_mined_cost_one(self) -> None:
        s = _FakeState(_make_items_mined(5, 3, 0))
        cost = diversity_cost(s, s, _PARAMS)
        assert float(cost[0]) == 1.0

    def test_names_match_dimensions(self) -> None:
        names = diversity_cost_names()
        s = _FakeState(_make_items_mined(0, 0, 0))
        cost = diversity_cost(s, s, _PARAMS)
        assert len(names) == cost.shape[0]
