"""Tests for :mod:`baselines.rocket.scripted.recipe_planning`.

Exercises the bill-of-materials, production-schedule, and inventory
arithmetic helpers used by the advanced agent to derive bootstrap
quantities from a :class:`RecipeBook`.
"""

from __future__ import annotations

import pytest

from baselines.rocket.scripted.recipe_planning import (
    bill_of_materials,
    production_schedule,
    scale_inventory,
    sum_inventories,
)
from factoriax.constants import ItemType, MachineType
from factoriax.recipes import (
    BASE_RECIPE_BOOK,
    Recipe,
    RecipeBalance,
    RecipeBook,
    RecipeOverride,
)

# ---------------------------------------------------------------------------
# bill_of_materials
# ---------------------------------------------------------------------------


class TestBillOfMaterials:
    def test_empty_targets_returns_empty(self) -> None:
        assert bill_of_materials({}) == {}

    def test_leaf_passes_through(self) -> None:
        """Items with no recipe (raw ores) appear unchanged in the BOM."""
        bom = bill_of_materials({int(ItemType.IRON_ORE): 5})
        assert bom == {int(ItemType.IRON_ORE): 5}

    def test_single_furnace_recipe(self) -> None:
        """IRON_PLATE → 1 IRON_ORE + 1 COAL per plate (default balance)."""
        bom = bill_of_materials({int(ItemType.IRON_PLATE): 3})
        assert bom == {int(ItemType.IRON_ORE): 3, int(ItemType.COAL): 3}

    def test_chained_recipe_rolls_through_intermediate(self) -> None:
        """WIRE = 1 COPPER_PLATE + 1 TIN_PLATE; each plate = 1 ore + 1 coal.
        So 1 WIRE → 1 COPPER_ORE + 1 TIN_ORE + 2 COAL (one coal per smelt).
        """
        bom = bill_of_materials({int(ItemType.WIRE): 1})
        assert bom == {
            int(ItemType.COPPER_ORE): 1,
            int(ItemType.TIN_ORE): 1,
            int(ItemType.COAL): 2,
        }

    def test_multi_target_sums_intermediates(self) -> None:
        """Two targets that share an intermediate (COAL) sum cleanly."""
        bom = bill_of_materials(
            {
                int(ItemType.IRON_PLATE): 2,
                int(ItemType.COPPER_PLATE): 3,
            }
        )
        # 2 iron smelts: 2 IRON_ORE + 2 COAL
        # 3 copper smelts: 3 COPPER_ORE + 3 COAL
        assert bom == {
            int(ItemType.IRON_ORE): 2,
            int(ItemType.COPPER_ORE): 3,
            int(ItemType.COAL): 5,
        }

    def test_output_count_ceil_division(self) -> None:
        """A recipe with output_count=2 producing odd target rounds up."""
        # Synthetic book: IRON_PLATE yields 2 plates per smelt instead of 1.
        balance = RecipeBalance(
            overrides=((int(ItemType.IRON_PLATE), RecipeOverride(output_count=2)),)
        )
        book = BASE_RECIPE_BOOK.with_balance(balance)
        # Asking for 5 plates needs 3 cycles (ceil(5/2)) which yields 6.
        # 3 cycles consume 3 IRON_ORE + 3 COAL.
        bom = bill_of_materials({int(ItemType.IRON_PLATE): 5}, book)
        assert bom == {int(ItemType.IRON_ORE): 3, int(ItemType.COAL): 3}

    def test_balance_overlay_doubles_input(self) -> None:
        """If IRON_PLATE.input_counts becomes (2, 1), ore demand doubles."""
        balance = RecipeBalance(
            overrides=(
                (
                    int(ItemType.IRON_PLATE),
                    RecipeOverride(input_counts=(2, 1)),
                ),
            )
        )
        book = BASE_RECIPE_BOOK.with_balance(balance)
        bom = bill_of_materials({int(ItemType.IRON_PLATE): 5}, book)
        assert bom == {int(ItemType.IRON_ORE): 10, int(ItemType.COAL): 5}

    def test_zero_qty_skipped(self) -> None:
        """Targets with qty=0 contribute nothing."""
        bom = bill_of_materials({int(ItemType.IRON_PLATE): 0})
        assert bom == {}

    def test_negative_qty_treated_as_zero(self) -> None:
        """Defensive: a stray negative count does not subtract."""
        bom = bill_of_materials({int(ItemType.IRON_PLATE): -3})
        assert bom == {}

    def test_miner_full_chain(self) -> None:
        """MINER recipe: 1 IRON_PLATE + 1 WIRE.
        WIRE: 1 COPPER_PLATE + 1 TIN_PLATE.
        So 1 MINER ->
            1 IRON_PLATE (1 IRON_ORE + 1 COAL)
            + 1 WIRE (1 COPPER_PLATE + 1 TIN_PLATE
                      = 1 COPPER_ORE + 1 TIN_ORE + 2 COAL)
            = 1 IRON_ORE + 1 COPPER_ORE + 1 TIN_ORE + 3 COAL.
        """
        bom = bill_of_materials({int(ItemType.MINER): 1})
        assert bom == {
            int(ItemType.IRON_ORE): 1,
            int(ItemType.COPPER_ORE): 1,
            int(ItemType.TIN_ORE): 1,
            int(ItemType.COAL): 3,
        }

    def test_cycle_in_synthetic_book_raises(self) -> None:
        """A book where A's recipe consumes B and B's recipe consumes A
        must raise ValueError naming the cycle. Using two unused
        ItemType values to avoid colliding with shipped recipes.
        """
        synthetic = RecipeBook(
            recipes=(
                Recipe(
                    output=int(ItemType.SCIENCE_LAB),
                    inputs=((int(ItemType.ROCKET_CORE), 1), (int(ItemType.HULL), 1)),
                    ticks=4,
                    name="cyclic-lab",
                ),
                Recipe(
                    output=int(ItemType.ROCKET_CORE),
                    inputs=(
                        (int(ItemType.SCIENCE_LAB), 1),
                        (int(ItemType.AVIONICS), 1),
                    ),
                    ticks=4,
                    name="cyclic-core",
                ),
            )
        )
        with pytest.raises(ValueError, match="Recipe cycle detected"):
            bill_of_materials({int(ItemType.SCIENCE_LAB): 1}, synthetic)


# ---------------------------------------------------------------------------
# production_schedule
# ---------------------------------------------------------------------------


class TestProductionSchedule:
    def test_empty_targets(self) -> None:
        assert production_schedule({}) == []

    def test_leaf_target_has_no_schedule(self) -> None:
        """Raw ores aren't crafted, so no schedule entry."""
        assert production_schedule({int(ItemType.IRON_ORE): 5}) == []

    def test_single_furnace_recipe(self) -> None:
        sched = production_schedule({int(ItemType.IRON_PLATE): 3})
        assert sched == [
            (int(ItemType.IRON_PLATE), 3, int(MachineType.FURNACE)),
        ]

    def test_intermediate_emitted_before_consumer(self) -> None:
        """MINER consumes WIRE — WIRE must come first."""
        sched = production_schedule({int(ItemType.MINER): 2})
        items = [item for item, _, _ in sched]
        # WIRE comes before MINER
        assert items.index(int(ItemType.WIRE)) < items.index(int(ItemType.MINER))
        # IRON_PLATE comes before MINER
        assert items.index(int(ItemType.IRON_PLATE)) < items.index(int(ItemType.MINER))

    def test_intermediate_qty_aggregated_across_consumers(self) -> None:
        """Two targets that both consume WIRE produce one combined
        WIRE entry sized to cover both demands.
        """
        sched = production_schedule(
            {
                int(ItemType.MINER): 2,  # consumes 2 WIRE
                int(ItemType.PALLET): 3,  # consumes 3 WIRE
            }
        )
        wire_entries = [
            (item, qty, m) for item, qty, m in sched if item == int(ItemType.WIRE)
        ]
        assert len(wire_entries) == 1
        _, qty, machine = wire_entries[0]
        assert qty == 5  # 2 + 3
        assert machine == int(MachineType.ASSEMBLER)

    def test_machine_type_per_entry(self) -> None:
        """Each schedule entry carries the right machine_type from the
        recipe — plates run on FURNACE, everything else on ASSEMBLER.
        """
        sched = production_schedule({int(ItemType.MINER): 1})
        for item, _, machine in sched:
            if item in {
                int(ItemType.IRON_PLATE),
                int(ItemType.COPPER_PLATE),
                int(ItemType.TIN_PLATE),
                int(ItemType.WAFER),
                int(ItemType.REFRACTORY),
            }:
                assert machine == int(MachineType.FURNACE)
            else:
                assert machine == int(MachineType.ASSEMBLER)

    def test_output_count_rounds_up_qty(self) -> None:
        """If output_count=2 and target=3, schedule reports qty=4
        (one extra cycle's surplus).
        """
        balance = RecipeBalance(
            overrides=((int(ItemType.IRON_PLATE), RecipeOverride(output_count=2)),)
        )
        book = BASE_RECIPE_BOOK.with_balance(balance)
        sched = production_schedule({int(ItemType.IRON_PLATE): 3}, book)
        plates = [(i, q) for i, q, _ in sched if i == int(ItemType.IRON_PLATE)]
        assert plates == [(int(ItemType.IRON_PLATE), 4)]

    def test_topological_order_validates(self) -> None:
        """For any schedule entry, every recipe input that is itself
        a scheduled item must appear earlier in the list.
        """
        sched = production_schedule(
            {
                int(ItemType.MINER): 2,
                int(ItemType.PALLET): 3,
                int(ItemType.FURNACE): 1,
            }
        )
        positions = {item: idx for idx, (item, _, _) in enumerate(sched)}
        from factoriax.recipes import BASE_RECIPES

        recipe_by_output = {r.output: r for r in BASE_RECIPES}
        for item, _, _ in sched:
            recipe = recipe_by_output[item]
            for input_item, _ in recipe.inputs:
                if int(input_item) in positions:
                    assert positions[int(input_item)] < positions[item], (
                        f"{ItemType(item).name} appears before its input "
                        f"{ItemType(int(input_item)).name}"
                    )

    def test_cycle_raises(self) -> None:
        synthetic = RecipeBook(
            recipes=(
                Recipe(
                    output=int(ItemType.SCIENCE_LAB),
                    inputs=((int(ItemType.ROCKET_CORE), 1), (int(ItemType.HULL), 1)),
                    ticks=4,
                    name="cyclic-lab",
                ),
                Recipe(
                    output=int(ItemType.ROCKET_CORE),
                    inputs=(
                        (int(ItemType.SCIENCE_LAB), 1),
                        (int(ItemType.AVIONICS), 1),
                    ),
                    ticks=4,
                    name="cyclic-core",
                ),
            )
        )
        with pytest.raises(ValueError, match="Recipe cycle"):
            production_schedule({int(ItemType.SCIENCE_LAB): 1}, synthetic)


# ---------------------------------------------------------------------------
# Inventory arithmetic helpers
# ---------------------------------------------------------------------------


class TestSumInventories:
    def test_no_inputs(self) -> None:
        assert sum_inventories() == {}

    def test_single_input_passes_through(self) -> None:
        assert sum_inventories({1: 5, 2: 3}) == {1: 5, 2: 3}

    def test_disjoint_keys_merged(self) -> None:
        assert sum_inventories({1: 5}, {2: 3}) == {1: 5, 2: 3}

    def test_overlapping_keys_summed(self) -> None:
        assert sum_inventories({1: 5}, {1: 3, 2: 2}, {1: 1}) == {1: 9, 2: 2}


class TestScaleInventory:
    def test_factor_zero_zeroes_all(self) -> None:
        assert scale_inventory({1: 5, 2: 3}, 0) == {1: 0, 2: 0}

    def test_factor_one_passthrough(self) -> None:
        assert scale_inventory({1: 5, 2: 3}, 1) == {1: 5, 2: 3}

    def test_factor_three(self) -> None:
        assert scale_inventory({1: 5, 2: 3}, 3) == {1: 15, 2: 9}

    def test_negative_factor_raises(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            scale_inventory({1: 5}, -1)
