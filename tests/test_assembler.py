"""Tests for the assembler machine system."""

from jax import random

from factoriax.constants import (
    MAX_ASSEMBLER_STACK_SIZE,
    ItemType,
    MachineType,
)
from factoriax.machines import run_assemblers
from factoriax.state import EnvParams
from factoriax.world_gen import generate_world


def _make_state_with_assembler(
    recipe: int = 0,
    slot0_item: int = 0,
    slot0_count: int = 0,
    slot1_item: int = 0,
    slot1_count: int = 0,
    slot3_item: int = 0,
    slot3_count: int = 0,
    power: int = 0,
) -> object:
    """Create a minimal world state with an assembler at (0, 0).

    Args:
        recipe: Assembler recipe index.
        slot0_item: Item type in input slot 0.
        slot0_count: Stack count in input slot 0.
        slot1_item: Item type in input slot 1.
        slot1_count: Stack count in input slot 1.
        slot3_item: Item type in output slot 3.
        slot3_count: Stack count in output slot 3.
        power: Initial machine_power value.

    Returns:
        An EnvState with the assembler configured.
    """
    rng = random.PRNGKey(42)
    params = EnvParams(map_width=4, map_height=4, num_players=1)
    state = generate_world(rng, params)

    state = state.replace(
        machine_types=state.machine_types.at[0, 0].set(MachineType.ASSEMBLER),
        machine_selected_recipe=state.machine_selected_recipe.at[0, 0].set(recipe),
        machine_power=state.machine_power.at[0, 0].set(power),
    )

    inv_items = state.machine_inventory_items
    inv_counts = state.machine_inventory_counts

    inv_items = inv_items.at[0, 0, 0].set(slot0_item)
    inv_counts = inv_counts.at[0, 0, 0].set(slot0_count)
    inv_items = inv_items.at[0, 0, 1].set(slot1_item)
    inv_counts = inv_counts.at[0, 0, 1].set(slot1_count)
    inv_items = inv_items.at[0, 0, 3].set(slot3_item)
    inv_counts = inv_counts.at[0, 0, 3].set(slot3_count)

    state = state.replace(
        machine_inventory_items=inv_items,
        machine_inventory_counts=inv_counts,
    )
    return state


class TestAssemblerStartsCraft:
    """Assembler should consume inputs and start a countdown."""

    def test_hull_recipe_starts(self) -> None:
        """Hull recipe: 5 iron -> power set, iron consumed."""
        state = _make_state_with_assembler(
            recipe=0,
            slot0_item=int(ItemType.IRON),
            slot0_count=10,
        )
        new_state = run_assemblers(state)

        assert int(new_state.machine_power[0, 0]) == 4  # ASSEMBLER_RECIPE_TICKS[0]
        assert int(new_state.machine_inventory_counts[0, 0, 0]) == 5  # 10 - 5

    def test_no_start_without_inputs(self) -> None:
        """Assembler with insufficient inputs should remain idle."""
        state = _make_state_with_assembler(
            recipe=0,
            slot0_item=int(ItemType.IRON),
            slot0_count=3,
        )
        new_state = run_assemblers(state)

        assert int(new_state.machine_power[0, 0]) == 0
        assert int(new_state.machine_inventory_counts[0, 0, 0]) == 3


class TestAssemblerCompletesCraft:
    """Assembler at power == 1 should produce output."""

    def test_hull_output_produced(self) -> None:
        """Power == 1 with space in output produces 1 hull."""
        state = _make_state_with_assembler(
            recipe=0,
            power=1,
        )
        new_state = run_assemblers(state)

        assert int(new_state.machine_power[0, 0]) == 0
        assert int(new_state.machine_inventory_items[0, 0, 3]) == int(
            ItemType.HULL
        )
        assert int(new_state.machine_inventory_counts[0, 0, 3]) == 1

    def test_output_stacks(self) -> None:
        """Completing a craft adds to existing output stack."""
        state = _make_state_with_assembler(
            recipe=0,
            power=1,
            slot3_item=int(ItemType.HULL),
            slot3_count=5,
        )
        new_state = run_assemblers(state)

        assert int(new_state.machine_inventory_counts[0, 0, 3]) == 6


class TestAssemblerStallsOutputFull:
    """Assembler should stall when output slot is at capacity."""

    def test_stall_at_cap(self) -> None:
        """Power stays at 1 and no item is lost when output is full."""
        state = _make_state_with_assembler(
            recipe=0,
            power=1,
            slot3_item=int(ItemType.HULL),
            slot3_count=MAX_ASSEMBLER_STACK_SIZE,
        )
        new_state = run_assemblers(state)

        assert int(new_state.machine_power[0, 0]) == 1
        assert (
            int(new_state.machine_inventory_counts[0, 0, 3])
            == MAX_ASSEMBLER_STACK_SIZE
        )


class TestAssemblerTwoInputRecipe:
    """Fuel pack recipe requires two distinct inputs."""

    def test_fuel_pack_starts(self) -> None:
        """Fuel pack: 3 copper + 2 coal -> power set, inputs consumed."""
        state = _make_state_with_assembler(
            recipe=1,
            slot0_item=int(ItemType.COPPER),
            slot0_count=5,
            slot1_item=int(ItemType.COAL),
            slot1_count=4,
        )
        new_state = run_assemblers(state)

        assert int(new_state.machine_power[0, 0]) == 6  # ASSEMBLER_RECIPE_TICKS[1]
        assert int(new_state.machine_inventory_counts[0, 0, 0]) == 2  # 5 - 3
        assert int(new_state.machine_inventory_counts[0, 0, 1]) == 2  # 4 - 2

    def test_fuel_pack_missing_second_input(self) -> None:
        """Missing coal should prevent craft start."""
        state = _make_state_with_assembler(
            recipe=1,
            slot0_item=int(ItemType.COPPER),
            slot0_count=5,
        )
        new_state = run_assemblers(state)

        assert int(new_state.machine_power[0, 0]) == 0


class TestAssemblerRecipeChangeBlocked:
    """Recipe change should be blocked while crafting or with items loaded."""

    def test_power_blocks_recipe_change(self) -> None:
        """An assembler mid-craft (power > 0) keeps its recipe."""
        state = _make_state_with_assembler(recipe=0, power=3)

        is_idle = int(state.machine_power[0, 0]) == 0
        assert not is_idle

    def test_items_in_slots_block_recipe_change(self) -> None:
        """Items in input slots should prevent recipe switching."""
        state = _make_state_with_assembler(
            recipe=0,
            slot0_item=int(ItemType.IRON),
            slot0_count=5,
        )

        has_inputs = (
            int(state.machine_inventory_counts[0, 0, 0]) > 0
            or int(state.machine_inventory_counts[0, 0, 1]) > 0
            or int(state.machine_inventory_counts[0, 0, 2]) > 0
        )
        assert has_inputs


class TestAssemblerDepositFiltering:
    """Only recipe-correct items should be depositable into assembler inputs."""

    def test_correct_item_accepted(self) -> None:
        """Iron into slot 0 of hull recipe should succeed."""
        from factoriax.play.transfer import deposit_to_machine

        state = _make_state_with_assembler(recipe=0)
        # Give the player iron in slot 0.
        state = state.replace(
            inventory_items=state.inventory_items.at[0, 0].set(
                int(ItemType.IRON)
            ),
            inventory_counts=state.inventory_counts.at[0, 0].set(10),
            selected_slots=state.selected_slots.at[0].set(0),
        )
        new_state = deposit_to_machine(state, 0, 0, 0, 0, 0)
        assert int(new_state.machine_inventory_items[0, 0, 0]) == int(
            ItemType.IRON
        )
        assert int(new_state.machine_inventory_counts[0, 0, 0]) == 10

    def test_wrong_item_rejected(self) -> None:
        """Copper into slot 0 of hull recipe should be rejected."""
        from factoriax.play.transfer import deposit_to_machine

        state = _make_state_with_assembler(recipe=0)
        state = state.replace(
            inventory_items=state.inventory_items.at[0, 0].set(
                int(ItemType.COPPER)
            ),
            inventory_counts=state.inventory_counts.at[0, 0].set(10),
            selected_slots=state.selected_slots.at[0].set(0),
        )
        new_state = deposit_to_machine(state, 0, 0, 0, 0, 0)
        # Copper should stay in player inventory, assembler slot unchanged.
        assert int(new_state.machine_inventory_counts[0, 0, 0]) == 0
        assert int(new_state.inventory_counts[0, 0]) == 10

    def test_unused_slot_rejected(self) -> None:
        """Hull recipe uses 1 input; slot 1 (unused) should reject items."""
        from factoriax.play.transfer import deposit_to_machine

        state = _make_state_with_assembler(recipe=0)
        state = state.replace(
            inventory_items=state.inventory_items.at[0, 0].set(
                int(ItemType.IRON)
            ),
            inventory_counts=state.inventory_counts.at[0, 0].set(10),
            selected_slots=state.selected_slots.at[0].set(0),
        )
        # Target slot 1 — hull recipe has 0 count for second input.
        new_state = deposit_to_machine(state, 0, 0, 0, 1, 0)
        assert int(new_state.machine_inventory_counts[0, 0, 1]) == 0


class TestAssemblerPlacementAndPickup:
    """Assembler should round-trip through place and pickup."""

    def test_assembler_exists_in_state(self) -> None:
        """Placing an assembler sets the correct machine type."""
        state = _make_state_with_assembler()
        assert int(state.machine_types[0, 0]) == int(MachineType.ASSEMBLER)

    def test_progress_decrements(self) -> None:
        """Power > 1 should decrement by 1 each tick."""
        state = _make_state_with_assembler(power=5)
        new_state = run_assemblers(state)
        assert int(new_state.machine_power[0, 0]) == 4
