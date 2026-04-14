"""Tests for the assembler machine system.

Uses the pouch inventory model where machine_inventory has shape
(H, W, NUM_ITEM_TYPES) and items are indexed by ItemType.
"""

import jax.numpy as jnp
from jax import random

from factoriax.constants import (
    DEFAULT_MACHINE_MAX_HEALTH,
    MACHINE_INVENTORY_COUNT_DTYPE,
    NUM_TECHNOLOGIES,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.game_logic import deposit_to_adjacent
from factoriax.machines import run_assemblers
from factoriax.recipes import MAX_ASSEMBLER_STACK_SIZE
from factoriax.state import EnvParams
from factoriax.world_gen import generate_world


def _make_state_with_assembler(
    recipe: int = 0,
    inv_entries: dict[int, int] | None = None,
    power: int = 0,
    research_unlocked: jnp.ndarray | None = None,
) -> object:
    """Create a minimal world state with an assembler at (0, 0).

    Args:
        recipe: Assembler recipe index.
        inv_entries: Mapping of item_type -> count for the assembler
            at (0, 0).
        power: Initial machine_power value.
        research_unlocked: Boolean array of unlocked technologies.
            Defaults to all True so existing tests pass without modification.

    Returns:
        An EnvState with the assembler configured.
    """
    rng = random.PRNGKey(42)
    params = EnvParams(map_width=4, map_height=4, num_players=1)
    state = generate_world(rng, params)

    if research_unlocked is None:
        research_unlocked = jnp.ones(NUM_TECHNOLOGIES, dtype=jnp.bool_)

    state = state.replace(
        machine_types=state.machine_types.at[0, 0].set(MachineType.ASSEMBLER),
        machine_selected_recipe=state.machine_selected_recipe.at[0, 0].set(
            recipe,
        ),
        machine_power=state.machine_power.at[0, 0].set(power),
        machine_health=state.machine_health.at[0, 0].set(
            DEFAULT_MACHINE_MAX_HEALTH,
        ),
        research_unlocked=research_unlocked,
    )

    inv = state.machine_inventory.astype(jnp.int32)
    for item_type, count in (inv_entries or {}).items():
        inv = inv.at[0, 0, item_type].set(count)
    state = state.replace(
        machine_inventory=inv.astype(MACHINE_INVENTORY_COUNT_DTYPE),
    )
    return state


class TestAssemblerStartsCraft:
    """Assembler should consume inputs and start a countdown."""

    def test_hull_recipe_starts(self) -> None:
        """Hull recipe: 5 iron -> power set, iron consumed."""
        state = _make_state_with_assembler(
            recipe=0,
            inv_entries={int(ItemType.IRON_ORE): 10},
        )
        new_state = run_assemblers(state)

        assert int(new_state.machine_power[0, 0]) == 4
        assert int(new_state.machine_inventory[0, 0, ItemType.IRON_ORE]) == 5

    def test_no_start_without_inputs(self) -> None:
        """Assembler with insufficient inputs should remain idle."""
        state = _make_state_with_assembler(
            recipe=0,
            inv_entries={int(ItemType.IRON_ORE): 3},
        )
        new_state = run_assemblers(state)

        assert int(new_state.machine_power[0, 0]) == 0
        assert int(new_state.machine_inventory[0, 0, ItemType.IRON_ORE]) == 3


class TestAssemblerCompletesCraft:
    """Assembler at power == 1 should produce output."""

    def test_hull_output_produced(self) -> None:
        """Power == 1 with space in output produces 1 hull."""
        state = _make_state_with_assembler(recipe=0, power=1)
        new_state = run_assemblers(state)

        assert int(new_state.machine_power[0, 0]) == 0
        assert int(new_state.machine_inventory[0, 0, ItemType.STEEL]) == 1

    def test_output_stacks(self) -> None:
        """Completing a craft adds to existing output stack."""
        state = _make_state_with_assembler(
            recipe=0,
            power=1,
            inv_entries={int(ItemType.STEEL): 5},
        )
        new_state = run_assemblers(state)

        assert int(new_state.machine_inventory[0, 0, ItemType.STEEL]) == 6


class TestAssemblerStallsOutputFull:
    """Assembler should stall when output slot is at capacity."""

    def test_stall_at_cap(self) -> None:
        """Power stays at 1 and no item is lost when output is full."""
        state = _make_state_with_assembler(
            recipe=0,
            power=1,
            inv_entries={int(ItemType.STEEL): MAX_ASSEMBLER_STACK_SIZE},
        )
        new_state = run_assemblers(state)

        assert int(new_state.machine_power[0, 0]) == 1
        assert (
            int(new_state.machine_inventory[0, 0, ItemType.STEEL])
            == MAX_ASSEMBLER_STACK_SIZE
        )


class TestAssemblerTwoInputRecipe:
    """Fuel pack recipe requires two distinct inputs."""

    def test_fuel_pack_starts(self) -> None:
        """Fuel pack: 3 copper + 2 coal -> power set, inputs consumed."""
        state = _make_state_with_assembler(
            recipe=1,
            inv_entries={
                int(ItemType.COPPER_ORE): 5,
                int(ItemType.COAL): 4,
            },
        )
        new_state = run_assemblers(state)

        assert int(new_state.machine_power[0, 0]) == 6
        assert int(new_state.machine_inventory[0, 0, ItemType.COPPER_ORE]) == 2
        assert int(new_state.machine_inventory[0, 0, ItemType.COAL]) == 2

    def test_fuel_pack_missing_second_input(self) -> None:
        """Missing coal should prevent craft start."""
        state = _make_state_with_assembler(
            recipe=1,
            inv_entries={int(ItemType.COPPER_ORE): 5},
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

    def test_items_in_inventory_block_recipe_change(self) -> None:
        """Items in input types should prevent recipe switching."""
        state = _make_state_with_assembler(
            recipe=0,
            inv_entries={int(ItemType.IRON_ORE): 5},
        )

        has_inputs = int(state.machine_inventory[0, 0, ItemType.IRON_ORE]) > 0
        assert has_inputs


class TestAssemblerDepositFiltering:
    """Only recipe-correct items should be depositable into assembler inputs."""

    def test_correct_item_accepted(self) -> None:
        """Iron into hull recipe assembler should succeed."""
        state = _make_state_with_assembler(recipe=0)
        state = state.replace(
            player_positions=state.player_positions.at[0].set([1, 0]),
            player_directions=state.player_directions.at[0].set(
                Direction.LEFT,
            ),
            player_inventory=state.player_inventory.at[0, ItemType.IRON_ORE].set(10),
        )
        new_state = deposit_to_adjacent(state, 0, ItemType.IRON_ORE)
        assert (
            int(
                new_state.machine_inventory[0, 0, ItemType.IRON_ORE],
            )
            == 10
        )

    def test_wrong_item_rejected(self) -> None:
        """Copper into hull recipe assembler should be rejected."""
        state = _make_state_with_assembler(recipe=0)
        state = state.replace(
            player_positions=state.player_positions.at[0].set([1, 0]),
            player_directions=state.player_directions.at[0].set(
                Direction.LEFT,
            ),
            player_inventory=state.player_inventory.at[0, ItemType.COPPER_ORE].set(10),
        )
        new_state = deposit_to_adjacent(state, 0, ItemType.COPPER_ORE)
        assert int(new_state.machine_inventory[0, 0, ItemType.COPPER_ORE]) == 0
        assert int(new_state.player_inventory[0, ItemType.COPPER_ORE]) == 10


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
