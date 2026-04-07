"""Tests for the DEPOSIT and WITHDRAW compound actions.

Covers deposit into chests, miners (fuel), and assemblers (recipe-filtered
inputs), as well as withdraw from various machine types. Uses the pouch
inventory model where player_inventory has shape (P, NUM_ITEM_TYPES) and
machine_inventory has shape (H, W, NUM_ITEM_TYPES).
"""

from __future__ import annotations

import jax.numpy as jnp

from factoriax import Action, BlockType, Direction, ItemType
from factoriax.constants import (
    MACHINE_INVENTORY_COUNT_DTYPE,
    MAX_MACHINE_STACK_SIZE,
    NUM_ITEM_TYPES,
    MachineType,
)
from factoriax.game_logic import deposit_to_adjacent, withdraw_from_adjacent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIRT_3X3 = jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32)


def _machine_types(w: int, h: int, placements: dict[tuple[int, int], int]):
    """Build a machine_types grid with specific placements."""
    arr = jnp.full((h, w), MachineType.NONE, dtype=jnp.int32)
    for (x, y), mtype in placements.items():
        arr = arr.at[y, x].set(mtype)
    return arr


def _machine_inv(
    h: int,
    w: int,
    **items: dict[tuple[int, int], int],
) -> jnp.ndarray:
    """Build a machine inventory pouch.

    Each keyword maps a (y, x) tuple to item-type counts.
    The keyword format is not used here; instead use the
    positional helper below.
    """
    return jnp.zeros((h, w, NUM_ITEM_TYPES), dtype=MACHINE_INVENTORY_COUNT_DTYPE)


def _set_machine_inv(
    h: int,
    w: int,
    entries: dict[tuple[int, int, int], int],
) -> jnp.ndarray:
    """Build a machine inventory pouch with specific item counts.

    Args:
        h: Grid height.
        w: Grid width.
        entries: Mapping of (y, x, item_type) -> count.

    Returns:
        Machine inventory array of shape (h, w, NUM_ITEM_TYPES).
    """
    inv = jnp.zeros((h, w, NUM_ITEM_TYPES), dtype=MACHINE_INVENTORY_COUNT_DTYPE)
    for (y, x, item_type), count in entries.items():
        inv = inv.at[y, x, item_type].set(count)
    return inv


def _player_inv(num_players: int, entries: dict[int, int]) -> jnp.ndarray:
    """Build a player inventory pouch.

    Args:
        num_players: Number of players.
        entries: Mapping of item_type -> count for player 0.

    Returns:
        Player inventory of shape (num_players, NUM_ITEM_TYPES).
    """
    inv = jnp.zeros((num_players, NUM_ITEM_TYPES), dtype=jnp.int32)
    for item_type, count in entries.items():
        inv = inv.at[0, item_type].set(count)
    return inv


# ===========================================================================
# DEPOSIT tests
# ===========================================================================


class TestDepositToChest:
    """Deposit items from player inventory into a chest."""

    def test_deposit_coal_into_empty_chest(self, state_factory) -> None:
        """Coal should transfer into the chest's pouch."""
        p_inv = _player_inv(1, {ItemType.COAL: 10})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
        )

        state = deposit_to_adjacent(state, 0, ItemType.COAL)

        assert int(state.player_inventory[0, ItemType.COAL]) == 0
        assert int(state.machine_inventory[1, 2, ItemType.COAL]) == 10

    def test_deposit_stacks_into_matching_type(self, state_factory) -> None:
        """Depositing coal should merge with existing coal in the chest."""
        p_inv = _player_inv(1, {ItemType.COAL: 5})
        m_inv = _set_machine_inv(3, 3, {(1, 2, ItemType.COAL): 10})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
            machine_inventory=m_inv,
        )

        state = deposit_to_adjacent(state, 0, ItemType.COAL)

        assert int(state.player_inventory[0, ItemType.COAL]) == 0
        assert int(state.machine_inventory[1, 2, ItemType.COAL]) == 15

    def test_deposit_respects_stack_cap(self, state_factory) -> None:
        """Deposit should cap at MAX_MACHINE_STACK_SIZE, leaving remainder."""
        p_inv = _player_inv(1, {ItemType.COAL: 20})
        m_inv = _set_machine_inv(
            3,
            3,
            {(1, 2, ItemType.COAL): MAX_MACHINE_STACK_SIZE - 5},
        )
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
            machine_inventory=m_inv,
        )

        state = deposit_to_adjacent(state, 0, ItemType.COAL)

        assert int(state.machine_inventory[1, 2, ItemType.COAL]) == (
            MAX_MACHINE_STACK_SIZE
        )
        assert int(state.player_inventory[0, ItemType.COAL]) == 15

    def test_deposit_noop_no_machine(self, state_factory) -> None:
        """Deposit into empty tile should be a no-op."""
        p_inv = _player_inv(1, {ItemType.COAL: 10})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
        )

        state = deposit_to_adjacent(state, 0, ItemType.COAL)

        assert int(state.player_inventory[0, ItemType.COAL]) == 10

    def test_deposit_noop_empty_type(self, state_factory) -> None:
        """Deposit with zero count of the item type should be a no-op."""
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
        )

        state = deposit_to_adjacent(state, 0, ItemType.COAL)

        assert int(state.machine_inventory[1, 2, ItemType.COAL]) == 0

    def test_deposit_noop_out_of_bounds(self, state_factory) -> None:
        """Deposit facing out of bounds should be a no-op."""
        p_inv = _player_inv(1, {ItemType.COAL: 10})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(2, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
        )

        state = deposit_to_adjacent(state, 0, ItemType.COAL)

        assert int(state.player_inventory[0, ItemType.COAL]) == 10


class TestDepositToMiner:
    """Deposit coal into a miner as fuel."""

    def test_deposit_coal_as_fuel(self, state_factory) -> None:
        """Coal deposited into a miner should be accepted."""
        p_inv = _player_inv(1, {ItemType.COAL: 5})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.MINER}),
        )

        state = deposit_to_adjacent(state, 0, ItemType.COAL)

        assert int(state.machine_inventory[1, 2, ItemType.COAL]) == 5
        assert int(state.player_inventory[0, ItemType.COAL]) == 0

    def test_deposit_non_fuel_rejected_by_miner(self, state_factory) -> None:
        """Iron deposited into a miner should be rejected (not fuel)."""
        p_inv = _player_inv(1, {ItemType.IRON: 5})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.MINER}),
        )

        state = deposit_to_adjacent(state, 0, ItemType.IRON)

        assert int(state.player_inventory[0, ItemType.IRON]) == 5
        assert int(state.machine_inventory[1, 2, ItemType.IRON]) == 0


class TestDepositToAssembler:
    """Deposit items into an assembler with recipe filtering."""

    def test_deposit_iron_into_hull_assembler(self, state_factory) -> None:
        """Iron should be accepted by a hull assembler (recipe 0 needs iron)."""
        p_inv = _player_inv(1, {ItemType.IRON: 10})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.ASSEMBLER}),
            machine_selected_recipe=jnp.zeros((3, 3), dtype=jnp.int32),
        )

        state = deposit_to_adjacent(state, 0, ItemType.IRON)

        assert int(state.machine_inventory[1, 2, ItemType.IRON]) == 10
        assert int(state.player_inventory[0, ItemType.IRON]) == 0

    def test_deposit_wrong_item_rejected(self, state_factory) -> None:
        """Copper should be rejected by a hull assembler (needs only iron)."""
        p_inv = _player_inv(1, {ItemType.COPPER: 10})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.ASSEMBLER}),
            machine_selected_recipe=jnp.zeros((3, 3), dtype=jnp.int32),
        )

        state = deposit_to_adjacent(state, 0, ItemType.COPPER)

        assert int(state.player_inventory[0, ItemType.COPPER]) == 10
        assert int(state.machine_inventory[1, 2, ItemType.COPPER]) == 0

    def test_deposit_fuel_pack_recipe_both_inputs(self, state_factory) -> None:
        """Fuel pack recipe needs copper and coal; both should deposit."""
        # Deposit copper first.
        p_inv = _player_inv(1, {ItemType.COPPER: 3})
        recipe = jnp.zeros((3, 3), dtype=jnp.int32).at[1, 2].set(1)
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.ASSEMBLER}),
            machine_selected_recipe=recipe,
        )
        state = deposit_to_adjacent(state, 0, ItemType.COPPER)
        assert int(state.machine_inventory[1, 2, ItemType.COPPER]) == 3

        # Now deposit coal.
        state = state.replace(
            player_inventory=state.player_inventory.at[0, ItemType.COAL].set(2),
        )
        state = deposit_to_adjacent(state, 0, ItemType.COAL)
        assert int(state.machine_inventory[1, 2, ItemType.COAL]) == 2


# ===========================================================================
# WITHDRAW tests
# ===========================================================================


class TestWithdrawFromChest:
    """Withdraw items from a chest into player inventory."""

    def test_withdraw_iron_from_chest(self, state_factory) -> None:
        """Should take the specified item type from the chest."""
        m_inv = _set_machine_inv(3, 3, {(1, 2, ItemType.IRON): 10})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
            machine_inventory=m_inv,
        )

        state = withdraw_from_adjacent(state, 0, ItemType.IRON)

        assert int(state.player_inventory[0, ItemType.IRON]) == 10
        assert int(state.machine_inventory[1, 2, ItemType.IRON]) == 0

    def test_withdraw_noop_empty_machine(self, state_factory) -> None:
        """Withdraw from empty chest should be a no-op."""
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
        )

        state = withdraw_from_adjacent(state, 0, ItemType.IRON)

        assert int(state.player_inventory[0, ItemType.IRON]) == 0

    def test_withdraw_noop_no_machine(self, state_factory) -> None:
        """Withdraw with no machine in front should be a no-op."""
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
        )

        state = withdraw_from_adjacent(state, 0, ItemType.IRON)

        assert int(state.player_inventory[0, ItemType.IRON]) == 0


class TestWithdrawFromMiner:
    """Withdraw from miners."""

    def test_withdraw_ore_from_miner(self, state_factory) -> None:
        """Should be able to withdraw mined ore from a miner."""
        m_inv = _set_machine_inv(3, 3, {(1, 2, ItemType.IRON): 10})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.MINER}),
            machine_inventory=m_inv,
        )

        state = withdraw_from_adjacent(state, 0, ItemType.IRON)

        assert int(state.player_inventory[0, ItemType.IRON]) == 10
        assert int(state.machine_inventory[1, 2, ItemType.IRON]) == 0


class TestWithdrawFromAssembler:
    """Withdraw from assemblers respects recipe output filtering."""

    def test_withdraw_output_from_assembler(self, state_factory) -> None:
        """Withdrawing the recipe output (hull) should work."""
        m_inv = _set_machine_inv(3, 3, {(1, 2, ItemType.HULL): 5})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.ASSEMBLER}),
            machine_inventory=m_inv,
            machine_selected_recipe=jnp.zeros((3, 3), dtype=jnp.int32),
        )

        state = withdraw_from_adjacent(state, 0, ItemType.HULL)

        assert int(state.player_inventory[0, ItemType.HULL]) == 5
        assert int(state.machine_inventory[1, 2, ItemType.HULL]) == 0

    def test_withdraw_input_from_assembler_rejected(self, state_factory) -> None:
        """Withdrawing a recipe input (iron) should be rejected."""
        m_inv = _set_machine_inv(3, 3, {(1, 2, ItemType.IRON): 5})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.ASSEMBLER}),
            machine_inventory=m_inv,
            machine_selected_recipe=jnp.zeros((3, 3), dtype=jnp.int32),
        )

        state = withdraw_from_adjacent(state, 0, ItemType.IRON)

        assert int(state.player_inventory[0, ItemType.IRON]) == 0
        assert int(state.machine_inventory[1, 2, ItemType.IRON]) == 5


class TestWithdrawMergesIntoInventory:
    """Withdrawn items should merge with existing player stacks."""

    def test_withdraw_merges_with_existing_stack(self, state_factory) -> None:
        """Items withdrawn should add to existing count in player pouch."""
        p_inv = _player_inv(1, {ItemType.IRON: 3})
        m_inv = _set_machine_inv(3, 3, {(1, 2, ItemType.IRON): 7})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
            machine_inventory=m_inv,
        )

        state = withdraw_from_adjacent(state, 0, ItemType.IRON)

        assert int(state.player_inventory[0, ItemType.IRON]) == 10
        assert int(state.machine_inventory[1, 2, ItemType.IRON]) == 0


class TestDepositWithdrawViaStep:
    """End-to-end tests dispatched through factoriax_step."""

    def test_deposit_via_step(self, state_factory) -> None:
        """DEPOSIT_COAL action through the full step pipeline."""
        import jax

        from factoriax.game_logic import factoriax_step
        from factoriax.state import EnvParams

        p_inv = _player_inv(1, {ItemType.COAL: 5})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            player_inventory=p_inv,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
        )
        params = EnvParams(map_width=3, map_height=3, num_players=1)
        rng = jax.random.PRNGKey(0)

        state = factoriax_step(rng, state, Action.DEPOSIT_COAL, params)

        assert int(state.machine_inventory[1, 2, ItemType.COAL]) == 5
        assert int(state.player_inventory[0, ItemType.COAL]) == 0

    def test_withdraw_via_step(self, state_factory) -> None:
        """WITHDRAW_IRON action through the full step pipeline."""
        import jax

        from factoriax.game_logic import factoriax_step
        from factoriax.state import EnvParams

        m_inv = _set_machine_inv(3, 3, {(1, 2, ItemType.IRON): 10})
        state = state_factory(
            world_map=_DIRT_3X3,
            player_position=(1, 1),
            player_direction=Direction.RIGHT,
            machine_types=_machine_types(3, 3, {(2, 1): MachineType.CHEST}),
            machine_inventory=m_inv,
        )
        params = EnvParams(map_width=3, map_height=3, num_players=1)
        rng = jax.random.PRNGKey(0)

        state = factoriax_step(rng, state, Action.WITHDRAW_IRON, params)

        assert int(state.player_inventory[0, ItemType.IRON]) == 10
        assert int(state.machine_inventory[1, 2, ItemType.IRON]) == 0
