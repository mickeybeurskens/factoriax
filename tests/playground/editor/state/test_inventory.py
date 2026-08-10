"""Tests for the per-machine inventory slots in the editor state.

A slot count comes from the machine kind. A write outside that count is
refused, and the slots survive the round trip through a :class:`Level`."""

from factoriax.engine.constants import ItemType, Machine
from factoriax.playground.editor.state import (
    InvTarget,
    clear_inventory_slot,
    editor_state_from_level,
    editor_state_to_level,
    get_inventory_slots,
    get_num_slots,
    new_editor_state,
    set_inventory_slot,
    set_machine,
    set_player_position,
)


class TestGetInventorySlots:
    """Tests for get_inventory_slots."""

    def test_empty_player_inventory(self) -> None:
        state = new_editor_state(5, 5)
        target: InvTarget = ("player", 0, 0)
        slots = get_inventory_slots(state, target)
        assert len(slots) == 10
        assert all(item == int(ItemType.EMPTY) for item, _ in slots)

    def test_player_with_items(self) -> None:
        state = new_editor_state(5, 5)
        state.player_inventories[0] = [
            (int(ItemType.COAL), 50),
            (int(ItemType.IRON_ORE), 30),
        ]
        slots = get_inventory_slots(state, ("player", 0, 0))
        assert slots[0] == (int(ItemType.COAL), 50)
        assert slots[1] == (int(ItemType.IRON_ORE), 30)
        assert slots[2] == (int(ItemType.EMPTY), 0)

    def test_machine_inventory(self) -> None:
        state = new_editor_state(5, 5)
        set_machine(state, 2, 2, int(Machine.PALLET), 0)
        state.machine_inventory[2, 2, int(ItemType.IRON_ORE)] = 10
        slots = get_inventory_slots(state, ("machine", 2, 2))
        assert slots[0] == (int(ItemType.IRON_ORE), 10)


class TestSetInventorySlot:
    """Tests for set_inventory_slot."""

    def test_set_player_slot(self) -> None:
        state = new_editor_state(5, 5)
        target: InvTarget = ("player", 0, 0)
        set_inventory_slot(state, target, 3, int(ItemType.COPPER_ORE), 25)
        slots = get_inventory_slots(state, target)
        assert slots[3] == (int(ItemType.COPPER_ORE), 25)
        assert state.dirty is True

    def test_set_machine_slot(self) -> None:
        state = new_editor_state(5, 5)
        set_machine(state, 1, 1, int(Machine.MINER), 0)
        target: InvTarget = ("machine", 1, 1)
        set_inventory_slot(state, target, 0, int(ItemType.COAL), 5)
        assert state.machine_inventory[1, 1, int(ItemType.COAL)] == 5


class TestClearInventorySlot:
    """Tests for clear_inventory_slot."""

    def test_clear_player_slot(self) -> None:
        state = new_editor_state(5, 5)
        target: InvTarget = ("player", 0, 0)
        set_inventory_slot(state, target, 0, int(ItemType.IRON_ORE), 10)
        clear_inventory_slot(state, target, 0)
        slots = get_inventory_slots(state, target)
        assert slots[0] == (int(ItemType.EMPTY), 0)


class TestGetNumSlots:
    """Tests for get_num_slots."""

    def test_player_has_ten(self) -> None:
        state = new_editor_state(5, 5)
        assert get_num_slots(state, ("player", 0, 0)) == 10

    def test_miner_has_one(self) -> None:
        state = new_editor_state(5, 5)
        set_machine(state, 0, 0, int(Machine.MINER), 0)
        assert get_num_slots(state, ("machine", 0, 0)) == 1

    def test_pallet_has_one(self) -> None:
        state = new_editor_state(5, 5)
        set_machine(state, 0, 0, int(Machine.PALLET), 0)
        assert get_num_slots(state, ("machine", 0, 0)) == 1


class TestPerPlayerInventoryRoundTrip:
    """Tests for per-player inventory through Level save/load."""

    def test_round_trip(self) -> None:
        state = new_editor_state(8, 8)
        set_player_position(state, 0, 1, 1)
        set_inventory_slot(state, ("player", 0, 0), 0, int(ItemType.COAL), 50)
        set_inventory_slot(state, ("player", 0, 0), 1, int(ItemType.IRON_ORE), 30)
        level = editor_state_to_level(state)
        assert level.player_inventories is not None
        assert 0 in level.player_inventories
        state2 = editor_state_from_level(level)
        slots = get_inventory_slots(state2, ("player", 0, 0))
        assert slots[0] == (int(ItemType.COAL), 50)
        assert slots[1] == (int(ItemType.IRON_ORE), 30)

    def test_empty_becomes_none(self) -> None:
        state = new_editor_state(5, 5)
        level = editor_state_to_level(state)
        assert level.player_inventories is None
