"""Editor state: mutable numpy arrays for the level editor.

The editor works on plain numpy arrays so there is no JAX dependency
during editing.  Conversion to/from :class:`~factoriax.engine.levels.Level`
happens only at save/load/play boundaries.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from factoriax.engine.constants import (
    BLOCK_MAX_RESOURCES,
    NUM_ITEM_TYPES,
    BlockType,
    ItemType,
    Machine,
)
from factoriax.engine.levels import Level, default_resources
from factoriax.playground.editor.slot_display import MAX_MACHINE_INVENTORY_SLOTS

#: Number of inventory slots shown per player in the editor. The engine
#: player inventory is item-indexed (it has no slot concept); this is
#: purely the editor's display capacity.
NUM_INVENTORY_SLOTS: int = 10

#: Target type for inventory operations.
#: ``("player", player_idx, 0)`` or ``("machine", tile_x, tile_y)``.
InvTarget = tuple[str, int, int]


@dataclasses.dataclass
class ResourceBrush:
    """Controls how resource amounts are assigned when painting ore tiles."""

    mode: str = "exact"
    exact_value: int = BLOCK_MAX_RESOURCES
    range_min: int = 1
    range_max: int = BLOCK_MAX_RESOURCES


_ORE_BLOCKS = frozenset(
    {int(BlockType.IRON), int(BlockType.COPPER), int(BlockType.COAL)}
)


def sample_resource(brush: ResourceBrush, rng: np.random.Generator) -> int:
    """Return a resource amount from the brush settings.

    Parameters
    ----------
    brush :
        Active resource brush
    rng :
        Numpy random generator for range mode
    brush: ResourceBrush :

    rng: np.random.Generator :


    Returns
    -------
    type
        Integer resource amount.

    """
    if brush.mode == "range":
        return int(rng.integers(brush.range_min, brush.range_max + 1))
    return brush.exact_value


@dataclasses.dataclass
class EditorState:
    """Mutable editor state representing a level being edited.

    Every field mirrors the field of :class:`~factoriax.engine.levels.Level`
    that carries the same name, and holds it in the same shape. A conversion
    therefore copies, and never translates. All arrays use numpy (never JAX)
    and are mutated in place for responsiveness.

    ``machine_inventory`` indexes the contents of a machine by item, as the
    level does, and records no slot. The engine assigns the slots when it
    builds the state, by the part each item plays in the recipe of that
    machine. A slot order held here could not survive a save, and would not
    match the order the engine picks.
    """

    name: str
    map_width: int
    map_height: int
    block_map: np.ndarray
    block_resources: np.ndarray
    machine_types: np.ndarray
    machine_directions: np.ndarray
    machine_inventory: np.ndarray
    player_inventory: list[tuple[int, int]] | None = None
    player_inventories: dict[int, list[tuple[int, int]]] = dataclasses.field(
        default_factory=dict,
    )
    player_positions: dict[int, tuple[int, int]] = dataclasses.field(
        default_factory=dict,
    )
    dirty: bool = False


def new_editor_state(width: int, height: int, name: str = "untitled") -> EditorState:
    """Create a blank editor state filled with dirt.

    Parameters
    ----------
    width :
        Map width in tiles.
    height :
        Map height in tiles.
    name :
        Level name.
    width: int :

    height: int :

    name: str :
         (Default value = "untitled")

    Returns
    -------
    Fresh
        class:`EditorState` with all-dirt terrain and no machines.

    """
    block_map = np.full((height, width), int(BlockType.DIRT), dtype=np.int32)
    return EditorState(
        name=name,
        map_width=width,
        map_height=height,
        block_map=block_map,
        block_resources=np.zeros((height, width), dtype=np.int32),
        machine_types=np.full((height, width), int(Machine.NONE), dtype=np.int32),
        machine_directions=np.zeros((height, width), dtype=np.int32),
        machine_inventory=np.zeros((height, width, NUM_ITEM_TYPES), dtype=np.int32),
    )


def editor_state_from_level(level: Level) -> EditorState:
    """Convert a loaded :class:`Level` into a mutable :class:`EditorState`.

    Missing optional arrays are filled with sensible defaults.

    Parameters
    ----------
    level :
        Source level.
    level: Level :


    Returns
    -------

        class:`EditorState` mirroring the level data.

    """
    resources = (
        level.block_resources.copy()
        if level.block_resources is not None
        else default_resources(level.block_map)
    )
    machines = (
        level.machine_types.copy()
        if level.machine_types is not None
        else np.full(
            (level.map_height, level.map_width),
            int(Machine.NONE),
            dtype=np.int32,
        )
    )
    directions = (
        level.machine_directions.copy()
        if level.machine_directions is not None
        else np.zeros((level.map_height, level.map_width), dtype=np.int32)
    )
    inventory = (
        level.machine_inventory.copy()
        if level.machine_inventory is not None
        else np.zeros(
            (level.map_height, level.map_width, NUM_ITEM_TYPES),
            dtype=np.int32,
        )
    )
    return EditorState(
        name=level.name,
        map_width=level.map_width,
        map_height=level.map_height,
        block_map=level.block_map.copy(),
        block_resources=resources.astype(np.int32),
        machine_types=machines.astype(np.int32),
        machine_directions=directions.astype(np.int32),
        machine_inventory=inventory.astype(np.int32),
        # Copy, so an edit in the editor cannot reach back into the level.
        player_inventory=(
            list(level.player_inventory)
            if level.player_inventory is not None
            else None
        ),
        player_inventories=(
            {k: list(v) for k, v in level.player_inventories.items()}
            if level.player_inventories
            else {}
        ),
        player_positions=(
            {i: (p[0], p[1]) for i, p in enumerate(level.player_positions)}
            if level.player_positions
            else {}
        ),
    )


def editor_state_to_level(state: EditorState) -> Level:
    """Convert the editor state back to a :class:`Level` for saving or play.

    All-zero optional arrays are stored as ``None`` to keep the JSON
    compact.

    Parameters
    ----------
    state :
        Current editor state.
    state: EditorState :


    Returns
    -------
    A
        class:`Level` ready for serialization or ``build_state``.

    """
    # ``None`` tells build_state to fill the ore tiles itself, and zeros tell
    # it the world starts with nothing. Drop the array only when it already
    # holds what the default would produce, so a depleted deposit stays
    # depleted and the saved file still stays compact.
    resources: np.ndarray | None = state.block_resources.copy()
    if np.array_equal(resources, default_resources(state.block_map)):
        resources = None

    machines: np.ndarray | None = state.machine_types.copy()
    if np.all(machines == int(Machine.NONE)):
        machines = None

    directions: np.ndarray | None = state.machine_directions.copy()
    if np.all(directions == 0):
        directions = None

    machine_inv: np.ndarray | None = state.machine_inventory.copy()
    if np.all(machine_inv == 0):
        machine_inv = None

    # A level numbers its players by list position, so a gap in the editor
    # keys closes here. The contents follow the same move, or a player would
    # inherit the items of the one that was removed.
    pp: list[tuple[int, int]] | None = None
    inventories: dict[int, list[tuple[int, int]]] | None = None
    if state.player_positions:
        order = sorted(state.player_positions)
        pp = [state.player_positions[k] for k in order]
        renumbered = {
            new: list(state.player_inventories[old])
            for new, old in enumerate(order)
            if old in state.player_inventories
        }
        inventories = renumbered or None
    elif state.player_inventories:
        inventories = {k: list(v) for k, v in state.player_inventories.items()}
    return Level(
        name=state.name,
        map_width=state.map_width,
        map_height=state.map_height,
        block_map=state.block_map.copy(),
        block_resources=resources,
        machine_types=machines,
        machine_directions=directions,
        machine_inventory=machine_inv,
        player_inventory=(
            list(state.player_inventory)
            if state.player_inventory is not None
            else None
        ),
        player_inventories=inventories,
        player_positions=pp,
    )


def set_tile(
    state: EditorState,
    x: int,
    y: int,
    block: int,
    brush: ResourceBrush,
    rng: np.random.Generator,
) -> None:
    """Paint a single tile, updating block type and resources.

    Non-ore blocks always receive 0 resources.  Ore blocks use the
    active :class:`ResourceBrush` to determine the amount.

    Parameters
    ----------
    state :
        Editor state (mutated in place).
    x :
        Tile column.
    y :
        Tile row.
    block :
        ``BlockType`` integer value.
    brush :
        Active resource brush.
    rng :
        Numpy random generator for range-mode sampling.
    state: EditorState :

    x: int :

    y: int :

    block: int :

    brush: ResourceBrush :

    rng: np.random.Generator :


    Returns
    -------

    """
    if not (0 <= x < state.map_width and 0 <= y < state.map_height):
        return
    state.block_map[y, x] = block
    if block in _ORE_BLOCKS:
        state.block_resources[y, x] = sample_resource(brush, rng)
    else:
        state.block_resources[y, x] = 0
    state.dirty = True


def set_machine(
    state: EditorState, x: int, y: int, machine: int, direction: int
) -> None:
    """Place or replace a machine on a tile.

    Parameters
    ----------
    state :
        Editor state (mutated in place).
    x :
        Tile column.
    y :
        Tile row.
    machine :
        ``Machine`` integer value.
    direction :
        ``Action`` direction value for the machine facing.
    state: EditorState :

    x: int :

    y: int :

    machine: int :

    direction: int :


    Returns
    -------

    """
    if not (0 <= x < state.map_width and 0 <= y < state.map_height):
        return
    state.machine_types[y, x] = machine
    state.machine_directions[y, x] = direction
    state.machine_inventory[y, x] = 0
    state.dirty = True


def fill_rect_tiles(
    state: EditorState,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    block: int,
    brush: ResourceBrush,
    rng: np.random.Generator,
) -> None:
    """Fill a rectangular area with a block type using the resource brush.

    Coordinates are inclusive and automatically clamped to map bounds.

    Parameters
    ----------
    state :
        Editor state (mutated in place).
    x0 :
        Left column of the rectangle.
    y0 :
        Top row.
    x1 :
        Right column (inclusive).
    y1 :
        Bottom row (inclusive).
    block :
        ``BlockType`` integer value.
    brush :
        Active resource brush.
    rng :
        Numpy random generator for range-mode sampling.
    state: EditorState :

    x0: int :

    y0: int :

    x1: int :

    y1: int :

    block: int :

    brush: ResourceBrush :

    rng: np.random.Generator :


    Returns
    -------

    """
    lx = max(0, min(x0, x1))
    ly = max(0, min(y0, y1))
    rx = min(state.map_width - 1, max(x0, x1))
    ry = min(state.map_height - 1, max(y0, y1))
    if lx > rx or ly > ry:
        return

    state.block_map[ly : ry + 1, lx : rx + 1] = block

    if block in _ORE_BLOCKS:
        h = ry - ly + 1
        w = rx - lx + 1
        if brush.mode == "range":
            state.block_resources[ly : ry + 1, lx : rx + 1] = rng.integers(
                brush.range_min, brush.range_max + 1, size=(h, w)
            )
        else:
            state.block_resources[ly : ry + 1, lx : rx + 1] = brush.exact_value
    else:
        state.block_resources[ly : ry + 1, lx : rx + 1] = 0

    state.dirty = True


def erase_block(state: EditorState, x: int, y: int) -> None:
    """Reset a tile's terrain to dirt, leaving any machine untouched.

    Parameters
    ----------
    state :
        Editor state (mutated in place).
    x :
        Tile column.
    y :
        Tile row.
    state: EditorState :

    x: int :

    y: int :


    Returns
    -------

    """
    if not (0 <= x < state.map_width and 0 <= y < state.map_height):
        return
    state.block_map[y, x] = int(BlockType.DIRT)
    state.block_resources[y, x] = 0
    state.dirty = True


def erase_machine(state: EditorState, x: int, y: int) -> None:
    """Remove a machine from a tile, leaving the terrain untouched.

    Parameters
    ----------
    state :
        Editor state (mutated in place).
    x :
        Tile column.
    y :
        Tile row.
    state: EditorState :

    x: int :

    y: int :


    Returns
    -------

    """
    if not (0 <= x < state.map_width and 0 <= y < state.map_height):
        return
    state.machine_types[y, x] = int(Machine.NONE)
    state.machine_directions[y, x] = 0
    state.machine_inventory[y, x] = 0
    state.dirty = True


def erase_tile(state: EditorState, x: int, y: int) -> None:
    """Reset a tile to dirt and remove any machine.

    Convenience function that clears both layers.

    Parameters
    ----------
    state :
        Editor state (mutated in place).
    x :
        Tile column.
    y :
        Tile row.
    state: EditorState :

    x: int :

    y: int :


    Returns
    -------

    """
    erase_block(state, x, y)
    erase_machine(state, x, y)


def add_column(state: EditorState) -> None:
    """Append one dirt column to the right edge of the map.

    All four arrays are extended in place and ``map_width`` is
    incremented.

    Parameters
    ----------
    state :
        Editor state (mutated in place).
    state: EditorState :


    Returns
    -------

    """
    h = state.map_height
    state.block_map = np.concatenate(
        [state.block_map, np.full((h, 1), int(BlockType.DIRT), dtype=np.int32)],
        axis=1,
    )
    state.block_resources = np.concatenate(
        [state.block_resources, np.zeros((h, 1), dtype=np.int32)],
        axis=1,
    )
    state.machine_types = np.concatenate(
        [
            state.machine_types,
            np.full((h, 1), int(Machine.NONE), dtype=np.int32),
        ],
        axis=1,
    )
    state.machine_directions = np.concatenate(
        [state.machine_directions, np.zeros((h, 1), dtype=np.int32)],
        axis=1,
    )
    state.machine_inventory = np.concatenate(
        [state.machine_inventory, np.zeros((h, 1, NUM_ITEM_TYPES), dtype=np.int32)],
        axis=1,
    )
    state.map_width += 1
    state.dirty = True


def remove_column(state: EditorState) -> None:
    """Remove the rightmost column from the map.

    No-op if the map is 1 tile wide.

    Parameters
    ----------
    state :
        Editor state (mutated in place).
    state: EditorState :


    Returns
    -------

    """
    if state.map_width <= 1:
        return
    state.block_map = state.block_map[:, :-1]
    state.block_resources = state.block_resources[:, :-1]
    state.machine_types = state.machine_types[:, :-1]
    state.machine_directions = state.machine_directions[:, :-1]
    state.machine_inventory = state.machine_inventory[:, :-1, :]
    state.map_width -= 1
    _clip_entities(state)
    state.dirty = True


def add_row(state: EditorState) -> None:
    """Append one dirt row to the bottom edge of the map.

    All four arrays are extended in place and ``map_height`` is
    incremented.

    Parameters
    ----------
    state :
        Editor state (mutated in place).
    state: EditorState :


    Returns
    -------

    """
    w = state.map_width
    state.block_map = np.concatenate(
        [state.block_map, np.full((1, w), int(BlockType.DIRT), dtype=np.int32)],
        axis=0,
    )
    state.block_resources = np.concatenate(
        [state.block_resources, np.zeros((1, w), dtype=np.int32)],
        axis=0,
    )
    state.machine_types = np.concatenate(
        [
            state.machine_types,
            np.full((1, w), int(Machine.NONE), dtype=np.int32),
        ],
        axis=0,
    )
    state.machine_directions = np.concatenate(
        [state.machine_directions, np.zeros((1, w), dtype=np.int32)],
        axis=0,
    )
    state.machine_inventory = np.concatenate(
        [state.machine_inventory, np.zeros((1, w, NUM_ITEM_TYPES), dtype=np.int32)],
        axis=0,
    )
    state.map_height += 1
    state.dirty = True


def remove_row(state: EditorState) -> None:
    """Remove the bottom row from the map.

    No-op if the map is 1 tile tall.

    Parameters
    ----------
    state :
        Editor state (mutated in place).
    state: EditorState :


    Returns
    -------

    """
    if state.map_height <= 1:
        return
    state.block_map = state.block_map[:-1, :]
    state.block_resources = state.block_resources[:-1, :]
    state.machine_types = state.machine_types[:-1, :]
    state.machine_directions = state.machine_directions[:-1, :]
    state.machine_inventory = state.machine_inventory[:-1, :, :]
    state.map_height -= 1
    _clip_entities(state)
    state.dirty = True


# ---------------------------------------------------------------------------
# Entity placement
# ---------------------------------------------------------------------------


def _clip_entities(state: EditorState) -> None:
    """Remove entities that fall outside the current map bounds.

    Parameters
    ----------
    state: EditorState :


    Returns
    -------

    """
    state.player_positions = {
        k: v
        for k, v in state.player_positions.items()
        if 0 <= v[0] < state.map_width and 0 <= v[1] < state.map_height
    }


def set_player_position(state: EditorState, player_idx: int, x: int, y: int) -> None:
    """Place or move a player start position.

    If player *player_idx* already has a position it is moved.

    Parameters
    ----------
    state :
        Editor state (mutated in place).
    player_idx :
        Player index (0-7).
    x :
        Tile column.
    y :
        Tile row.
    state: EditorState :

    player_idx: int :

    x: int :

    y: int :


    Returns
    -------

    """
    if not (0 <= x < state.map_width and 0 <= y < state.map_height):
        return
    state.player_positions[player_idx] = (x, y)
    state.dirty = True


def remove_player_at(state: EditorState, x: int, y: int) -> None:
    """Remove any player start at the given tile.

    Parameters
    ----------
    state :
        Editor state (mutated in place).
    x :
        Tile column.
    y :
        Tile row.
    state: EditorState :

    x: int :

    y: int :


    Returns
    -------

    """
    to_remove = [k for k, v in state.player_positions.items() if v == (x, y)]
    for k in to_remove:
        del state.player_positions[k]
    if to_remove:
        state.dirty = True


def erase_entity(state: EditorState, x: int, y: int) -> None:
    """Remove all entities (players) at the given tile.

    Parameters
    ----------
    state :
        Editor state (mutated in place).
    x :
        Tile column.
    y :
        Tile row.
    state: EditorState :

    x: int :

    y: int :


    Returns
    -------

    """
    remove_player_at(state, x, y)


# ---------------------------------------------------------------------------
# Inventory helpers
# ---------------------------------------------------------------------------


def get_inventory_slots(state: EditorState, target: InvTarget) -> list[tuple[int, int]]:
    """Return the inventory as a list of ``(ItemType, count)`` pairs.

    Parameters
    ----------
    state :
        Editor state
    target :
        player
    machine :
        tile_x
    state: EditorState :

    target: InvTarget :


    Returns
    -------
    type
        List of ``(item_type, count)`` per slot, padded to the
        slot count with ``(EMPTY, 0)``.

    """
    kind = target[0]
    if kind == "player":
        player_idx = target[1]
        slots = state.player_inventories.get(player_idx, [])
        padded = list(slots) + [(int(ItemType.EMPTY), 0)] * (
            NUM_INVENTORY_SLOTS - len(slots)
        )
        return padded[:NUM_INVENTORY_SLOTS]
    x, y = target[1], target[2]
    pouch = state.machine_inventory[y, x]
    filled = [(it, int(pouch[it])) for it in np.nonzero(pouch)[0] if it != 0]
    width = max(MAX_MACHINE_INVENTORY_SLOTS, len(filled))
    padded = filled + [(int(ItemType.EMPTY), 0)] * (width - len(filled))
    return [(int(i), int(c)) for i, c in padded]


def get_num_slots(state: EditorState, target: InvTarget) -> int:
    """Return the number of active slots for a target.

    Parameters
    ----------
    state :
        Editor state
    target :
        Inventory target
    state: EditorState :

    target: InvTarget :


    Returns
    -------
    type
        Slot count (10 for players, machine-type-dependent for machines).

    """
    from factoriax.playground.editor.slot_display import MACHINE_NUM_SLOTS

    if target[0] == "player":
        return NUM_INVENTORY_SLOTS
    mt = int(state.machine_types[target[2], target[1]])
    return int(MACHINE_NUM_SLOTS[mt])


def set_inventory_slot(
    state: EditorState,
    target: InvTarget,
    slot: int,
    item_type: int,
    count: int,
) -> None:
    """Set an inventory slot to a specific item and count.

    Parameters
    ----------
    state :
        Editor state (mutated in place).
    target :
        Inventory target.
    slot :
        Slot index.
    item_type :
        ``ItemType`` integer.
    count :
        Stack count.
    state: EditorState :

    target: InvTarget :

    slot: int :

    item_type: int :

    count: int :


    Returns
    -------

    """
    kind = target[0]
    if kind == "player":
        player_idx = target[1]
        slots = list(state.player_inventories.get(player_idx, []))
        while len(slots) < NUM_INVENTORY_SLOTS:
            slots.append((int(ItemType.EMPTY), 0))
        slots[slot] = (item_type, count)
        state.player_inventories[player_idx] = slots
    else:
        x, y = target[1], target[2]
        current = get_inventory_slots(state, target)
        if slot < len(current):
            # Free the item this row showed, so a row rewrite does not leave
            # the old item behind under its own index.
            state.machine_inventory[y, x, current[slot][0]] = 0
        if item_type != int(ItemType.EMPTY) and count > 0:
            state.machine_inventory[y, x, item_type] = count
    state.dirty = True


def clear_inventory_slot(state: EditorState, target: InvTarget, slot: int) -> None:
    """Clear an inventory slot to empty.

    Parameters
    ----------
    state :
        Editor state (mutated in place).
    target :
        Inventory target.
    slot :
        Slot index.
    state: EditorState :

    target: InvTarget :

    slot: int :


    Returns
    -------

    """
    set_inventory_slot(state, target, slot, int(ItemType.EMPTY), 0)
