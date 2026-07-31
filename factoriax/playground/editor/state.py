"""The level under edit, as mutable numpy arrays.

The editor holds a level in plain numpy arrays, so an edit needs no JAX. The
editor converts to or from :class:`~factoriax.engine.levels.Level` at three
moments only: it loads a file, it saves a file, or it starts a play session.

Every field of :class:`EditorState` has the name and the shape of the level
field that it holds. A conversion therefore copies, and does not translate.
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

#: Number of inventory rows that the editor shows for one player. The engine
#: holds a player inventory by item and has no slot, so this number is a
#: display width and not a limit of the engine.
NUM_INVENTORY_SLOTS: int = 10

#: Target of an inventory operation. It is ``("player", player_index, 0)`` for
#: a player, or ``("machine", tile_x, tile_y)`` for a machine.
InvTarget = tuple[str, int, int]


@dataclasses.dataclass
class ResourceBrush:
    """Amount of ore that the editor writes to each tile that it paints.

    Attributes
    ----------
    mode
        ``"exact"`` gives every tile the same amount. ``"range"`` gives each
        tile its own random amount.
    exact_value
        Amount for one tile in ``"exact"`` mode.
    range_min, range_max
        Lowest and highest amount in ``"range"`` mode. Both ends are included.
    """

    mode: str = "exact"
    exact_value: int = BLOCK_MAX_RESOURCES
    range_min: int = 1
    range_max: int = BLOCK_MAX_RESOURCES


_ORE_BLOCKS = frozenset(
    {int(BlockType.IRON), int(BlockType.COPPER), int(BlockType.COAL)}
)


def sample_resource(brush: ResourceBrush, rng: np.random.Generator) -> int:
    """Return the ore amount for one tile.

    Parameters
    ----------
    brush
        Brush that gives the mode and the limits.
    rng
        Random generator. The function reads it in ``"range"`` mode only.

    Returns
    -------
    int
        Ore amount for one tile.
    """
    if brush.mode == "range":
        return int(rng.integers(brush.range_min, brush.range_max + 1))
    return brush.exact_value


@dataclasses.dataclass
class EditorState:
    """The level that the editor has open.

    Every field has the name and the shape of the field of
    :class:`~factoriax.engine.levels.Level` that it holds. The arrays are
    numpy and not JAX, and the functions of this module write to them in
    place.

    ``machine_inventory`` records the contents of a machine by item, as the
    level does, and records no slot. When the engine builds the state, it
    gives each item a slot. It selects that slot from the part that the item
    has in the recipe of that machine. A slot order in this class can
    therefore not survive a save, and does not agree with the engine.

    Attributes
    ----------
    name
        Identifier of the level. A save writes it to the file.
    map_width, map_height
        Tile dimensions. Every array field has these dimensions.
    block_map
        :class:`~factoriax.engine.constants.BlockType` of each tile. Shape
        ``(map_height, map_width)``.
    block_resources
        Ore units in each tile. A zero means an empty deposit, and not "use
        the default".
    machine_types
        :class:`~factoriax.engine.constants.Machine` on each tile, and
        ``NONE`` where the tile is clear.
    machine_directions
        Facing of each tile. The value has no meaning on a tile with no
        machine.
    machine_inventory
        Contents of the machine on each tile, by item. Shape
        ``(map_height, map_width, NUM_ITEM_TYPES)``.
    player_inventory
        ``(item, count)`` pairs that every player starts with, or ``None``.
    player_inventories
        Contents for one player, with the player index as the key. For a
        player that it names, it replaces ``player_inventory``.
    player_positions
        ``(x, y)`` spawn tile of each player, with the player index as the
        key. A save numbers the players by list position, so a gap in these
        keys closes on the way out.
    dirty
        ``True`` after an edit that no save has written yet.
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
    """Return a new editor state of dirt tiles.

    Parameters
    ----------
    width
        Map width in tiles.
    height
        Map height in tiles.
    name
        Identifier of the level.

    Returns
    -------
    EditorState
        A level of dirt tiles, with no ore, no machine, and no player.
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
    """Return an editor state that holds the contents of a level.

    A level field that holds ``None`` gives the editor the value that the
    engine applies by default. For ``block_resources`` this is a full deposit
    on each ore tile, and for the other fields it is an empty array.

    Every array and every list is a copy, so an edit in the editor cannot
    reach the level that this function read.

    Parameters
    ----------
    level
        Level to read. The function does not modify it.

    Returns
    -------
    EditorState
        The same level, in the form that the editor writes to.
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
            list(level.player_inventory) if level.player_inventory is not None else None
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
    """Return a level that holds the contents of the editor state.

    An optional field becomes ``None`` when it holds what the engine applies
    by default. This keeps the saved file small, and the value that comes back
    is the same.

    ``block_resources`` is the one field where a zero and ``None`` are
    different instructions. ``None`` tells the engine to fill each ore tile,
    and a zero tells it that the deposit is empty. The function therefore
    compares against the default, and does not test for zeros.

    Parameters
    ----------
    state
        Editor state to read. The function does not modify it.

    Returns
    -------
    Level
        A level that a save or :func:`~factoriax.engine.levels.build_state`
        can read.
    """
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

    # A level numbers its players by list position, so a gap closes here. The
    # inventories move with the positions, or a player gets the items of the
    # player that was removed.
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
            list(state.player_inventory) if state.player_inventory is not None else None
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
    """Paint one tile with a block type, and set the ore of that tile.

    An ore block gets its amount from the brush. Every other block gets zero.
    A column or a row outside the map has no effect.

    Parameters
    ----------
    state
        Editor state. The function writes to it.
    x
        Tile column.
    y
        Tile row.
    block
        :class:`~factoriax.engine.constants.BlockType` value to write.
    brush
        Brush that gives the ore amount.
    rng
        Random generator that the brush reads in ``"range"`` mode.
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
    """Put a machine on one tile, and clear the contents of that tile.

    A machine that was on the tile is lost, together with its contents. A
    column or a row outside the map has no effect.

    Parameters
    ----------
    state
        Editor state. The function writes to it.
    x
        Tile column.
    y
        Tile row.
    machine
        :class:`~factoriax.engine.constants.Machine` value to write.
    direction
        :class:`~factoriax.engine.constants.Direction` value for the facing.
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
    """Paint a rectangle of tiles with a block type.

    The rectangle includes all four edges. If an edge is outside the map, the
    function moves that edge to the map. A rectangle fully outside the map has
    no effect.

    In ``"range"`` mode each tile of an ore rectangle gets its own random
    amount.

    Parameters
    ----------
    state
        Editor state. The function writes to it.
    x0, y0
        Column and row of one corner.
    x1, y1
        Column and row of the opposite corner.
    block
        :class:`~factoriax.engine.constants.BlockType` value to write.
    brush
        Brush that gives the ore amount.
    rng
        Random generator that the brush reads in ``"range"`` mode.
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
    """Set one tile to dirt, and keep the machine on that tile.

    The ore of the tile becomes zero. A column or a row outside the map has no
    effect.

    Parameters
    ----------
    state
        Editor state. The function writes to it.
    x
        Tile column.
    y
        Tile row.
    """
    if not (0 <= x < state.map_width and 0 <= y < state.map_height):
        return
    state.block_map[y, x] = int(BlockType.DIRT)
    state.block_resources[y, x] = 0
    state.dirty = True


def erase_machine(state: EditorState, x: int, y: int) -> None:
    """Remove the machine from one tile, and keep the block of that tile.

    The contents of the machine are lost. A column or a row outside the map
    has no effect.

    Parameters
    ----------
    state
        Editor state. The function writes to it.
    x
        Tile column.
    y
        Tile row.
    """
    if not (0 <= x < state.map_width and 0 <= y < state.map_height):
        return
    state.machine_types[y, x] = int(Machine.NONE)
    state.machine_directions[y, x] = 0
    state.machine_inventory[y, x] = 0
    state.dirty = True


def erase_tile(state: EditorState, x: int, y: int) -> None:
    """Set one tile to dirt, and remove the machine on that tile.

    This function calls :func:`erase_block` and then :func:`erase_machine`.

    Parameters
    ----------
    state
        Editor state. The function writes to it.
    x
        Tile column.
    y
        Tile row.
    """
    erase_block(state, x, y)
    erase_machine(state, x, y)


def add_column(state: EditorState) -> None:
    """Add one column of dirt tiles to the right edge of the map.

    The function makes every array one column wider, and adds 1 to
    ``map_width``.

    Parameters
    ----------
    state
        Editor state. The function writes to it.
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
    """Remove the right column of the map.

    A player on that column is removed with it. A map one tile wide does not
    change.

    Parameters
    ----------
    state
        Editor state. The function writes to it.
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
    """Add one row of dirt tiles to the bottom edge of the map.

    The function makes every array one row taller, and adds 1 to
    ``map_height``.

    Parameters
    ----------
    state
        Editor state. The function writes to it.
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
    """Remove the bottom row of the map.

    A player on that row is removed with it. A map one tile tall does not
    change.

    Parameters
    ----------
    state
        Editor state. The function writes to it.
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
    """Remove each player that is outside the map.

    A call to :func:`remove_column` or :func:`remove_row` can put a player
    outside the map, and this function then removes that player.

    Parameters
    ----------
    state
        Editor state. The function writes to it.
    """
    state.player_positions = {
        k: v
        for k, v in state.player_positions.items()
        if 0 <= v[0] < state.map_width and 0 <= v[1] < state.map_height
    }


def set_player_position(state: EditorState, player_idx: int, x: int, y: int) -> None:
    """Set the spawn tile of one player.

    If the player has a spawn tile, the function moves that player to the new
    tile. A column or a row outside the map has no effect.

    Parameters
    ----------
    state
        Editor state. The function writes to it.
    player_idx
        Index of the player.
    x
        Tile column.
    y
        Tile row.
    """
    if not (0 <= x < state.map_width and 0 <= y < state.map_height):
        return
    state.player_positions[player_idx] = (x, y)
    state.dirty = True


def remove_player_at(state: EditorState, x: int, y: int) -> None:
    """Remove each player that has its spawn tile at one tile.

    This can leave a gap in the player numbering.
    :func:`editor_state_to_level` closes that gap.

    Parameters
    ----------
    state
        Editor state. The function writes to it.
    x
        Tile column.
    y
        Tile row.
    """
    to_remove = [k for k, v in state.player_positions.items() if v == (x, y)]
    for k in to_remove:
        del state.player_positions[k]
    if to_remove:
        state.dirty = True


def erase_entity(state: EditorState, x: int, y: int) -> None:
    """Remove each entity at one tile.

    A player is the only entity that the editor puts on a tile, so this
    function calls :func:`remove_player_at`.

    Parameters
    ----------
    state
        Editor state. The function writes to it.
    x
        Tile column.
    y
        Tile row.
    """
    remove_player_at(state, x, y)


# ---------------------------------------------------------------------------
# Inventory helpers
# ---------------------------------------------------------------------------


def get_inventory_slots(state: EditorState, target: InvTarget) -> list[tuple[int, int]]:
    """Return the contents of a player or a machine, as one row for each slot.

    For a machine the rows are a view over ``machine_inventory``. The machine
    records its contents by item, so each item fills one row, and the rows
    follow the item order. An item can therefore not hold two rows.

    The list holds at least :data:`NUM_INVENTORY_SLOTS` rows for a player, and
    at least ``MAX_MACHINE_INVENTORY_SLOTS`` rows for a machine. A machine
    with more items than that gets one row for each item, so no item is lost.

    Parameters
    ----------
    state
        Editor state to read.
    target
        Player or machine to read.

    Returns
    -------
    list[tuple[int, int]]
        One ``(item, count)`` pair for each row. An empty row is
        ``(ItemType.EMPTY, 0)``.
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
    """Return the number of slots that a player or a machine has.

    Parameters
    ----------
    state
        Editor state to read.
    target
        Player or machine to read.

    Returns
    -------
    int
        :data:`NUM_INVENTORY_SLOTS` for a player. For a machine, the slot
        count of that machine type.
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
    """Put one item and its count in one slot.

    For a machine the slot is a row of the view that
    :func:`get_inventory_slots` returns. The write removes the item that the
    row showed. If the same item is already in another row, the machine keeps
    one entry for that item, and this count replaces the earlier one.

    A count of 0, or an item of ``ItemType.EMPTY``, clears the row.

    Parameters
    ----------
    state
        Editor state. The function writes to it.
    target
        Player or machine to write to.
    slot
        Index of the row.
    item_type
        :class:`~factoriax.engine.constants.ItemType` value to write.
    count
        Number of items.
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
    """Clear one slot.

    Parameters
    ----------
    state
        Editor state. The function writes to it.
    target
        Player or machine to write to.
    slot
        Index of the row.
    """
    set_inventory_slot(state, target, slot, int(ItemType.EMPTY), 0)
