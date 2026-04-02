"""Editor state: mutable numpy arrays for the level editor.

The editor works on plain numpy arrays so there is no JAX dependency
during editing.  Conversion to/from :class:`~factoriax.levels.Level`
happens only at save/load/play boundaries.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from factoriax.constants import (
    BLOCK_MAX_RESOURCES,
    MAX_MACHINE_INVENTORY_SLOTS,
    BlockType,
    MachineType,
)
from factoriax.levels import Level, default_resources


@dataclasses.dataclass
class ResourceBrush:
    """Controls how resource amounts are assigned when painting ore tiles.

    Attributes:
        mode: ``"exact"`` assigns a fixed value, ``"range"`` samples
            uniformly from ``[range_min, range_max]``.
        exact_value: Resource amount used in exact mode.
        range_min: Lower bound (inclusive) for range mode.
        range_max: Upper bound (inclusive) for range mode.
    """

    mode: str = "exact"
    exact_value: int = BLOCK_MAX_RESOURCES
    range_min: int = 0
    range_max: int = BLOCK_MAX_RESOURCES


_ORE_BLOCKS = frozenset(
    {int(BlockType.IRON), int(BlockType.COPPER), int(BlockType.COAL)}
)


def sample_resource(brush: ResourceBrush, rng: np.random.Generator) -> int:
    """Return a resource amount from the brush settings.

    Args:
        brush: Active resource brush.
        rng: Numpy random generator for range mode.

    Returns:
        Integer resource amount.
    """
    if brush.mode == "range":
        return int(rng.integers(brush.range_min, brush.range_max + 1))
    return brush.exact_value


@dataclasses.dataclass
class EditorState:
    """Mutable editor state representing a level being edited.

    All arrays use numpy (never JAX) and are mutated in place for
    responsiveness.

    Attributes:
        name: Human-readable level name.
        map_width: Number of tile columns.
        map_height: Number of tile rows.
        block_map: Block types, shape ``(H, W)`` int32.
        block_resources: Per-tile resource amounts, shape ``(H, W)`` int32.
        machine_types: Machine type per tile, shape ``(H, W)`` int32.
        machine_directions: Machine facing direction per tile, shape ``(H, W)`` int32.
        machine_inventory_items: Item types per machine slot,
            shape ``(H, W, MAX_MACHINE_INVENTORY_SLOTS)`` int32.
        machine_inventory_counts: Stack counts per machine slot,
            shape ``(H, W, MAX_MACHINE_INVENTORY_SLOTS)`` int32.
        machine_selected_recipe: Selected assembler recipe per tile,
            shape ``(H, W)`` int32.
        dirty: ``True`` when unsaved changes exist.
    """

    name: str
    map_width: int
    map_height: int
    block_map: np.ndarray
    block_resources: np.ndarray
    machine_types: np.ndarray
    machine_directions: np.ndarray
    machine_inventory_items: np.ndarray
    machine_inventory_counts: np.ndarray
    machine_selected_recipe: np.ndarray
    player_inventory: list[tuple[int, int]] | None = None
    player_positions: dict[int, tuple[int, int]] = dataclasses.field(
        default_factory=dict,
    )
    biter_positions: list[tuple[int, int]] = dataclasses.field(
        default_factory=list,
    )
    dirty: bool = False


def new_editor_state(width: int, height: int, name: str = "untitled") -> EditorState:
    """Create a blank editor state filled with dirt.

    Args:
        width: Map width in tiles.
        height: Map height in tiles.
        name: Level name.

    Returns:
        Fresh :class:`EditorState` with all-dirt terrain and no machines.
    """
    block_map = np.full((height, width), int(BlockType.DIRT), dtype=np.int32)
    inv_shape = (height, width, MAX_MACHINE_INVENTORY_SLOTS)
    return EditorState(
        name=name,
        map_width=width,
        map_height=height,
        block_map=block_map,
        block_resources=np.zeros((height, width), dtype=np.int32),
        machine_types=np.full((height, width), int(MachineType.NONE), dtype=np.int32),
        machine_directions=np.zeros((height, width), dtype=np.int32),
        machine_inventory_items=np.zeros(inv_shape, dtype=np.int32),
        machine_inventory_counts=np.zeros(inv_shape, dtype=np.int32),
        machine_selected_recipe=np.zeros((height, width), dtype=np.int32),
    )


def editor_state_from_level(level: Level) -> EditorState:
    """Convert a loaded :class:`Level` into a mutable :class:`EditorState`.

    Missing optional arrays are filled with sensible defaults.

    Args:
        level: Source level.

    Returns:
        :class:`EditorState` mirroring the level data.
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
            int(MachineType.NONE),
            dtype=np.int32,
        )
    )
    directions = (
        level.machine_directions.copy()
        if level.machine_directions is not None
        else np.zeros((level.map_height, level.map_width), dtype=np.int32)
    )
    inv_shape = (level.map_height, level.map_width, MAX_MACHINE_INVENTORY_SLOTS)
    inv_items = (
        level.machine_inventory_items.copy()
        if level.machine_inventory_items is not None
        else np.zeros(inv_shape, dtype=np.int32)
    )
    inv_counts = (
        level.machine_inventory_counts.copy()
        if level.machine_inventory_counts is not None
        else np.zeros(inv_shape, dtype=np.int32)
    )
    recipe = (
        level.machine_selected_recipe.copy()
        if level.machine_selected_recipe is not None
        else np.zeros((level.map_height, level.map_width), dtype=np.int32)
    )
    return EditorState(
        name=level.name,
        map_width=level.map_width,
        map_height=level.map_height,
        block_map=level.block_map.copy(),
        block_resources=resources.astype(np.int32),
        machine_types=machines.astype(np.int32),
        machine_directions=directions.astype(np.int32),
        machine_inventory_items=inv_items.astype(np.int32),
        machine_inventory_counts=inv_counts.astype(np.int32),
        machine_selected_recipe=recipe.astype(np.int32),
        player_inventory=level.player_inventory,
        player_positions=(
            {i: (p[0], p[1]) for i, p in enumerate(level.player_positions)}
            if level.player_positions
            else {}
        ),
        biter_positions=(
            [(p[0], p[1]) for p in level.biter_positions]
            if level.biter_positions
            else []
        ),
    )


def editor_state_to_level(state: EditorState) -> Level:
    """Convert the editor state back to a :class:`Level` for saving or play.

    All-zero optional arrays are stored as ``None`` to keep the JSON
    compact.

    Args:
        state: Current editor state.

    Returns:
        A :class:`Level` ready for serialization or ``build_state``.
    """
    resources: np.ndarray | None = state.block_resources.copy()
    if np.all(resources == 0):
        resources = None

    machines: np.ndarray | None = state.machine_types.copy()
    if np.all(machines == int(MachineType.NONE)):
        machines = None

    directions: np.ndarray | None = state.machine_directions.copy()
    if np.all(directions == 0):
        directions = None

    inv_items: np.ndarray | None = state.machine_inventory_items.copy()
    if np.all(inv_items == 0):
        inv_items = None

    inv_counts: np.ndarray | None = state.machine_inventory_counts.copy()
    if np.all(inv_counts == 0):
        inv_counts = None

    recipe: np.ndarray | None = state.machine_selected_recipe.copy()
    if np.all(recipe == 0):
        recipe = None

    pp: list[tuple[int, int]] | None = None
    if state.player_positions:
        pp = [
            state.player_positions[k]
            for k in sorted(state.player_positions)
        ]
    bp: list[tuple[int, int]] | None = (
        list(state.biter_positions) if state.biter_positions else None
    )

    return Level(
        name=state.name,
        map_width=state.map_width,
        map_height=state.map_height,
        block_map=state.block_map.copy(),
        block_resources=resources,
        machine_types=machines,
        machine_directions=directions,
        machine_inventory_items=inv_items,
        machine_inventory_counts=inv_counts,
        machine_selected_recipe=recipe,
        player_inventory=state.player_inventory,
        player_positions=pp,
        biter_positions=bp,
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

    Args:
        state: Editor state (mutated in place).
        x: Tile column.
        y: Tile row.
        block: ``BlockType`` integer value.
        brush: Active resource brush.
        rng: Numpy random generator for range-mode sampling.
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

    Args:
        state: Editor state (mutated in place).
        x: Tile column.
        y: Tile row.
        machine: ``MachineType`` integer value.
        direction: ``Action`` direction value for the machine facing.
    """
    if not (0 <= x < state.map_width and 0 <= y < state.map_height):
        return
    state.machine_types[y, x] = machine
    state.machine_directions[y, x] = direction
    state.machine_inventory_items[y, x] = 0
    state.machine_inventory_counts[y, x] = 0
    state.machine_selected_recipe[y, x] = 0
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

    Args:
        state: Editor state (mutated in place).
        x0: Left column of the rectangle.
        y0: Top row.
        x1: Right column (inclusive).
        y1: Bottom row (inclusive).
        block: ``BlockType`` integer value.
        brush: Active resource brush.
        rng: Numpy random generator for range-mode sampling.
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

    Args:
        state: Editor state (mutated in place).
        x: Tile column.
        y: Tile row.
    """
    if not (0 <= x < state.map_width and 0 <= y < state.map_height):
        return
    state.block_map[y, x] = int(BlockType.DIRT)
    state.block_resources[y, x] = 0
    state.dirty = True


def erase_machine(state: EditorState, x: int, y: int) -> None:
    """Remove a machine from a tile, leaving the terrain untouched.

    Args:
        state: Editor state (mutated in place).
        x: Tile column.
        y: Tile row.
    """
    if not (0 <= x < state.map_width and 0 <= y < state.map_height):
        return
    state.machine_types[y, x] = int(MachineType.NONE)
    state.machine_directions[y, x] = 0
    state.machine_inventory_items[y, x] = 0
    state.machine_inventory_counts[y, x] = 0
    state.machine_selected_recipe[y, x] = 0
    state.dirty = True


def erase_tile(state: EditorState, x: int, y: int) -> None:
    """Reset a tile to dirt and remove any machine.

    Convenience function that clears both layers.

    Args:
        state: Editor state (mutated in place).
        x: Tile column.
        y: Tile row.
    """
    erase_block(state, x, y)
    erase_machine(state, x, y)


def add_column(state: EditorState) -> None:
    """Append one dirt column to the right edge of the map.

    All four arrays are extended in place and ``map_width`` is
    incremented.

    Args:
        state: Editor state (mutated in place).
    """
    h = state.map_height
    s = MAX_MACHINE_INVENTORY_SLOTS
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
            np.full((h, 1), int(MachineType.NONE), dtype=np.int32),
        ],
        axis=1,
    )
    state.machine_directions = np.concatenate(
        [state.machine_directions, np.zeros((h, 1), dtype=np.int32)],
        axis=1,
    )
    state.machine_inventory_items = np.concatenate(
        [state.machine_inventory_items, np.zeros((h, 1, s), dtype=np.int32)],
        axis=1,
    )
    state.machine_inventory_counts = np.concatenate(
        [state.machine_inventory_counts, np.zeros((h, 1, s), dtype=np.int32)],
        axis=1,
    )
    state.machine_selected_recipe = np.concatenate(
        [state.machine_selected_recipe, np.zeros((h, 1), dtype=np.int32)],
        axis=1,
    )
    state.map_width += 1
    state.dirty = True


def remove_column(state: EditorState) -> None:
    """Remove the rightmost column from the map.

    No-op if the map is 1 tile wide.

    Args:
        state: Editor state (mutated in place).
    """
    if state.map_width <= 1:
        return
    state.block_map = state.block_map[:, :-1]
    state.block_resources = state.block_resources[:, :-1]
    state.machine_types = state.machine_types[:, :-1]
    state.machine_directions = state.machine_directions[:, :-1]
    state.machine_inventory_items = state.machine_inventory_items[:, :-1, :]
    state.machine_inventory_counts = state.machine_inventory_counts[:, :-1, :]
    state.machine_selected_recipe = state.machine_selected_recipe[:, :-1]
    state.map_width -= 1
    _clip_entities(state)
    state.dirty = True


def add_row(state: EditorState) -> None:
    """Append one dirt row to the bottom edge of the map.

    All four arrays are extended in place and ``map_height`` is
    incremented.

    Args:
        state: Editor state (mutated in place).
    """
    w = state.map_width
    s = MAX_MACHINE_INVENTORY_SLOTS
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
            np.full((1, w), int(MachineType.NONE), dtype=np.int32),
        ],
        axis=0,
    )
    state.machine_directions = np.concatenate(
        [state.machine_directions, np.zeros((1, w), dtype=np.int32)],
        axis=0,
    )
    state.machine_inventory_items = np.concatenate(
        [state.machine_inventory_items, np.zeros((1, w, s), dtype=np.int32)],
        axis=0,
    )
    state.machine_inventory_counts = np.concatenate(
        [state.machine_inventory_counts, np.zeros((1, w, s), dtype=np.int32)],
        axis=0,
    )
    state.machine_selected_recipe = np.concatenate(
        [state.machine_selected_recipe, np.zeros((1, w), dtype=np.int32)],
        axis=0,
    )
    state.map_height += 1
    state.dirty = True


def remove_row(state: EditorState) -> None:
    """Remove the bottom row from the map.

    No-op if the map is 1 tile tall.

    Args:
        state: Editor state (mutated in place).
    """
    if state.map_height <= 1:
        return
    state.block_map = state.block_map[:-1, :]
    state.block_resources = state.block_resources[:-1, :]
    state.machine_types = state.machine_types[:-1, :]
    state.machine_directions = state.machine_directions[:-1, :]
    state.machine_inventory_items = state.machine_inventory_items[:-1, :, :]
    state.machine_inventory_counts = state.machine_inventory_counts[:-1, :, :]
    state.machine_selected_recipe = state.machine_selected_recipe[:-1, :]
    state.map_height -= 1
    _clip_entities(state)
    state.dirty = True


# ---------------------------------------------------------------------------
# Entity placement
# ---------------------------------------------------------------------------


def _clip_entities(state: EditorState) -> None:
    """Remove entities that fall outside the current map bounds."""
    state.player_positions = {
        k: v
        for k, v in state.player_positions.items()
        if 0 <= v[0] < state.map_width and 0 <= v[1] < state.map_height
    }
    state.biter_positions = [
        (x, y)
        for x, y in state.biter_positions
        if 0 <= x < state.map_width and 0 <= y < state.map_height
    ]


def set_player_position(
    state: EditorState, player_idx: int, x: int, y: int
) -> None:
    """Place or move a player start position.

    If player *player_idx* already has a position it is moved.

    Args:
        state: Editor state (mutated in place).
        player_idx: Player index (0-7).
        x: Tile column.
        y: Tile row.
    """
    if not (0 <= x < state.map_width and 0 <= y < state.map_height):
        return
    state.player_positions[player_idx] = (x, y)
    state.dirty = True


def remove_player_at(state: EditorState, x: int, y: int) -> None:
    """Remove any player start at the given tile.

    Args:
        state: Editor state (mutated in place).
        x: Tile column.
        y: Tile row.
    """
    to_remove = [k for k, v in state.player_positions.items() if v == (x, y)]
    for k in to_remove:
        del state.player_positions[k]
    if to_remove:
        state.dirty = True


def add_biter(state: EditorState, x: int, y: int) -> None:
    """Add a biter at the given tile.

    Args:
        state: Editor state (mutated in place).
        x: Tile column.
        y: Tile row.
    """
    if not (0 <= x < state.map_width and 0 <= y < state.map_height):
        return
    state.biter_positions.append((x, y))
    state.dirty = True


def remove_biters_at(state: EditorState, x: int, y: int) -> None:
    """Remove all biters at the given tile.

    Args:
        state: Editor state (mutated in place).
        x: Tile column.
        y: Tile row.
    """
    before = len(state.biter_positions)
    state.biter_positions = [
        (bx, by)
        for bx, by in state.biter_positions
        if (bx, by) != (x, y)
    ]
    if len(state.biter_positions) < before:
        state.dirty = True


def erase_entity(state: EditorState, x: int, y: int) -> None:
    """Remove all entities (players and biters) at the given tile.

    Args:
        state: Editor state (mutated in place).
        x: Tile column.
        y: Tile row.
    """
    remove_player_at(state, x, y)
    remove_biters_at(state, x, y)
