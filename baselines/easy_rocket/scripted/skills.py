"""Atomic action-emitting helpers for the scripted agent.

Each function returns one :class:`Action` per call. The phase
driver composes them (navigate → face → interact) across multiple
ticks; skills here are deliberately stateless so the phase logic
stays explicit about which sub-step it's in.

Conventions:

- Tiles are addressed as ``(x, y)`` where ``x`` is the column.
- "Adjacent" means manhattan distance 1 (orthogonal neighbours).
- "In front" of the player means the tile one step along the
  player's current facing direction; ``MINE``, ``PICKUP``, and
  ``PLACE_*`` all act on that tile.
"""

from __future__ import annotations

from collections import deque

from baselines.easy_rocket.scripted.state_reader import player_pos, tile_free
from factoriax.constants import Action, Direction, ItemType, MachineType
from factoriax.state import EnvState

# Per-direction unit vector in (dx, dy). Direction is 1..4; index 0
# is reserved for "no direction".
DIR_OFFSETS: dict[int, tuple[int, int]] = {
    int(Direction.LEFT): (-1, 0),
    int(Direction.RIGHT): (1, 0),
    int(Direction.UP): (0, -1),
    int(Direction.DOWN): (0, 1),
}

# Movement actions per direction (player walks one tile that way).
DIR_TO_MOVE_ACTION: dict[int, int] = {
    int(Direction.LEFT): int(Action.LEFT),
    int(Direction.RIGHT): int(Action.RIGHT),
    int(Direction.UP): int(Action.UP),
    int(Direction.DOWN): int(Action.DOWN),
}

# Facing actions per direction (player rotates without moving).
DIR_TO_FACE_ACTION: dict[int, int] = {
    int(Direction.LEFT): int(Action.FACE_LEFT),
    int(Direction.RIGHT): int(Action.FACE_RIGHT),
    int(Direction.UP): int(Action.FACE_UP),
    int(Direction.DOWN): int(Action.FACE_DOWN),
}

# Hand-craftable items and their CRAFT actions. Items that can only
# be produced by an assembler or furnace (HULL, ENGINE_UNIT, AVIONICS,
# ROCKET_CORE, REFRACTORY) deliberately have no entry here.
ITEM_TO_CRAFT_ACTION: dict[int, int] = {
    int(ItemType.IRON_PLATE): int(Action.CRAFT_IRON_PLATE),
    int(ItemType.COPPER_PLATE): int(Action.CRAFT_COPPER_PLATE),
    int(ItemType.TIN_PLATE): int(Action.CRAFT_TIN_PLATE),
    int(ItemType.WAFER): int(Action.CRAFT_WAFER),
    int(ItemType.FRAME): int(Action.CRAFT_FRAME),
    int(ItemType.CIRCUIT): int(Action.CRAFT_CIRCUIT),
    int(ItemType.WIRE): int(Action.CRAFT_WIRE),
    int(ItemType.MOTOR): int(Action.CRAFT_MOTOR),
    int(ItemType.SENSOR): int(Action.CRAFT_SENSOR),
    int(ItemType.CONVEYOR_BELT): int(Action.CRAFT_BELT),
    int(ItemType.MINER): int(Action.CRAFT_MINER),
    int(ItemType.ASSEMBLER): int(Action.CRAFT_ASSEMBLER),
    int(ItemType.PALLET): int(Action.CRAFT_PALLET),
    int(ItemType.ARM): int(Action.CRAFT_ARM),
    int(ItemType.FURNACE): int(Action.CRAFT_FURNACE),
    int(ItemType.BASIC_SCIENCE_PACK): int(Action.CRAFT_BASIC_SCIENCE),
    int(ItemType.ADVANCED_SCIENCE_PACK): int(Action.CRAFT_ADV_SCIENCE),
    int(ItemType.ROCKET): int(Action.CRAFT_ROCKET),
    int(ItemType.SCIENCE_LAB): int(Action.CRAFT_SCIENCE_LAB),
    int(ItemType.SPLITTER): int(Action.CRAFT_SPLITTER),
    int(ItemType.CROSSING): int(Action.CRAFT_CROSSING),
}

# Placeable items (machines) and their PLACE actions.
ITEM_TO_PLACE_ACTION: dict[int, int] = {
    int(ItemType.MINER): int(Action.PLACE_MINER),
    int(ItemType.PALLET): int(Action.PLACE_PALLET),
    int(ItemType.CONVEYOR_BELT): int(Action.PLACE_BELT),
    int(ItemType.ASSEMBLER): int(Action.PLACE_ASSEMBLER),
    int(ItemType.ARM): int(Action.PLACE_ARM),
    int(ItemType.ROCKET): int(Action.PLACE_ROCKET),
    int(ItemType.FURNACE): int(Action.PLACE_FURNACE),
    int(ItemType.SCIENCE_LAB): int(Action.PLACE_SCIENCE_LAB),
    int(ItemType.SPLITTER): int(Action.PLACE_SPLITTER),
    int(ItemType.CROSSING): int(Action.PLACE_CROSSING),
}

# Each machine type's item form, for "pick the right CRAFT/PLACE
# action given a MachineType the planner specified".
MACHINE_TO_ITEM: dict[int, int] = {
    int(MachineType.MINER): int(ItemType.MINER),
    int(MachineType.PALLET): int(ItemType.PALLET),
    int(MachineType.ASSEMBLER): int(ItemType.ASSEMBLER),
    int(MachineType.CONVEYOR_BELT): int(ItemType.CONVEYOR_BELT),
    int(MachineType.ARM): int(ItemType.ARM),
    int(MachineType.ROCKET): int(ItemType.ROCKET),
    int(MachineType.FURNACE): int(ItemType.FURNACE),
    int(MachineType.SCIENCE_LAB): int(ItemType.SCIENCE_LAB),
    int(MachineType.SPLITTER): int(ItemType.SPLITTER),
    int(MachineType.CROSSING): int(ItemType.CROSSING),
}


def is_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True when ``a`` and ``b`` differ by one tile along one axis."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def direction_to(from_pos: tuple[int, int], to_pos: tuple[int, int]) -> int | None:
    """Return the ``Direction`` from ``from_pos`` toward ``to_pos``.

    Only defined for orthogonal neighbours. Returns ``None`` when
    ``from_pos == to_pos`` or the two tiles are not adjacent.
    """
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    if abs(dx) + abs(dy) != 1:
        return None
    if dx == 1:
        return int(Direction.RIGHT)
    if dx == -1:
        return int(Direction.LEFT)
    if dy == 1:
        return int(Direction.DOWN)
    return int(Direction.UP)


def stand_tile_for(target: tuple[int, int], facing: int) -> tuple[int, int]:
    """Return the tile the player must stand on to act on ``target``
    while facing ``facing``.

    Action targets the tile one step ahead of the player's facing, so
    ``stand_tile = target - unit_vec(facing)``. Used by the place /
    pickup / mine sub-steps to compute where to navigate before
    facing and emitting the action.
    """
    dx, dy = DIR_OFFSETS[facing]
    return (target[0] - dx, target[1] - dy)


def face_action(direction: int) -> int:
    """Return the FACE_* action that rotates the player to ``direction``."""
    return DIR_TO_FACE_ACTION[direction]


def move_action(direction: int) -> int:
    """Return the move action that walks one tile in ``direction``."""
    return DIR_TO_MOVE_ACTION[direction]


def craft_action(item: int) -> int:
    """Return the ``CRAFT_*`` action that produces one ``item``.

    Raises:
        KeyError: When ``item`` has no hand-craft action (i.e. it can
            only come from an assembler / furnace recipe).
    """
    return ITEM_TO_CRAFT_ACTION[item]


def place_action(item: int) -> int:
    """Return the ``PLACE_*`` action for placing ``item`` as a machine.

    Raises:
        KeyError: When ``item`` is not placeable.
    """
    return ITEM_TO_PLACE_ACTION[item]


def step_toward_adjacent(state: EnvState, tx: int, ty: int, player: int = 0) -> int:
    """Emit one movement action that brings the player closer to a
    tile orthogonally adjacent to ``(tx, ty)``.

    Uses BFS over walkable tiles. Returns ``Action.NOOP`` if the
    player is already adjacent or no path exists. The target tile
    itself need not be walkable — only the goal set of tiles
    around it.
    """
    start = player_pos(state, player)
    if is_adjacent(start, (tx, ty)):
        return int(Action.NOOP)

    goal_set = {(tx + dx, ty + dy) for dx, dy in DIR_OFFSETS.values()}
    # BFS over free tiles to any tile in goal_set.
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    frontier: deque[tuple[int, int]] = deque([start])
    found: tuple[int, int] | None = None
    while frontier:
        cur = frontier.popleft()
        if cur in goal_set:
            found = cur
            break
        for dx, dy in DIR_OFFSETS.values():
            nx, ny = cur[0] + dx, cur[1] + dy
            if (nx, ny) in came_from:
                continue
            if not tile_free(state, nx, ny):
                continue
            came_from[(nx, ny)] = cur
            frontier.append((nx, ny))
    if found is None:
        return int(Action.NOOP)

    # Walk back along came_from to find the first step from start.
    step = found
    while came_from[step] != start:
        prev = came_from[step]
        assert prev is not None
        step = prev
    direction = direction_to(start, step)
    if direction is None:
        return int(Action.NOOP)
    return move_action(direction)
