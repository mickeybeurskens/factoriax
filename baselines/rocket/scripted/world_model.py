"""Observation-derived world view for the scripted rocket agent.

The scripted agent is map-agnostic: nothing it does is allowed to
reference the initial level layout directly. Instead it consumes the
``global_array`` observation each tick, reconstructs per-tile grids,
and runs simple graph searches on the walkable mask.

The data model is a :class:`WorldView` — an immutable snapshot of the
environment as it appears in one observation. Call
:func:`decode_observation` on a fresh obs to produce a new view.

Why rebuild every tick rather than mutate: the observation is the
authoritative source, and tracking a partial in-memory world model
risks drifting out of sync with the engine (e.g. machines placed by
other players, ore patches depleted by nearby miners). Decoding is
cheap — a handful of numpy reshapes — and keeps the agent's decisions
grounded in whatever the env most recently reported.
"""

from __future__ import annotations

import dataclasses
from collections import deque

import numpy as np

from factoriax.constants import (
    BLOCK_MAX_RESOURCES,
    BLOCK_TO_ITEM,
    NUM_ITEM_TYPES,
    PLAYER_MAX_STACK,
    Action,
    BlockType,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.observations import NUM_PLAYER_SCALARS, NUM_SPATIAL_CHANNELS
from factoriax.recipes import NUM_RECIPES

# Normalization constants used by ``factoriax.observations.global_array``.
# Duplicated here so the decoder is self-contained and doesn't reach into
# module-private names.
_MAP_NORM: float = float(max(BlockType))
_MACHINE_NORM: float = float(max(MachineType))
_DIR_NORM: float = 4.0
_BUF_COUNT_NORM: float = 64.0
# Slot counts on the 10-channel obs are normalized by a larger constant
# than ``_BUF_COUNT_NORM`` so assembler-input and pallet storage (which
# can exceed 64) round-trip cleanly.
_SLOT_COUNT_NORM: float = 1024.0
# Player inventory is per-item normalized by PLAYER_MAX_STACK[i]; convert
# once to a plain numpy array.
_PLAYER_MAX_STACK_NP: np.ndarray = np.asarray(PLAYER_MAX_STACK, dtype=np.float32)

# Action enum offsets for compound actions. Keep as bare ints so tests
# don't need to re-import the enum.
_ITEM_TO_DEPOSIT: dict[int, int] = {
    int(ItemType.COAL): int(Action.DEPOSIT_COAL),
    int(ItemType.IRON_ORE): int(Action.DEPOSIT_IRON_ORE),
    int(ItemType.COPPER_ORE): int(Action.DEPOSIT_COPPER_ORE),
    int(ItemType.TIN_ORE): int(Action.DEPOSIT_TIN_ORE),
    int(ItemType.SILICON): int(Action.DEPOSIT_SILICON),
    int(ItemType.IRON_PLATE): int(Action.DEPOSIT_IRON_PLATE),
    int(ItemType.COPPER_PLATE): int(Action.DEPOSIT_COPPER_PLATE),
    int(ItemType.TIN_PLATE): int(Action.DEPOSIT_TIN_PLATE),
    int(ItemType.WAFER): int(Action.DEPOSIT_WAFER),
    int(ItemType.FRAME): int(Action.DEPOSIT_FRAME),
    int(ItemType.CIRCUIT): int(Action.DEPOSIT_CIRCUIT),
    int(ItemType.WIRE): int(Action.DEPOSIT_WIRE),
    int(ItemType.MOTOR): int(Action.DEPOSIT_MOTOR),
    int(ItemType.SENSOR): int(Action.DEPOSIT_SENSOR),
    int(ItemType.CONVEYOR_BELT): int(Action.DEPOSIT_BELT),
    int(ItemType.MINER): int(Action.DEPOSIT_MINER),
    int(ItemType.ASSEMBLER): int(Action.DEPOSIT_ASSEMBLER),
    int(ItemType.PALLET): int(Action.DEPOSIT_PALLET),
    int(ItemType.ARM): int(Action.DEPOSIT_ARM),
    int(ItemType.BASIC_SCIENCE_PACK): int(Action.DEPOSIT_BASIC_SCIENCE),
    int(ItemType.ADVANCED_SCIENCE_PACK): int(Action.DEPOSIT_ADV_SCIENCE),
    int(ItemType.ROCKET): int(Action.DEPOSIT_ROCKET),
    int(ItemType.FURNACE): int(Action.DEPOSIT_FURNACE),
    int(ItemType.REFRACTORY): int(Action.DEPOSIT_REFRACTORY),
    int(ItemType.HULL): int(Action.DEPOSIT_HULL),
    int(ItemType.ENGINE_UNIT): int(Action.DEPOSIT_ENGINE_UNIT),
    int(ItemType.AVIONICS): int(Action.DEPOSIT_AVIONICS),
    int(ItemType.ROCKET_CORE): int(Action.DEPOSIT_ROCKET_CORE),
    # ItemType.LIMESTONE = 30 was inserted between SCIENCE_LAB (29)
    # and SPLITTER (31). The Action enum's DEPOSIT_* slots weren't
    # renumbered, so the dispatch math
    # ``action - DEPOSIT_BASE + ItemType.COAL`` for action 71
    # (named DEPOSIT_SPLITTER) actually deposits ItemType 30 =
    # LIMESTONE. The agent only ever *places* SPLITTERs (never
    # deposits them), so the SPLITTER deposit slot is otherwise
    # unused — we redirect ItemType.LIMESTONE through it so
    # ProduceInFurnace(REFRACTORY) (LIMESTONE + COAL recipe) works.
    int(ItemType.LIMESTONE): int(Action.DEPOSIT_SPLITTER),
}
_MACHINE_TO_PLACE: dict[int, int] = {
    int(MachineType.MINER): int(Action.PLACE_MINER),
    int(MachineType.PALLET): int(Action.PLACE_PALLET),
    int(MachineType.CONVEYOR_BELT): int(Action.PLACE_BELT),
    int(MachineType.ASSEMBLER): int(Action.PLACE_ASSEMBLER),
    int(MachineType.ARM): int(Action.PLACE_ARM),
    int(MachineType.ROCKET): int(Action.PLACE_ROCKET),
    int(MachineType.SPLITTER): int(Action.PLACE_SPLITTER),
    int(MachineType.CROSSING): int(Action.PLACE_CROSSING),
    int(MachineType.FURNACE): int(Action.PLACE_FURNACE),
}
# Reverse mapping for ore-block → item type (e.g. BlockType.IRON → ItemType.IRON_ORE).
_BLOCK_ITEM: dict[int, int] = {int(k): int(v) for k, v in BLOCK_TO_ITEM.items()}
# Compass direction → (dx, dy) offset matching factoriax.constants.DIRECTIONS.
_DIR_OFFSETS: dict[int, tuple[int, int]] = {
    int(Direction.LEFT): (-1, 0),
    int(Direction.RIGHT): (1, 0),
    int(Direction.UP): (0, -1),
    int(Direction.DOWN): (0, 1),
}
# The player moves one tile in the given direction using a single Action.
_DIR_TO_MOVE_ACTION: dict[int, int] = {
    int(Direction.LEFT): int(Action.LEFT),
    int(Direction.RIGHT): int(Action.RIGHT),
    int(Direction.UP): int(Action.UP),
    int(Direction.DOWN): int(Action.DOWN),
}
_DIR_TO_FACE_ACTION: dict[int, int] = {
    int(Direction.LEFT): int(Action.FACE_LEFT),
    int(Direction.RIGHT): int(Action.FACE_RIGHT),
    int(Direction.UP): int(Action.FACE_UP),
    int(Direction.DOWN): int(Action.FACE_DOWN),
}


# ---------------------------------------------------------------------------
# Parsed observation containers
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PlayerScalars:
    """De-normalized player-scalar fields from the observation vector."""

    pos_x: int
    pos_y: int
    direction: int
    timestep: int
    facing_machine_type: int
    facing_buffer_type: int
    facing_buffer_count: int
    inventory: np.ndarray  # (NUM_ITEM_TYPES,) int32, counts

    @property
    def pos(self) -> tuple[int, int]:
        return self.pos_x, self.pos_y

    def held(self, item: int | ItemType) -> int:
        """Count of *item* in the player's inventory."""
        return int(self.inventory[int(item)])


@dataclasses.dataclass(frozen=True)
class WorldView:
    """Immutable snapshot of the world derived from one observation.

    The machine slot layout is uniform across machine types:

    - Slot 0: first input (assembler/furnace ingredient A). Zero elsewhere.
    - Slot 1: second input (assembler/furnace ingredient B). Zero elsewhere.
    - Slot 2: output — assembler/furnace output or the ``ent_buf`` of
      buffer machines (miner/pallet/belt).

    ``buffer_type`` is kept as an alias for ``slot2_type`` so existing
    callers that only care about the output-facing slot keep working.

    Attributes:
        block_type: ``(H, W)`` int32, values from :class:`BlockType`.
        machine_type: ``(H, W)`` int32, values from :class:`MachineType`.
            ``NONE`` where no machine is placed.
        block_resources: ``(H, W)`` int32, ore remaining on each tile.
        slot0_type/slot0_count: ``(H, W)`` int32, input-A slot contents.
        slot1_type/slot1_count: ``(H, W)`` int32, input-B slot contents.
        slot2_type/slot2_count: ``(H, W)`` int32, output/buffer slot.
        buffer_type: Alias of ``slot2_type`` — item in the output-facing
            slot on that tile, or 0.
        machine_direction: ``(H, W)`` int32, compass direction each
            placed machine is facing. 0 on empty tiles.
        player: Decoded player scalars.
        walkable: ``(H, W)`` bool, True where the player can stand.
    """

    block_type: np.ndarray
    machine_type: np.ndarray
    block_resources: np.ndarray
    slot0_type: np.ndarray
    slot0_count: np.ndarray
    slot1_type: np.ndarray
    slot1_count: np.ndarray
    slot2_type: np.ndarray
    slot2_count: np.ndarray
    machine_direction: np.ndarray
    buffer_type: np.ndarray
    player: PlayerScalars
    walkable: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.block_type.shape

    # ------------------------------------------------------------------
    # Resource queries
    # ------------------------------------------------------------------

    def ore_tiles(self, item_type: int | ItemType) -> list[tuple[int, int]]:
        """List of ``(x, y)`` tile coords for the block yielding *item_type*.

        Only tiles with positive ``block_resources`` are returned.
        """
        item = int(item_type)
        # Inverse lookup: find the block type whose BLOCK_TO_ITEM == item.
        block_matches = [b for b, it in _BLOCK_ITEM.items() if it == item]
        if not block_matches:
            return []
        mask = np.zeros_like(self.block_type, dtype=bool)
        for b in block_matches:
            mask |= self.block_type == b
        mask &= self.block_resources > 0
        ys, xs = np.nonzero(mask)
        return list(zip(xs.tolist(), ys.tolist(), strict=True))

    def tiles_with_machine(
        self,
        machine_type: int | MachineType,
    ) -> list[tuple[int, int]]:
        """List of ``(x, y)`` tiles holding a machine of *machine_type*."""
        mask = self.machine_type == int(machine_type)
        ys, xs = np.nonzero(mask)
        return list(zip(xs.tolist(), ys.tolist(), strict=True))

    def buffered_tiles(
        self,
        item_type: int | ItemType,
    ) -> list[tuple[int, int]]:
        """List of ``(x, y)`` tiles where a machine holds *item_type*."""
        mask = self.buffer_type == int(item_type)
        ys, xs = np.nonzero(mask)
        return list(zip(xs.tolist(), ys.tolist(), strict=True))

    def total_machines(self) -> int:
        """Count placed machines of any type on the map."""
        return int((self.machine_type != int(MachineType.NONE)).sum())

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------

    def is_in_bounds(self, tile: tuple[int, int]) -> bool:
        x, y = tile
        h, w = self.shape
        return 0 <= x < w and 0 <= y < h

    def adjacent_tiles(
        self,
        tile: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """4-connected in-bounds neighbours of *tile* (no walkability check)."""
        x, y = tile
        h, w = self.shape
        candidates = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [(cx, cy) for cx, cy in candidates if 0 <= cx < w and 0 <= cy < h]

    def plan_path(
        self,
        goal: tuple[int, int],
        *,
        mode: str = "adjacent_or_on",
    ) -> list[int] | None:
        """BFS from the player to *goal*, returning movement actions.

        Three modes control where the player stops:

        - ``"adjacent_or_on"`` — end adjacent OR on the goal. Used
          when either is fine (e.g. approaching an occupied machine
          to withdraw items; walkable tile or blocked tile both OK).
        - ``"adjacent_only"`` — never step onto the goal, always end
          adjacent. Used when the goal tile needs to remain empty for
          a subsequent PLACE/DEPOSIT action.
        - ``"exactly_on"`` — target must be walkable; end on it. Used
          for MINE, which reads the block under the player's feet.

        Args:
            goal: Target ``(x, y)`` in tile coordinates.
            mode: One of the strings above.

        Returns:
            List of :class:`Action` movement ints. An empty list means
            the player is already at an acceptable stop tile. ``None``
            means no path exists.
        """
        start = self.player.pos
        if not self.is_in_bounds(goal):
            return None
        if mode not in ("adjacent_or_on", "adjacent_only", "exactly_on"):
            raise ValueError(f"unknown plan_path mode: {mode!r}")

        if mode == "exactly_on":
            if not self.walkable[goal[1], goal[0]]:
                return None
            target_set = {goal}
            walkable = self.walkable
        else:
            adj = {
                (tx, ty)
                for (tx, ty) in self.adjacent_tiles(goal)
                if self.walkable[ty, tx]
            }
            if mode == "adjacent_or_on":
                if self.walkable[goal[1], goal[0]]:
                    adj.add(goal)
                walkable = self.walkable
            else:  # "adjacent_only"
                # Block the goal tile so the BFS can't end on it.
                walkable = self.walkable.copy()
                walkable[goal[1], goal[0]] = False
            target_set = adj

        if not target_set:
            return None
        if start in target_set:
            return []

        return _bfs(walkable, start, target_set)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


def _scalar_sections() -> dict[str, slice]:
    """Return the scalar-field layout used by ``_player_scalars``.

    Sections in order: fixed (pos_x, pos_y, dir, timestep, affords),
    facing (9), player inventory (NUM_ITEM_TYPES).
    """
    i = 0
    sections: dict[str, slice] = {}
    sections["pos_x"] = slice(i, i + 1)
    i += 1
    sections["pos_y"] = slice(i, i + 1)
    i += 1
    sections["direction"] = slice(i, i + 1)
    i += 1
    sections["timestep"] = slice(i, i + 1)
    i += 1
    sections["afford"] = slice(i, i + NUM_RECIPES)
    i += NUM_RECIPES
    sections["facing"] = slice(i, i + 9)
    i += 9  # 9 facing fields
    sections["inventory"] = slice(i, i + NUM_ITEM_TYPES)
    return sections


_SCALAR_SECTIONS: dict[str, slice] = _scalar_sections()


def decode_observation(
    obs: np.ndarray,
    map_height: int,
    map_width: int,
    max_timesteps: int,
) -> WorldView:
    """Parse a ``global_array`` observation into a :class:`WorldView`.

    Args:
        obs: 1-D float32 array as produced by
            :func:`factoriax.observations.global_array` for a single
            player.
        map_height: Number of tile rows.
        map_width: Number of tile columns.
        max_timesteps: Value of ``EnvParams.max_timesteps``; needed to
            de-normalize the timestep scalar.

    Returns:
        A :class:`WorldView` with de-normalized integer grids and the
        player scalars needed for planning.
    """
    obs = np.asarray(obs, dtype=np.float32)
    expected_spatial = NUM_SPATIAL_CHANNELS * map_height * map_width
    expected_total = expected_spatial + NUM_PLAYER_SCALARS
    if obs.shape != (expected_total,):
        raise ValueError(
            f"Expected obs shape ({expected_total},) for map "
            f"({map_height}, {map_width}); got {obs.shape}.",
        )

    spatial = obs[:expected_spatial].reshape(
        NUM_SPATIAL_CHANNELS,
        map_height,
        map_width,
    )
    # Re-scale with np.round to undo the float division in global_array.
    block_type = np.round(spatial[0] * _MAP_NORM).astype(np.int32)
    machine_type = np.round(spatial[1] * _MACHINE_NORM).astype(np.int32)
    block_resources = np.round(spatial[2] * float(BLOCK_MAX_RESOURCES)).astype(
        np.int32,
    )
    item_norm = float(NUM_ITEM_TYPES)
    slot0_type = np.round(spatial[3] * item_norm).astype(np.int32)
    slot0_count = np.round(spatial[4] * _SLOT_COUNT_NORM).astype(np.int32)
    slot1_type = np.round(spatial[5] * item_norm).astype(np.int32)
    slot1_count = np.round(spatial[6] * _SLOT_COUNT_NORM).astype(np.int32)
    slot2_type = np.round(spatial[7] * item_norm).astype(np.int32)
    slot2_count = np.round(spatial[8] * _SLOT_COUNT_NORM).astype(np.int32)
    machine_direction = np.round(spatial[9] * _DIR_NORM).astype(np.int32)
    # Legacy alias: callers that check the "buffer" of a machine tile
    # read the output-facing slot.
    buffer_type = slot2_type

    scalars = obs[expected_spatial:]

    pos_x = int(round(float(scalars[_SCALAR_SECTIONS["pos_x"]][0]) * map_width))
    pos_y = int(round(float(scalars[_SCALAR_SECTIONS["pos_y"]][0]) * map_height))
    direction = int(round(float(scalars[_SCALAR_SECTIONS["direction"]][0]) * _DIR_NORM))
    timestep = int(
        round(float(scalars[_SCALAR_SECTIONS["timestep"]][0]) * max_timesteps),
    )

    facing = scalars[_SCALAR_SECTIONS["facing"]]
    facing_machine_type = int(round(float(facing[0]) * _MACHINE_NORM))
    facing_buffer_type = int(round(float(facing[1]) * float(NUM_ITEM_TYPES)))
    facing_buffer_count = int(round(float(facing[2]) * _BUF_COUNT_NORM))

    inv_norm = np.asarray(
        scalars[_SCALAR_SECTIONS["inventory"]],
        dtype=np.float32,
    )
    inventory = np.round(inv_norm * _PLAYER_MAX_STACK_NP).astype(np.int32)

    player = PlayerScalars(
        pos_x=pos_x,
        pos_y=pos_y,
        direction=direction,
        timestep=timestep,
        facing_machine_type=facing_machine_type,
        facing_buffer_type=facing_buffer_type,
        facing_buffer_count=facing_buffer_count,
        inventory=inventory,
    )

    walkable = _compute_walkable(block_type, machine_type)

    return WorldView(
        block_type=block_type,
        machine_type=machine_type,
        block_resources=block_resources,
        slot0_type=slot0_type,
        slot0_count=slot0_count,
        slot1_type=slot1_type,
        slot1_count=slot1_count,
        slot2_type=slot2_type,
        slot2_count=slot2_count,
        machine_direction=machine_direction,
        buffer_type=buffer_type,
        player=player,
        walkable=walkable,
    )


def _compute_walkable(block_type: np.ndarray, machine_type: np.ndarray) -> np.ndarray:
    """Tiles a player can stand on.

    Water, out-of-bounds markers, and machine-occupied tiles are
    blocked. Belts are the one exception — the engine's
    :func:`factoriax.game_logic.is_position_walkable` treats
    ``MachineType.CONVEYOR_BELT`` as walkable so items can flow
    through tiles the agent later walks across, and the planner has
    to match that to plan paths through laid trunks. DIRT + ore tiles
    are walkable (the engine lets the player stand on ore; mining is
    the primary way it drops).
    """
    blocked_machines = (machine_type != int(MachineType.NONE)) & (
        machine_type != int(MachineType.CONVEYOR_BELT)
    )
    blocked = (
        (block_type == int(BlockType.WATER))
        | (block_type == int(BlockType.OUT_OF_BOUNDS))
        | (block_type == int(BlockType.INVALID))
        | blocked_machines
    )
    return ~blocked


# ---------------------------------------------------------------------------
# BFS pathfinding
# ---------------------------------------------------------------------------


def _bfs(
    walkable: np.ndarray,
    start: tuple[int, int],
    target_tiles: set[tuple[int, int]],
) -> list[int] | None:
    """BFS on a 4-connected grid; returns the action sequence or None."""
    h, w = walkable.shape
    # parent maps tile → (prev_tile, action_taken).
    parent: dict[tuple[int, int], tuple[tuple[int, int] | None, int | None]] = {
        start: (None, None),
    }
    queue: deque[tuple[int, int]] = deque([start])
    found: tuple[int, int] | None = None

    while queue:
        cur = queue.popleft()
        if cur in target_tiles:
            found = cur
            break
        cx, cy = cur
        for direction, (dx, dy) in _DIR_OFFSETS.items():
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            nxt = (nx, ny)
            if nxt in parent:
                continue
            # The next tile must be walkable OR itself be one of the target
            # tiles (some callers allow the target tile to be non-walkable).
            if not walkable[ny, nx] and nxt not in target_tiles:
                continue
            parent[nxt] = (cur, _DIR_TO_MOVE_ACTION[direction])
            queue.append(nxt)

    if found is None:
        return None

    # Reconstruct the action chain from target back to start.
    actions: list[int] = []
    cur = found
    while True:
        prev, action = parent[cur]
        if prev is None:
            break
        assert action is not None
        actions.append(action)
        cur = prev
    actions.reverse()
    return actions


# ---------------------------------------------------------------------------
# Small helpers reused by higher layers
# ---------------------------------------------------------------------------


def direction_toward(
    src: tuple[int, int],
    dst: tuple[int, int],
) -> int | None:
    """Compass direction from *src* to *dst* when they're 4-adjacent."""
    dx, dy = dst[0] - src[0], dst[1] - src[1]
    for d, (ox, oy) in _DIR_OFFSETS.items():
        if (ox, oy) == (dx, dy):
            return d
    return None


def face_action(direction: int) -> int:
    """``FACE_*`` action that orients the player in *direction*."""
    return _DIR_TO_FACE_ACTION[int(direction)]


def withdraw_action() -> int:
    """Single ``WITHDRAW`` action — engine pulls whatever is in the output slot."""
    return int(Action.WITHDRAW)


def deposit_action(item: int | ItemType) -> int:
    """``DEPOSIT_<item>`` action for the given item type."""
    return _ITEM_TO_DEPOSIT[int(item)]


def place_action(machine: int | MachineType) -> int:
    """``PLACE_<machine>`` action for the given placeable machine type."""
    return _MACHINE_TO_PLACE[int(machine)]
