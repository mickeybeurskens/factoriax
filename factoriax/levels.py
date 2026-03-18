"""Level system for FactoriaX.

A :class:`Level` is a compact, JSON-serializable description of a world:
block layout, optional initial resource amounts, and optional initial
machines.  It is purely a world description — player count and placement
are runtime concerns supplied via :class:`~factoriax.state.EnvParams`
when calling :func:`build_state`.  This separation means the same level
works unchanged for 1, 2, or N agents.

Typical usage::

    # Load a built-in level and build a state for 2 players.
    level = get_level("15x15_resources")
    params = EnvParams(num_players=2, max_timesteps=500)
    state = build_state(level, params)

    # Define a custom level programmatically.
    level = (
        LevelBuilder(8, 8)
        .fill_rect(0, 0, 3, 3, BlockType.COAL)
        .build("coal_corner")
    )

    # Round-trip through JSON.
    save_level(level, Path("my_level.json"))
    level = load_level(Path("my_level.json"))

Procedural generation (previously in ``world_gen``) is also here as
:func:`generate_state`, which produces an :class:`~factoriax.state.EnvState`
directly from a JAX key — fully JAX-native and JIT-compatible.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orjson
from jax import random

from factoriax.achievements import NUM_ACHIEVEMENTS
from factoriax.constants import (
    BLOCK_MAX_RESOURCES,
    MAX_MACHINE_INVENTORY_SLOTS,
    MINEABLE_BLOCKS,
    NUM_INVENTORY_SLOTS,
    NUM_ITEM_TYPES,
    Action,
    BlockType,
    MachineType,
)
from factoriax.state import EnvParams, EnvState

# ---------------------------------------------------------------------------
# Level dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Level:
    """Serializable description of an initial world state.

    A Level captures only the static world geometry — block layout and
    optional pre-set resource amounts or machines.  Player placement is
    not part of the level; use :func:`build_state` with an
    :class:`~factoriax.state.EnvParams` to materialise the full
    :class:`~factoriax.state.EnvState`.

    Attributes:
        name: Human-readable identifier used in the level registry.
        map_width: Number of tiles along the x-axis.
        map_height: Number of tiles along the y-axis.
        block_map: Integer block-type grid of shape ``(map_height, map_width)``.
        block_resources: Per-tile resource amounts of shape
            ``(map_height, map_width)``, or ``None`` to auto-fill: ore
            tiles receive ``BLOCK_MAX_RESOURCES``, others receive 0.
        machine_types: Per-tile machine-type grid of shape
            ``(map_height, map_width)``, or ``None`` for an empty world.
    """

    name: str
    map_width: int
    map_height: int
    block_map: np.ndarray
    block_resources: np.ndarray | None = None
    machine_types: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate array shapes match declared dimensions.

        Raises:
            ValueError: If any array has an unexpected shape.
        """
        expected = (self.map_height, self.map_width)
        if self.block_map.shape != expected:
            raise ValueError(f"block_map shape {self.block_map.shape} != {expected}")
        if self.block_resources is not None and self.block_resources.shape != expected:
            raise ValueError(
                f"block_resources shape {self.block_resources.shape} != {expected}"
            )
        if self.machine_types is not None and self.machine_types.shape != expected:
            raise ValueError(
                f"machine_types shape {self.machine_types.shape} != {expected}"
            )


# ---------------------------------------------------------------------------
# LevelBuilder
# ---------------------------------------------------------------------------


class LevelBuilder:
    """Fluent builder for constructing :class:`Level` objects programmatically.

    All mutating methods return ``self`` to support method chaining.  Call
    :meth:`build` to produce the final :class:`Level`.

    Example::

        level = (
            LevelBuilder(15, 15)
            .fill_rect(0, 0, 4, 4, BlockType.COAL)
            .fill_rect(11, 0, 4, 4, BlockType.COPPER)
            .build("my_level")
        )
    """

    def __init__(
        self,
        width: int,
        height: int,
        default_block: BlockType = BlockType.DIRT,
    ) -> None:
        """Initialise a blank canvas filled with *default_block*.

        Args:
            width: Map width in tiles.
            height: Map height in tiles.
            default_block: Block type to fill the canvas with.
        """
        self._width = width
        self._height = height
        self._block_map = np.full((height, width), int(default_block), dtype=np.int32)
        self._block_resources: np.ndarray | None = None
        self._machine_types: np.ndarray | None = None

    def fill_rect(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        block: BlockType,
        resources: int | None = None,
    ) -> LevelBuilder:
        """Fill a rectangular region with *block*, optionally overriding resources.

        The rectangle is clipped to the map boundary so callers do not
        need to guard against out-of-bounds coordinates.

        Args:
            x: Left column (inclusive, 0-indexed).
            y: Top row (inclusive, 0-indexed).
            w: Width of the rectangle in tiles.
            h: Height of the rectangle in tiles.
            block: Block type to place.
            resources: If given, set every tile in the region to this resource
                count instead of the default (``BLOCK_MAX_RESOURCES`` for ore,
                0 for non-ore).

        Returns:
            ``self`` for chaining.
        """
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self._width, x + w)
        y1 = min(self._height, y + h)
        self._block_map[y0:y1, x0:x1] = int(block)
        if resources is not None:
            if self._block_resources is None:
                self._block_resources = _default_resources(self._block_map)
            self._block_resources[y0:y1, x0:x1] = resources
        return self

    def set_resources(self, x: int, y: int, amount: int) -> LevelBuilder:
        """Override the resource amount at a single tile.

        If no resource array has been set yet, one is created with the
        auto-fill defaults (ore tiles → ``BLOCK_MAX_RESOURCES``, others → 0)
        before applying the override.

        Args:
            x: Column (0-indexed).
            y: Row (0-indexed).
            amount: Resource amount to place.

        Returns:
            ``self`` for chaining.

        Raises:
            IndexError: If ``(x, y)`` is outside the map.
        """
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise IndexError(
                f"Tile ({x}, {y}) is outside the {self._width}x{self._height} map."
            )
        if self._block_resources is None:
            self._block_resources = _default_resources(self._block_map)
        self._block_resources[y, x] = amount
        return self

    def build(self, name: str) -> Level:
        """Finalise and return the :class:`Level`.

        Args:
            name: Human-readable identifier for the level.

        Returns:
            The constructed :class:`Level`.
        """
        return Level(
            name=name,
            map_width=self._width,
            map_height=self._height,
            block_map=self._block_map.copy(),
            block_resources=(
                self._block_resources.copy()
                if self._block_resources is not None
                else None
            ),
            machine_types=(
                self._machine_types.copy() if self._machine_types is not None else None
            ),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_resources(block_map: np.ndarray) -> np.ndarray:
    """Build a resource array from a block map using natural defaults.

    Ore tiles (COAL, IRON, COPPER) receive ``BLOCK_MAX_RESOURCES``; all
    other tiles receive 0.

    Args:
        block_map: Integer block-type grid of shape ``(H, W)``.

    Returns:
        int32 resource array of the same shape.
    """
    mineable = np.isin(
        block_map, [int(BlockType.COAL), int(BlockType.IRON), int(BlockType.COPPER)]
    )
    return np.where(mineable, BLOCK_MAX_RESOURCES, 0).astype(np.int32)


def _place_players(
    block_map: np.ndarray, num_players: int
) -> tuple[np.ndarray, np.ndarray]:
    """Place *num_players* players near the centre of the map on dirt tiles.

    Players are spread horizontally around the centre column.  Each spawn
    tile is forced to DIRT so players never appear inside a wall or ore
    body (the block_map is not modified in place; a copy is returned).

    Args:
        block_map: Integer block-type grid of shape ``(H, W)``.
        num_players: Number of players to place.

    Returns:
        Tuple ``(block_map_copy, player_positions)`` where
        ``player_positions`` has shape ``(num_players, 2)`` and dtype
        int32.  Column ``0`` is x (column index); column ``1`` is y (row
        index).
    """
    h, w = block_map.shape
    block_map = block_map.copy()
    center_x = w // 2
    center_y = h // 2
    positions: list[list[int]] = []
    for i in range(num_players):
        offset = i - num_players // 2
        px = int(np.clip(center_x + offset, 0, w - 1))
        py = center_y
        block_map[py, px] = int(BlockType.DIRT)
        positions.append([px, py])
    return block_map, np.array(positions, dtype=np.int32)


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------


def build_state(level: Level, params: EnvParams) -> EnvState:
    """Construct a JAX :class:`~factoriax.state.EnvState` from a :class:`Level`.

    Players are placed at the centre of the map, spread horizontally,
    each guaranteed to land on a DIRT tile.  All dynamic fields
    (inventory, craft progress, machine state) are zero-initialised.

    Args:
        level: Level definition.  Its ``map_width`` and ``map_height``
            must match ``params.map_width`` and ``params.map_height``.
        params: Environment parameters, including ``num_players``.

    Returns:
        A fully initialised :class:`~factoriax.state.EnvState`.

    Raises:
        ValueError: If the level dimensions do not match ``params``.
    """
    if level.map_width != params.map_width or level.map_height != params.map_height:
        raise ValueError(
            f"Level dimensions ({level.map_width}x{level.map_height}) do not "
            f"match params ({params.map_width}x{params.map_height})."
        )

    block_map, player_positions_np = _place_players(level.block_map, params.num_players)

    resources_np = (
        level.block_resources
        if level.block_resources is not None
        else _default_resources(block_map)
    )
    machine_types_np = (
        level.machine_types
        if level.machine_types is not None
        else np.full(
            (level.map_height, level.map_width), int(MachineType.NONE), dtype=np.int32
        )
    )

    map_shape = (level.map_height, level.map_width)
    inv_shape = (params.num_players, NUM_INVENTORY_SLOTS)
    player_shape = (params.num_players,)
    machine_inv_shape = (level.map_height, level.map_width, MAX_MACHINE_INVENTORY_SLOTS)

    return EnvState(
        map=jnp.array(block_map, dtype=jnp.int32),
        player_positions=jnp.array(player_positions_np, dtype=jnp.int32),
        player_directions=jnp.full(player_shape, int(Action.DOWN), dtype=jnp.int32),
        timestep=0,
        inventory_items=jnp.zeros(inv_shape, dtype=jnp.int32),
        inventory_counts=jnp.zeros(inv_shape, dtype=jnp.int32),
        selected_player=0,
        selected_slots=jnp.zeros(player_shape, dtype=jnp.int32),
        selected_recipes=jnp.zeros(player_shape, dtype=jnp.int32),
        craft_progress=jnp.zeros(player_shape, dtype=jnp.int32),
        block_resources=jnp.array(resources_np, dtype=jnp.int16),
        machine_types=jnp.array(machine_types_np, dtype=jnp.int32),
        machine_power=jnp.zeros(map_shape, dtype=jnp.int32),
        machine_inventory_items=jnp.zeros(machine_inv_shape, dtype=jnp.int32),
        machine_inventory_counts=jnp.zeros(machine_inv_shape, dtype=jnp.int16),
        machine_selected_recipe=jnp.zeros(map_shape, dtype=jnp.int32),
        machine_selected_slot=jnp.zeros(map_shape, dtype=jnp.int32),
        achievements_unlocked=jnp.zeros(NUM_ACHIEVEMENTS, dtype=jnp.bool_),
        items_mined=jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# Procedural generation (moved from world_gen)
# ---------------------------------------------------------------------------


def generate_state(rng: jax.Array, params: EnvParams) -> EnvState:
    """Generate a procedural world state from a random key.

    JAX-native and JIT-compatible.  Players spawn near the centre of the
    map; spawn tiles are forced to DIRT after random terrain generation.

    Args:
        rng: JAX random key for reproducible generation.
        params: Environment parameters including map dimensions and
            terrain probabilities.

    Returns:
        Initial :class:`~factoriax.state.EnvState` with a randomly
        generated map and players near the centre.
    """
    rng_map, _ = random.split(rng)
    world_map = _generate_terrain(rng_map, params)

    center_x = params.map_width // 2
    center_y = params.map_height // 2
    player_positions = []
    for i in range(params.num_players):
        offset = i - params.num_players // 2
        px = jnp.clip(center_x + offset, 0, params.map_width - 1)
        py = center_y
        player_positions.append([px, py])
        world_map = world_map.at[py, px].set(BlockType.DIRT)

    player_positions_arr = jnp.array(player_positions, dtype=jnp.int32)
    player_directions = jnp.full(params.num_players, int(Action.DOWN), dtype=jnp.int32)

    is_mineable = jnp.isin(world_map, MINEABLE_BLOCKS)
    block_resources = jnp.where(is_mineable, BLOCK_MAX_RESOURCES, 0).astype(jnp.int16)

    map_shape = (params.map_height, params.map_width)
    inv_shape = (params.num_players, NUM_INVENTORY_SLOTS)
    player_shape = (params.num_players,)
    machine_inv_shape = (
        params.map_height,
        params.map_width,
        MAX_MACHINE_INVENTORY_SLOTS,
    )

    return EnvState(
        map=world_map,
        player_positions=player_positions_arr,
        player_directions=player_directions,
        timestep=0,
        inventory_items=jnp.zeros(inv_shape, dtype=jnp.int32),
        inventory_counts=jnp.zeros(inv_shape, dtype=jnp.int32),
        selected_player=0,
        selected_slots=jnp.zeros(player_shape, dtype=jnp.int32),
        selected_recipes=jnp.zeros(player_shape, dtype=jnp.int32),
        craft_progress=jnp.zeros(player_shape, dtype=jnp.int32),
        block_resources=block_resources,
        machine_types=jnp.full(map_shape, int(MachineType.NONE), dtype=jnp.int32),
        machine_power=jnp.zeros(map_shape, dtype=jnp.int32),
        machine_inventory_items=jnp.zeros(machine_inv_shape, dtype=jnp.int32),
        machine_inventory_counts=jnp.zeros(machine_inv_shape, dtype=jnp.int16),
        machine_selected_recipe=jnp.zeros(map_shape, dtype=jnp.int32),
        machine_selected_slot=jnp.zeros(map_shape, dtype=jnp.int32),
        achievements_unlocked=jnp.zeros(NUM_ACHIEVEMENTS, dtype=jnp.bool_),
        items_mined=jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32),
    )


def _generate_terrain(rng: jax.Array, params: EnvParams) -> jax.Array:
    """Generate a random terrain map.

    Uses cumulative probability thresholds to assign block types.  Water
    takes priority, then iron, copper, coal, with dirt as the fallback.

    Args:
        rng: JAX random key.
        params: Environment parameters with terrain probabilities.

    Returns:
        2D int32 array of shape ``(map_height, map_width)``.
    """
    random_values = random.uniform(rng, (params.map_height, params.map_width))

    water_threshold = params.water_probability
    iron_threshold = water_threshold + params.iron_probability
    copper_threshold = iron_threshold + params.copper_probability
    coal_threshold = copper_threshold + params.coal_probability

    terrain = jnp.full(
        (params.map_height, params.map_width), int(BlockType.DIRT), dtype=jnp.int32
    )
    terrain = jnp.where(random_values < coal_threshold, int(BlockType.COAL), terrain)
    terrain = jnp.where(
        random_values < copper_threshold, int(BlockType.COPPER), terrain
    )
    terrain = jnp.where(random_values < iron_threshold, int(BlockType.IRON), terrain)
    terrain = jnp.where(random_values < water_threshold, int(BlockType.WATER), terrain)

    return terrain


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def save_level(level: Level, path: Path) -> None:
    """Serialize a :class:`Level` to a JSON file using orjson.

    Arrays are stored as nested integer lists.  The file is
    human-readable and can be edited in any text editor.

    Args:
        path: Destination file path.  Parent directories are created if
            they do not exist.
        level: Level to serialize.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": level.name,
        "map_width": level.map_width,
        "map_height": level.map_height,
        "block_map": level.block_map.tolist(),
        "block_resources": (
            level.block_resources.tolist()
            if level.block_resources is not None
            else None
        ),
        "machine_types": (
            level.machine_types.tolist() if level.machine_types is not None else None
        ),
    }
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def load_level(path: Path) -> Level:
    """Deserialize a :class:`Level` from a JSON file written by :func:`save_level`.

    Args:
        path: Path to the JSON file.

    Returns:
        The reconstructed :class:`Level`.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    payload = orjson.loads(Path(path).read_bytes())
    return Level(
        name=payload["name"],
        map_width=payload["map_width"],
        map_height=payload["map_height"],
        block_map=np.array(payload["block_map"], dtype=np.int32),
        block_resources=(
            np.array(payload["block_resources"], dtype=np.int32)
            if payload["block_resources"] is not None
            else None
        ),
        machine_types=(
            np.array(payload["machine_types"], dtype=np.int32)
            if payload["machine_types"] is not None
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Built-in levels
# ---------------------------------------------------------------------------

#: 15x15 map with 4x4 ore patches in three corners and dirt elsewhere.
#: Coal: top-left. Copper: top-right. Iron: bottom-left.
_15X15_RESOURCES: Level = (
    LevelBuilder(15, 15)
    .fill_rect(0, 0, 4, 4, BlockType.COAL, resources=3)
    .fill_rect(11, 0, 4, 4, BlockType.COPPER, resources=3)
    .fill_rect(0, 11, 4, 4, BlockType.IRON, resources=3)
    .build("15x15_resources")
)

#: Registry of all built-in levels, keyed by name.
LEVELS: dict[str, Level] = {
    _15X15_RESOURCES.name: _15X15_RESOURCES,
}


def get_level(name: str) -> Level:
    """Look up a built-in level by name.

    Args:
        name: Level name as registered in :data:`LEVELS`.

    Returns:
        The corresponding :class:`Level`.

    Raises:
        KeyError: If *name* is not found.  The error message lists
            available names.
    """
    if name not in LEVELS:
        available = ", ".join(sorted(LEVELS))
        raise KeyError(f"Unknown level {name!r}. Available: {available}")
    return LEVELS[name]
