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

from factoriax.constants import (
    BLOCK_MAX_RESOURCES,
    DEFAULT_MACHINE_MAX_HEALTH,
    DEFAULT_MAX_BITERS,
    MAX_ACHIEVEMENTS,
    MAX_MACHINE_INVENTORY_SLOTS,
    MINEABLE_BLOCKS,
    NUM_INVENTORY_SLOTS,
    NUM_ITEM_TYPES,
    NUM_TECHNOLOGIES,
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
        player_inventory: Starting items for every player, as a list of
            ``(ItemType, count)`` pairs.  Each pair fills one inventory
            slot, applied in order.  ``None`` (default) means empty.
        player_positions: Explicit spawn positions as a list of
            ``(x, y)`` tuples, one per player.  ``None`` (default)
            uses the automatic centre-of-map placement.
    """

    name: str
    map_width: int
    map_height: int
    block_map: np.ndarray
    block_resources: np.ndarray | None = None
    machine_types: np.ndarray | None = None
    machine_directions: np.ndarray | None = None
    machine_inventory_items: np.ndarray | None = None
    machine_inventory_counts: np.ndarray | None = None
    machine_selected_recipe: np.ndarray | None = None
    player_inventory: list[tuple[int, int]] | None = None
    player_positions: list[tuple[int, int]] | None = None

    def __post_init__(self) -> None:
        """Validate array shapes match declared dimensions.

        Raises:
            ValueError: If any array has an unexpected shape.
        """
        expected = (self.map_height, self.map_width)
        inv_expected = (self.map_height, self.map_width, MAX_MACHINE_INVENTORY_SLOTS)
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
        if (
            self.machine_directions is not None
            and self.machine_directions.shape != expected
        ):
            raise ValueError(
                f"machine_directions shape "
                f"{self.machine_directions.shape} != {expected}"
            )
        if (
            self.machine_inventory_items is not None
            and self.machine_inventory_items.shape != inv_expected
        ):
            raise ValueError(
                f"machine_inventory_items shape "
                f"{self.machine_inventory_items.shape} != {inv_expected}"
            )
        if (
            self.machine_inventory_counts is not None
            and self.machine_inventory_counts.shape != inv_expected
        ):
            raise ValueError(
                f"machine_inventory_counts shape "
                f"{self.machine_inventory_counts.shape} != {inv_expected}"
            )
        if (
            self.machine_selected_recipe is not None
            and self.machine_selected_recipe.shape != expected
        ):
            raise ValueError(
                f"machine_selected_recipe shape "
                f"{self.machine_selected_recipe.shape} != {expected}"
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
        self._machine_directions: np.ndarray | None = None
        self._machine_inv_items: np.ndarray | None = None
        self._machine_inv_counts: np.ndarray | None = None
        self._machine_selected_recipe: np.ndarray | None = None
        self._player_positions: list[tuple[int, int]] | None = None

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
                self._block_resources = default_resources(self._block_map)
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
            self._block_resources = default_resources(self._block_map)
        self._block_resources[y, x] = amount
        return self

    def set_machine_inventory(
        self,
        x: int,
        y: int,
        slot: int,
        item_type: int,
        count: int,
    ) -> LevelBuilder:
        """Set the contents of a machine inventory slot.

        Args:
            x: Column (0-indexed).
            y: Row (0-indexed).
            slot: Slot index (0 to ``MAX_MACHINE_INVENTORY_SLOTS - 1``).
            item_type: ``ItemType`` integer value.
            count: Stack count.

        Returns:
            ``self`` for chaining.

        Raises:
            IndexError: If ``(x, y)`` is outside the map or slot is invalid.
        """
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise IndexError(
                f"Tile ({x}, {y}) is outside the {self._width}x{self._height} map."
            )
        if not (0 <= slot < MAX_MACHINE_INVENTORY_SLOTS):
            raise IndexError(
                f"Slot {slot} is outside range [0, {MAX_MACHINE_INVENTORY_SLOTS})."
            )
        if self._machine_inv_items is None:
            shape = (self._height, self._width, MAX_MACHINE_INVENTORY_SLOTS)
            self._machine_inv_items = np.zeros(shape, dtype=np.int32)
            self._machine_inv_counts = np.zeros(shape, dtype=np.int32)
        self._machine_inv_items[y, x, slot] = item_type
        self._machine_inv_counts[y, x, slot] = count  # type: ignore[index]
        return self

    def set_machine_recipe(self, x: int, y: int, recipe_idx: int) -> LevelBuilder:
        """Set the selected assembler recipe for a machine tile.

        Args:
            x: Column (0-indexed).
            y: Row (0-indexed).
            recipe_idx: Assembler recipe index.

        Returns:
            ``self`` for chaining.

        Raises:
            IndexError: If ``(x, y)`` is outside the map.
        """
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise IndexError(
                f"Tile ({x}, {y}) is outside the {self._width}x{self._height} map."
            )
        if self._machine_selected_recipe is None:
            self._machine_selected_recipe = np.zeros(
                (self._height, self._width), dtype=np.int32
            )
        self._machine_selected_recipe[y, x] = recipe_idx
        return self

    def place_machine(
        self,
        x: int,
        y: int,
        machine_type: int,
        direction: int = 0,
    ) -> LevelBuilder:
        """Place a machine on a tile with an optional facing direction.

        Args:
            x: Column (0-indexed).
            y: Row (0-indexed).
            machine_type: ``MachineType`` integer value.
            direction: Facing direction as an ``Action`` integer value
                (e.g. ``Action.RIGHT``).  Defaults to 0 (no direction).

        Returns:
            ``self`` for chaining.

        Raises:
            IndexError: If ``(x, y)`` is outside the map.
        """
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise IndexError(
                f"Tile ({x}, {y}) is outside the {self._width}x{self._height} map."
            )
        if self._machine_types is None:
            self._machine_types = np.full(
                (self._height, self._width),
                int(MachineType.NONE),
                dtype=np.int32,
            )
        if self._machine_directions is None:
            self._machine_directions = np.zeros(
                (self._height, self._width), dtype=np.int32
            )
        self._machine_types[y, x] = machine_type
        self._machine_directions[y, x] = direction
        return self

    def set_player_position(self, x: int, y: int) -> LevelBuilder:
        """Set the spawn position for the first player.

        For multi-player levels, call this method once per player in
        order.  Each call appends a position to the list.

        Args:
            x: Column (0-indexed).
            y: Row (0-indexed).

        Returns:
            ``self`` for chaining.

        Raises:
            IndexError: If ``(x, y)`` is outside the map.
        """
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise IndexError(
                f"Position ({x}, {y}) is outside the "
                f"{self._width}x{self._height} map."
            )
        if self._player_positions is None:
            self._player_positions = []
        self._player_positions.append((x, y))
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
            machine_directions=(
                self._machine_directions.copy()
                if self._machine_directions is not None
                else None
            ),
            machine_inventory_items=(
                self._machine_inv_items.copy()
                if self._machine_inv_items is not None
                else None
            ),
            machine_inventory_counts=(
                self._machine_inv_counts.copy()
                if self._machine_inv_counts is not None
                else None
            ),
            machine_selected_recipe=(
                self._machine_selected_recipe.copy()
                if self._machine_selected_recipe is not None
                else None
            ),
            player_positions=(
                list(self._player_positions)
                if self._player_positions is not None
                else None
            ),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def default_resources(block_map: np.ndarray) -> np.ndarray:
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

    if level.player_positions is not None:
        block_map = level.block_map.copy()
        player_positions_np = np.array(
            level.player_positions, dtype=np.int32
        )
        # Ensure each spawn tile is walkable.
        for px, py in level.player_positions:
            block_map[py, px] = int(BlockType.DIRT)
    else:
        block_map, player_positions_np = _place_players(
            level.block_map, params.num_players
        )

    resources_np = (
        level.block_resources
        if level.block_resources is not None
        else default_resources(block_map)
    )
    machine_types_np = (
        level.machine_types
        if level.machine_types is not None
        else np.full(
            (level.map_height, level.map_width), int(MachineType.NONE), dtype=np.int32
        )
    )
    machine_dirs_np = (
        level.machine_directions
        if level.machine_directions is not None
        else np.zeros((level.map_height, level.map_width), dtype=np.int32)
    )

    map_shape = (level.map_height, level.map_width)
    inv_shape = (params.num_players, NUM_INVENTORY_SLOTS)
    player_shape = (params.num_players,)
    machine_inv_shape = (level.map_height, level.map_width, MAX_MACHINE_INVENTORY_SLOTS)

    machine_inv_items_np = (
        level.machine_inventory_items
        if level.machine_inventory_items is not None
        else np.zeros(machine_inv_shape, dtype=np.int32)
    )
    machine_inv_counts_np = (
        level.machine_inventory_counts
        if level.machine_inventory_counts is not None
        else np.zeros(machine_inv_shape, dtype=np.int32)
    )
    machine_recipe_np = (
        level.machine_selected_recipe
        if level.machine_selected_recipe is not None
        else np.zeros(map_shape, dtype=np.int32)
    )

    inv_items_np = np.zeros(inv_shape, dtype=np.int32)
    inv_counts_np = np.zeros(inv_shape, dtype=np.int32)
    if level.player_inventory is not None:
        for slot_idx, (item_type, count) in enumerate(level.player_inventory):
            if slot_idx >= NUM_INVENTORY_SLOTS:
                break
            for p in range(params.num_players):
                inv_items_np[p, slot_idx] = item_type
                inv_counts_np[p, slot_idx] = count

    return EnvState(
        map=jnp.array(block_map, dtype=jnp.int32),
        player_positions=jnp.array(player_positions_np, dtype=jnp.int32),
        player_directions=jnp.full(player_shape, int(Action.DOWN), dtype=jnp.int32),
        timestep=0,
        inventory_items=jnp.array(inv_items_np, dtype=jnp.int32),
        inventory_counts=jnp.array(inv_counts_np, dtype=jnp.int32),
        selected_player=0,
        selected_slots=jnp.zeros(player_shape, dtype=jnp.int32),
        crafting_recipe=jnp.zeros(player_shape, dtype=jnp.int32),
        craft_progress=jnp.zeros(player_shape, dtype=jnp.int32),
        block_resources=jnp.array(resources_np, dtype=jnp.int16),
        machine_types=jnp.array(machine_types_np, dtype=jnp.int32),
        machine_power=jnp.zeros(map_shape, dtype=jnp.int32),
        machine_inventory_items=jnp.array(machine_inv_items_np, dtype=jnp.int32),
        machine_inventory_counts=jnp.array(machine_inv_counts_np, dtype=jnp.int16),
        machine_selected_recipe=jnp.array(machine_recipe_np, dtype=jnp.int32),
        machine_selected_slot=jnp.zeros(map_shape, dtype=jnp.int32),
        machine_direction=jnp.array(machine_dirs_np, dtype=jnp.int32),
        achievements_unlocked=jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_),
        items_mined=jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32),
        research_progress=jnp.zeros(NUM_TECHNOLOGIES, dtype=jnp.int32),
        research_unlocked=jnp.zeros(NUM_TECHNOLOGIES, dtype=jnp.bool_),
        machine_health=jnp.where(
            jnp.array(machine_types_np, dtype=jnp.int32) != int(MachineType.NONE),
            DEFAULT_MACHINE_MAX_HEALTH,
            0,
        ).astype(jnp.int32),
        biter_positions=jnp.zeros((DEFAULT_MAX_BITERS, 2), dtype=jnp.int32),
        biter_health=jnp.zeros(DEFAULT_MAX_BITERS, dtype=jnp.int32),
        scent_field=jnp.zeros(map_shape, dtype=jnp.float32),
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

    # Place nests on dirt tiles away from the center.
    rng_nest, _ = random.split(rng_map)
    nest_noise = random.uniform(
        rng_nest, shape=(params.map_height, params.map_width)
    )
    ys = jnp.broadcast_to(
        jnp.arange(params.map_height)[:, None],
        (params.map_height, params.map_width),
    )
    xs = jnp.broadcast_to(
        jnp.arange(params.map_width)[None, :],
        (params.map_height, params.map_width),
    )
    dist_from_center = jnp.abs(xs - center_x) + jnp.abs(ys - center_y)
    min_nest_dist = max(params.map_width, params.map_height) // 4
    is_dirt = world_map == BlockType.DIRT
    nest_eligible = is_dirt & (dist_from_center >= min_nest_dist)
    place_nest = nest_eligible & (nest_noise < params.nest_probability)
    world_map = jnp.where(place_nest, BlockType.NEST, world_map)

    is_mineable = jnp.isin(world_map, MINEABLE_BLOCKS)
    block_resources = jnp.where(is_mineable, params.base_resources, 0).astype(jnp.int16)

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
        crafting_recipe=jnp.zeros(player_shape, dtype=jnp.int32),
        craft_progress=jnp.zeros(player_shape, dtype=jnp.int32),
        block_resources=block_resources,
        machine_types=jnp.full(map_shape, int(MachineType.NONE), dtype=jnp.int32),
        machine_power=jnp.zeros(map_shape, dtype=jnp.int32),
        machine_inventory_items=jnp.zeros(machine_inv_shape, dtype=jnp.int32),
        machine_inventory_counts=jnp.zeros(machine_inv_shape, dtype=jnp.int16),
        machine_selected_recipe=jnp.zeros(map_shape, dtype=jnp.int32),
        machine_selected_slot=jnp.zeros(map_shape, dtype=jnp.int32),
        machine_direction=jnp.zeros(map_shape, dtype=jnp.int32),
        achievements_unlocked=jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_),
        items_mined=jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32),
        research_progress=jnp.zeros(NUM_TECHNOLOGIES, dtype=jnp.int32),
        research_unlocked=jnp.zeros(NUM_TECHNOLOGIES, dtype=jnp.bool_),
        machine_health=jnp.zeros(map_shape, dtype=jnp.int32),
        biter_positions=jnp.zeros((params.max_biters, 2), dtype=jnp.int32),
        biter_health=jnp.zeros(params.max_biters, dtype=jnp.int32),
        scent_field=jnp.zeros(map_shape, dtype=jnp.float32),
    )


def _generate_terrain(rng: jax.Array, params: EnvParams) -> jax.Array:
    """Generate a random terrain map using patch-based noise.

    Delegates to :func:`_generate_terrain_patched`, the default
    algorithm that produces natural-looking resource clusters and
    water bodies.  See also :func:`_generate_terrain_uniform` for
    the original per-tile random approach.

    Args:
        rng: JAX random key.
        params: Environment parameters with terrain probabilities.

    Returns:
        2D int32 array of shape ``(map_height, map_width)``.
    """
    return _generate_terrain_patched(rng, params)


# ---------------------------------------------------------------------------
# Terrain generation algorithms
#
# Each algorithm takes the same (rng, params) signature and returns
# an int32 terrain grid.  _generate_terrain delegates to one of these.
# ---------------------------------------------------------------------------


def _generate_terrain_uniform(
    rng: jax.Array,
    params: EnvParams,
) -> jax.Array:
    """Generate terrain with independent per-tile random rolls.

    Every tile gets a uniform random value and is assigned a block
    type via cumulative probability thresholds.  Produces a scattered
    salt-and-pepper distribution with no spatial coherence.

    Priority (highest to lowest): water, iron, copper, coal.

    Args:
        rng: JAX random key.
        params: Environment parameters with terrain probabilities.

    Returns:
        2D int32 array of shape ``(map_height, map_width)``.
    """
    random_values = random.uniform(
        rng,
        (params.map_height, params.map_width),
    )

    water_threshold = params.water_probability
    iron_threshold = water_threshold + params.iron_probability
    copper_threshold = iron_threshold + params.copper_probability
    coal_threshold = copper_threshold + params.coal_probability

    terrain = jnp.full(
        (params.map_height, params.map_width),
        int(BlockType.DIRT),
        dtype=jnp.int32,
    )
    terrain = jnp.where(
        random_values < coal_threshold,
        int(BlockType.COAL),
        terrain,
    )
    terrain = jnp.where(
        random_values < copper_threshold,
        int(BlockType.COPPER),
        terrain,
    )
    terrain = jnp.where(
        random_values < iron_threshold,
        int(BlockType.IRON),
        terrain,
    )
    terrain = jnp.where(
        random_values < water_threshold,
        int(BlockType.WATER),
        terrain,
    )

    return terrain


def _smooth_noise(
    rng: jax.Array,
    height: int,
    width: int,
    scale: int = 4,
) -> jax.Array:
    """Generate a smooth 2D noise field via low-res sampling and upscale.

    A small random grid is bilinearly upscaled to the full map size,
    producing natural-looking blobs suitable for patch-based terrain.

    Args:
        rng: JAX random key.
        height: Output height in tiles.
        width: Output width in tiles.
        scale: Downscale factor.  Larger values produce bigger, smoother
            patches.  The low-res grid is ``ceil(dim / scale) + 1``.

    Returns:
        Float32 array of shape ``(height, width)`` in ``[0, 1)``.
    """
    lo_h = height // scale + 2
    lo_w = width // scale + 2
    lo = random.uniform(rng, (lo_h, lo_w))
    hi = jax.image.resize(lo, (height, width), method="bilinear")
    # Rescale to [0, 1) so probability thresholds work as expected.
    lo_val = jnp.min(hi)
    hi_val = jnp.max(hi)
    span = jnp.maximum(hi_val - lo_val, 1e-6)
    return (hi - lo_val) / span


def _generate_terrain_patched(
    rng: jax.Array,
    params: EnvParams,
) -> jax.Array:
    """Generate terrain with smooth resource patches and water bodies.

    Each terrain type gets its own smooth noise field so deposits form
    organic-looking clusters instead of single scattered tiles.  Water
    uses a coarser noise scale to produce larger lakes.

    Priority (highest to lowest): water, iron, copper, coal.

    Args:
        rng: JAX random key.
        params: Environment parameters with terrain probabilities.

    Returns:
        2D int32 array of shape ``(map_height, map_width)``.
    """
    h, w = params.map_height, params.map_width
    k_water, k_iron, k_copper, k_coal = random.split(rng, 4)

    noise_water = _smooth_noise(k_water, h, w, scale=6)
    noise_iron = _smooth_noise(k_iron, h, w, scale=4)
    noise_copper = _smooth_noise(k_copper, h, w, scale=4)
    noise_coal = _smooth_noise(k_coal, h, w, scale=4)

    terrain = jnp.full((h, w), int(BlockType.DIRT), dtype=jnp.int32)
    terrain = jnp.where(
        noise_coal < params.coal_probability,
        int(BlockType.COAL),
        terrain,
    )
    terrain = jnp.where(
        noise_copper < params.copper_probability,
        int(BlockType.COPPER),
        terrain,
    )
    terrain = jnp.where(
        noise_iron < params.iron_probability,
        int(BlockType.IRON),
        terrain,
    )
    terrain = jnp.where(
        noise_water < params.water_probability,
        int(BlockType.WATER),
        terrain,
    )

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
        "machine_directions": (
            level.machine_directions.tolist()
            if level.machine_directions is not None
            else None
        ),
        "machine_inventory_items": (
            level.machine_inventory_items.tolist()
            if level.machine_inventory_items is not None
            else None
        ),
        "machine_inventory_counts": (
            level.machine_inventory_counts.tolist()
            if level.machine_inventory_counts is not None
            else None
        ),
        "machine_selected_recipe": (
            level.machine_selected_recipe.tolist()
            if level.machine_selected_recipe is not None
            else None
        ),
        "player_inventory": level.player_inventory,
        "player_positions": level.player_positions,
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
    raw_dirs = payload.get("machine_directions")
    raw_inv_items = payload.get("machine_inventory_items")
    raw_inv_counts = payload.get("machine_inventory_counts")
    raw_recipe = payload.get("machine_selected_recipe")
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
        machine_directions=(
            np.array(raw_dirs, dtype=np.int32) if raw_dirs is not None else None
        ),
        machine_inventory_items=(
            np.array(raw_inv_items, dtype=np.int32)
            if raw_inv_items is not None
            else None
        ),
        machine_inventory_counts=(
            np.array(raw_inv_counts, dtype=np.int32)
            if raw_inv_counts is not None
            else None
        ),
        machine_selected_recipe=(
            np.array(raw_recipe, dtype=np.int32) if raw_recipe is not None else None
        ),
        player_inventory=payload.get("player_inventory"),
        player_positions=(
            [tuple(p) for p in raw_pos]
            if (raw_pos := payload.get("player_positions")) is not None
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
