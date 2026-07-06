"""Level system for FactoriaX.

A :class:`Level` is a compact, JSON-serializable description of a world:
block layout, optional initial resource amounts, and optional initial
machines.  It is purely a world description — player count and placement
are runtime concerns supplied via :class:`~factoriax.engine.state.EnvParams`
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

Procedural generation lives here as :func:`generate_state`, which
produces an :class:`~factoriax.engine.state.EnvState` directly from a JAX
key — fully JAX-native and JIT-compatible.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orjson
from jax import random

from factoriax.engine.constants import (
    BLOCK_MAX_RESOURCES,
    MAX_ACHIEVEMENTS,
    NUM_ITEM_TYPES,
    NUM_SCIENCE_PACK_TYPES,
    BlockType,
    Direction,
    Machine,
)
from factoriax.engine.machine_spec import MACHINE_MAX_HEALTH
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.tables import MINEABLE_BLOCKS

# ---------------------------------------------------------------------------
# Level dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Level:
    """Serializable description of an initial world state.
    
    A Level captures only the static world geometry — block layout and
    optional pre-set resource amounts or machines.  Player placement is
    not part of the level; use :func:`build_state` with an
    :class:`~factoriax.engine.state.EnvParams` to materialise the full
    :class:`~factoriax.engine.state.EnvState`.

    Parameters
    ----------

    Returns
    -------

    
    >>> from factoriax import FactoriaxEnv, get_level
        >>> env = FactoriaxEnv(level=get_level("15x15_resources"))
        >>> # ``env`` is bound to the registered Level for the lifetime
        >>> # of the env; ``factoriax.LEVELS`` lists every built-in.
    """

    name: str
    map_width: int
    map_height: int
    block_map: np.ndarray
    block_resources: np.ndarray | None = None
    machine_types: np.ndarray | None = None
    machine_directions: np.ndarray | None = None
    machine_inventory: np.ndarray | None = None
    machine_selected_recipe: np.ndarray | None = None
    player_inventory: list[tuple[int, int]] | None = None
    player_inventories: dict[int, list[tuple[int, int]]] | None = None
    player_positions: list[tuple[int, int]] | None = None
    biter_positions: list[tuple[int, int]] | None = None

    def __post_init__(self) -> None:
        """Validate array shapes match declared dimensions.

        Raises
        ------
            ValueError: If any array has an unexpected shape.
        """
        expected = (self.map_height, self.map_width)
        inv_expected = (self.map_height, self.map_width, NUM_ITEM_TYPES)
        if self.block_map.shape != expected:
            raise ValueError(
                f"block_map shape {self.block_map.shape} != {expected}",
            )
        for name, arr, exp in [
            ("block_resources", self.block_resources, expected),
            ("machine_types", self.machine_types, expected),
            ("machine_directions", self.machine_directions, expected),
            ("machine_inventory", self.machine_inventory, inv_expected),
            ("machine_selected_recipe", self.machine_selected_recipe, expected),
        ]:
            if arr is not None and arr.shape != exp:
                raise ValueError(
                    f"{name} shape {arr.shape} != {exp}",
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

    Parameters
    ----------

    Returns
    -------

    
    """

    def __init__(
        self,
        width: int,
        height: int,
        default_block: BlockType = BlockType.DIRT,
    ) -> None:
        """Initialise a blank canvas filled with *default_block*.

        Parameters
        ----------
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
        self._machine_inv: np.ndarray | None = None
        self._machine_selected_recipe: np.ndarray | None = None
        self._player_positions: list[tuple[int, int]] | None = None
        self._biter_positions: list[tuple[int, int]] | None = None

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

        Parameters
        ----------
        x :
            Left column (inclusive, 0-indexed).
        y :
            Top row (inclusive, 0-indexed).
        w :
            Width of the rectangle in tiles.
        h :
            Height of the rectangle in tiles.
        block :
            Block type to place.
        resources :
            If given, set every tile in the region to this resource
            count instead of the default (``BLOCK_MAX_RESOURCES`` for ore,
            0 for non-ore).
        x : int :
            
        y : int :
            
        w : int :
            
        h : int :
            
        block : BlockType :
            
        resources : int | None :
            (Default value = None)
        x: int :
            
        y: int :
            
        w: int :
            
        h: int :
            
        block: BlockType :
            
        resources: int | None :
             (Default value = None)

        Returns
        -------

        
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

        Parameters
        ----------
        x :
            Column (0-indexed).
        y :
            Row (0-indexed).
        amount :
            Resource amount to place.
        x : int :
            
        y : int :
            
        amount : int :
            
        x: int :
            
        y: int :
            
        amount: int :
            

        Returns
        -------

        
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
        item_type: int,
        count: int,
    ) -> LevelBuilder:
        """Set the count of an item type in a machine's pouch inventory.

        Parameters
        ----------
        x :
            Column (0-indexed).
        y :
            Row (0-indexed).
        item_type :
            ``ItemType`` integer value (1-14).
        count :
            Stack count.
        x : int :
            
        y : int :
            
        item_type : int :
            
        count : int :
            
        x: int :
            
        y: int :
            
        item_type: int :
            
        count: int :
            

        Returns
        -------

        
        """
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise IndexError(
                f"Tile ({x}, {y}) is outside the {self._width}x{self._height} map.",
            )
        if self._machine_inv is None:
            shape = (self._height, self._width, NUM_ITEM_TYPES)
            self._machine_inv = np.zeros(shape, dtype=np.int32)
        self._machine_inv[y, x, item_type] = count
        return self

    def set_machine_recipe(self, x: int, y: int, recipe_idx: int) -> LevelBuilder:
        """Set the selected assembler recipe for a machine tile.

        Parameters
        ----------
        x :
            Column (0-indexed).
        y :
            Row (0-indexed).
        recipe_idx :
            Assembler recipe index.
        x : int :
            
        y : int :
            
        recipe_idx : int :
            
        x: int :
            
        y: int :
            
        recipe_idx: int :
            

        Returns
        -------

        
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

        Parameters
        ----------
        x :
            Column (0-indexed).
        y :
            Row (0-indexed).
        machine_type :
            ``Machine`` integer value.
        direction :
            Facing direction as an ``Action`` integer value
            (e.g. ``Direction.RIGHT``).  Defaults to 0 (no direction).
        x : int :
            
        y : int :
            
        machine_type : int :
            
        direction : int :
            (Default value = 0)
        x: int :
            
        y: int :
            
        machine_type: int :
            
        direction: int :
             (Default value = 0)

        Returns
        -------

        
        """
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise IndexError(
                f"Tile ({x}, {y}) is outside the {self._width}x{self._height} map."
            )
        if self._machine_types is None:
            self._machine_types = np.full(
                (self._height, self._width),
                int(Machine.NONE),
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

        Parameters
        ----------
        x :
            Column (0-indexed).
        y :
            Row (0-indexed).
        x : int :
            
        y : int :
            
        x: int :
            
        y: int :
            

        Returns
        -------

        
        """
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise IndexError(
                f"Position ({x}, {y}) is outside the {self._width}x{self._height} map."
            )
        if self._player_positions is None:
            self._player_positions = []
        self._player_positions.append((x, y))
        return self

    def add_biter(self, x: int, y: int) -> LevelBuilder:
        """Add a biter spawn position.

        Parameters
        ----------
        x :
            Column (0-indexed).
        y :
            Row (0-indexed).
        x : int :
            
        y : int :
            
        x: int :
            
        y: int :
            

        Returns
        -------

        
        """
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise IndexError(
                f"Position ({x}, {y}) is outside the {self._width}x{self._height} map."
            )
        if self._biter_positions is None:
            self._biter_positions = []
        self._biter_positions.append((x, y))
        return self

    def set_player_inventory(self, items: list[tuple[int, int]]) -> LevelBuilder:
        """Set starting inventory for all players.

        Parameters
        ----------
        items :
            List of ``(ItemType, count)`` tuples.
        items : list[tuple[int :
            
        int]] :
            
        items: list[tuple[int :
            

        Returns
        -------

        
        """
        self._player_inventory = list(items)
        return self

    def build(self, name: str) -> Level:
        """Finalise and return the :class:`Level`.

        Parameters
        ----------
        name :
            Human-readable identifier for the level.
        name : str :
            
        name: str :
            

        Returns
        -------

        
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
            machine_inventory=(
                self._machine_inv.copy() if self._machine_inv is not None else None
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
            biter_positions=(
                list(self._biter_positions)
                if self._biter_positions is not None
                else None
            ),
            player_inventory=getattr(self, "_player_inventory", None),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def default_resources(block_map: np.ndarray) -> np.ndarray:
    """Build a resource array from a block map using natural defaults.
    
    Ore tiles (COAL, IRON, COPPER) receive ``BLOCK_MAX_RESOURCES``; all
    other tiles receive 0.

    Parameters
    ----------
    block_map :
        Integer block-type grid of shape ``(H, W)``.
    block_map : np.ndarray :
        
    block_map: np.ndarray :
        

    Returns
    -------

    
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

    Parameters
    ----------
    block_map :
        Integer block-type grid of shape ``(H, W)``.
    num_players :
        Number of players to place.
    block_map : np.ndarray :
        
    num_players : int :
        
    block_map: np.ndarray :
        
    num_players: int :
        

    Returns
    -------

    
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


def build_state(level: Level, num_players: int, max_machines: int = 0) -> EnvState:
    """Construct a JAX :class:`~factoriax.engine.state.EnvState` from a :class:`Level`.

    Players are placed at the centre of the map, spread horizontally,
    each guaranteed to land on a DIRT tile.  All dynamic fields
    (inventory, craft progress, machine state) are zero-initialised.

    Returns
    -------
    EnvState
        A fully initialised :class:`~factoriax.engine.state.EnvState`.

    
    >>> import factoriax
        >>> level = factoriax.LevelBuilder(8, 8).build("tiny")
        >>> _, params = factoriax.make("EasyRocket-v1")
        >>> params = EnvParams(num_players=1)
        >>> state = factoriax.build_state(level, params)
        >>> state.map.shape
        (8, 8)
    """
    if level.player_positions is not None:
        block_map = level.block_map.copy()
        player_positions_np = np.array(level.player_positions, dtype=np.int32)
        # Ensure each spawn tile is walkable.
        for px, py in level.player_positions:
            block_map[py, px] = int(BlockType.DIRT)
    else:
        block_map, player_positions_np = _place_players(
            level.block_map, num_players
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
            (level.map_height, level.map_width), int(Machine.NONE), dtype=np.int32
        )
    )
    machine_dirs_np = (
        level.machine_directions
        if level.machine_directions is not None
        else np.zeros((level.map_height, level.map_width), dtype=np.int32)
    )

    map_shape = (level.map_height, level.map_width)
    inv_shape = (num_players, NUM_ITEM_TYPES)
    player_shape = (num_players,)
    # Build player pouch inventories from (item_type, count) pairs.
    player_inv_np = np.zeros(inv_shape, dtype=np.int32)
    if level.player_inventory is not None:
        for item_type, count in level.player_inventory:
            for p in range(num_players):
                player_inv_np[p, item_type] += count
    if level.player_inventories is not None:
        for p_idx, items in level.player_inventories.items():
            if p_idx >= num_players:
                continue
            player_inv_np[p_idx] = 0
            for item_type, count in items:
                player_inv_np[p_idx, item_type] += count

    h, w = level.map_height, level.map_width
    mm = max_machines if max_machines > 0 else max(64, h * w // 4)

    # Build entity arrays from grid machine data.
    mt_jnp = jnp.array(machine_types_np, dtype=jnp.int8)
    ent_y = jnp.full(mm, -1, dtype=jnp.int16)
    ent_x = jnp.full(mm, -1, dtype=jnp.int16)
    ent_type = jnp.zeros(mm, dtype=jnp.int8)
    ent_dir = jnp.zeros(mm, dtype=jnp.int8)
    ent_health = jnp.zeros(mm, dtype=jnp.int16)
    max_health_arr = MACHINE_MAX_HEALTH
    tile_ent = jnp.full(map_shape, -1, dtype=jnp.int16)

    # Entity inventory arrays (populated from level.machine_inventory).
    ent_buf_type_np = np.zeros(mm, dtype=np.int8)
    ent_buf_count_np = np.zeros(mm, dtype=np.int16)
    ent_asm_in_type_np = np.zeros((mm, 2), dtype=np.int8)
    ent_asm_in_count_np = np.zeros((mm, 2), dtype=np.int16)

    machine_inv = level.machine_inventory

    # Populate entities from grid (Python loop, only at build time)
    idx = 0
    for y in range(map_shape[0]):
        for x in range(map_shape[1]):
            mt = int(machine_types_np[y, x])
            if mt == int(Machine.NONE):
                continue
            if idx >= mm:
                break
            ent_y = ent_y.at[idx].set(jnp.int16(y))
            ent_x = ent_x.at[idx].set(jnp.int16(x))
            ent_type = ent_type.at[idx].set(jnp.int8(mt))
            ent_dir = ent_dir.at[idx].set(
                jnp.int8(machine_dirs_np[y, x]),
            )
            ent_health = ent_health.at[idx].set(max_health_arr[mt])
            tile_ent = tile_ent.at[y, x].set(jnp.int16(idx))

            # Populate inventory from level data.
            if machine_inv is not None:
                inv_row = machine_inv[y, x]
                if mt == int(Machine.MINER):
                    # Miners have one output buffer for mined ore.
                    for it in range(1, NUM_ITEM_TYPES):
                        if int(inv_row[it]) > 0:
                            ent_buf_type_np[idx] = it
                            ent_buf_count_np[idx] = int(inv_row[it])
                            break
                elif mt in (
                    int(Machine.ASSEMBLER),
                    int(Machine.FURNACE),
                ):
                    slot = 0
                    for it in range(1, NUM_ITEM_TYPES):
                        if int(inv_row[it]) > 0 and slot < 2:
                            ent_asm_in_type_np[idx, slot] = it
                            ent_asm_in_count_np[idx, slot] = int(
                                inv_row[it],
                            )
                            slot += 1
                else:
                    # Pallet, belt, etc: first non-zero item to buffer.
                    for it in range(1, NUM_ITEM_TYPES):
                        if int(inv_row[it]) > 0:
                            ent_buf_type_np[idx] = it
                            ent_buf_count_np[idx] = int(inv_row[it])
                            break

            idx += 1

    return EnvState(
        map=jnp.array(block_map, dtype=jnp.int8),
        block_resources=jnp.array(resources_np, dtype=jnp.int16),
        machine_types=mt_jnp,
        tile_entity=tile_ent,
        ent_y=ent_y,
        ent_x=ent_x,
        ent_type=ent_type,
        ent_direction=ent_dir,
        ent_power=jnp.zeros(mm, dtype=jnp.int16),
        ent_buf_type=jnp.array(ent_buf_type_np, dtype=jnp.int8),
        ent_buf_count=jnp.array(ent_buf_count_np, dtype=jnp.int16),
        ent_asm_in_type=jnp.array(ent_asm_in_type_np, dtype=jnp.int8),
        ent_asm_in_count=jnp.array(ent_asm_in_count_np, dtype=jnp.int16),
        ent_asm_out_type=jnp.zeros(mm, dtype=jnp.int8),
        ent_asm_out_count=jnp.zeros(mm, dtype=jnp.int16),
        ent_health=ent_health,
        player_positions=jnp.array(player_positions_np, dtype=jnp.int16),
        player_directions=jnp.full(
            player_shape,
            int(Direction.DOWN),
            dtype=jnp.int8,
        ),
        player_inventory=jnp.array(player_inv_np, dtype=jnp.int16),
        selected_player=jnp.int32(0),
        timestep=jnp.int32(0),
        items_mined=jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32),
        science_consumed_step=jnp.zeros(NUM_SCIENCE_PACK_TYPES, dtype=jnp.int32),
        achievements_unlocked=jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_),
    )


# ---------------------------------------------------------------------------
# Procedural generation
# ---------------------------------------------------------------------------


def initial_state(
    world_map: jax.Array,
    params: EnvParams,
    num_players: int,
    max_machines: int,
) -> EnvState:
    """Assemble an initial :class:`EnvState` from a generated block map."""
    h, w = world_map.shape
    center_x = w // 2
    center_y = h // 2
    player_positions = []
    for i in range(num_players):
        offset = i - num_players // 2
        px = jnp.clip(center_x + offset, 0, w - 1)
        py = center_y
        player_positions.append([px, py])
        world_map = world_map.at[py, px].set(jnp.int8(BlockType.DIRT))

    player_positions_arr = jnp.array(player_positions, dtype=jnp.int32)
    player_directions = jnp.full(
        num_players, int(Direction.DOWN), dtype=jnp.int32
    )

    is_mineable = jnp.isin(world_map, MINEABLE_BLOCKS)
    block_resources = jnp.where(
        is_mineable,
        params.base_resources,
        0,
    ).astype(jnp.int16)

    map_shape = (h, w)
    inv_shape = (num_players, NUM_ITEM_TYPES)
    mm = max_machines if max_machines > 0 else max(64, h * w // 4)

    return EnvState(
        map=world_map.astype(jnp.int8),
        block_resources=block_resources,
        machine_types=jnp.full(map_shape, int(Machine.NONE), dtype=jnp.int8),
        tile_entity=jnp.full(map_shape, -1, dtype=jnp.int16),
        ent_y=jnp.full(mm, -1, dtype=jnp.int16),
        ent_x=jnp.full(mm, -1, dtype=jnp.int16),
        ent_type=jnp.zeros(mm, dtype=jnp.int8),
        ent_direction=jnp.zeros(mm, dtype=jnp.int8),
        ent_power=jnp.zeros(mm, dtype=jnp.int16),
        ent_buf_type=jnp.zeros(mm, dtype=jnp.int8),
        ent_buf_count=jnp.zeros(mm, dtype=jnp.int16),
        ent_asm_in_type=jnp.zeros((mm, 2), dtype=jnp.int8),
        ent_asm_in_count=jnp.zeros((mm, 2), dtype=jnp.int16),
        ent_asm_out_type=jnp.zeros(mm, dtype=jnp.int8),
        ent_asm_out_count=jnp.zeros(mm, dtype=jnp.int16),
        ent_health=jnp.zeros(mm, dtype=jnp.int16),
        player_positions=player_positions_arr.astype(jnp.int16),
        player_directions=player_directions.astype(jnp.int8),
        player_inventory=jnp.zeros(inv_shape, dtype=jnp.int16),
        selected_player=jnp.int32(0),
        timestep=jnp.int32(0),
        items_mined=jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32),
        science_consumed_step=jnp.zeros(NUM_SCIENCE_PACK_TYPES, dtype=jnp.int32),
        achievements_unlocked=jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_),
    )





# ---------------------------------------------------------------------------
# Terrain generation algorithms
#
# Each algorithm takes the same (rng, params) signature and returns
# an int32 terrain grid.  generate_terrain delegates to one of these.
# ---------------------------------------------------------------------------


def _generate_terrain_uniform(
    rng: jax.Array,
    params: EnvParams,
    map_height: int,
    map_width: int,
) -> jax.Array:
    random_values = random.uniform(rng, (map_height, map_width))

    water_threshold = params.water_probability
    iron_threshold = water_threshold + params.iron_probability
    copper_threshold = iron_threshold + params.copper_probability
    coal_threshold = copper_threshold + params.coal_probability
    tin_threshold = coal_threshold + params.tin_probability
    silicon_threshold = tin_threshold + params.silicon_probability

    terrain = jnp.full(
        (map_height, map_width),
        int(BlockType.DIRT),
        dtype=jnp.int32,
    )
    terrain = jnp.where(
        random_values < silicon_threshold,
        int(BlockType.SILICON),
        terrain,
    )
    terrain = jnp.where(
        random_values < tin_threshold,
        int(BlockType.TIN),
        terrain,
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

    Parameters
    ----------
    rng :
        JAX random key.
    height :
        Output height in tiles.
    width :
        Output width in tiles.
    scale :
        Downscale factor.  Larger values produce bigger, smoother
        patches.  The low-res grid is ``ceil(dim / scale) + 1``.
    rng : jax.Array :
        
    height : int :
        
    width : int :
        
    scale : int :
        (Default value = 4)
    rng: jax.Array :
        
    height: int :
        
    width: int :
        
    scale: int :
         (Default value = 4)

    Returns
    -------

    
    """
    lo_h = height // scale + 2
    lo_w = width // scale + 2
    lo = random.uniform(rng, (lo_h, lo_w))
    hi = jax.image.resize(lo, (height, width), method="bilinear")
    # Rescale to [0, 1) so probability thresholds work as expected.
    lo_val = jnp.min(hi)
    hi_val = jnp.max(hi)
    span = jnp.maximum(hi_val - lo_val, 1e-6)
    result: jax.Array = (hi - lo_val) / span
    return result


def generate_terrain(
    rng: jax.Array,
    params: EnvParams,
    map_height: int,
    map_width: int,
) -> jax.Array:
    h, w = map_height, map_width
    keys = random.split(rng, 6)

    noise_water = _smooth_noise(keys[0], h, w, scale=6)
    noise_iron = _smooth_noise(keys[1], h, w, scale=4)
    noise_copper = _smooth_noise(keys[2], h, w, scale=4)
    noise_coal = _smooth_noise(keys[3], h, w, scale=4)
    noise_tin = _smooth_noise(keys[4], h, w, scale=4)
    noise_silicon = _smooth_noise(keys[5], h, w, scale=4)

    terrain = jnp.full((h, w), int(BlockType.DIRT), dtype=jnp.int32)
    terrain = jnp.where(
        noise_silicon < params.silicon_probability,
        int(BlockType.SILICON),
        terrain,
    )
    terrain = jnp.where(
        noise_tin < params.tin_probability,
        int(BlockType.TIN),
        terrain,
    )
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
# Convenience: procedural state for tests and scripting
# ---------------------------------------------------------------------------


def generate_state(
    rng: jax.Array,
    params: EnvParams,
    map_height: int = 32,
    map_width: int = 32,
    num_players: int = 1,
    max_machines: int = 0,
) -> EnvState:
    """Generate a procedural EnvState for use in tests and scripts."""
    from factoriax.engine.envs.base import FactoriaxEnv  # avoid circular import

    env = FactoriaxEnv(
        num_players=num_players,
        max_machines=max_machines,
        map_height=map_height,
        map_width=map_width,
    )
    _, state = env.reset_env(rng, params)
    return state


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def save_level(level: Level, path: Path) -> None:
    """Serialize a :class:`Level` to a JSON file using orjson.
    
    Arrays are stored as nested integer lists.  The file is
    human-readable and can be edited in any text editor.

    Parameters
    ----------
    path :
        Destination file path.  Parent directories are created if
        they do not exist.
    level :
        Level to serialize.
        Examples
        --------
    level : Level :
        
    path : Path :
        
    level: Level :
        
    path: Path :
        

    Returns
    -------

    
    >>> from pathlib import Path
        >>> import tempfile, factoriax
        >>> level = factoriax.LevelBuilder(8, 8).build("tiny")
        >>> with tempfile.TemporaryDirectory() as d:
        ...     factoriax.save_level(level, Path(d) / "tiny.json")
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
        "machine_inventory": (
            level.machine_inventory.tolist()
            if level.machine_inventory is not None
            else None
        ),
        "machine_selected_recipe": (
            level.machine_selected_recipe.tolist()
            if level.machine_selected_recipe is not None
            else None
        ),
        "player_inventory": level.player_inventory,
        "player_inventories": (
            {str(k): v for k, v in level.player_inventories.items()}
            if level.player_inventories
            else None
        ),
        "player_positions": level.player_positions,
        "biter_positions": level.biter_positions,
    }
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def load_level(path: Path) -> Level:
    """Deserialize a :class:`Level` from a JSON file written by :func:`save_level`.

    Parameters
    ----------
    path :
        Path to the JSON file.
    path : Path :
        
    path: Path :
        

    Returns
    -------
    The reconstructed
        class:`Level`.

    
    >>> from pathlib import Path
        >>> import tempfile, factoriax
        >>> level = factoriax.LevelBuilder(8, 8).build("tiny")
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "tiny.json"
        ...     factoriax.save_level(level, p)
        ...     factoriax.load_level(p).name
        'tiny'
    """
    payload = orjson.loads(Path(path).read_bytes())
    raw_dirs = payload.get("machine_directions")
    raw_inv = payload.get("machine_inventory")
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
        machine_inventory=(
            np.array(raw_inv, dtype=np.int32) if raw_inv is not None else None
        ),
        machine_selected_recipe=(
            np.array(raw_recipe, dtype=np.int32) if raw_recipe is not None else None
        ),
        player_inventory=payload.get("player_inventory"),
        player_inventories=(
            {int(k): [tuple(p) for p in v] for k, v in raw_pi.items()}
            if (raw_pi := payload.get("player_inventories")) is not None
            else None
        ),
        player_positions=(
            [tuple(p) for p in raw_pos]
            if (raw_pos := payload.get("player_positions")) is not None
            else None
        ),
        biter_positions=(
            [tuple(p) for p in raw_bp]
            if (raw_bp := payload.get("biter_positions")) is not None
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

    Parameters
    ----------
    name :
        Level name as registered in :data:`LEVELS`.
    name : str :
        
    name: str :
        

    Returns
    -------
    The corresponding
        class:`Level`.

    
    >>> import factoriax
        >>> factoriax.get_level("15x15_resources").name
        '15x15_resources'
    """
    if name not in LEVELS:
        available = ", ".join(sorted(LEVELS))
        raise KeyError(f"Unknown level {name!r}. Available: {available}")
    return LEVELS[name]
