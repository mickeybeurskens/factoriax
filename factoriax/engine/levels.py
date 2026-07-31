"""Describe a starting world and turn it into an :class:`EnvState`.

A :class:`Level` is the static half of a world: the block layout, and, where
the author gives them, the ore amounts, the machines already on the map, and
the spawn tiles of the players. It holds no live state and no timestep. It is
plain numpy, so it writes to JSON and reads back unchanged.

The level does not fix the number of players. The build decides that.
:func:`build_state` takes the count as an argument, so one level serves a
single-agent run and a four-agent run with no edit.

There are two ways in. :class:`LevelBuilder` writes a level tile by tile, and
the editor and the hand-written scenarios use it. :func:`generate_state` uses
no :class:`Level` at all and makes a state from a random key. An environment
calls it when a scenario supplies no level.

Typical usage::

    from pathlib import Path

    from factoriax.engine.constants import BlockType
    from factoriax.engine.levels import (
        LevelBuilder, build_state, get_level, load_level, save_level,
    )

    # Load a built-in level and build a state for two players.
    level = get_level("15x15_resources")
    state = build_state(level, num_players=2)

    # Define a custom level programmatically.
    level = (
        LevelBuilder(8, 8)
        .fill_rect(0, 0, 3, 3, BlockType.COAL)
        .build("coal_corner")
    )

    # Round-trip through JSON.
    save_level(level, Path("my_level.json"))
    level = load_level(Path("my_level.json"))
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
from factoriax.engine.recipes import OUTPUT_TO_RECIPE, RECIPE_MACHINE_TYPE
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.tables import MACHINE_MAX_HEALTH, MINEABLE_BLOCKS

#: Machine kind that produces each item, indexed by ``ItemType`` value, with
#: ``Machine.NONE`` for an item that no recipe produces. :func:`build_state`
#: reads it to separate a finished output from an input, because an item indexes
#: ``Level.machine_inventory`` and that field records no slot. The values come
#: from the default recipes of the engine. A scenario with its own recipe book
#: does not change them.
_ITEM_PRODUCER: np.ndarray = np.full(NUM_ITEM_TYPES, int(Machine.NONE), dtype=np.int32)
_out_to_recipe = np.asarray(OUTPUT_TO_RECIPE)
_recipe_machine = np.asarray(RECIPE_MACHINE_TYPE)
for _item in range(NUM_ITEM_TYPES):
    _recipe = int(_out_to_recipe[_item])
    if _recipe >= 0:
        _ITEM_PRODUCER[_item] = int(_recipe_machine[_recipe])

# ---------------------------------------------------------------------------
# Level dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Level:
    """A starting world, as plain arrays that survive a JSON round trip.

    Every field after ``block_map`` is optional. A field that holds ``None``
    means "let :func:`build_state` decide", and not "empty". With no
    ``block_resources``, :func:`default_resources` fills the ore tiles. With no
    ``player_positions``, the build spreads the players along the middle row. A
    field that holds an array of zeros is a different instruction. It means
    that the world starts with nothing.

    An array is indexed ``[row, column]``, so ``block_map[y][x]``. A position
    list holds ``(x, y)`` pairs. The two orders are not the same, and that
    matters when a person writes a level by hand.

    Attributes
    ----------
    name
        Identifier for the level. :func:`get_level` finds a level by it.
    map_width, map_height
        Tile dimensions. The class validates every array field against them.
    block_map
        :class:`~factoriax.engine.constants.BlockType` of each tile. Shape
        ``(map_height, map_width)``. This is the one required field.
    block_resources
        Ore units in each tile, or ``None`` to take them from ``block_map``.
    machine_types
        :class:`~factoriax.engine.constants.Machine` on each tile, ``NONE``
        where the tile is clear, or ``None`` for a world with no machines.
    machine_directions
        Facing of each tile, 0 where nothing set it. The value has no meaning
        on a tile with no machine.
    machine_inventory
        Items in the machine on each tile, shape
        ``(map_height, map_width, NUM_ITEM_TYPES)``. An item indexes it and it
        records no slot, so :func:`build_state` must decide which slot holds
        each item.
    player_inventory
        ``(item, count)`` pairs for every player.
    player_inventories
        Contents for one player, with the player index as the key. For a player
        that it names, it replaces ``player_inventory`` and does not add to it.
        An index past the player count has no effect.
    player_positions
        ``(x, y)`` spawn tile of each player. The build sets each listed tile to
        DIRT, so a spawn never lands in ore or water. When the field is set, the
        list must hold exactly one entry for each player of the state.

    Raises
    ------
    ValueError
        If an array field disagrees with ``map_width`` or ``map_height``.
        ``__post_init__`` raises it, so a malformed level fails at construction
        and not at render time.
    """

    name: str
    map_width: int
    map_height: int
    block_map: np.ndarray
    block_resources: np.ndarray | None = None
    machine_types: np.ndarray | None = None
    machine_directions: np.ndarray | None = None
    machine_inventory: np.ndarray | None = None
    player_inventory: list[tuple[int, int]] | None = None
    player_inventories: dict[int, list[tuple[int, int]]] | None = None
    player_positions: list[tuple[int, int]] | None = None

    def __post_init__(self) -> None:
        """Test every array that the caller gave against the declared dimensions.

        This method runs at construction, so no level can exist in a shape that
        the rest of the module has to guard against. A field that holds
        ``None`` passes, and the method does not treat it as empty.

        Raises
        ------
        ValueError
            If an array disagrees with ``map_width`` or ``map_height``. The
            message names the field at fault and both shapes.
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
        ]:
            if arr is not None and arr.shape != exp:
                raise ValueError(
                    f"{name} shape {arr.shape} != {exp}",
                )


# ---------------------------------------------------------------------------
# LevelBuilder
# ---------------------------------------------------------------------------


class LevelBuilder:
    """Build a :class:`Level` a tile at a time.

    Every method that writes returns ``self``, so the calls chain, and
    :meth:`build` ends the chain. The builder starts from a canvas of one block
    type. It allocates an optional array only when a method first writes to it.
    A field that no method touched therefore stays ``None`` in the finished
    level, and :func:`build_state` applies its own default.

    A builder is reusable. :meth:`build` copies each array out, so a later edit
    cannot change a level that the builder already produced.

    A coordinate is ``(x, y)``, with the origin at the top left. This is the
    opposite order from the ``[row, column]`` arrays below it.

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
        """Open a blank canvas of one block type.

        Parameters
        ----------
        width
            Map width in tiles.
        height
            Map height in tiles.
        default_block
            Block type that covers the whole canvas at the start.
        """
        self._width = width
        self._height = height
        self._block_map = np.full((height, width), int(default_block), dtype=np.int32)
        self._block_resources: np.ndarray | None = None
        self._machine_types: np.ndarray | None = None
        self._machine_directions: np.ndarray | None = None
        self._machine_inv: np.ndarray | None = None
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
        """Write a block type across a rectangle, and set its ore if asked.

        The method clips the rectangle to the map. A caller can therefore give
        coordinates that pass an edge, and the method writes the part inside
        the map. A rectangle fully outside the map writes nothing and still
        returns ``self``. This is the one placement method that raises nothing
        on a bad coordinate.

        Parameters
        ----------
        x
            Left column, included in the rectangle.
        y
            Top row, included in the rectangle.
        w
            Width in tiles. A value of zero or less fills nothing.
        h
            Height in tiles. A value of zero or less fills nothing.
        block
            Block type to write across the region.
        resources
            Ore units for every tile in the region. Without this argument the
            tiles keep the values that the resource array already held. If no
            resource array exists, the method creates none, and
            :func:`default_resources` sets the amounts at build time.

        Returns
        -------
        LevelBuilder
            ``self``, so the calls chain.
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
        elif self._block_resources is not None:
            # The array is a copy of one moment, so new ore needs its default
            # value now. Without it the tile keeps the zero from its creation.
            self._block_resources[y0:y1, x0:x1] = default_resources(
                self._block_map[y0:y1, x0:x1],
            )
        return self

    def set_resources(self, x: int, y: int, amount: int) -> LevelBuilder:
        """Set the ore left in one tile.

        This call creates the resource array. A level that calls it one time
        therefore carries a full array from that point, and
        :func:`build_state` stops taking the amounts from the block map. Every
        other tile keeps its default value, and ore that a later call paints
        also gets one, because :meth:`fill_rect` keeps the array in step.

        Parameters
        ----------
        x
            Column.
        y
            Row.
        amount
            Ore units to store. The method does not clamp the value, so a value
            larger than ``BLOCK_MAX_RESOURCES`` stands. A tile that is not ore
            can also take a count that no machine ever mines.

        Returns
        -------
        LevelBuilder
            ``self``, so the calls chain.

        Raises
        ------
        IndexError
            If the tile is outside the map.
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
        """Give the machine on a tile a starting stock of one item.

        The field records the contents by item and holds no slot, so this call
        gives what a machine holds, and not where. :func:`build_state` selects
        the slot when it builds the entity. For an assembler or a furnace it
        reads the recipe book to separate a finished output from an input.

        The method does not test that a machine stands on the tile. A stock on
        an empty tile passes through save and load, and the build then drops
        it.

        A machine with one buffer, which is every kind except an assembler and
        a furnace, holds one item kind. This method accepts a second kind, and
        :func:`build_state` refuses it, because the kind on the tile can change
        after this call.

        Parameters
        ----------
        x
            Column.
        y
            Row.
        item_type
            :class:`~factoriax.engine.constants.ItemType` value.
        count
            Number of items. The method replaces any count for this item on
            this tile, and does not add to it. It sets no limit, so a value
            above the stack limit of the machine clamps later.

        Returns
        -------
        LevelBuilder
            ``self``, so the calls chain.

        Raises
        ------
        IndexError
            If the tile is outside the map.
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

    def place_machine(
        self,
        x: int,
        y: int,
        machine_type: int,
        direction: int = 0,
    ) -> LevelBuilder:
        """Put a machine on a tile, with a facing.

        A call on an occupied tile replaces the machine that was there. The
        block below stays as it was, so this method places a machine on water
        or on ore, where the placement rules of the engine refuse one.

        Parameters
        ----------
        x
            Column.
        y
            Row.
        machine_type
            :class:`~factoriax.engine.constants.Machine` value.
        direction
            :class:`~factoriax.engine.constants.Direction` value. The default 0
            means "no direction", and a machine that ignores its facing keeps
            it. A belt or a miner at 0 moves nothing.

        Returns
        -------
        LevelBuilder
            ``self``, so the calls chain.

        Raises
        ------
        IndexError
            If the tile is outside the map.
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
        """Add one player spawn to the end of the list.

        The call order is the player order, so the first call sets player 0. To
        set the spawn of one player, a caller must also set the spawn of every
        earlier player. No call removes a spawn.

        One spawn commits the level to all of them. :func:`build_state` refuses
        a level whose spawn count does not match the number of players that the
        caller asked for.

        Parameters
        ----------
        x
            Column.
        y
            Row.

        Returns
        -------
        LevelBuilder
            ``self``, so the calls chain.

        Raises
        ------
        IndexError
            If the position is outside the map.
        """
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise IndexError(
                f"Position ({x}, {y}) is outside the {self._width}x{self._height} map."
            )
        if self._player_positions is None:
            self._player_positions = []
        self._player_positions.append((x, y))
        return self

    def set_player_inventory(self, items: list[tuple[int, int]]) -> LevelBuilder:
        """Give every player the same start inventory.

        The method replaces the inventory of an earlier call, and does not add
        to it. A repeated item inside one list does add up, because
        :func:`build_state` adds the pairs as it reads them.

        Parameters
        ----------
        items
            ``(item, count)`` pairs. An empty list leaves the players with
            nothing, and no call to this method has the same result.

        Returns
        -------
        LevelBuilder
            ``self``, so the calls chain.
        """
        self._player_inventory = list(items)
        return self

    def build(self, name: str) -> Level:
        """End the chain and return the finished level.

        The method copies each array out, so the builder stays usable and a
        later edit cannot change the level that it just returned. A field that
        no method touched stays ``None`` and does not become an empty array.
        :func:`build_state` therefore separates "not set" from "empty".

        Parameters
        ----------
        name
            Identifier for the level. The method does not test it against
            :data:`LEVELS` for uniqueness.

        Returns
        -------
        Level
            A level whose shapes ``Level.__post_init__`` validated.
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
            player_positions=(
                list(self._player_positions)
                if self._player_positions is not None
                else None
            ),
            player_inventory=getattr(self, "_player_inventory", None),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def default_resources(block_map: np.ndarray) -> np.ndarray:
    """Fill every ore tile with a full deposit, and every other tile with none.

    Ore means every block in ``factoriax.engine.tables.MINEABLE_BLOCKS``. The
    generated path tests against the same set, so a hand-built level and a
    generated level agree on which tiles hold something to mine.

    Parameters
    ----------
    block_map
        Block type of each tile, shape ``(H, W)``.

    Returns
    -------
    np.ndarray
        Shape ``(H, W)``, int32. An ore tile holds ``BLOCK_MAX_RESOURCES`` and
        every other tile holds 0. This is a new array, and the function does
        not modify the input.
    """
    mineable = np.isin(block_map, np.asarray(MINEABLE_BLOCKS))
    return np.where(mineable, BLOCK_MAX_RESOURCES, 0).astype(np.int32)


def _place_players(
    block_map: np.ndarray, num_players: int
) -> tuple[np.ndarray, np.ndarray]:
    """Spread the players along the middle row and clear the tiles under them.

    The function sets each spawn tile to DIRT, so a player never starts in
    water, in ore, or in a wall. That edit is the reason to return a block map
    and not the positions alone.

    Parameters
    ----------
    block_map
        Block type of each tile, shape ``(H, W)``. The function does not modify
        it and returns a copy.
    num_players
        Number of players to place. The positions sit around the centre
        column. On a map narrower than the player count they clamp to the edge,
        and two or more players then share a tile.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        The new block map, shape ``(H, W)``, and the spawns as ``(x, y)`` rows,
        shape ``(num_players, 2)``, int32.
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


def _fill_buffer(
    level: Level,
    x: int,
    y: int,
    machine_type: int,
    inv_row: np.ndarray,
    idx: int,
    buf_type: np.ndarray,
    buf_count: np.ndarray,
) -> None:
    """Load the stock of a one-buffer machine, and refuse a load it cannot hold."""
    items = [it for it in range(1, NUM_ITEM_TYPES) if int(inv_row[it]) > 0]
    if len(items) > 1:
        raise ValueError(
            f"Level {level.name!r} gives the {Machine(machine_type).name} at "
            f"({x}, {y}) {len(items)} item types, but the machine holds one "
            f"item. Store the rest elsewhere.",
        )
    if items:
        buf_type[idx] = items[0]
        buf_count[idx] = int(inv_row[items[0]])


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------


def build_state(level: Level, num_players: int, max_machines: int = 0) -> EnvState:
    """Turn a level into a live state, ready to step.

    This function turns the per-tile view of the level into the per-entity view
    of the engine. Each occupied tile gets one entry in the entity arrays, and
    ``tile_entity`` records the join. Everything that the level does not
    describe starts empty. No craft is in progress, no player mined an item,
    and no achievement is unlocked.

    Where the level leaves a field unset, the default applies, so a plain block
    map still gives a playable world. Where the level sets a field, the level
    wins and the function follows it exactly. The mismatches below therefore
    raise, and the function does not correct them.

    Parameters
    ----------
    level
        World to build from. The function does not modify it.
    num_players
        Number of players to spawn. This sets the length of every player array.
        It must match ``level.player_positions`` when that field is set.
    max_machines
        Size of the entity table. The default 0 means "select one", which is
        ``max(64, tiles // 4)``, enough for a quarter of the map. A level
        cannot place more machines than this number.

    Returns
    -------
    EnvState
        A state with ``timestep`` at 0, and with every player facing DOWN.

    Raises
    ------
    ValueError
        If the level places more machines than ``max_machines`` allows, lists a
        spawn count other than ``num_players``, or gives a one-buffer machine
        more than one kind of item. Each case describes a world that the state
        cannot hold, so the function trims none of them.

    Examples
    --------
    >>> from factoriax.engine.levels import LevelBuilder, build_state
    >>> level = LevelBuilder(8, 8).build("tiny")
    >>> state = build_state(level, num_players=1)
    >>> state.map.shape
    (8, 8)
    >>> int(state.timestep)
    0
    """
    if level.player_positions is not None:
        if len(level.player_positions) != num_players:
            raise ValueError(
                f"Level {level.name!r} lists {len(level.player_positions)} "
                f"player positions but the state needs {num_players}. Give the "
                f"level one spawn per player or leave player_positions unset.",
            )
        block_map = level.block_map.copy()
        player_positions_np = np.array(level.player_positions, dtype=np.int32)
        # Make each spawn tile walkable.
        for px, py in level.player_positions:
            block_map[py, px] = int(BlockType.DIRT)
    else:
        block_map, player_positions_np = _place_players(level.block_map, num_players)

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
    # Build the player inventories from the (item_type, count) pairs.
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

    # Build the entity arrays from the machine data on the grid.
    mt_jnp = jnp.array(machine_types_np, dtype=jnp.int8)
    ent_y = jnp.full(mm, -1, dtype=jnp.int16)
    ent_x = jnp.full(mm, -1, dtype=jnp.int16)
    ent_type = jnp.zeros(mm, dtype=jnp.int8)
    ent_dir = jnp.zeros(mm, dtype=jnp.int8)
    ent_health = jnp.zeros(mm, dtype=jnp.int16)
    max_health_arr = MACHINE_MAX_HEALTH
    tile_ent = jnp.full(map_shape, -1, dtype=jnp.int16)

    # The entity inventory arrays, which level.machine_inventory fills.
    ent_buf_type_np = np.zeros(mm, dtype=np.int8)
    ent_buf_count_np = np.zeros(mm, dtype=np.int16)
    ent_asm_in_type_np = np.zeros((mm, 2), dtype=np.int8)
    ent_asm_in_count_np = np.zeros((mm, 2), dtype=np.int16)
    ent_asm_out_type_np = np.zeros(mm, dtype=np.int8)
    ent_asm_out_count_np = np.zeros(mm, dtype=np.int16)

    machine_inv = level.machine_inventory

    # Refuse a level that is larger than the entity table. A cut instead leaves
    # machine_types with machines that no entity holds.
    machine_count = int(np.count_nonzero(machine_types_np != int(Machine.NONE)))
    if machine_count > mm:
        raise ValueError(
            f"Level {level.name!r} places {machine_count} machines but "
            f"max_machines is {mm}. Raise max_machines or remove machines.",
        )

    # Fill the entities from the grid. This Python loop runs at build time only.
    idx = 0
    for y in range(map_shape[0]):
        for x in range(map_shape[1]):
            mt = int(machine_types_np[y, x])
            if mt == int(Machine.NONE):
                continue
            ent_y = ent_y.at[idx].set(jnp.int16(y))
            ent_x = ent_x.at[idx].set(jnp.int16(x))
            ent_type = ent_type.at[idx].set(jnp.int8(mt))
            ent_dir = ent_dir.at[idx].set(
                jnp.int8(machine_dirs_np[y, x]),
            )
            ent_health = ent_health.at[idx].set(max_health_arr[mt])
            tile_ent = tile_ent.at[y, x].set(jnp.int16(idx))

            # Fill the inventory from the level data.
            if machine_inv is not None:
                inv_row = machine_inv[y, x]
                if mt == int(Machine.MINER):
                    _fill_buffer(
                        level, x, y, mt, inv_row, idx, ent_buf_type_np, ent_buf_count_np
                    )
                elif mt in (
                    int(Machine.ASSEMBLER),
                    int(Machine.FURNACE),
                ):
                    # An item indexes the inventory, and the field holds no
                    # slot. An item that the recipes of this machine produce is
                    # therefore its finished output, and every other item is an
                    # input. Without that split, an output fills an input slot
                    # and pushes a real input out of the two that exist.
                    slot = 0
                    for it in range(1, NUM_ITEM_TYPES):
                        count = int(inv_row[it])
                        if count <= 0:
                            continue
                        if int(_ITEM_PRODUCER[it]) == mt:
                            if int(ent_asm_out_count_np[idx]) == 0:
                                ent_asm_out_type_np[idx] = it
                                ent_asm_out_count_np[idx] = count
                        elif slot < 2:
                            ent_asm_in_type_np[idx, slot] = it
                            ent_asm_in_count_np[idx, slot] = count
                            slot += 1
                else:
                    _fill_buffer(
                        level, x, y, mt, inv_row, idx, ent_buf_type_np, ent_buf_count_np
                    )

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
        ent_asm_out_type=jnp.array(ent_asm_out_type_np, dtype=jnp.int8),
        ent_asm_out_count=jnp.array(ent_asm_out_count_np, dtype=jnp.int16),
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
    """Wrap a generated block map in a state with no machines standing.

    This function is the generated counterpart of :func:`build_state`. It takes
    a finished block map and not a :class:`Level`, so it places nothing. It
    allocates the entity table, leaves it empty, and the world starts with no
    machines.

    The ore comes from ``MINEABLE_BLOCKS`` and ``params.base_resources``.
    :func:`default_resources` uses the same set of blocks but a different
    amount, so a generated tile and a hand-built tile can hold different
    amounts of the same ore.

    Parameters
    ----------
    world_map
        Block type of each tile, shape ``(H, W)``. The function writes DIRT to
        each spawn tile, and the state carries that new copy.
    params
        The function reads ``base_resources`` only.
    num_players
        Number of players to spawn. They stand along the middle row, as in
        :func:`_place_players`.
    max_machines
        Size of the entity table. A value of 0 means ``max(64, tiles // 4)``.

    Returns
    -------
    EnvState
        A state with an empty entity table and ``timestep`` at 0. Every player
        faces DOWN and holds an empty inventory.
    """
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
    player_directions = jnp.full(num_players, int(Direction.DOWN), dtype=jnp.int32)

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
# Terrain generation
#
# generate_terrain is the one algorithm here. It draws one smooth-noise field
# for each resource and tests each field against its own probability. The draws
# are independent, but the results go down in layers, silicon first and water
# last, and each layer writes over the one before. Only water, the last layer,
# covers the share that its probability names. Every earlier layer keeps only
# the part that no later layer took, so silicon reaches about half its figure.
# This behaviour stands as it is.
# ---------------------------------------------------------------------------


def _smooth_noise(
    rng: jax.Array,
    height: int,
    width: int,
    scale: int = 4,
) -> jax.Array:
    """Draw a field of smooth blobs whose values are spread evenly over [0, 1).

    Two properties matter to the caller, and they pull against each other. Two
    tiles next to each other hold close values, which makes patches and not
    speckle. Across the whole field the values spread evenly, so ``field < p``
    selects a fraction ``p`` of the map for any ``p``. A probability in
    :class:`~factoriax.engine.state.EnvParams` therefore means what it says.

    Parameters
    ----------
    rng
        Key for the draw. The same key and shape give the same field.
    height, width
        Output size in tiles.
    scale
        Blob size. The grid before the resize is ``height // scale + 2`` by
        ``width // scale + 2``, so a larger scale gives fewer and wider
        patches.

    Returns
    -------
    jax.Array
        Shape ``(height, width)``, float32, with values in (0, 1). The spread
        is even by construction, and not even on average, so even a small map
        divides correctly at any threshold.
    """
    lo_h = height // scale + 2
    lo_w = width // scale + 2
    lo = random.uniform(rng, (lo_h, lo_w))
    hi = jax.image.resize(lo, (height, width), method="bilinear")
    # Replace each value by its rank. An interpolation of uniform samples
    # gathers them around the mean. A rescale of that range to [0, 1] leaves
    # the gathering in place, so a threshold of 0.1 once selected far less than
    # a tenth of the map. The rank flattens the whole distribution and not only
    # its bounds. It is monotonic, so the blobs keep their shape and only their
    # values move.
    flat = hi.ravel()
    order = jnp.argsort(flat)
    ranks = jnp.zeros_like(order).at[order].set(jnp.arange(flat.size))
    result: jax.Array = ((ranks + 0.5) / flat.size).reshape(height, width)
    return result


def generate_terrain(
    rng: jax.Array,
    params: EnvParams,
    map_height: int,
    map_width: int,
) -> jax.Array:
    """Draw a random block map of ore patches, water, and dirt.

    Each block type gets its own noise field and its own threshold, so the
    draws are independent. The results then go down in a fixed order, silicon
    first and water last, and each layer writes over the one before. Only water
    covers the share that its probability names. Every earlier layer keeps only
    the tiles that no later layer took, so silicon, the first layer, reaches
    about half its figure. Dirt covers every tile that no layer took.

    Parameters
    ----------
    rng
        Key for the draw. The function splits it six ways, one for each block
        type, so the same key gives the same map.
    params
        The function reads the six ``*_probability`` fields. Each one is the
        share of the map that the block covers on its own, before a later layer
        writes over it.
    map_height, map_width
        Map size in tiles.

    Returns
    -------
    jax.Array
        Block type of each tile, shape ``(map_height, map_width)``, int32. The
        result holds terrain only: no machines, no players, and no ore amounts.
    """
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
# A generated state, for tests and scripts
# ---------------------------------------------------------------------------


def generate_state(
    rng: jax.Array,
    params: EnvParams,
    map_height: int = 32,
    map_width: int = 32,
    num_players: int = 1,
    max_machines: int = 0,
) -> EnvState:
    """Generate a random world and return its state.

    This is a short wrapper for tests and scripts. It builds a
    :class:`~factoriax.engine.envs.base.FactoriaxEnv` and resets it. The result
    therefore matches what an environment produces, and this function assembles
    no state of its own. The import happens inside the call, because
    ``envs.base`` imports this module.

    JAX cannot trace this function. It builds an environment from Python
    integers, so it cannot run under ``jax.jit``. Only the reset that it calls
    is traceable.

    Parameters
    ----------
    rng
        Key for the terrain draw. The same key gives the same world.
    params
        The function passes this to the reset, which reads the terrain
        probabilities from it.
    map_height, map_width
        Map dimensions in tiles.
    num_players
        Number of players to spawn.
    max_machines
        Size of the entity table. A value of 0 means ``max(64, tiles // 4)``.

    Returns
    -------
    EnvState
        A state on new terrain, with no machines on the map.
    """
    from factoriax.engine.envs.base import FactoriaxEnv  # breaks a circular import

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
    """Write a level to a JSON file.

    Each array becomes a nested list of integers, and the output is indented. A
    person can therefore read and edit a level by hand. The function writes
    every optional field, and writes ``null`` where a field is unset.
    :func:`load_level` also accepts a file that leaves a field out.

    The format carries no version marker. A later change to the field set
    therefore has no way to announce itself to an older reader.

    Parameters
    ----------
    level
        Level to write. The function does not modify it.
    path
        Destination file. The function creates the parent directories, and
        writes over an existing file with no warning.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> from factoriax.engine.levels import LevelBuilder, save_level
    >>> level = LevelBuilder(8, 8).build("tiny")
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     path = Path(directory) / "tiny.json"
    ...     save_level(level, path)
    ...     path.exists()
    True
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
        "player_inventory": level.player_inventory,
        "player_inventories": (
            {str(k): v for k, v in level.player_inventories.items()}
            if level.player_inventories
            else None
        ),
        "player_positions": level.player_positions,
    }
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def load_level(path: Path) -> Level:
    """Read a :class:`Level` from a JSON file that :func:`save_level` wrote.

    Only ``name``, ``map_width``, ``map_height``, and ``block_map`` are
    required. Every other field can hold ``null`` or be absent, and both mean
    "not set". A hand-written file therefore holds only the fields that it
    needs.

    An array comes back as int32 and a position list as tuples, which is what
    :class:`LevelBuilder` produces. A level therefore passes through a save and
    a load unchanged. ``Level.__post_init__`` tests the shapes, so a file whose
    arrays disagree with its dimensions fails here.

    Parameters
    ----------
    path
        File to read.

    Returns
    -------
    Level
        The level that the file describes.

    Raises
    ------
    KeyError
        If a required field is absent.
    ValueError
        If an array disagrees with the declared dimensions.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> from factoriax.engine.levels import LevelBuilder, load_level, save_level
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     path = Path(directory) / "tiny.json"
    ...     save_level(LevelBuilder(8, 8).build("tiny"), path)
    ...     load_level(path).name
    'tiny'
    """
    payload = orjson.loads(Path(path).read_bytes())
    # Only the name, the dimensions, and the block map are required. Every
    # other field is optional on Level, so a file can leave one out or set it
    # to null.
    raw_res = payload.get("block_resources")
    raw_types = payload.get("machine_types")
    raw_dirs = payload.get("machine_directions")
    raw_inv = payload.get("machine_inventory")
    return Level(
        name=payload["name"],
        map_width=payload["map_width"],
        map_height=payload["map_height"],
        block_map=np.array(payload["block_map"], dtype=np.int32),
        block_resources=(
            np.array(raw_res, dtype=np.int32) if raw_res is not None else None
        ),
        machine_types=(
            np.array(raw_types, dtype=np.int32) if raw_types is not None else None
        ),
        machine_directions=(
            np.array(raw_dirs, dtype=np.int32) if raw_dirs is not None else None
        ),
        machine_inventory=(
            np.array(raw_inv, dtype=np.int32) if raw_inv is not None else None
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
    )


# ---------------------------------------------------------------------------
# Built-in levels
# ---------------------------------------------------------------------------

#: 15x15 map with 4x4 ore patches in three corners, and dirt on every other
#: tile. Coal is top-left, copper is top-right, and iron is bottom-left.
_15X15_RESOURCES: Level = (
    LevelBuilder(15, 15)
    .fill_rect(0, 0, 4, 4, BlockType.COAL, resources=BLOCK_MAX_RESOURCES)
    .fill_rect(11, 0, 4, 4, BlockType.COPPER, resources=BLOCK_MAX_RESOURCES)
    .fill_rect(0, 11, 4, 4, BlockType.IRON, resources=BLOCK_MAX_RESOURCES)
    .build("15x15_resources")
)

#: Every built-in level, with the name as the key.
LEVELS: dict[str, Level] = {
    _15X15_RESOURCES.name: _15X15_RESOURCES,
}


def get_level(name: str) -> Level:
    """Find a built-in level by name.

    Parameters
    ----------
    name
        Key into :data:`LEVELS`.

    Returns
    -------
    Level
        The registered level itself, and not a copy.

        CAUTION: Do not write to the arrays of the result. Every caller shares
        this object with :data:`LEVELS`, so a write changes the level for the
        whole process. :func:`build_state` only reads it.

    Raises
    ------
    KeyError
        If no level carries that name. The message lists the names that exist.

    Examples
    --------
    >>> from factoriax.engine.levels import get_level
    >>> get_level("15x15_resources").name
    '15x15_resources'
    """
    if name not in LEVELS:
        available = ", ".join(sorted(LEVELS))
        raise KeyError(f"Unknown level {name!r}. Available: {available}")
    return LEVELS[name]
