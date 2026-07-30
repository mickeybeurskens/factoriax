"""Describe a starting world and turn it into an :class:`EnvState`.

A :class:`Level` is the static half of a world: the block layout, and
optionally the ore amounts, the machines already standing, and where players
spawn. It holds no live state, has no timestep, and is plain numpy, so it
serializes to JSON and survives a round trip.

How many players actually spawn is decided at build time, not by the level.
:func:`build_state` takes the count as an argument, so one level serves a
single-agent run and a four-agent run without editing.

Two ways in. :class:`LevelBuilder` writes a level tile by tile, which is what
the editor and the hand-written scenarios use. :func:`generate_state` skips
:class:`Level` entirely and produces a state from a random key, which is what
an environment falls back to when a scenario ships no level.

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
#: ``Machine.NONE`` for items no recipe outputs. :func:`build_state` reads it to
#: tell a combiner's finished output from its inputs, because
#: ``Level.machine_inventory`` is item-indexed and records no slot. Derived from
#: the engine's default recipes; a scenario shipping its own recipe book does
#: not change this classification.
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

    Every field past ``block_map`` is optional, and ``None`` means "no opinion,
    let :func:`build_state` decide" rather than "empty". Leaving
    ``block_resources`` unset fills ore tiles from :func:`default_resources`;
    leaving ``player_positions`` unset spreads players along the middle row.
    Setting a field to an all-zero array is a different instruction, and means
    the world really does start with nothing.

    Arrays are indexed ``[row, column]``, so ``block_map[y][x]``, while the
    position lists hold ``(x, y)`` pairs. The two orders do not match, which
    matters when hand-writing a level.

    Attributes
    ----------
    name
        Identifier for the level. :func:`get_level` looks levels up by it.
    map_width, map_height
        Tile dimensions. Every array field is validated against them.
    block_map
        :class:`~factoriax.engine.constants.BlockType` per tile. Shape
        ``(map_height, map_width)``. The only required field.
    block_resources
        Ore units per tile, or ``None`` to derive them from ``block_map``.
    machine_types
        :class:`~factoriax.engine.constants.Machine` per tile, ``NONE`` where
        the tile is clear, or ``None`` for a world with no machines standing.
    machine_directions
        Facing per tile, 0 where unset. Meaningless where no machine stands.
    machine_inventory
        Items held by the machine on each tile, shape
        ``(map_height, map_width, NUM_ITEM_TYPES)``. Item-indexed with no slot,
        so :func:`build_state` has to infer which slot an item belongs in.
    player_inventory
        ``(item, count)`` pairs given to every player.
    player_inventories
        Per-player contents, keyed by player index. For a player it names it
        replaces ``player_inventory`` rather than adding to it, and an index
        past the player count is ignored.
    player_positions
        ``(x, y)`` spawn per player. Each listed tile is forced to DIRT, so a
        spawn never lands inside ore or water. When set, the list must hold
        exactly one entry per player the state is built for.

    Raises
    ------
    ValueError
        An array field disagrees with ``map_width`` or ``map_height``. Raised
        from ``__post_init__``, so a malformed level fails at construction
        rather than at render time.
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
        """Check every supplied array against the declared dimensions.

        Runs on construction, so a level cannot exist in a shape the rest of
        the module would have to guard against. Fields left ``None`` are
        skipped rather than treated as empty.

        Raises
        ------
        ValueError
            An array disagrees with ``map_width`` or ``map_height``. The
            message names the offending field and both shapes.
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

    Every mutating method returns ``self``, so calls chain, and :meth:`build`
    closes the chain. The builder starts from a canvas of one block type and
    allocates the optional arrays only when a method first touches them, which
    is what keeps an untouched field ``None`` in the finished level and lets
    :func:`build_state` apply its own default.

    A builder is reusable: :meth:`build` copies each array out, so a later
    edit does not reach back into a level already built.

    Coordinates are ``(x, y)`` with the origin at the top left, the opposite
    order from the ``[row, column]`` arrays underneath.

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
            Block type covering the whole canvas to begin with.
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
        """Write a block type across a rectangle, optionally setting its ore.

        The rectangle is clipped to the map, so a caller may pass coordinates
        that hang off an edge and get the overlapping part. A rectangle
        entirely outside the map writes nothing and still returns ``self``.
        This is the one placement method that does not raise on a bad
        coordinate.

        Parameters
        ----------
        x
            Left column, inclusive.
        y
            Top row, inclusive.
        w
            Width in tiles. Zero or negative fills nothing.
        h
            Height in tiles. Zero or negative fills nothing.
        block
            Block type to write across the region.
        resources
            Ore units for every tile in the region. When omitted the tiles
            keep whatever the resource array already held, and if no resource
            array exists yet none is created, which leaves the amounts to
            :func:`default_resources` at build time.

        Returns
        -------
        LevelBuilder
            ``self``, so calls chain.
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
            # The array is a snapshot, so newly painted ore needs its default
            # written now or the tile would keep the zero it was created with.
            self._block_resources[y0:y1, x0:x1] = default_resources(
                self._block_map[y0:y1, x0:x1],
            )
        return self

    def set_resources(self, x: int, y: int, amount: int) -> LevelBuilder:
        """Set the ore left in one tile.

        Creating the resource array is what makes it explicit, so a level that
        calls this once carries a full array from then on and
        :func:`build_state` stops deriving amounts from the block map. Every
        other tile keeps its default, and ore painted later still picks one up,
        because :meth:`fill_rect` keeps the array in step.

        Parameters
        ----------
        x
            Column.
        y
            Row.
        amount
            Ore units to store. Not clamped, so a value above
            ``BLOCK_MAX_RESOURCES`` stands, and a non-ore tile can be given a
            count that no machine will ever mine.

        Returns
        -------
        LevelBuilder
            ``self``, so calls chain.

        Raises
        ------
        IndexError
            The tile is outside the map.
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

        Contents are recorded per item with no slot, so this says what a
        machine holds but not where. :func:`build_state` decides the slot when
        it builds the entity, and for a combiner it uses the recipe book to
        tell a finished output from an input.

        Nothing checks that a machine stands here. Stock set on an empty tile
        is carried through save and load and then dropped at build time.

        A machine with one buffer, which is everything but an assembler or a
        furnace, holds one item kind. Recording a second is allowed here and
        refused by :func:`build_state`, because the kind on the tile can change
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
            Number of items. Replaces any count already set for this item on
            this tile rather than adding to it, and is not capped, so a value
            above the machine's stack limit is clamped later.

        Returns
        -------
        LevelBuilder
            ``self``, so calls chain.

        Raises
        ------
        IndexError
            The tile is outside the map.
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
        """Stand a machine on a tile, facing a direction.

        Placing over an occupied tile replaces what was there. The block
        underneath is untouched, so a machine can be placed on water or ore
        even where the engine's own placement rules would refuse it.

        Parameters
        ----------
        x
            Column.
        y
            Row.
        machine_type
            :class:`~factoriax.engine.constants.Machine` value.
        direction
            :class:`~factoriax.engine.constants.Direction` value. The default
            0 means unset, which is what a machine that does not care about
            facing should keep. A belt or miner left at 0 moves nothing.

        Returns
        -------
        LevelBuilder
            ``self``, so calls chain.

        Raises
        ------
        IndexError
            The tile is outside the map.
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
        """Append a player spawn.

        Call order is player order: the first call is player 0. There is no
        way to set one player's spawn without setting every earlier player's,
        and no call removes one.

        Setting any spawn commits the level to setting all of them:
        :func:`build_state` refuses a level whose spawn count does not match
        the number of players it is asked for.

        Parameters
        ----------
        x
            Column.
        y
            Row.

        Returns
        -------
        LevelBuilder
            ``self``, so calls chain.

        Raises
        ------
        IndexError
            The position is outside the map.
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
        """Give every player the same starting inventory.

        Replaces any inventory set by an earlier call rather than adding to
        it. Repeating an item within one list does add up, since
        :func:`build_state` accumulates the pairs as it reads them.

        Parameters
        ----------
        items
            ``(item, count)`` pairs. An empty list leaves players
            empty-handed, which is also what never calling this does.

        Returns
        -------
        LevelBuilder
            ``self``, so calls chain.
        """
        self._player_inventory = list(items)
        return self

    def build(self, name: str) -> Level:
        """Close the chain and return the finished level.

        Arrays are copied on the way out, so the builder stays usable and a
        later edit does not reach into the level just returned. A field no
        method touched stays ``None`` rather than becoming an empty array,
        which is what lets :func:`build_state` tell "unset" from "empty".

        Parameters
        ----------
        name
            Identifier for the level. Not checked for uniqueness against
            :data:`LEVELS`.

        Returns
        -------
        Level
            A level whose shapes are validated by ``Level.__post_init__``.
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
    """Fill every ore tile with a full deposit and everything else with none.

    Ore means anything in ``factoriax.engine.tables.MINEABLE_BLOCKS``, the same
    set the procedural path tests against, so a hand-built level and a
    generated one agree on which tiles carry something to mine.

    Parameters
    ----------
    block_map
        Block type per tile, shape ``(H, W)``.

    Returns
    -------
    np.ndarray
        Shape ``(H, W)``, int32. Ore tiles hold ``BLOCK_MAX_RESOURCES`` and
        the rest hold 0. A fresh array; the input is not modified.
    """
    mineable = np.isin(block_map, np.asarray(MINEABLE_BLOCKS))
    return np.where(mineable, BLOCK_MAX_RESOURCES, 0).astype(np.int32)


def _place_players(
    block_map: np.ndarray, num_players: int
) -> tuple[np.ndarray, np.ndarray]:
    """Spread players along the middle row and clear the tiles they land on.

    Each spawn is forced to DIRT, so a player never starts inside water, ore,
    or a wall. That edit is the point of returning a block map rather than
    just positions.

    Parameters
    ----------
    block_map
        Block type per tile, shape ``(H, W)``. Not modified; a copy is
        returned.
    num_players
        How many to place. Positions are packed around the centre column, so
        on a map narrower than the player count they clamp to the edge and
        several players share a tile.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        The edited block map, shape ``(H, W)``, and the spawns as ``(x, y)``
        rows, shape ``(num_players, 2)``, int32.
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
    """Load a single-buffer machine's stock, refusing a load it cannot hold."""
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

    This is where the level's per-tile view becomes the engine's per-entity
    view. Each occupied tile gets one entry in the entity arrays, and
    ``tile_entity`` records the join. Everything the level does not describe
    starts empty: no craft is in progress, no items have been mined, and no
    achievement is unlocked.

    Where the level leaves a field unset the default applies, so a bare block
    map still yields a playable world. Where it does set one, the level wins
    and is taken literally, which is why the mismatches below raise instead of
    being papered over.

    Parameters
    ----------
    level
        World to build from. Not modified.
    num_players
        How many players to spawn. Decides the length of every player array,
        and must match ``level.player_positions`` when that is set.
    max_machines
        Entity table size. The default 0 means "choose one", which is
        ``max(64, tiles // 4)``, enough for a quarter of the map. A level
        cannot place more machines than this.

    Returns
    -------
    EnvState
        A state whose ``timestep`` is 0 and whose players all face DOWN.

    Raises
    ------
    ValueError
        The level places more machines than ``max_machines`` allows, lists a
        number of spawns other than ``num_players``, or gives a single-buffer
        machine more than one kind of item. Each case describes a world the
        state cannot represent, so none of them is silently trimmed.

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
        # Ensure each spawn tile is walkable.
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
    ent_asm_out_type_np = np.zeros(mm, dtype=np.int8)
    ent_asm_out_count_np = np.zeros(mm, dtype=np.int16)

    machine_inv = level.machine_inventory

    # Refuse a level that outgrows the entity table. Truncating instead would
    # leave machine_types naming machines that no entity backs.
    machine_count = int(np.count_nonzero(machine_types_np != int(Machine.NONE)))
    if machine_count > mm:
        raise ValueError(
            f"Level {level.name!r} places {machine_count} machines but "
            f"max_machines is {mm}. Raise max_machines or remove machines.",
        )

    # Populate entities from grid (Python loop, only at build time)
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

            # Populate inventory from level data.
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
                    # Item-indexed inventory carries no slot, so an item this
                    # machine's own recipes produce is its finished output and
                    # everything else is an input. Without the split, an output
                    # would fill an input slot and push a real input out of the
                    # two the engine has.
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

    The procedural counterpart to :func:`build_state`. It takes a finished
    block map rather than a :class:`Level`, so there is nothing to place: the
    entity table is allocated and left empty, and the world starts bare.

    Ore is derived here from ``MINEABLE_BLOCKS`` and
    ``params.base_resources``, which is the same set :func:`default_resources`
    uses but a different amount, so a generated tile and a hand-built one can
    hold different quantities of the same ore.

    Parameters
    ----------
    world_map
        Block type per tile, shape ``(H, W)``. Spawn tiles are overwritten
        with DIRT, and the edited copy is what the state carries.
    params
        Read for ``base_resources`` only.
    num_players
        How many players to spawn, along the middle row as in
        :func:`_place_players`.
    max_machines
        Entity table size. 0 means ``max(64, tiles // 4)``.

    Returns
    -------
    EnvState
        A state with an empty entity table, ``timestep`` 0, and every player
        facing DOWN with an empty inventory.
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
# generate_terrain is the only algorithm. It draws one smooth-noise field per
# resource and tests each against its own probability. The draws are
# independent, but the results are layered silicon first and water last, and
# each layer overwrites the one before. Only water, applied last, lands on the
# share its probability names; everything earlier keeps just the part no later
# layer claimed, so silicon realises about half its figure. Accepted as is.
# ---------------------------------------------------------------------------


def _smooth_noise(
    rng: jax.Array,
    height: int,
    width: int,
    scale: int = 4,
) -> jax.Array:
    """Draw a field of smooth blobs whose values are spread evenly over [0, 1).

    Two properties matter to the caller and they pull against each other.
    Neighbouring tiles hold close values, which is what makes patches instead
    of speckle. Across the whole field the values are spread evenly, so
    ``field < p`` selects a fraction ``p`` of the map for any ``p``, and a
    probability in :class:`~factoriax.engine.state.EnvParams` means what it
    says.

    Parameters
    ----------
    rng
        Key for the draw. The same key and shape give the same field.
    height, width
        Output size in tiles.
    scale
        Blob size. The grid drawn before upscaling is
        ``height // scale + 2`` by ``width // scale + 2``, so a larger scale
        means fewer, broader patches.

    Returns
    -------
    jax.Array
        Shape ``(height, width)``, float32, values in (0, 1). Evenly spread by
        construction rather than on average, so even one small map divides
        cleanly at any threshold.
    """
    lo_h = height // scale + 2
    lo_w = width // scale + 2
    lo = random.uniform(rng, (lo_h, lo_w))
    hi = jax.image.resize(lo, (height, width), method="bilinear")
    # Replace each value by its rank. Interpolating uniform samples piles them
    # up around the mean, and rescaling that range to [0, 1] leaves the pile in
    # place, so a threshold of 0.1 used to select far less than a tenth of the
    # map. Ranking flattens the distribution instead of just its bounds. It is
    # monotonic, so the blobs keep their shape and only their values move.
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
    draws are independent. The results are then layered in a fixed order,
    silicon first and water last, and each layer overwrites the one before.
    Only water lands on the share its probability names. Every earlier layer
    keeps just the tiles no later layer claimed, so silicon, drawn first,
    realises about half its figure. Dirt is whatever no layer claimed.

    Parameters
    ----------
    rng
        Key for the draw. Split six ways, one per block type, so the same key
        gives the same map.
    params
        Read for the six ``*_probability`` fields. Each is the share of the
        map that block would cover on its own, before later layers overwrite
        it.
    map_height, map_width
        Map size in tiles.

    Returns
    -------
    jax.Array
        Block type per tile, shape ``(map_height, map_width)``, int32. Holds
        terrain only: no machines, no players, no ore amounts.
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
    """Generate a random world and return its state.

    A convenience wrapper for tests and scripts. It builds a
    :class:`~factoriax.engine.envs.base.FactoriaxEnv` and resets it, so the
    result matches what an environment would produce rather than assembling
    one here. That import is deferred to the call because ``envs.base``
    imports this module.

    Not traceable. It constructs an environment from Python integers, so it
    cannot be placed under ``jax.jit``; only the reset it delegates to is
    traceable.

    Parameters
    ----------
    rng
        Key for the terrain draw. The same key gives the same world.
    params
        Passed to the reset, which reads the terrain probabilities from it.
    map_height, map_width
        Map dimensions in tiles.
    num_players
        How many players to spawn.
    max_machines
        Entity table size. 0 means ``max(64, tiles // 4)``.

    Returns
    -------
    EnvState
        A state on freshly generated terrain, with no machines standing.
    """
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
    """Write a level to a JSON file.

    Arrays become nested integer lists and the output is indented, so a level
    can be read and edited by hand. Every optional field is written, as
    ``null`` where unset, though :func:`load_level` accepts a file that omits
    one entirely.

    The format carries no version marker, so a future change to the field set
    has no way to announce itself to an older reader.

    Parameters
    ----------
    level
        Level to write. Not modified.
    path
        Destination. Missing parent directories are created, and an existing
        file is overwritten without warning.

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
    """Deserialize a :class:`Level` from a JSON file written by :func:`save_level`.

    Only ``name``, ``map_width``, ``map_height``, and ``block_map`` are
    required. Every other field may be ``null`` or absent, both meaning unset,
    so a hand-written file need only carry what it wants to say.

    Arrays come back as int32 and position lists as tuples, which is what
    :class:`LevelBuilder` produces, so a level survives a round trip
    unchanged. The shapes are checked by ``Level.__post_init__``, so a file
    whose arrays disagree with its dimensions fails here.

    Parameters
    ----------
    path
        File to read.

    Returns
    -------
    Level
        The level the file describes.

    Raises
    ------
    KeyError
        A required field is missing.
    ValueError
        An array disagrees with the declared dimensions.

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
    # Only name, dimensions, and the block map are required. Every other field
    # is optional on Level, so a file may omit it as readily as write null.
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

#: 15x15 map with 4x4 ore patches in three corners and dirt elsewhere.
#: Coal: top-left. Copper: top-right. Iron: bottom-left.
_15X15_RESOURCES: Level = (
    LevelBuilder(15, 15)
    .fill_rect(0, 0, 4, 4, BlockType.COAL, resources=BLOCK_MAX_RESOURCES)
    .fill_rect(11, 0, 4, 4, BlockType.COPPER, resources=BLOCK_MAX_RESOURCES)
    .fill_rect(0, 11, 4, 4, BlockType.IRON, resources=BLOCK_MAX_RESOURCES)
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
    name
        Key into :data:`LEVELS`.

    Returns
    -------
    Level
        The registered level itself, not a copy. It is shared with every other
        caller and with :data:`LEVELS`, so mutating its arrays changes the
        level for the whole process. :func:`build_state` only reads it.

    Raises
    ------
    KeyError
        No level is registered under that name. The message lists the names
        that are.

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
