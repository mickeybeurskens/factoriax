"""Derived JAX arrays and lookup tables for the FactoriaX engine.

:mod:`factoriax.engine.constants` holds the pure-Python definitions. This module
projects them into the ``jnp`` arrays engine code reads, and every array is
built once at import.

The arrays come in two shapes. A gather table is sized over the full range of
whatever indexes it, an ``ItemType``, a ``Machine``, a ``BlockType``, or a
``Direction``, and holds a neutral value at every position with no entry. A
traced index therefore never falls outside it and a lookup needs no membership
test first. A membership set, such as :data:`MINEABLE_BLOCKS` or
:data:`SOLID_BLOCKS`, holds only the values that belong and is compared against
rather than indexed.

Both shapes keep the lookups traced code needs branchless under ``jax.jit``.
"""

from __future__ import annotations

import jax.numpy as jnp

from factoriax.engine.constants import (
    BLOCK_TO_ITEM,
    ITEM_TO_MACHINE,
    NUM_ITEM_TYPES,
    PLACEABLE_ITEM_LIST,
    SCIENCE_PACK_TYPES,
    BlockType,
    Direction,
    ItemType,
    Machine,
)

#: ``(dx, dy)`` step per :class:`~factoriax.engine.constants.Direction` value.
#: Shape ``(5, 2)``, int32. Row 0 is ``(0, 0)`` for the value ``Direction``
#: leaves unnumbered, so a machine with no direction set moves nothing. ``dy``
#: grows downward, matching the row order of ``EnvState.map``: ``UP`` is
#: ``(0, -1)``.
DIRECTIONS = jnp.array(
    [
        [0, 0],
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
    ],
    dtype=jnp.int32,
)

# ---------------------------------------------------------------------------
# Splitter and crossing direction decoding
# ---------------------------------------------------------------------------
#
# ``EnvState.ent_direction`` holds one int8 per entity. A conveyor belt reads it
# as a plain facing, but a splitter pushes to two sides at once and a crossing
# carries two independent flows, so both need the byte unpacked into a pair of
# Direction values. The tables below do that unpacking as a constant array
# indexed by the byte, which keeps the belt tick in
# ``factoriax.engine.machines.run_conveyor_belts`` branchless and traceable
# under jax.jit. Each reserves row 0 for the value Direction leaves unnumbered,
# which stored state uses to mean "no direction".

#: Output sides of a splitter, indexed by ``ent_direction``. Row ``d`` holds the
#: two :class:`~factoriax.engine.constants.Direction` values perpendicular to
#: facing ``d``, which are the sides a splitter facing ``d`` pushes to. Shape
#: ``(5, 2)``, dtype int8. A splitter stores the direction items travel through
#: it, the same convention a conveyor belt uses: a splitter facing ``d`` takes
#: items from the neighbour on its ``-d`` side. The two entries sit in ascending
#: ``Direction`` order and the order carries no meaning, since a splitter treats
#: both sides alike. Row 0 is ``(0, 0)``, a pair that matches no direction, so an
#: unset entity never reads as pushing to a real side.
SPLITTER_PERP_OUTPUTS = jnp.array(
    [
        [0, 0],  # NONE
        [int(Direction.UP), int(Direction.DOWN)],  # facing LEFT  -> outputs N + S
        [int(Direction.UP), int(Direction.DOWN)],  # facing RIGHT -> outputs N + S
        [int(Direction.LEFT), int(Direction.RIGHT)],  # facing UP   -> outputs W + E
        [int(Direction.LEFT), int(Direction.RIGHT)],  # facing DOWN -> outputs W + E
    ],
    dtype=jnp.int8,
)

#: Output direction of each crossing axis, indexed by the packed
#: ``ent_direction``. Row ``e`` is ``(vertical_output, horizontal_output)`` for
#: encoding ``e`` in 1..4. Shape ``(5, 2)``, dtype int8. A crossing runs a
#: vertical flow and a horizontal flow at the same time and packs both into the
#: one direction byte. An axis takes items from the neighbour opposite its
#: output direction, so ``vertical_output == DOWN`` means the vertical flow
#: reads the tile above and writes the tile below. The two input sides therefore
#: always meet at a corner. Row 0 is ``(0, 0)``: a crossing with an unset
#: direction moves nothing on either axis.
CROSSING_AXIS_DIRS = jnp.array(
    [
        [0, 0],  # NONE / unset
        [int(Direction.DOWN), int(Direction.RIGHT)],  # 1: N->S + W->E
        [int(Direction.DOWN), int(Direction.LEFT)],  # 2: N->S + E->W
        [int(Direction.UP), int(Direction.RIGHT)],  # 3: S->N + W->E
        [int(Direction.UP), int(Direction.LEFT)],  # 4: S->N + E->W
    ],
    dtype=jnp.int8,
)

#: Diagonal glyph per crossing encoding, for drawing only. A backslash marks a
#: NW-SE diagonal, a forward slash a NE-SW one. The diagonal runs from the corner
#: where the encoding's two input sides meet to the corner across from it. Index
#: 0 is the empty string, because an unset crossing has no input sides. The
#: simulation never reads this value; only the crossing icon does.
CROSSING_DIAGONAL: tuple[str, ...] = (
    "",  # NONE
    "\\",  # 1: input pair (N, W)
    "/",  # 2: input pair (N, E)
    "/",  # 3: input pair (S, W)
    "\\",  # 4: input pair (S, E)
)

#: Slot index of the vertical flow buffer in ``EnvState.ent_asm_in_type`` and
#: ``EnvState.ent_asm_in_count``. A crossing reuses the two assembler input
#: slots as one buffer per axis, which is why the two flows cannot mix: each
#: axis only ever reads and writes its own slot.
CROSSING_VERT_SLOT: int = 0
#: Slot index of the horizontal flow buffer, in the same two arrays as
#: :data:`CROSSING_VERT_SLOT`.
CROSSING_HORIZ_SLOT: int = 1


#: The mineable block types, as a set to test membership against rather than a
#: table to index. These are the keys of
#: :data:`~factoriax.engine.constants.BLOCK_TO_ITEM`; mining any other block
#: yields nothing.
MINEABLE_BLOCKS = jnp.array([int(b) for b in BLOCK_TO_ITEM], dtype=jnp.int32)
#: The block types that nothing can stand on or build on, likewise a membership
#: set. Both :func:`factoriax.engine.step.is_position_walkable` and
#: :func:`factoriax.engine.placement.is_valid_placement_tile` read it, so the
#: two agree on what is solid.
SOLID_BLOCKS = jnp.array([BlockType.WATER, BlockType.OUT_OF_BOUNDS], dtype=jnp.int32)

_block_to_item = [int(ItemType.EMPTY)] * len(BlockType)
for _block, _item in BLOCK_TO_ITEM.items():
    _block_to_item[int(_block)] = int(_item)
#: Item each block yields when mined, indexed by ``BlockType``. Shape
#: ``(len(BlockType),)``, int32. A block that is not mineable holds
#: ``ItemType.EMPTY``, so a mine action on it produces item 0 and the caller
#: needs no separate check.
BLOCK_TO_ITEM_ARRAY = jnp.array(_block_to_item, dtype=jnp.int32)

# Built in one pass so the forward and inverse arrays cannot disagree.
_item_to_machine = [int(Machine.NONE)] * NUM_ITEM_TYPES
_machine_to_item = [int(ItemType.EMPTY)] * len(Machine)
for _it, _machine in ITEM_TO_MACHINE.items():
    _item_to_machine[int(_it)] = int(_machine)
    _machine_to_item[int(_machine)] = int(_it)
#: Machine an item becomes when placed, indexed by ``ItemType``. Shape
#: ``(NUM_ITEM_TYPES,)``, int32. An item that is not placeable holds
#: ``Machine.NONE``, which is how placement rejects it.
ITEM_TO_MACHINE_ARRAY = jnp.array(_item_to_machine, dtype=jnp.int32)
#: Item a machine returns when picked up, indexed by ``Machine`` value. Shape
#: ``(len(Machine),)``, int32. The inverse of :data:`ITEM_TO_MACHINE_ARRAY`,
#: with ``ItemType.EMPTY`` at ``Machine.NONE``.
MACHINE_TO_ITEM_ARRAY = jnp.array(_machine_to_item, dtype=jnp.int32)

#: The placeable item ids, as a membership set. Same contents as
#: :data:`~factoriax.engine.constants.PLACEABLE_ITEM_LIST`, in the same order,
#: which is the play UI's palette order and carries no meaning here.
PLACEABLE_ITEMS = jnp.array(PLACEABLE_ITEM_LIST, dtype=jnp.int32)

_science_pack_index = [-1] * NUM_ITEM_TYPES
for _pos, _pack in enumerate(SCIENCE_PACK_TYPES):
    _science_pack_index[_pack] = _pos
#: Position of an item in
#: :data:`~factoriax.engine.constants.SCIENCE_PACK_TYPES`, indexed by
#: ``ItemType``. Shape ``(NUM_ITEM_TYPES,)``, int8. An item that is not a
#: science pack holds ``-1``. The step function uses the position as a segment
#: id when it totals ``EnvState.science_consumed_step``.
SCIENCE_PACK_INDEX = jnp.array(_science_pack_index, dtype=jnp.int8)

# Machines and the large rocket components are bulky, so a stack holds fewer.
_DEFAULT_PLAYER_STACK = 1024
_BULKY_PLAYER_STACK = 128
_PLAYER_STACK_OVERRIDES: dict[int, int] = {
    int(ItemType.EMPTY): 0,
    int(ItemType.CONVEYOR_BELT): _BULKY_PLAYER_STACK,
    int(ItemType.MINER): _BULKY_PLAYER_STACK,
    int(ItemType.ASSEMBLER): _BULKY_PLAYER_STACK,
    int(ItemType.PALLET): _BULKY_PLAYER_STACK,
    int(ItemType.ARM): _BULKY_PLAYER_STACK,
    int(ItemType.ROCKET): _BULKY_PLAYER_STACK,
    int(ItemType.FURNACE): _BULKY_PLAYER_STACK,
    int(ItemType.HULL): _BULKY_PLAYER_STACK,
    int(ItemType.ENGINE_UNIT): _BULKY_PLAYER_STACK,
    int(ItemType.AVIONICS): _BULKY_PLAYER_STACK,
    int(ItemType.ROCKET_CORE): _BULKY_PLAYER_STACK,
    int(ItemType.SCIENCE_LAB): _BULKY_PLAYER_STACK,
    int(ItemType.SPLITTER): _BULKY_PLAYER_STACK,
    int(ItemType.CROSSING): _BULKY_PLAYER_STACK,
}
#: Items of one type a player can carry, indexed by ``ItemType``. Shape
#: ``(NUM_ITEM_TYPES,)``, int32. Counted in items, not stacks. ``EMPTY`` holds
#: 0, so nothing accumulates in the slot that marks an empty one. A craft or
#: pickup that would pass the cap is refused rather than clipped: the craft's
#: inputs stay in the inventory, and the machine stays on the map.
PLAYER_MAX_STACK = jnp.array(
    [
        _PLAYER_STACK_OVERRIDES.get(i, _DEFAULT_PLAYER_STACK)
        for i in range(NUM_ITEM_TYPES)
    ],
    dtype=jnp.int32,
)

# ---------------------------------------------------------------------------
# Per-machine capacities
# ---------------------------------------------------------------------------
#
# Both arrays below are indexed by Machine value and are built by iterating
# Machine, so a kind added to the enum without an entry here raises KeyError at
# import rather than silently reading a neighbour's row.

# Items one buffer slot holds, per machine kind.
_MACHINE_BUFFER_STACK: dict[Machine, int] = {
    Machine.NONE: 0,
    Machine.MINER: 64,
    Machine.PALLET: 256,
    Machine.CONVEYOR_BELT: 3,
    Machine.ASSEMBLER: 1000,
    Machine.ARM: 1,
    Machine.ROCKET: 0,
    Machine.FURNACE: 1000,
    Machine.SCIENCE_LAB: 1000,
    Machine.SPLITTER: 2,
    Machine.CROSSING: 2,
}

#: Items one buffer slot holds, indexed by ``Machine`` value. Shape
#: ``(len(Machine),)``, int16 to match ``EnvState.ent_buf_count``. Counted in
#: items rather than stacks. Zero means the machine holds nothing, and
#: ``ARM`` holds one because it carries a single item in transit. Transfers in
#: :mod:`factoriax.engine.machines` compare a destination's count against this
#: and refuse a move into a full buffer, leaving the source untouched.
MACHINE_MAX_STACK = jnp.array(
    [_MACHINE_BUFFER_STACK[m] for m in Machine],
    dtype=jnp.int16,
)

#: Hit points a machine is placed with. The same for every machine kind;
#: :data:`MACHINE_MAX_HEALTH` spreads it over the ``Machine`` range so a caller
#: can index it with a machine type without special-casing.
MACHINE_HEALTH: int = 256

#: Hit points a placed machine starts with, indexed by ``Machine`` value. Shape
#: ``(len(Machine),)``, int16 to match ``EnvState.ent_health``. Repairs clamp to
#: it, and :mod:`factoriax.engine.placement` only allows a pickup at full health.
MACHINE_MAX_HEALTH = jnp.full(len(Machine), MACHINE_HEALTH, dtype=jnp.int16)
