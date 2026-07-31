"""Derived JAX arrays and lookup tables for the FactoriaX engine.

:mod:`factoriax.engine.constants` holds the pure-Python definitions. This module
turns them into the ``jnp`` arrays that engine code reads. It builds every
array once, at import.

The arrays have two shapes. A gather table covers the full range of the value
that indexes it: an ``ItemType``, a ``Machine``, a ``BlockType``, or a
``Direction``. It holds a neutral value at every position that has no entry. A
traced index therefore never falls outside it, and a lookup needs no membership
test first. A membership set, such as :data:`MINEABLE_BLOCKS` or
:data:`SOLID_BLOCKS`, holds only the values that belong to the set. Code
compares a value against such an array and never indexes it.

Both shapes keep the lookups of traced code branchless under ``jax.jit``.
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
#: leaves unnumbered, so a machine with no direction moves nothing. ``dy``
#: increases downward, which matches the row order of ``EnvState.map``. ``UP``
#: is therefore ``(0, -1)``.
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
# ``EnvState.ent_direction`` holds one int8 for each entity. A conveyor belt
# reads it as a plain facing. A splitter pushes to two sides at the same time,
# and a crossing carries two separate flows, so both need a pair of Direction
# values out of that one byte. The tables below hold those pairs as constant
# arrays, indexed by the byte. The belt tick in
# ``factoriax.engine.machines.run_conveyor_belts`` therefore stays branchless
# and traceable under jax.jit. Each table keeps row 0 for the value Direction
# leaves unnumbered, which stored state uses for "no direction".

#: Output sides of a splitter, indexed by ``ent_direction``. Row ``d`` holds the
#: two :class:`~factoriax.engine.constants.Direction` values that are
#: perpendicular to facing ``d``, and a splitter with facing ``d`` pushes to
#: those two sides. Shape ``(5, 2)``, dtype int8. A splitter stores the
#: direction in which items move through it, the same convention that a conveyor
#: belt uses. A splitter with facing ``d`` therefore takes items from the
#: neighbour on its ``-d`` side. The two entries are in ascending ``Direction``
#: order, and that order carries no meaning, because a splitter treats both
#: sides in the same way. Row 0 is ``(0, 0)``, a pair that matches no direction,
#: so an entity with no direction never pushes to a real side.
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
#: vertical flow and a horizontal flow at the same time, and packs both into the
#: one direction byte. An axis takes items from the neighbour opposite to its
#: output direction. If ``vertical_output`` is ``DOWN``, the vertical flow reads
#: the tile above and writes the tile below. The two input sides therefore
#: always meet at a corner. Row 0 is ``(0, 0)``. A crossing with no direction
#: moves nothing on either axis.
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

#: Diagonal glyph for each crossing encoding, for the display only. A backslash
#: marks a NW-SE diagonal. A forward slash marks a NE-SW diagonal. The diagonal
#: runs from the corner where the two input sides of the encoding meet, to the
#: opposite corner. Index 0 is the empty string, because a crossing with no
#: direction has no input sides. The simulation never reads this value. Only the
#: crossing icon reads it.
CROSSING_DIAGONAL: tuple[str, ...] = (
    "",  # NONE
    "\\",  # 1: input pair (N, W)
    "/",  # 2: input pair (N, E)
    "/",  # 3: input pair (S, W)
    "\\",  # 4: input pair (S, E)
)

#: Slot index of the vertical flow buffer in ``EnvState.ent_asm_in_type`` and
#: ``EnvState.ent_asm_in_count``. A crossing uses the two assembler input slots
#: as one buffer for each axis. Each axis reads and writes its own slot only, so
#: the two flows cannot mix.
CROSSING_VERT_SLOT: int = 0
#: Slot index of the horizontal flow buffer, in the same two arrays as
#: :data:`CROSSING_VERT_SLOT`.
CROSSING_HORIZ_SLOT: int = 1


#: The block types that a player can mine, as a membership set and not as a
#: table to index. These are the keys of
#: :data:`~factoriax.engine.constants.BLOCK_TO_ITEM`. Any other block gives
#: nothing.
MINEABLE_BLOCKS = jnp.array([int(b) for b in BLOCK_TO_ITEM], dtype=jnp.int32)
#: The block types that hold no player and no machine, also a membership set.
#: Both :func:`factoriax.engine.step.is_position_walkable` and
#: :func:`factoriax.engine.placement.is_valid_placement_tile` read it, so the
#: two agree on what is solid.
SOLID_BLOCKS = jnp.array([BlockType.WATER, BlockType.OUT_OF_BOUNDS], dtype=jnp.int32)

_block_to_item = [int(ItemType.EMPTY)] * len(BlockType)
for _block, _item in BLOCK_TO_ITEM.items():
    _block_to_item[int(_block)] = int(_item)
#: Item that each block gives to the player who mines it, indexed by
#: ``BlockType``. Shape ``(len(BlockType),)``, int32. A block that a player
#: cannot mine holds ``ItemType.EMPTY``. A mine action on it therefore gives
#: item 0, and the caller needs no separate test.
BLOCK_TO_ITEM_ARRAY = jnp.array(_block_to_item, dtype=jnp.int32)

# One pass builds both, so the forward and inverse arrays cannot disagree.
_item_to_machine = [int(Machine.NONE)] * NUM_ITEM_TYPES
_machine_to_item = [int(ItemType.EMPTY)] * len(Machine)
for _it, _machine in ITEM_TO_MACHINE.items():
    _item_to_machine[int(_it)] = int(_machine)
    _machine_to_item[int(_machine)] = int(_it)
#: Machine that an item becomes on the map, indexed by ``ItemType``. Shape
#: ``(NUM_ITEM_TYPES,)``, int32. An item that no action can place holds
#: ``Machine.NONE``, and that is how placement refuses it.
ITEM_TO_MACHINE_ARRAY = jnp.array(_item_to_machine, dtype=jnp.int32)
#: Item that a machine gives back after a pickup, indexed by ``Machine`` value.
#: Shape ``(len(Machine),)``, int32. This is the inverse of
#: :data:`ITEM_TO_MACHINE_ARRAY`, with ``ItemType.EMPTY`` at ``Machine.NONE``.
MACHINE_TO_ITEM_ARRAY = jnp.array(_machine_to_item, dtype=jnp.int32)

#: The placeable item ids, as a membership set. The contents and the order are
#: the same as :data:`~factoriax.engine.constants.PLACEABLE_ITEM_LIST`. That
#: order is the palette order of the play UI and carries no meaning here.
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

# Machines and the large rocket parts are bulky, so a stack holds fewer of them.
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
#: Number of items of one type that a player can carry, indexed by ``ItemType``.
#: Shape ``(NUM_ITEM_TYPES,)``, int32. The unit is items, not stacks. ``EMPTY``
#: holds 0, so no count can build up in the slot that marks an empty slot. The
#: engine refuses a craft or a pickup that passes this limit, and does not clip
#: it. The inputs of the craft stay in the inventory, and the machine stays on
#: the map.
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
# A Machine value indexes both arrays below, and both walk Machine to fill
# themselves. A new kind with no entry here therefore raises KeyError at
# import, and never reads the row of its neighbour.

# Items that one buffer slot holds, for each machine kind.
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

#: Items that one buffer slot holds, indexed by ``Machine`` value. Shape
#: ``(len(Machine),)``, int16 to match ``EnvState.ent_buf_count``. The unit is
#: items, not stacks. Zero means that the machine holds nothing. ``ARM`` holds
#: one, because it carries a single item in transit. A transfer in
#: :mod:`factoriax.engine.machines` compares the count at the destination
#: against this value. It refuses a move into a full buffer and leaves the
#: source unchanged.
MACHINE_MAX_STACK = jnp.array(
    [_MACHINE_BUFFER_STACK[m] for m in Machine],
    dtype=jnp.int16,
)

# Machines that take deliveries into the two ``ent_asm_in`` slots and not into
# ``ent_buf``: assemblers, furnaces, and science labs.
_MACHINE_HAS_INPUT_SLOTS: dict[Machine, bool] = {
    Machine.NONE: False,
    Machine.MINER: False,
    Machine.PALLET: False,
    Machine.CONVEYOR_BELT: False,
    Machine.ASSEMBLER: True,
    Machine.ARM: False,
    Machine.ROCKET: False,
    Machine.FURNACE: True,
    Machine.SCIENCE_LAB: True,
    Machine.SPLITTER: False,
    Machine.CROSSING: False,
}

#: Whether a machine takes deliveries into ``ent_asm_in``, indexed by
#: ``Machine`` value. Shape ``(len(Machine),)``, bool. The value is True for
#: assemblers, furnaces, and science labs, and that is the one property those
#: three share. An assembler and a furnace consume the slots to run a recipe. A
#: science lab consumes them directly, in
#: :func:`factoriax.engine.step.run_labs`.
#:
#: Every path into those slots reads this array, and no path lists the three
#: machine kinds again. A new kind with input slots therefore needs one change
#: only. This array does not tell a caller whether a machine runs recipes,
#: because a science lab does not run recipes.
#: :mod:`factoriax.engine.machines` tests ``ASSEMBLER`` and ``FURNACE``
#: directly for that question.
#:
#: A ``CROSSING`` is False here, although it keeps its two axis buffers in the
#: same two columns. Nothing delivers into a crossing by slot. The belt pass
#: selects the axis from the direction of travel.
MACHINE_HAS_INPUT_SLOTS = jnp.array(
    [_MACHINE_HAS_INPUT_SLOTS[m] for m in Machine],
    dtype=jnp.bool_,
)

#: Hit points of a machine at the moment of placement. The value is the same for
#: every machine kind. :data:`MACHINE_MAX_HEALTH` repeats it over the
#: ``Machine`` range, so a caller can index it with a machine type and needs no
#: special case.
MACHINE_HEALTH: int = 256

#: Hit points of a placed machine at the start, indexed by ``Machine`` value.
#: Shape ``(len(Machine),)``, int16 to match ``EnvState.ent_health``. A repair
#: clamps to this value, and :mod:`factoriax.engine.placement` allows a pickup
#: only at full health.
MACHINE_MAX_HEALTH = jnp.full(len(Machine), MACHINE_HEALTH, dtype=jnp.int16)
