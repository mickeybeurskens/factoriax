"""Derived JAX arrays for the FactoriaX engine.

:mod:`factoriax.constants` holds pure-Python definitions; this is the JAX layer
that projects them into the jnp gather tables and state-array dtypes the engine
uses. It imports only ``constants`` and is imported by the engine modules, so
it sits one layer above ``constants`` and below everything else.
"""

from __future__ import annotations

import jax.numpy as jnp

from factoriax.constants import (
    BLOCK_TO_ITEM,
    ITEM_TO_MACHINE,
    NUM_ITEM_TYPES,
    PLACEABLE_ITEM_LIST,
    SCIENCE_PACK_TYPES,
    BlockType,
    ItemType,
    Machine,
)

# State-array dtypes.
MACHINE_INVENTORY_COUNT_DTYPE = jnp.int16
BLOCK_RESOURCE_DTYPE = jnp.int16

# (dx, dy) offset per Direction value; index 0 is the unused NONE slot.
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

# Mineable ore blocks (the BLOCK_TO_ITEM keys) and impassable blocks.
MINEABLE_BLOCKS = jnp.array([int(b) for b in BLOCK_TO_ITEM], dtype=jnp.int32)
SOLID_BLOCKS = jnp.array([BlockType.WATER, BlockType.OUT_OF_BOUNDS], dtype=jnp.int32)

# Block -> mined item, over the full BlockType range, EMPTY where not mineable.
_block_to_item = [int(ItemType.EMPTY)] * len(BlockType)
for _block, _item in BLOCK_TO_ITEM.items():
    _block_to_item[int(_block)] = int(_item)
BLOCK_TO_ITEM_ARRAY = jnp.array(_block_to_item, dtype=jnp.int32)

# Item <-> machine, built in one pass so forward and inverse cannot disagree.
# Each is indexed over its full enum range; absent slots hold the NONE/EMPTY id.
_item_to_machine = [int(Machine.NONE)] * NUM_ITEM_TYPES
_machine_to_item = [int(ItemType.EMPTY)] * len(Machine)
for _it, _machine in ITEM_TO_MACHINE.items():
    _item_to_machine[int(_it)] = int(_machine)
    _machine_to_item[int(_machine)] = int(_it)
ITEM_TO_MACHINE_ARRAY = jnp.array(_item_to_machine, dtype=jnp.int32)
MACHINE_TO_ITEM_ARRAY = jnp.array(_machine_to_item, dtype=jnp.int32)

# Placeable item membership.
PLACEABLE_ITEMS = jnp.array(PLACEABLE_ITEM_LIST, dtype=jnp.int32)

# ItemType -> position in SCIENCE_PACK_TYPES, -1 for non-pack items.
_science_pack_index = [-1] * NUM_ITEM_TYPES
for _pos, _pack in enumerate(SCIENCE_PACK_TYPES):
    _science_pack_index[_pack] = _pos
SCIENCE_PACK_INDEX = jnp.array(_science_pack_index, dtype=jnp.int8)

# Per-item player inventory stack caps; machines and large rocket components
# are bulky, so the player carries fewer per stack.
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
PLAYER_MAX_STACK = jnp.array(
    [
        _PLAYER_STACK_OVERRIDES.get(i, _DEFAULT_PLAYER_STACK)
        for i in range(NUM_ITEM_TYPES)
    ],
    dtype=jnp.int32,
)
