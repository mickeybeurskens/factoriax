"""Observation constructors for FactoriaX.

Each function maps ``(state, params, player_idx) -> jax.Array``. Each one is
JAX-native and works under JIT. ``rgb`` is the exception. It returns a NumPy RGB
image from the pixel renderer, and JIT cannot compile it.

Two profiles and two view extents give four top-level builders.

- **x_ray** sees through every machine to its slot contents, and sees the ore
  left under each terrain tile. It has 10 spatial channels and 81 player
  scalars.
- **superficial** sees only what is visible from outside. It has 3 spatial
  channels and 72 player scalars, with no view into a slot and no view of the
  tile in front.

Each profile pairs with a view extent. ``global_*`` flattens the whole map.
``local_*`` takes a ``(2r+1) x (2r+1)`` window with the selected player at the
centre.

``global_*`` keeps the map in absolute coordinates, and gives the position of
the observing player as two scalars only. It fits a function approximator that
uses the spatial channels, such as a CNN over the unflattened grid. The
training code must add a player-position plane for that CNN. A flat MLP usually
cannot find the player in this encoding. The ``local_*`` variants are
egocentric and translation-invariant, and they fit an MLP better. With
``radius >= map size - 1`` the window still covers the whole map, and the
``OUT_OF_BOUNDS`` padding marks the edges, so the absolute position stays
readable. One measurement: on Mining-v1, PPO with a 64x64 MLP stops near 7/30
on ``superficial_global`` and reaches 30/30 on ``superficial_local``.

The 10 x_ray spatial channels:

- ``block_type``, the terrain.
- ``machine_type``, the ``Machine`` on each tile, or NONE.
- ``block_resources``, the ore count under the tile.
- ``slot{0,1,2}_type`` and ``slot{0,1,2}_count``, a uniform 3-slot view of the
  contents of every machine. Slots 0 and 1 carry the two ``ent_asm_in``
  columns, which covers the inputs of an assembler, the packs of a science lab,
  and the two axis buffers of a crossing. Slot 2 carries ``ent_asm_out`` where
  a recipe runs, and ``ent_buf`` for every other machine that holds items.
- ``machine_direction``, the ``ent_direction`` of each tile, 1 to 4 or 0.

The 3 superficial spatial channels: ``block_type``, ``machine_type``, and
``machine_direction``.

The x_ray scalars are the pose, the affordability of each item, the player
inventory, and the readouts of the machine in front: 81 floats. The superficial
scalars are the pose, the affordability of each item, and the player inventory:
72 floats. The 9 facing readouts are absent.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from factoriax.engine.constants import (
    BLOCK_MAX_RESOURCES,
    NUM_ITEM_TYPES,
    BlockType,
    Machine,
)
from factoriax.engine.crafting import can_afford_recipe
from factoriax.engine.placement import get_tile_in_front
from factoriax.engine.renderer import JaxRenderer
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.tables import (
    MACHINE_HAS_INPUT_SLOTS,
    MACHINE_MAX_STACK,
    PLAYER_MAX_STACK,
)

# JaxRenderer holds one texture atlas on the device for each tile size. Keep
# one renderer for each ``block_pixel_size``, so the JIT compile and the atlas
# build happen on the first call for that size only.
_RENDERER_CACHE: dict[int, JaxRenderer] = {}

_MAP_NORM: float = float(max(BlockType))
_MACHINE_NORM: float = float(max(Machine))
_DIR_NORM: float = 4.0
# Count normalisation for the slot channels. The value is large enough to hold
# a pallet count of up to 1000, and small enough that a small buffer still
# shows. 1024 matches PLAYER_MAX_STACK for the common items.
_SLOT_COUNT_NORM: float = 1024.0
_PLAYER_MAX_STACK_F: jnp.ndarray = jnp.maximum(
    PLAYER_MAX_STACK.astype(jnp.float32),
    1.0,
)


_SlotGrids = tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]


def _reconstruct_slot_grids(state: EnvState) -> _SlotGrids:
    """Put the contents of every machine onto uniform 3-slot grids.

    Machines keep their contents in different fields, and an observation needs
    one shape for all of them. An assembler and a furnace map ``ent_asm_in[0]``
    to slot 0, ``ent_asm_in[1]`` to slot 1, and ``ent_asm_out`` to slot 2. A
    machine that only stores leaves slots 0 and 1 empty and puts ``ent_buf`` in
    slot 2.

    Slots 0 and 1 carry the contents of the two ``ent_asm_in`` columns. They
    therefore cover the packs of a science lab and the two axis buffers of a
    crossing, as well as the inputs of an assembler. Slot 2 carries
    ``ent_asm_out`` where a recipe runs, and ``ent_buf`` for every other
    machine that holds items.

    Parameters
    ----------
    state
        State to read.

    Returns
    -------
    tuple of jnp.ndarray
        ``(slot0_type, slot0_count, slot1_type, slot1_count, slot2_type,
        slot2_count)``, each with the shape ``(H, W)``. A type is an int8 item
        id and a count is an int16. Both are zero on a tile with no machine.
        These are raw values, and the callers divide them.
    """
    h, w = state.map.shape
    active = state.ent_y >= 0
    ey = jnp.clip(state.ent_y, 0, h - 1)
    ex = jnp.clip(state.ent_x, 0, w - 1)

    machine_idx = state.ent_type.astype(jnp.int32)
    # Every machine that stores items in the two ``ent_asm_in`` columns: the
    # machines with real input slots, and a crossing, which keeps one buffer
    # for each axis in the same two columns.
    uses_in_columns = MACHINE_HAS_INPUT_SLOTS[machine_idx] | (
        state.ent_type == Machine.CROSSING
    )
    # Only an assembler and a furnace produce into ``ent_asm_out``. Every other
    # machine that holds items holds them in ``ent_buf``.
    runs_recipes = (state.ent_type == Machine.ASSEMBLER) | (
        state.ent_type == Machine.FURNACE
    )
    has_buffer = ~uses_in_columns & (MACHINE_MAX_STACK[machine_idx] > 0)

    # Slots 0 and 1: the two ent_asm_in columns, for every use of them.
    in0_t = jnp.where(
        active & uses_in_columns,
        state.ent_asm_in_type[..., 0],
        jnp.int8(0),
    )
    in0_c = jnp.where(
        active & uses_in_columns,
        state.ent_asm_in_count[..., 0],
        jnp.int16(0),
    )
    in1_t = jnp.where(
        active & uses_in_columns,
        state.ent_asm_in_type[..., 1],
        jnp.int8(0),
    )
    in1_c = jnp.where(
        active & uses_in_columns,
        state.ent_asm_in_count[..., 1],
        jnp.int16(0),
    )

    # Slot 2: asm_out where a recipe runs, and ent_buf in every other case.
    out_t = jnp.where(
        active & runs_recipes,
        state.ent_asm_out_type,
        jnp.where(active & has_buffer, state.ent_buf_type, jnp.int8(0)),
    )
    out_c = jnp.where(
        active & runs_recipes,
        state.ent_asm_out_count,
        jnp.where(active & has_buffer, state.ent_buf_count, jnp.int16(0)),
    )

    zero_t = jnp.zeros((h, w), dtype=jnp.int8)
    zero_c = jnp.zeros((h, w), dtype=jnp.int16)
    # Add, and do not assign. Every free slot clips onto tile (0, 0) and adds
    # zero, so an add leaves a real machine on that tile unchanged. A set
    # competes with the stale writes and usually loses.
    return (
        zero_t.at[ey, ex].add(in0_t),
        zero_c.at[ey, ex].add(in0_c),
        zero_t.at[ey, ex].add(in1_t),
        zero_c.at[ey, ex].add(in1_c),
        zero_t.at[ey, ex].add(out_t),
        zero_c.at[ey, ex].add(out_c),
    )


def _reconstruct_machine_direction_grid(state: EnvState) -> jnp.ndarray:
    """Put the facing of each machine onto a per-tile grid.

    The facing decides where a machine sends its output. An agent therefore
    needs it to read a factory as a chain, and not as a set of separate tiles.

    The function adds and does not assign, for the reason that
    :func:`_reconstruct_slot_grids` gives.

    Parameters
    ----------
    state
        State to read.

    Returns
    -------
    jnp.ndarray
        Shape ``(H, W)``, int8. ``Direction`` values 1 to 4, and 0 on a tile
        with no machine. These are raw values.
    """
    h, w = state.map.shape
    grid = jnp.zeros((h, w), dtype=jnp.int8)
    active = state.ent_y >= 0
    ey = jnp.clip(state.ent_y, 0, h - 1)
    ex = jnp.clip(state.ent_x, 0, w - 1)
    vals = jnp.where(active, state.ent_direction, jnp.int8(0))
    # Add, for the reason that ``_reconstruct_slot_grids`` gives.
    return grid.at[ey, ex].add(vals)


# The channel names and scalar names of each profile.
_X_RAY_SPATIAL_CHANNEL_NAMES: tuple[str, ...] = (
    "block_type",
    "machine_type",
    "block_resources",
    "slot0_type",
    "slot0_count",
    "slot1_type",
    "slot1_count",
    "slot2_type",
    "slot2_count",
    "machine_direction",
)
_SUPERFICIAL_SPATIAL_CHANNEL_NAMES: tuple[str, ...] = (
    "block_type",
    "machine_type",
    "machine_direction",
)
NUM_SPATIAL_CHANNELS: dict[str, int] = {
    "x_ray": len(_X_RAY_SPATIAL_CHANNEL_NAMES),
    "superficial": len(_SUPERFICIAL_SPATIAL_CHANNEL_NAMES),
}

# The scalar fields of each helper, in vector order.
_COMMON_SCALAR_FIELDS: tuple[str, ...] = (
    "pos_x",
    "pos_y",
    "direction",
    "timestep",
    *(f"afford_{i}" for i in range(NUM_ITEM_TYPES)),
    *(f"player_inv_{i}" for i in range(NUM_ITEM_TYPES)),
)
_FACING_SCALAR_FIELDS: tuple[str, ...] = (
    "facing_machine_type",
    "facing_buffer_type",
    "facing_buffer_count",
    "facing_asm_in0_type",
    "facing_asm_in0_count",
    "facing_asm_in1_type",
    "facing_asm_in1_count",
    "facing_asm_out_type",
    "facing_asm_out_count",
)
#: Number of scalars in each profile. The number of recipes in the
#: :class:`~factoriax.engine.recipes.RecipeTable` of a scenario does not change
#: it, because an item type indexes the ``afford`` block, not a recipe id.
NUM_PLAYER_SCALARS: dict[str, int] = {
    "x_ray": len(_COMMON_SCALAR_FIELDS) + len(_FACING_SCALAR_FIELDS),
    "superficial": len(_COMMON_SCALAR_FIELDS),
}


def _common_scalars(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
) -> jax.Array:
    """Build the scalar block that both observation profiles share.

    The block holds four pose and clock values, then one affordability flag for
    each item type, then the inventory of the player. The total length is
    ``NUM_PLAYER_SCALARS["superficial"]`` floats, which is 72 today.

    An item type indexes the affordability block, not a recipe id. A scenario
    supplies its own recipe book, and a recipe id changes between books while
    an item id does not. An item that no recipe in the active book produces
    reads 0.

    Every value normalises to about 0 to 1. The function divides a position by
    the map size, a direction by 4, the timestep by ``max_timesteps``, and an
    inventory count by the ``PLAYER_MAX_STACK`` of that item.

    Parameters
    ----------
    state
        State to read.
    params
        Supplies ``max_timesteps`` and ``recipe_table``.
    player_idx
        Player that observes.

    Returns
    -------
    jax.Array
        1-D float32, of length ``NUM_PLAYER_SCALARS["superficial"]``.
    """
    pos = state.player_positions[player_idx]
    _h, _w = state.map.shape
    pose_time = jnp.array(
        [
            pos[0] / _w,
            pos[1] / _h,
            state.player_directions[player_idx] / _DIR_NORM,
            state.timestep / params.max_timesteps,
        ],
        dtype=jnp.float32,
    )
    # An item indexes this, not a recipe. A recipe id changes with the book.
    recipe_for_item = params.recipe_table.output_to_recipe
    afford = jax.vmap(
        lambda r: can_afford_recipe(state, params, player_idx, r).astype(
            jnp.float32,
        ),
    )(jnp.maximum(recipe_for_item, 0))
    afford = jnp.where(recipe_for_item >= 0, afford, 0.0)
    inv = state.player_inventory[player_idx].astype(jnp.float32) / _PLAYER_MAX_STACK_F
    return jnp.concatenate([pose_time, afford, inv])


def _facing_scalars(
    state: EnvState,
    player_idx: int | jax.Array,
) -> jax.Array:
    """Read the contents of the machine in front of the player, for x_ray only.

    The block holds nine floats: the machine kind, then its buffer and its
    three craft slots, each as a type and a count. An x_ray agent reads these
    to choose between a deposit and a withdraw, and needs no memory of the
    machine.

    A tile with no machine reads zero on every field. This holds both for a
    tile outside the map and for an empty tile, because the mask covers both.

    The function divides a count by ``_SLOT_COUNT_NORM``, which is equal to or
    larger than the ``MACHINE_MAX_STACK`` of every machine. The values
    therefore stay inside 0 to 1 and match the bounds that
    :meth:`FactoriaxEnv.observation_space` declares.

    Parameters
    ----------
    state
        State to read.
    player_idx
        Player that observes.

    Returns
    -------
    jax.Array
        1-D float32 of length 9, in ``_FACING_SCALAR_FIELDS`` order.
    """
    tx, ty = get_tile_in_front(state, player_idx)
    map_h, map_w = state.map.shape
    in_bounds = (tx >= 0) & (tx < map_w) & (ty >= 0) & (ty < map_h)
    sx = jnp.clip(tx, 0, map_w - 1)
    sy = jnp.clip(ty, 0, map_h - 1)
    mask = in_bounds.astype(jnp.float32)

    max_e = state.ent_y.shape[0]
    eidx_raw = state.tile_entity[sy, sx]
    # A tile inside the map with no machine holds -1, and the clip turns that
    # into entity 0. Mask on the presence of an entity, and not on the bounds
    # alone. Without that mask every empty tile reports the contents of slot 0.
    mask = (in_bounds & (eidx_raw >= 0)).astype(jnp.float32)
    eidx = jnp.clip(eidx_raw, 0, max_e - 1)
    item_norm = float(NUM_ITEM_TYPES)

    return jnp.array(
        [
            state.machine_types[sy, sx].astype(jnp.float32) * mask / _MACHINE_NORM,
            state.ent_buf_type[eidx].astype(jnp.float32) * mask / item_norm,
            state.ent_buf_count[eidx].astype(jnp.float32) * mask / _SLOT_COUNT_NORM,
            state.ent_asm_in_type[eidx, 0].astype(jnp.float32) * mask / item_norm,
            state.ent_asm_in_count[eidx, 0].astype(jnp.float32)
            * mask
            / _SLOT_COUNT_NORM,
            state.ent_asm_in_type[eidx, 1].astype(jnp.float32) * mask / item_norm,
            state.ent_asm_in_count[eidx, 1].astype(jnp.float32)
            * mask
            / _SLOT_COUNT_NORM,
            state.ent_asm_out_type[eidx].astype(jnp.float32) * mask / item_norm,
            state.ent_asm_out_count[eidx].astype(jnp.float32) * mask / _SLOT_COUNT_NORM,
        ]
    )


def _x_ray_scalars(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
) -> jax.Array:
    """Build the x_ray scalar block: the shared block and the facing readouts.

    Parameters
    ----------
    state
        State to read.
    params
        The function passes this to :func:`_common_scalars`.
    player_idx
        Player that observes.

    Returns
    -------
    jax.Array
        1-D float32, of length ``NUM_PLAYER_SCALARS["x_ray"]``, which is 81
        today: 72 shared values, and then 9 facing readouts.
    """
    return jnp.concatenate(
        [
            _common_scalars(state, params, player_idx),
            _facing_scalars(state, player_idx),
        ]
    )


def _superficial_scalars(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
) -> jax.Array:
    """Build the superficial scalar block, which is the shared block alone.

    The facing readouts are absent on purpose. A superficial agent must move to
    a machine and act on it to learn its contents.

    Parameters
    ----------
    state
        State to read.
    params
        The function passes this to :func:`_common_scalars`.
    player_idx
        Player that observes.

    Returns
    -------
    jax.Array
        1-D float32, of length ``NUM_PLAYER_SCALARS["superficial"]``, which is
        72 today.
    """
    return _common_scalars(state, params, player_idx)


def observation_size(
    *,
    profile: str,
    view: str,
    radius: int = 7,
    map_height: int = 32,
    map_width: int = 32,
) -> int:
    """Return the flat observation size for one profile, view, and map size."""
    if view == "global":
        spatial_tiles = map_width * map_height
    elif view == "local":
        side = 2 * radius + 1
        spatial_tiles = side * side
    else:
        raise ValueError(f"view must be 'global' or 'local'; got {view!r}")
    return NUM_SPATIAL_CHANNELS[profile] * spatial_tiles + NUM_PLAYER_SCALARS[profile]


def global_x_ray(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
) -> jax.Array:
    """Full-map x_ray observation for one player.

    The result holds ten flat spatial channels, and then the x_ray scalar
    vector. The channels are the block type, the machine type, the ore
    resources, the six slot channels, and the machine direction.

    Parameters
    ----------
    state
        State to read.
    params
        The function passes this to the scalar builders.
    player_idx
        Player that observes. Only the scalar block depends on it. The spatial
        channels are the same for every player.

    Returns
    -------
    jax.Array
        1-D float32, of length
        ``observation_size(profile="x_ray", view="global", ...)``. It holds ten
        flat ``(H, W)`` channels in ``_X_RAY_SPATIAL_CHANNEL_NAMES`` order, and
        then the x_ray scalars. A reader that wants the grid back must reshape
        it.
    """
    flat_blocks = state.map.flatten().astype(jnp.float32) / _MAP_NORM
    flat_machines = state.machine_types.flatten().astype(jnp.float32) / _MACHINE_NORM
    flat_resources = state.block_resources.flatten().astype(jnp.float32) / float(
        BLOCK_MAX_RESOURCES
    )
    s0t, s0c, s1t, s1c, s2t, s2c = _reconstruct_slot_grids(state)
    item_norm = float(NUM_ITEM_TYPES)
    flat_s0t = s0t.flatten().astype(jnp.float32) / item_norm
    flat_s0c = s0c.flatten().astype(jnp.float32) / _SLOT_COUNT_NORM
    flat_s1t = s1t.flatten().astype(jnp.float32) / item_norm
    flat_s1c = s1c.flatten().astype(jnp.float32) / _SLOT_COUNT_NORM
    flat_s2t = s2t.flatten().astype(jnp.float32) / item_norm
    flat_s2c = s2c.flatten().astype(jnp.float32) / _SLOT_COUNT_NORM
    dir_grid = _reconstruct_machine_direction_grid(state)
    flat_direction = dir_grid.flatten().astype(jnp.float32) / _DIR_NORM
    spatial = jnp.concatenate(
        [
            flat_blocks,
            flat_machines,
            flat_resources,
            flat_s0t,
            flat_s0c,
            flat_s1t,
            flat_s1c,
            flat_s2t,
            flat_s2c,
            flat_direction,
        ],
    )
    return jnp.concatenate(
        [spatial, _x_ray_scalars(state, params, player_idx)],
    )


def local_x_ray(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
    radius: int = 10,
) -> jax.Array:
    """Local x_ray observation in a window around one player.

    The function takes a ``(2*radius+1) x (2*radius+1)`` patch from the ten
    x_ray spatial channels, and adds the x_ray scalar vector after it.

    Parameters
    ----------
    state
        State to read.
    params
        The function passes this to the scalar builders.
    player_idx
        Player that observes. This player is at the centre of the window.
    radius
        Half-width of the window in tiles. The window is
        ``(2 * radius + 1)`` tiles square, so it always has a centre tile. The
        value is static under ``jit``, and a new value traces again.

    Returns
    -------
    jax.Array
        1-D float32, of length
        ``observation_size(profile="x_ray", view="local", radius=radius)``. The
        padding for a tile past the map edge is ``BlockType.OUT_OF_BOUNDS`` on
        the terrain channel and zero on every other channel, so an agent can
        still find the edge.
    """
    size = 2 * radius + 1
    pw = ((radius, radius), (radius, radius))

    padded_map = (
        jnp.pad(state.map, pw, constant_values=BlockType.OUT_OF_BOUNDS).astype(
            jnp.float32
        )
        / _MAP_NORM
    )
    padded_machines = (
        jnp.pad(state.machine_types, pw, constant_values=Machine.NONE).astype(
            jnp.float32
        )
        / _MACHINE_NORM
    )
    padded_resources = jnp.pad(state.block_resources, pw, constant_values=0).astype(
        jnp.float32
    ) / float(BLOCK_MAX_RESOURCES)
    s0t, s0c, s1t, s1c, s2t, s2c = _reconstruct_slot_grids(state)
    item_norm = float(NUM_ITEM_TYPES)
    padded_s0t = jnp.pad(s0t, pw, constant_values=0).astype(jnp.float32) / item_norm
    padded_s0c = (
        jnp.pad(s0c, pw, constant_values=0).astype(jnp.float32) / _SLOT_COUNT_NORM
    )
    padded_s1t = jnp.pad(s1t, pw, constant_values=0).astype(jnp.float32) / item_norm
    padded_s1c = (
        jnp.pad(s1c, pw, constant_values=0).astype(jnp.float32) / _SLOT_COUNT_NORM
    )
    padded_s2t = jnp.pad(s2t, pw, constant_values=0).astype(jnp.float32) / item_norm
    padded_s2c = (
        jnp.pad(s2c, pw, constant_values=0).astype(jnp.float32) / _SLOT_COUNT_NORM
    )
    dir_grid = _reconstruct_machine_direction_grid(state)
    padded_direction = (
        jnp.pad(dir_grid, pw, constant_values=0).astype(jnp.float32) / _DIR_NORM
    )

    pos = state.player_positions[player_idx]
    start = (pos[1], pos[0])
    slice_shape = (size, size)

    spatial = jnp.concatenate(
        [
            jax.lax.dynamic_slice(padded_map, start, slice_shape).ravel(),
            jax.lax.dynamic_slice(padded_machines, start, slice_shape).ravel(),
            jax.lax.dynamic_slice(padded_resources, start, slice_shape).ravel(),
            jax.lax.dynamic_slice(padded_s0t, start, slice_shape).ravel(),
            jax.lax.dynamic_slice(padded_s0c, start, slice_shape).ravel(),
            jax.lax.dynamic_slice(padded_s1t, start, slice_shape).ravel(),
            jax.lax.dynamic_slice(padded_s1c, start, slice_shape).ravel(),
            jax.lax.dynamic_slice(padded_s2t, start, slice_shape).ravel(),
            jax.lax.dynamic_slice(padded_s2c, start, slice_shape).ravel(),
            jax.lax.dynamic_slice(padded_direction, start, slice_shape).ravel(),
        ]
    )
    return jnp.concatenate(
        [spatial, _x_ray_scalars(state, params, player_idx)],
    )


def global_superficial(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
) -> jax.Array:
    """Full-map superficial observation for one player.

    The result holds three flat spatial channels, and then the superficial
    scalar vector. The channels are the block type, the machine type, and the
    machine direction.

    Parameters
    ----------
    state
        State to read.
    params
        The function passes this to the scalar builders.
    player_idx
        Player that observes.

    Returns
    -------
    jax.Array
        1-D float32, of length
        ``observation_size(profile="superficial", view="global", ...)``. It
        holds three flat ``(H, W)`` channels, and then the superficial scalars.
    """
    flat_blocks = state.map.flatten().astype(jnp.float32) / _MAP_NORM
    flat_machines = state.machine_types.flatten().astype(jnp.float32) / _MACHINE_NORM
    flat_direction = (
        _reconstruct_machine_direction_grid(state).flatten().astype(jnp.float32)
        / _DIR_NORM
    )
    spatial = jnp.concatenate([flat_blocks, flat_machines, flat_direction])
    return jnp.concatenate(
        [spatial, _superficial_scalars(state, params, player_idx)],
    )


def local_superficial(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
    radius: int = 10,
) -> jax.Array:
    """Local superficial observation in a window around one player.

    The function takes a ``(2*radius+1) x (2*radius+1)`` patch from the three
    superficial spatial channels, and adds the superficial scalar vector after
    it. That vector holds ``NUM_PLAYER_SCALARS["superficial"]`` floats.

    Parameters
    ----------
    state
        State to read.
    params
        The function passes this to the scalar builders.
    player_idx
        Player that observes. This player is at the centre of the window.
    radius
        Half-width of the window in tiles. The value is static under ``jit``.

    Returns
    -------
    jax.Array
        1-D float32, of length
        ``observation_size(profile="superficial", view="local",
        radius=radius)``. The padding is the same as in :func:`local_x_ray`.
    """
    size = 2 * radius + 1
    pw = ((radius, radius), (radius, radius))

    padded_map = (
        jnp.pad(state.map, pw, constant_values=BlockType.OUT_OF_BOUNDS).astype(
            jnp.float32
        )
        / _MAP_NORM
    )
    padded_machines = (
        jnp.pad(state.machine_types, pw, constant_values=Machine.NONE).astype(
            jnp.float32
        )
        / _MACHINE_NORM
    )
    padded_direction = (
        jnp.pad(
            _reconstruct_machine_direction_grid(state), pw, constant_values=0
        ).astype(jnp.float32)
        / _DIR_NORM
    )

    pos = state.player_positions[player_idx]
    start = (pos[1], pos[0])
    slice_shape = (size, size)

    spatial = jnp.concatenate(
        [
            jax.lax.dynamic_slice(padded_map, start, slice_shape).ravel(),
            jax.lax.dynamic_slice(padded_machines, start, slice_shape).ravel(),
            jax.lax.dynamic_slice(padded_direction, start, slice_shape).ravel(),
        ]
    )
    return jnp.concatenate(
        [spatial, _superficial_scalars(state, params, player_idx)],
    )


#: The observation variants. The environment takes the name, and a radius for a
#: ``_local`` variant, and dispatches through this dict.
OBSERVATIONS: dict[str, jax.Array] = {  # type: ignore[type-arg]
    "x_ray_global": global_x_ray,
    "x_ray_local": local_x_ray,
    "superficial_global": global_superficial,
    "superficial_local": local_superficial,
}


def rgb(state: EnvState, block_pixel_size: int = 32) -> np.ndarray:
    """Render the full map as an RGB image.

    This is the one observation here that is not JAX-native. It returns a NumPy
    array, and JAX cannot trace it, so a caller must run it outside a ``jit``
    boundary.

    The image shows less than the symbolic profiles carry: the terrain, the
    machine kind and facing, and the player position and facing. There is no
    inventory, no craft affordability, and no machine contents.

    CAUTION: A pixel agent cannot learn to craft from this observation alone.
    ``ISSUES.md`` records this.

    Parameters
    ----------
    state
        State to render.
    block_pixel_size
        Tile side length in pixels. Each different value builds its own
        renderer and holds it in a cache, so the atlas build and the JIT
        compile happen one time for each size, and not one time for each call.
        Many different sizes therefore grow the cache without bound.

    Returns
    -------
    numpy.ndarray
        Shape ``(H * block_pixel_size, W * block_pixel_size, 3)``, uint8 RGB.
    """
    renderer = _RENDERER_CACHE.get(block_pixel_size)
    if renderer is None:
        renderer = JaxRenderer(tile_px=block_pixel_size)
        _RENDERER_CACHE[block_pixel_size] = renderer
    return np.asarray(renderer.jit_render_map(state))
