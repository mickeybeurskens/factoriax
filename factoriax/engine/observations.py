"""Observation constructors for FactoriaX.

Each function maps ``(state, params, player_idx) -> jax.Array`` and is
JAX-native and JIT-compatible. ``rgb`` is the exception: it returns a
NumPy RGB image via the pixel renderer and cannot be JIT'd.

Two profiles compose four top-level builders:

- **x_ray** sees through every machine to its slot contents and the ore
  remaining under each terrain tile (10 spatial channels + 81 player
  scalars).
- **superficial** sees only what's outwardly visible (3 spatial channels
  + 72 player scalars; no slot peek, no facing-tile peek).

Each profile pairs with a view extent: ``global_*`` flattens the whole
map; ``local_*`` extracts a ``(2r+1) x (2r+1)`` window centred on the
selected player.

Choosing a view extent: ``global_*`` keeps the map in absolute
coordinates and encodes the observing player's position only as two
scalars — it is built for function approximators that exploit the
spatial channels (e.g. a CNN over the unflattened grid, ideally with a
player-position plane added by the training code). Flat MLPs generally
fail to localize the player in this encoding. The egocentric
``local_*`` variants are translation-invariant and are the better fit
for MLPs; with ``radius >= map size - 1`` the window still covers the
whole map (OUT_OF_BOUNDS padding marks the edges, so absolute position
stays recoverable). Reference point: on Mining-v1, PPO with a 64x64
MLP plateaus near 7/30 on ``superficial_global`` but reaches 30/30 on
``superficial_local``.

x_ray spatial channels (10):

- ``block_type`` — terrain.
- ``machine_type`` — ``Machine`` at each tile (or NONE).
- ``block_resources`` — ore count under the tile.
- ``slot{0,1,2}_type`` / ``slot{0,1,2}_count`` — a uniform 3-slot
  projection of every machine's contents. Slots 0 and 1 carry the two
  ``ent_asm_in`` columns, which covers an assembler's inputs, a science
  lab's packs, and a crossing's two axis buffers. Slot 2 carries
  ``ent_asm_out`` where a recipe runs and ``ent_buf`` for every other
  machine that holds items.
- ``machine_direction`` — ``ent_direction`` per tile (1..4 or 0).

superficial spatial channels (3): ``block_type``, ``machine_type``,
``machine_direction``.

x_ray scalars: pose + per-item affordability + player inventory +
facing-machine readouts (81 floats).
superficial scalars: pose + per-item affordability + player inventory
(72 floats; the 9 facing readouts are dropped).
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

# JaxRenderer caches device-resident texture atlases per tile size; keep
# one renderer per requested ``block_pixel_size`` so the JIT compile and
# atlas build only happen on the first call for that size.
_RENDERER_CACHE: dict[int, JaxRenderer] = {}

_MAP_NORM: float = float(max(BlockType))
_MACHINE_NORM: float = float(max(Machine))
_DIR_NORM: float = 4.0
# Count normalisation for slot channels. Big enough to keep pallet
# counts (up to 1000) finite, small enough that tiny buffers still
# register visibly. 1024 matches PLAYER_MAX_STACK for common items.
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
    """Project every machine's contents onto uniform 3-slot grids.

    Machines store their contents in different fields, and an observation
    needs one shape for all of them. Assemblers and furnaces map
    ``ent_asm_in[0]`` to slot 0, ``ent_asm_in[1]`` to slot 1, and
    ``ent_asm_out`` to slot 2. Machines that only store leave slots 0 and 1
    empty and put ``ent_buf`` in slot 2.

    Slots 0 and 1 carry whatever sits in the two ``ent_asm_in`` columns, so
    they cover a science lab's packs and a crossing's two axis buffers as
    well as an assembler's inputs. Slot 2 carries ``ent_asm_out`` where a
    recipe runs and ``ent_buf`` for every other machine that holds items.

    Parameters
    ----------
    state
        State to read.

    Returns
    -------
    tuple of jnp.ndarray
        ``(slot0_type, slot0_count, slot1_type, slot1_count, slot2_type,
        slot2_count)``, each shaped ``(H, W)``. Types are int8 item ids and
        counts are int16, both zero on a tile with no machine. Raw values,
        not normalised; the callers divide.
    """
    h, w = state.map.shape
    active = state.ent_y >= 0
    ey = jnp.clip(state.ent_y, 0, h - 1)
    ex = jnp.clip(state.ent_x, 0, w - 1)

    machine_idx = state.ent_type.astype(jnp.int32)
    # Anything that stores items in the two ``ent_asm_in`` columns: the
    # machines with real input slots, plus a crossing, which keeps one buffer
    # per axis in the same two columns.
    uses_in_columns = MACHINE_HAS_INPUT_SLOTS[machine_idx] | (
        state.ent_type == Machine.CROSSING
    )
    # Only assemblers and furnaces produce into ``ent_asm_out``; everything
    # else that holds items holds them in ``ent_buf``.
    runs_recipes = (state.ent_type == Machine.ASSEMBLER) | (
        state.ent_type == Machine.FURNACE
    )
    has_buffer = ~uses_in_columns & (MACHINE_MAX_STACK[machine_idx] > 0)

    # Slots 0 and 1: the two ent_asm_in columns, whatever they are used for.
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

    # Slot 2: asm_out where a recipe runs, ent_buf everywhere else.
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
    # Accumulate rather than assign. Free slots clip onto tile (0, 0) and
    # contribute zero, so an add leaves a real machine standing there intact,
    # where a set would race with the stale writes and usually lose.
    return (
        zero_t.at[ey, ex].add(in0_t),
        zero_c.at[ey, ex].add(in0_c),
        zero_t.at[ey, ex].add(in1_t),
        zero_c.at[ey, ex].add(in1_c),
        zero_t.at[ey, ex].add(out_t),
        zero_c.at[ey, ex].add(out_c),
    )


def _reconstruct_machine_direction_grid(state: EnvState) -> jnp.ndarray:
    """Lay out each machine's facing on a per-tile grid.

    Facing decides where a machine sends its output, so an agent needs it to
    read a factory as a chain rather than a set of unrelated tiles.

    Accumulates rather than assigns, for the reason
    :func:`_reconstruct_slot_grids` gives.

    Parameters
    ----------
    state
        State to read.

    Returns
    -------
    jnp.ndarray
        Shape ``(H, W)``, int8. ``Direction`` values 1 to 4, and 0 on a tile
        with no machine. Raw, not normalised.
    """
    h, w = state.map.shape
    grid = jnp.zeros((h, w), dtype=jnp.int8)
    active = state.ent_y >= 0
    ey = jnp.clip(state.ent_y, 0, h - 1)
    ex = jnp.clip(state.ent_x, 0, w - 1)
    vals = jnp.where(active, state.ent_direction, jnp.int8(0))
    # Accumulate, for the reason given in ``_reconstruct_slot_grids``.
    return grid.at[ey, ex].add(vals)


# Channel and scalar manifests, per profile.
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

# Scalar fields, in vector order, per helper.
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
#: Scalar count per profile. Independent of how many recipes a scenario's
#: :class:`~factoriax.engine.recipes.RecipeTable` holds, because the ``afford``
#: block is indexed by item type rather than by recipe id.
NUM_PLAYER_SCALARS: dict[str, int] = {
    "x_ray": len(_COMMON_SCALAR_FIELDS) + len(_FACING_SCALAR_FIELDS),
    "superficial": len(_COMMON_SCALAR_FIELDS),
}


def _common_scalars(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
) -> jax.Array:
    """Build the scalar block both observation profiles share.

    Four pose and clock values, then one affordability flag per item type,
    then the player's inventory. ``NUM_PLAYER_SCALARS["superficial"]``
    floats in total, currently 72.

    Affordability is indexed by item type rather than by recipe id, because
    a scenario ships its own recipe book and recipe ids shift between books
    while item ids do not. An item no recipe in the active book produces
    reads 0.

    Everything is normalised into roughly 0 to 1: positions by map size,
    direction by 4, timestep by ``max_timesteps``, and inventory by
    ``PLAYER_MAX_STACK`` per item.

    Parameters
    ----------
    state
        State to read.
    params
        Supplies ``max_timesteps`` and ``recipe_table``.
    player_idx
        Which player is observing.

    Returns
    -------
    jax.Array
        1-D float32, length ``NUM_PLAYER_SCALARS["superficial"]``.
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
    # Item-indexed, not recipe-indexed: recipe ids shift per scenario book.
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
    """Read the contents of the machine the player faces. x_ray only.

    Nine floats: the machine kind, then its buffer and its three crafting
    slots as type and count pairs. This is what lets an x_ray agent decide
    whether to deposit or withdraw without walking a machine's history.

    A tile with no machine reads zero on every field, whether it is off the
    map or simply empty. Both cases are masked.

    Counts are normalised by ``_SLOT_COUNT_NORM``, which is at or above every
    machine's ``MACHINE_MAX_STACK``, so these stay inside 0 to 1 and match
    the bounds :meth:`FactoriaxEnv.observation_space` declares.

    Parameters
    ----------
    state
        State to read.
    player_idx
        Which player is observing.

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
    # An in-bounds tile with no machine holds -1, which the clip would turn
    # into entity 0. Mask on there being an entity, not only on bounds, or
    # every empty tile reports slot 0's contents.
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
    """Build the x_ray scalar block: the shared block plus facing readouts.

    Parameters
    ----------
    state
        State to read.
    params
        Passed through to :func:`_common_scalars`.
    player_idx
        Which player is observing.

    Returns
    -------
    jax.Array
        1-D float32, length ``NUM_PLAYER_SCALARS["x_ray"]``, currently 81:
        72 shared values followed by 9 facing readouts.
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

    The facing readouts are dropped on purpose. A superficial agent has to
    walk up to a machine and act to learn what is inside it.

    Parameters
    ----------
    state
        State to read.
    params
        Passed through to :func:`_common_scalars`.
    player_idx
        Which player is observing.

    Returns
    -------
    jax.Array
        1-D float32, length ``NUM_PLAYER_SCALARS["superficial"]``,
        currently 72.
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
    """Flat observation size for ``(profile, view)`` with the given map dimensions."""
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

    Ten spatial channels (block type, machine type, ore resources, six
    slot channels, machine direction) flattened, followed by the x_ray scalar
    vector.

    Parameters
    ----------
    state
        State to read.
    params
        Passed through to the scalar builders.
    player_idx
        Which player is observing. Only the scalar block depends on it; the
        spatial channels are the same for every player.

    Returns
    -------
    jax.Array
        1-D float32 of length
        ``observation_size(profile="x_ray", view="global", ...)``. Ten
        flattened ``(H, W)`` channels in ``_X_RAY_SPATIAL_CHANNEL_NAMES``
        order, then the x_ray scalars. A consumer that wants the grid back
        has to reshape it itself.
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
    """Local windowed x_ray observation centered on one player.

    Extracts a ``(2*radius+1) x (2*radius+1)`` patch from the ten x_ray
    spatial channels and appends the x_ray scalar vector.

    Parameters
    ----------
    state
        State to read.
    params
        Passed through to the scalar builders.
    player_idx
        Which player is observing. The window centres on this player.
    radius
        Half-width of the window in tiles. The window is
        ``(2 * radius + 1)`` square, so it always has a centre tile. Static
        under ``jit``: changing it retraces.

    Returns
    -------
    jax.Array
        1-D float32 of length
        ``observation_size(profile="x_ray", view="local", radius=radius)``.
        Tiles beyond the map edge are padded with
        ``BlockType.OUT_OF_BOUNDS`` on the terrain channel and zero
        elsewhere, so an agent can still tell where the edge is.
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

    Three spatial channels (block type, machine type, machine direction)
    flattened, followed by the superficial scalar vector.

    Parameters
    ----------
    state
        State to read.
    params
        Passed through to the scalar builders.
    player_idx
        Which player is observing.

    Returns
    -------
    jax.Array
        1-D float32 of length
        ``observation_size(profile="superficial", view="global", ...)``.
        Three flattened ``(H, W)`` channels then the superficial scalars.
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
    """Local windowed superficial observation centered on one player.

    Extracts a ``(2*radius+1) x (2*radius+1)`` patch from the three
    superficial spatial channels and appends the 63-float superficial
    scalar vector.

    Parameters
    ----------
    state
        State to read.
    params
        Passed through to the scalar builders.
    player_idx
        Which player is observing. The window centres on this player.
    radius
        Half-width of the window in tiles. Static under ``jit``.

    Returns
    -------
    jax.Array
        1-D float32 of length
        ``observation_size(profile="superficial", view="local",
        radius=radius)``. Padded the same way as :func:`local_x_ray`.
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


#: Catalog of obs variants. The env takes the name (and a radius for
#: ``_local`` variants) and dispatches through this dict.
OBSERVATIONS: dict[str, jax.Array] = {  # type: ignore[type-arg]
    "x_ray_global": global_x_ray,
    "x_ray_local": local_x_ray,
    "superficial_global": global_superficial,
    "superficial_local": local_superficial,
}


def rgb(state: EnvState, block_pixel_size: int = 32) -> np.ndarray:
    """Render the full map as an RGB image.

    The one observation here that is not JAX-native. It returns a NumPy
    array and cannot be traced, so it belongs outside a ``jit`` boundary.

    What it shows is a strict subset of what the symbolic profiles carry:
    terrain, machine kind and facing, and player position and facing. There
    is no inventory, no craft affordability, and no machine contents, so a
    pixel agent cannot learn crafting from this alone. Recorded in
    ``ISSUES.md``.

    Parameters
    ----------
    state
        State to render.
    block_pixel_size
        Tile side length in pixels. Each distinct value builds and caches
        its own renderer, so the atlas build and JIT compile are paid once
        per size rather than once per call. Passing many sizes grows the
        cache without bound.

    Returns
    -------
    numpy.ndarray
        Shape ``(H * block_pixel_size, W * block_pixel_size, 3)``, uint8
        RGB.
    """
    renderer = _RENDERER_CACHE.get(block_pixel_size)
    if renderer is None:
        renderer = JaxRenderer(tile_px=block_pixel_size)
        _RENDERER_CACHE[block_pixel_size] = renderer
    return np.asarray(renderer.jit_render_map(state))
