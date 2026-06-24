"""Observation constructors for FactoriaX.

Each function maps ``(state, params, player_idx) -> jax.Array`` and is
JAX-native and JIT-compatible. ``rgb`` is the exception: it returns a
NumPy RGB image via the pixel renderer and cannot be JIT'd.

Two profiles compose four top-level builders:

- **x_ray** sees through every machine to its slot contents and the ore
  remaining under each terrain tile (10 spatial channels + 72 player
  scalars).
- **superficial** sees only what's outwardly visible (3 spatial channels
  + 63 player scalars; no slot peek, no facing-tile peek).

Each profile pairs with a view extent: ``global_*`` flattens the whole
map; ``local_*`` extracts a ``(2r+1) x (2r+1)`` window centred on the
selected player.

x_ray spatial channels (10):

- ``block_type`` — terrain.
- ``machine_type`` — ``Machine`` at each tile (or NONE).
- ``block_resources`` — ore count under the tile.
- ``slot{0,1,2}_type`` / ``slot{0,1,2}_count`` — a uniform 3-slot
  projection of every machine's contents. Combiners map
  ``ent_asm_in[0] → slot 0``, ``ent_asm_in[1] → slot 1``,
  ``ent_asm_out → slot 2``. Buffer machines (miner, pallet, belt)
  leave slots 0/1 at zero and write ``ent_buf`` into slot 2.
- ``machine_direction`` — ``ent_direction`` per tile (1..4 or 0).

superficial spatial channels (3): ``block_type``, ``machine_type``,
``machine_direction``.

x_ray scalars: pose + recipe affordability + player inventory +
facing-machine readouts (72 floats).
superficial scalars: pose + recipe affordability + player inventory
(63 floats; the 9 facing readouts are dropped).
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
from factoriax.engine.jax_renderer import JaxRenderer
from factoriax.engine.placement import get_tile_in_front
from factoriax.engine.recipes import NUM_RECIPES
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.tables import PLAYER_MAX_STACK

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

    Parameters
    ----------
    state :
        EnvState:
    state : EnvState :
        
    state: EnvState :
        

    Returns
    -------

    
    """
    h, w = state.map.shape
    active = state.ent_y >= 0
    ey = jnp.clip(state.ent_y, 0, h - 1)
    ex = jnp.clip(state.ent_x, 0, w - 1)

    is_combiner = (state.ent_type == Machine.ASSEMBLER) | (
        state.ent_type == Machine.FURNACE
    )
    is_buffer_machine = (
        (state.ent_type == Machine.MINER)
        | (state.ent_type == Machine.PALLET)
        | (state.ent_type == Machine.CONVEYOR_BELT)
    )

    # Slots 0 and 1: asm_in for combiners, else zero.
    in0_t = jnp.where(
        active & is_combiner,
        state.ent_asm_in_type[..., 0],
        jnp.int8(0),
    )
    in0_c = jnp.where(
        active & is_combiner,
        state.ent_asm_in_count[..., 0],
        jnp.int16(0),
    )
    in1_t = jnp.where(
        active & is_combiner,
        state.ent_asm_in_type[..., 1],
        jnp.int8(0),
    )
    in1_c = jnp.where(
        active & is_combiner,
        state.ent_asm_in_count[..., 1],
        jnp.int16(0),
    )

    # Slot 2: asm_out for combiners, ent_buf for buffer machines.
    out_t = jnp.where(
        active & is_combiner,
        state.ent_asm_out_type,
        jnp.where(active & is_buffer_machine, state.ent_buf_type, jnp.int8(0)),
    )
    out_c = jnp.where(
        active & is_combiner,
        state.ent_asm_out_count,
        jnp.where(active & is_buffer_machine, state.ent_buf_count, jnp.int16(0)),
    )

    zero_t = jnp.zeros((h, w), dtype=jnp.int8)
    zero_c = jnp.zeros((h, w), dtype=jnp.int16)
    return (
        zero_t.at[ey, ex].set(in0_t),
        zero_c.at[ey, ex].set(in0_c),
        zero_t.at[ey, ex].set(in1_t),
        zero_c.at[ey, ex].set(in1_c),
        zero_t.at[ey, ex].set(out_t),
        zero_c.at[ey, ex].set(out_c),
    )


def _reconstruct_machine_direction_grid(state: EnvState) -> jnp.ndarray:
    """Per-tile ``ent_direction`` for active machines.
    
    Values are :class:`Direction` ints (0 where no machine is
    placed). Lets agents plan push chains (miner → pallet, arm →
    furnace) from the obs alone.

    Parameters
    ----------
    state :
        EnvState:
    state : EnvState :
        
    state: EnvState :
        

    Returns
    -------

    
    """
    h, w = state.map.shape
    grid = jnp.zeros((h, w), dtype=jnp.int8)
    active = state.ent_y >= 0
    ey = jnp.clip(state.ent_y, 0, h - 1)
    ex = jnp.clip(state.ent_x, 0, w - 1)
    vals = jnp.where(active, state.ent_direction, jnp.int8(0))
    return grid.at[ey, ex].set(vals)


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
    *(f"afford_{i}" for i in range(NUM_RECIPES)),
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
# x_ray = common ++ facing (72); superficial = common (63).
NUM_PLAYER_SCALARS: dict[str, int] = {
    "x_ray": len(_COMMON_SCALAR_FIELDS) + len(_FACING_SCALAR_FIELDS),
    "superficial": len(_COMMON_SCALAR_FIELDS),
}


def _common_scalars(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
) -> jax.Array:
    """Pose, recipe affordability, and player inventory. 63 floats.
    
    Shared by x_ray and superficial profiles.
    
    Parameters
    ----------
        state: Current environment state.
    
    Parameters
    ----------
      player_idx: Index of the player
      state: EnvState:

    Parameters
    ----------
    player_idx :
        int
    state : EnvState :
        
    params : EnvParams :
        
    player_idx : int | jax.Array :
        
    state: EnvState :
        
    params: EnvParams :
        
    player_idx: int | jax.Array :
        

    Returns
    -------

    
    """
    pos = state.player_positions[player_idx]
    pose_time = jnp.array(
        [
            pos[0] / params.map_width,
            pos[1] / params.map_height,
            state.player_directions[player_idx] / _DIR_NORM,
            state.timestep / params.max_timesteps,
        ],
        dtype=jnp.float32,
    )
    afford = jax.vmap(
        lambda r: can_afford_recipe(state, params, player_idx, r).astype(
            jnp.float32,
        ),
    )(jnp.arange(NUM_RECIPES))
    inv = state.player_inventory[player_idx].astype(jnp.float32) / _PLAYER_MAX_STACK_F
    return jnp.concatenate([pose_time, afford, inv])


def _facing_scalars(
    state: EnvState,
    player_idx: int | jax.Array,
) -> jax.Array:
    """Facing-machine readouts. 9 floats. x_ray only.
    
    Reads the entity in front of the player and returns its machine
    type plus buffer/assembler-slot contents. Out-of-bounds tiles
    return zeros via a mask.

    Parameters
    ----------
    state :
        Current environment state
    player_idx :
        Index of the player
    state :
        EnvState
    player_idx :
        int
    state : EnvState :
        
    player_idx : int | jax.Array :
        
    state: EnvState :
        
    player_idx: int | jax.Array :
        

    Returns
    -------

    
    """
    tx, ty = get_tile_in_front(state, player_idx)
    map_h, map_w = state.map.shape
    in_bounds = (tx >= 0) & (tx < map_w) & (ty >= 0) & (ty < map_h)
    sx = jnp.clip(tx, 0, map_w - 1)
    sy = jnp.clip(ty, 0, map_h - 1)
    mask = in_bounds.astype(jnp.float32)

    max_e = state.ent_y.shape[0]
    eidx = jnp.clip(state.tile_entity[sy, sx], 0, max_e - 1)
    item_norm = float(NUM_ITEM_TYPES)

    return jnp.array(
        [
            state.machine_types[sy, sx].astype(jnp.float32) * mask / _MACHINE_NORM,
            state.ent_buf_type[eidx].astype(jnp.float32) * mask / item_norm,
            state.ent_buf_count[eidx].astype(jnp.float32) * mask / 64.0,
            state.ent_asm_in_type[eidx, 0].astype(jnp.float32) * mask / item_norm,
            state.ent_asm_in_count[eidx, 0].astype(jnp.float32) * mask / 64.0,
            state.ent_asm_in_type[eidx, 1].astype(jnp.float32) * mask / item_norm,
            state.ent_asm_in_count[eidx, 1].astype(jnp.float32) * mask / 64.0,
            state.ent_asm_out_type[eidx].astype(jnp.float32) * mask / item_norm,
            state.ent_asm_out_count[eidx].astype(jnp.float32) * mask / 64.0,
        ]
    )


def _x_ray_scalars(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
) -> jax.Array:
    """The 72-float x_ray scalar block: common (63) + facing (9).
    
    Parameters
    ----------
      state: EnvState:

    Parameters
    ----------
    player_idx :
        int
    state : EnvState :
        
    params : EnvParams :
        
    player_idx : int | jax.Array :
        
    state: EnvState :
        
    params: EnvParams :
        
    player_idx: int | jax.Array :
        

    Returns
    -------

    
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
    """The 63-float superficial scalar block: common only.
    
    Parameters
    ----------
      state: EnvState:

    Parameters
    ----------
    player_idx :
        int
    state : EnvState :
        
    params : EnvParams :
        
    player_idx : int | jax.Array :
        
    state: EnvState :
        
    params: EnvParams :
        
    player_idx: int | jax.Array :
        

    Returns
    -------

    
    """
    return _common_scalars(state, params, player_idx)


def observation_size(
    params: EnvParams,
    *,
    profile: str,
    view: str,
    radius: int = 7,
) -> int:
    """Flat observation size for ``(profile, view)`` on ``params``.
    
    Parameters
    ----------
    
    Parameters
    ----------
      profile: x_ray
      view: global
      radius: Half

    Parameters
    ----------
    profile :
        str
    view :
        str
    radius :
        int
    params : EnvParams :
        
    * :
        
    profile : str :
        
    view : str :
        
    radius : int :
        (Default value = 7)
    params: EnvParams :
        
    profile: str :
        
    view: str :
        
    radius: int :
         (Default value = 7)

    Returns
    -------

    
    """
    if view == "global":
        spatial_tiles = params.map_width * params.map_height
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
    slot channels, machine direction) flattened, followed by the 72-float
    x_ray scalar vector.
    
    Parameters
    ----------
        state: Current environment state.
    
    Parameters
    ----------
      player_idx: Index of the observing player
      state: EnvState:

    Parameters
    ----------
    player_idx :
        int
    state : EnvState :
        
    params : EnvParams :
        
    player_idx : int | jax.Array :
        
    state: EnvState :
        
    params: EnvParams :
        
    player_idx: int | jax.Array :
        

    Returns
    -------

    
    >>> import jax
        >>> import factoriax
        >>> env, params = factoriax.make("EasyRocket-v1")
        >>> _, state = env.reset_env(jax.random.PRNGKey(0), params)
        >>> obs = factoriax.global_x_ray(state, params, state.selected_player)
        >>> obs.ndim
        1
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
    spatial channels and appends the 72-float x_ray scalar vector.
    
    Parameters
    ----------
        state: Current environment state.
    
    Parameters
    ----------
      player_idx: Index of the observing player
      radius: Half
      state: EnvState:

    Parameters
    ----------
    player_idx :
        int
    radius :
        int
    state : EnvState :
        
    params : EnvParams :
        
    player_idx : int | jax.Array :
        
    radius : int :
        (Default value = 10)
    state: EnvState :
        
    params: EnvParams :
        
    player_idx: int | jax.Array :
        
    radius: int :
         (Default value = 10)

    Returns
    -------

    
    >>> import jax
        >>> import factoriax
        >>> env, params = factoriax.make("EasyRocket-v1")
        >>> _, state = env.reset_env(jax.random.PRNGKey(0), params)
        >>> obs = factoriax.local_x_ray(state, params, state.selected_player, radius=3)
        >>> obs.ndim
        1
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
    flattened, followed by the 63-float superficial scalar vector.
    
    Parameters
    ----------
        state: Current environment state.
    
    Parameters
    ----------
      player_idx: Index of the observing player
      state: EnvState:

    Parameters
    ----------
    player_idx :
        int
    state : EnvState :
        
    params : EnvParams :
        
    player_idx : int | jax.Array :
        
    state: EnvState :
        
    params: EnvParams :
        
    player_idx: int | jax.Array :
        

    Returns
    -------

    
    >>> import jax
        >>> import factoriax
        >>> env, params = factoriax.make("EasyRocket-v1")
        >>> _, state = env.reset_env(jax.random.PRNGKey(0), params)
        >>> obs = factoriax.global_superficial(state, params, state.selected_player)
        >>> obs.ndim
        1
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
        state: Current environment state.
    
    Parameters
    ----------
      player_idx: Index of the observing player
      radius: Half
      state: EnvState:

    Parameters
    ----------
    player_idx :
        int
    radius :
        int
    state : EnvState :
        
    params : EnvParams :
        
    player_idx : int | jax.Array :
        
    radius : int :
        (Default value = 10)
    state: EnvState :
        
    params: EnvParams :
        
    player_idx: int | jax.Array :
        
    radius: int :
         (Default value = 10)

    Returns
    -------

    
    >>> import jax
        >>> import factoriax
        >>> env, params = factoriax.make("EasyRocket-v1")
        >>> _, state = env.reset_env(jax.random.PRNGKey(0), params)
        >>> obs = factoriax.local_superficial(
        ...     state, params, state.selected_player, radius=3,
        ... )
        >>> obs.ndim
        1
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
    
    Routes through a process-wide :class:`JaxRenderer` cache keyed by
    ``block_pixel_size`` so vision-mode rollouts pay the atlas build
    and JIT compile cost once per tile size, not per call.

    Parameters
    ----------
    state :
        Current environment state.
    block_pixel_size :
        Tile side length in pixels.
    state :
        EnvState:
    block_pixel_size :
        int:  (Default value = 32)
    state : EnvState :
        
    block_pixel_size : int :
        (Default value = 32)
    state: EnvState :
        
    block_pixel_size: int :
         (Default value = 32)

    Returns
    -------

    
    >>> import jax
        >>> import factoriax
        >>> env, params = factoriax.make("EasyRocket-v1")
        >>> _, state = env.reset_env(jax.random.PRNGKey(0), params)
        >>> img = factoriax.rgb(state, block_pixel_size=8)
        >>> img.shape[2]
        3
    """
    renderer = _RENDERER_CACHE.get(block_pixel_size)
    if renderer is None:
        renderer = JaxRenderer(tile_px=block_pixel_size)
        _RENDERER_CACHE[block_pixel_size] = renderer
    return np.asarray(renderer.jit_render_map(state))
