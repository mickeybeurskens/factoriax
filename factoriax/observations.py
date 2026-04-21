"""Observation constructors for FactoriaX.

Each function maps ``(state, params, player_idx) -> jax.Array`` and is
JAX-native and JIT-compatible. ``rgb`` is the exception: it returns a
NumPy RGB image via the pixel renderer and cannot be JIT'd.

Spatial channels: block_type, machine_type, block_resources, buffer_type.
Player scalars: position, direction, timestep, recipe affordability,
facing-machine state, player inventory, research state.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from factoriax.constants import (
    BLOCK_MAX_RESOURCES,
    NUM_ITEM_TYPES,
    PLAYER_MAX_STACK,
    RESEARCH_COST,
    BlockType,
    MachineType,
)
from factoriax.crafting import can_afford_recipe
from factoriax.placement import get_tile_in_front
from factoriax.recipes import NUM_RECIPES
from factoriax.renderer import render_pixels
from factoriax.state import EnvParams, EnvState


def _reconstruct_buffer_type_grid(state: EnvState) -> jnp.ndarray:
    """Build a (H, W) buffer-type grid from entity arrays.

    Active entities scatter a single "what's available to withdraw
    here?" value. Combiner output (``ent_asm_out``) takes priority
    when present — with Phase 4 of ``run_combiners`` removed, the
    asm_out slot holds completed recipe output until withdrawn and
    ``ent_buf`` stays empty on combiners. For non-combiners (miners,
    pallets, belts) ``ent_asm_out`` is always empty and the fallback
    to ``ent_buf_type`` is what shows up.

    Args:
        state: Current environment state.

    Returns:
        int8 array of shape ``(H, W)`` with withdrawable item types.
    """
    h, w = state.map.shape
    grid = jnp.zeros((h, w), dtype=jnp.int8)
    active = state.ent_y >= 0
    ey = jnp.clip(state.ent_y, 0, h - 1)
    ex = jnp.clip(state.ent_x, 0, w - 1)
    out_has = state.ent_asm_out_count > 0
    buf_val = jnp.where(active, state.ent_buf_type, jnp.int8(0))
    out_val = jnp.where(active & out_has, state.ent_asm_out_type, jnp.int8(0))
    vals = jnp.where(out_val != jnp.int8(0), out_val, buf_val)
    return grid.at[ey, ex].set(vals)


_MAP_NORM: float = float(max(BlockType))
_MACHINE_NORM: float = float(max(MachineType))
_DIR_NORM: float = 4.0
_PLAYER_MAX_STACK_F: jnp.ndarray = jnp.maximum(
    PLAYER_MAX_STACK.astype(jnp.float32),
    1.0,
)

# Spatial channels shared by global_array and local_array.
_SPATIAL_CHANNEL_NAMES: tuple[str, ...] = (
    "block_type",
    "machine_type",
    "block_resources",
    "buffer_type",
)
NUM_SPATIAL_CHANNELS: int = len(_SPATIAL_CHANNEL_NAMES)

# Scalar fields in every observation vector.
_PLAYER_SCALAR_FIELDS: tuple[str, ...] = (
    "pos_x",
    "pos_y",
    "direction",
    "timestep",
    *(f"afford_{i}" for i in range(NUM_RECIPES)),
    "facing_machine_type",
    "facing_buffer_type",
    "facing_buffer_count",
    "facing_asm_in0_type",
    "facing_asm_in0_count",
    "facing_asm_in1_type",
    "facing_asm_in1_count",
    "facing_asm_out_type",
    "facing_asm_out_count",
    *(f"player_inv_{i}" for i in range(NUM_ITEM_TYPES)),
)
NUM_PLAYER_SCALARS: int = len(_PLAYER_SCALAR_FIELDS)


def _player_scalars(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
) -> jax.Array:
    """Build the per-player scalar vector.

    Args:
        state: Current environment state.
        params: Environment parameters.
        player_idx: Index of the player.

    Returns:
        Float32 array of shape ``(NUM_PLAYER_SCALARS
        + 2 * NUM_TECHNOLOGIES,)``.
    """
    pos = state.player_positions[player_idx]
    afford = jax.vmap(
        lambda r: can_afford_recipe(state, player_idx, r).astype(
            jnp.float32,
        ),
    )(jnp.arange(NUM_RECIPES))

    scalars = jnp.array(
        [
            pos[0] / params.map_width,
            pos[1] / params.map_height,
            state.player_directions[player_idx] / _DIR_NORM,
            state.timestep / params.max_timesteps,
        ],
        dtype=jnp.float32,
    )
    scalars = jnp.concatenate([scalars, afford])

    # Facing-machine state via entity lookup.
    tx, ty = get_tile_in_front(state, player_idx)
    map_h, map_w = state.map.shape
    in_bounds = (tx >= 0) & (tx < map_w) & (ty >= 0) & (ty < map_h)
    sx = jnp.clip(tx, 0, map_w - 1)
    sy = jnp.clip(ty, 0, map_h - 1)
    mask = in_bounds.astype(jnp.float32)

    max_e = state.ent_y.shape[0]
    eidx = jnp.clip(state.tile_entity[sy, sx], 0, max_e - 1)

    facing_mt = state.machine_types[sy, sx].astype(jnp.float32) * mask
    facing_bt = state.ent_buf_type[eidx].astype(jnp.float32) * mask
    facing_bc = state.ent_buf_count[eidx].astype(jnp.float32) * mask
    facing_asm_in0t = state.ent_asm_in_type[eidx, 0].astype(jnp.float32) * mask
    facing_asm_in0c = state.ent_asm_in_count[eidx, 0].astype(jnp.float32) * mask
    facing_asm_in1t = state.ent_asm_in_type[eidx, 1].astype(jnp.float32) * mask
    facing_asm_in1c = state.ent_asm_in_count[eidx, 1].astype(jnp.float32) * mask
    facing_asm_ot = state.ent_asm_out_type[eidx].astype(jnp.float32) * mask
    facing_asm_oc = state.ent_asm_out_count[eidx].astype(jnp.float32) * mask

    facing = jnp.array(
        [
            facing_mt / _MACHINE_NORM,
            facing_bt / float(NUM_ITEM_TYPES),
            facing_bc / 64.0,
            facing_asm_in0t / float(NUM_ITEM_TYPES),
            facing_asm_in0c / 64.0,
            facing_asm_in1t / float(NUM_ITEM_TYPES),
            facing_asm_in1c / 64.0,
            facing_asm_ot / float(NUM_ITEM_TYPES),
            facing_asm_oc / 64.0,
        ]
    )
    scalars = jnp.concatenate([scalars, facing])

    # Player inventory.
    player_inv = (
        state.player_inventory[player_idx].astype(jnp.float32) / _PLAYER_MAX_STACK_F
    )

    # Research state.
    research_unlocked = state.research_unlocked.astype(jnp.float32)
    research_progress = state.research_progress.astype(jnp.float32) / RESEARCH_COST
    return jnp.concatenate(
        [scalars, player_inv, research_unlocked, research_progress],
    )


def global_array(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
) -> jax.Array:
    """Full-map flat observation for one player.

    Four spatial channels (block type, machine type, resources, buffer type)
    followed by the player scalar vector.

    Args:
        state: Current environment state.
        params: Environment parameters.
        player_idx: Index of the observing player.

    Returns:
        Float32 array of shape ``(NUM_SPATIAL_CHANNELS * H * W
        + NUM_PLAYER_SCALARS + 2 * NUM_TECHNOLOGIES,)``.
    """
    flat_blocks = state.map.flatten().astype(jnp.float32) / _MAP_NORM
    flat_machines = state.machine_types.flatten().astype(jnp.float32) / _MACHINE_NORM
    flat_resources = state.block_resources.flatten().astype(jnp.float32) / float(
        BLOCK_MAX_RESOURCES
    )
    buf_grid = _reconstruct_buffer_type_grid(state)
    flat_buffer = buf_grid.flatten().astype(jnp.float32) / float(NUM_ITEM_TYPES)
    spatial = jnp.concatenate(
        [flat_blocks, flat_machines, flat_resources, flat_buffer],
    )
    return jnp.concatenate(
        [spatial, _player_scalars(state, params, player_idx)],
    )


def local_array(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
    radius: int = 10,
) -> jax.Array:
    """Local windowed observation centered on one player.

    Extracts a ``(2*radius+1) x (2*radius+1)`` patch from four spatial
    channels. Player scalars are appended.

    Args:
        state: Current environment state.
        params: Environment parameters.
        player_idx: Index of the observing player.
        radius: Half-width of the observation window.

    Returns:
        Float32 array of shape ``(NUM_SPATIAL_CHANNELS * (2r+1)^2
        + NUM_PLAYER_SCALARS + 2 * NUM_TECHNOLOGIES,)``.
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
        jnp.pad(state.machine_types, pw, constant_values=MachineType.NONE).astype(
            jnp.float32
        )
        / _MACHINE_NORM
    )
    padded_resources = jnp.pad(state.block_resources, pw, constant_values=0).astype(
        jnp.float32
    ) / float(BLOCK_MAX_RESOURCES)
    buf_grid = _reconstruct_buffer_type_grid(state)
    padded_buffer = jnp.pad(buf_grid, pw, constant_values=0).astype(
        jnp.float32
    ) / float(NUM_ITEM_TYPES)

    pos = state.player_positions[player_idx]
    start = (pos[1], pos[0])
    slice_shape = (size, size)

    spatial = jnp.concatenate(
        [
            jax.lax.dynamic_slice(padded_map, start, slice_shape).ravel(),
            jax.lax.dynamic_slice(padded_machines, start, slice_shape).ravel(),
            jax.lax.dynamic_slice(padded_resources, start, slice_shape).ravel(),
            jax.lax.dynamic_slice(padded_buffer, start, slice_shape).ravel(),
        ]
    )
    return jnp.concatenate(
        [spatial, _player_scalars(state, params, player_idx)],
    )


def rgb(state: EnvState, block_pixel_size: int = 32) -> np.ndarray:
    """Render the full map as an RGB image.

    Args:
        state: Current environment state.
        block_pixel_size: Tile side length in pixels.

    Returns:
        uint8 NumPy array of shape ``(H*px, W*px, 3)``.
    """
    return render_pixels(state, block_pixel_size=block_pixel_size)
