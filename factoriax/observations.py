"""Observation constructors for FactoriaX.

Each function maps ``(state, params, player_idx) -> jax.Array`` and is
JAX-native and JIT-compatible.  :func:`rgb` is the exception: it returns
a NumPy RGB image via the existing pixel renderer and cannot be JIT'd.

All three follow the same convention: ``state`` and ``params`` are the
standard FactoriaX types; ``player_idx`` selects whose perspective the
observation is built from.  Pure functions compose naturally with
``jax.vmap`` for simultaneous multi-agent observations::

    all_obs = jax.vmap(local_array, in_axes=(None, None, 0))(
        state, params, jnp.arange(params.num_players)
    )

``local_array`` requires ``radius`` to be a plain Python :class:`int` so
that pad widths and slice sizes remain concrete during JIT compilation.
Bind it with ``functools.partial`` before calling ``jax.jit`` if you need
a non-default radius::

    import functools
    obs_fn = functools.partial(local_array, radius=5)
    jit_obs = jax.jit(obs_fn)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from factoriax.constants import (
    BLOCK_MAX_RESOURCES,
    DEFAULT_MACHINE_MAX_HEALTH,
    MAX_MACHINE_INVENTORY_SLOTS,
    MAX_STACK_SIZE,
    NUM_INVENTORY_SLOTS,
    NUM_ITEM_TYPES,
    RESEARCH_COST,
    BlockType,
    MachineType,
)
from factoriax.crafting import can_afford_recipe
from factoriax.placement import get_tile_in_front
from factoriax.recipes import MAX_ASSEMBLER_STACK_SIZE, NUM_RECIPES, RECIPES
from factoriax.renderer import render_pixels
from factoriax.state import EnvParams, EnvState

_MAP_NORM: float = float(max(BlockType))
_MACHINE_NORM: float = float(max(MachineType))
_INV_ITEM_NORM: float = float(NUM_ITEM_TYPES)
_MAX_CRAFT_TICKS: float = max(1.0, float(max(r["ticks"] for r in RECIPES)))
_DIR_NORM: float = 4.0  # max directional Action value (LEFT=1..DOWN=4)
_HEALTH_NORM: float = float(DEFAULT_MACHINE_MAX_HEALTH)
_MACHINE_INV_COUNT_NORM: float = float(MAX_ASSEMBLER_STACK_SIZE)

# Spatial channels shared by global_array and local_array.
# Add new channels here; NUM_SPATIAL_CHANNELS updates automatically.
_SPATIAL_CHANNEL_NAMES: tuple[str, ...] = (
    "block_type",
    "machine_type",
    "block_resources",
    "biter_presence",
    "machine_direction",
    "machine_health",
)
NUM_SPATIAL_CHANNELS: int = len(_SPATIAL_CHANNEL_NAMES)


# Scalar fields prepended before inventory in every observation vector.
# Update this list when adding new per-player scalars.
_PLAYER_SCALAR_FIELDS: tuple[str, ...] = (
    "pos_x",
    "pos_y",
    "direction",
    "timestep",
    "selected_slot",
    "craft_progress",
    "afford_miner",
    "afford_chest",
    "afford_belt",
    "afford_arm",
    "afford_assembler",
    "facing_machine_type",
    "facing_machine_health",
    *(f"facing_inv_item_{i}" for i in range(MAX_MACHINE_INVENTORY_SLOTS)),
    *(f"facing_inv_count_{i}" for i in range(MAX_MACHINE_INVENTORY_SLOTS)),
)
NUM_PLAYER_SCALARS: int = len(_PLAYER_SCALAR_FIELDS)


def _player_scalars(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
) -> jax.Array:
    """Build the per-player scalar vector shared by all observation types.

    Includes position, direction, timestep, crafting state, recipe
    affordability, facing-machine state, inventory, and research.

    Args:
        state: Current environment state.
        params: Environment parameters.
        player_idx: Index of the player whose scalars to extract.

    Returns:
        Float32 array of shape ``(NUM_PLAYER_SCALARS
        + 2 * NUM_INVENTORY_SLOTS + 2 * NUM_TECHNOLOGIES,)``.
    """
    pos = state.player_positions[player_idx]
    # Per-recipe affordability: 1.0 if the player can afford it, else 0.0.
    afford = jax.vmap(
        lambda r: can_afford_recipe(state, player_idx, r).astype(
            jnp.float32
        )
    )(jnp.arange(NUM_RECIPES))

    scalars = jnp.array(
        [
            pos[0] / params.map_width,
            pos[1] / params.map_height,
            state.player_directions[player_idx] / _DIR_NORM,
            state.timestep / params.max_timesteps,
            state.selected_slots[player_idx] / NUM_INVENTORY_SLOTS,
            state.craft_progress[player_idx] / _MAX_CRAFT_TICKS,
        ],
        dtype=jnp.float32,
    )
    scalars = jnp.concatenate([scalars, afford])

    # Facing-machine state: type, health, and full inventory of the
    # machine on the tile directly in front of the player.
    tx, ty = get_tile_in_front(state, player_idx)
    map_h, map_w = state.map.shape
    in_bounds = (tx >= 0) & (tx < map_w) & (ty >= 0) & (ty < map_h)
    sx = jnp.clip(tx, 0, map_w - 1)
    sy = jnp.clip(ty, 0, map_h - 1)
    mask = in_bounds.astype(jnp.float32)

    facing_type = (
        state.machine_types[sy, sx].astype(jnp.float32) * mask
        / _MACHINE_NORM
    )
    facing_health = (
        state.machine_health[sy, sx].astype(jnp.float32) * mask
        / _HEALTH_NORM
    )
    facing_inv_items = (
        state.machine_inventory_items[sy, sx].astype(jnp.float32) * mask
        / _INV_ITEM_NORM
    )
    facing_inv_counts = (
        state.machine_inventory_counts[sy, sx].astype(jnp.float32) * mask
        / _MACHINE_INV_COUNT_NORM
    )
    facing = jnp.concatenate(
        [
            facing_type[None],
            facing_health[None],
            facing_inv_items,
            facing_inv_counts,
        ]
    )
    scalars = jnp.concatenate([scalars, facing])

    # Player inventory.
    inv_items = (
        state.inventory_items[player_idx].astype(jnp.float32)
        / _INV_ITEM_NORM
    )
    inv_counts = (
        state.inventory_counts[player_idx].astype(jnp.float32)
        / MAX_STACK_SIZE
    )
    # Research state: per-technology [unlocked, progress/cost].
    research_unlocked = state.research_unlocked.astype(jnp.float32)
    research_progress = (
        state.research_progress.astype(jnp.float32) / RESEARCH_COST
    )
    return jnp.concatenate(
        [scalars, inv_items, inv_counts, research_unlocked, research_progress]
    )


def _biter_grid(state: EnvState) -> jax.Array:
    """Binary (H, W) grid with 1.0 where active biters are present.

    Args:
        state: Current environment state.

    Returns:
        Float32 array of shape ``(map_h, map_w)`` with values in {0, 1}.
    """
    h, w = state.map.shape
    grid = jnp.zeros((h, w), dtype=jnp.float32)
    is_active = state.biter_health > 0
    bx = state.biter_positions[:, 0]
    by = state.biter_positions[:, 1]
    grid = grid.at[by, bx].add(is_active.astype(jnp.float32))
    return jnp.minimum(grid, 1.0)


def global_array(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
) -> jax.Array:
    """Full-map flat observation for one player.

    Concatenates six normalized spatial channels (block type, machine
    type, block resources, biter presence, machine direction, machine
    health) in row-major order, followed by the player scalar vector
    (position, inventory, facing-machine state, research).

    Args:
        state: Current environment state.
        params: Environment parameters.
        player_idx: Index of the observing player.

    Returns:
        Float32 array of shape ``(NUM_SPATIAL_CHANNELS * map_h * map_w
        + NUM_PLAYER_SCALARS + 2 * NUM_INVENTORY_SLOTS
        + 2 * NUM_TECHNOLOGIES,)``.
    """
    flat_blocks = state.map.flatten().astype(jnp.float32) / _MAP_NORM
    flat_machines = (
        state.machine_types.flatten().astype(jnp.float32) / _MACHINE_NORM
    )
    flat_resources = (
        state.block_resources.flatten().astype(jnp.float32)
        / float(BLOCK_MAX_RESOURCES)
    )
    flat_biters = _biter_grid(state).ravel()
    flat_directions = (
        state.machine_direction.flatten().astype(jnp.float32) / _DIR_NORM
    )
    flat_health = (
        state.machine_health.flatten().astype(jnp.float32) / _HEALTH_NORM
    )
    spatial = jnp.concatenate(
        [
            flat_blocks,
            flat_machines,
            flat_resources,
            flat_biters,
            flat_directions,
            flat_health,
        ]
    )
    return jnp.concatenate(
        [spatial, _player_scalars(state, params, player_idx)]
    )


def local_array(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
    radius: int = 10,
) -> jax.Array:
    """Local windowed observation centered on one player.

    Extracts a ``(2*radius+1) x (2*radius+1)`` patch around the player
    from six spatial channels: block type, machine type, block resources,
    biter presence, machine direction, and machine health.  Out-of-bounds
    tiles are filled with their natural sentinel values
    (``BlockType.OUT_OF_BOUNDS``, ``MachineType.NONE``, and 0 for the
    rest).  Player scalars are appended.

    The padding trick: padding each map by ``radius`` on all sides places
    the player at ``(pos[1] + radius, pos[0] + radius)`` in padded
    coordinates.  A window of size ``(2*radius+1, 2*radius+1)`` starting
    at the original ``(pos[1], pos[0])`` is therefore always centered on
    the player and always within bounds.

    JAX-native and JIT-compatible when ``radius`` is a Python ``int``.

    Args:
        state: Current environment state.
        params: Environment parameters.
        player_idx: Index of the observing player.
        radius: Half-width of the observation window (default 10).
            Must be a plain Python ``int``, not a traced JAX value.

    Returns:
        Float32 array of shape ``(NUM_SPATIAL_CHANNELS * (2*radius+1)**2
        + NUM_PLAYER_SCALARS + 2 * NUM_INVENTORY_SLOTS
        + 2 * NUM_TECHNOLOGIES,)``.
    """
    size = 2 * radius + 1
    pw = ((radius, radius), (radius, radius))

    padded_map = (
        jnp.pad(
            state.map,
            pw,
            mode="constant",
            constant_values=BlockType.OUT_OF_BOUNDS,
        ).astype(jnp.float32)
        / _MAP_NORM
    )
    padded_machines = (
        jnp.pad(
            state.machine_types,
            pw,
            mode="constant",
            constant_values=MachineType.NONE,
        ).astype(jnp.float32)
        / _MACHINE_NORM
    )
    padded_resources = (
        jnp.pad(
            state.block_resources,
            pw,
            mode="constant",
            constant_values=0,
        ).astype(jnp.float32)
        / float(BLOCK_MAX_RESOURCES)
    )
    padded_biters = jnp.pad(
        _biter_grid(state),
        pw,
        mode="constant",
        constant_values=0.0,
    )
    padded_directions = (
        jnp.pad(
            state.machine_direction,
            pw,
            mode="constant",
            constant_values=0,
        ).astype(jnp.float32)
        / _DIR_NORM
    )
    padded_health = (
        jnp.pad(
            state.machine_health,
            pw,
            mode="constant",
            constant_values=0,
        ).astype(jnp.float32)
        / _HEALTH_NORM
    )

    # Original player (row, col) = (pos[1], pos[0]) is the start index in
    # padded space for a centered window of size (2*radius+1).
    pos = state.player_positions[player_idx]
    start = (pos[1], pos[0])
    slice_shape = (size, size)

    window_map = jax.lax.dynamic_slice(padded_map, start, slice_shape)
    window_machines = jax.lax.dynamic_slice(
        padded_machines, start, slice_shape
    )
    window_resources = jax.lax.dynamic_slice(
        padded_resources, start, slice_shape
    )
    window_biters = jax.lax.dynamic_slice(
        padded_biters, start, slice_shape
    )
    window_directions = jax.lax.dynamic_slice(
        padded_directions, start, slice_shape
    )
    window_health = jax.lax.dynamic_slice(
        padded_health, start, slice_shape
    )

    spatial = jnp.concatenate(
        [
            window_map.ravel(),
            window_machines.ravel(),
            window_resources.ravel(),
            window_biters.ravel(),
            window_directions.ravel(),
            window_health.ravel(),
        ]
    )
    return jnp.concatenate(
        [spatial, _player_scalars(state, params, player_idx)]
    )


def rgb(state: EnvState, block_pixel_size: int = 32) -> np.ndarray:
    """Render the full map as an RGB image with all players visible.

    Wraps the existing pixel renderer.  The output is a NumPy array and
    cannot be JIT-compiled or used inside ``jax.vmap``.  Intended for
    visualization, video recording, and human play — not batched RL
    training.

    Args:
        state: Current environment state.
        block_pixel_size: Side length of each tile in pixels.

    Returns:
        uint8 NumPy array of shape
        ``(map_h * block_pixel_size, map_w * block_pixel_size, 3)``.
    """
    return render_pixels(state, block_pixel_size=block_pixel_size)
