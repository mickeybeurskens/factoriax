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
    MAX_STACK_SIZE,
    NUM_ACTIONS,
    NUM_INVENTORY_SLOTS,
    NUM_ITEM_TYPES,
    NUM_TECHNOLOGIES,
    RESEARCH_COST,
    BlockType,
    MachineType,
)
from factoriax.crafting import can_afford_recipe
from factoriax.recipes import NUM_RECIPES, RECIPES
from factoriax.renderer import render_pixels
from factoriax.state import EnvParams, EnvState

_MAP_NORM: float = float(BlockType.COAL)
_MACHINE_NORM: float = float(max(MachineType))
_INV_ITEM_NORM: float = float(NUM_ITEM_TYPES)
_MAX_CRAFT_TICKS: float = max(1.0, float(max(r["ticks"] for r in RECIPES)))


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
)
NUM_PLAYER_SCALARS: int = len(_PLAYER_SCALAR_FIELDS)


def _player_scalars(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
) -> jax.Array:
    """Build the per-player scalar vector shared by all observation types.

    Includes position, direction, timestep, inventory contents, and
    crafting UI state (selected recipe, selected slot, craft progress).

    Args:
        state: Current environment state.
        params: Environment parameters.
        player_idx: Index of the player whose scalars to extract.

    Returns:
        Float32 array of shape
        ``(NUM_PLAYER_SCALARS + 2 * NUM_INVENTORY_SLOTS,)``.
    """
    pos = state.player_positions[player_idx]
    # Per-recipe affordability: 1.0 if the player can afford it, else 0.0.
    # Vectorized over recipes so JAX can batch the inventory scans.
    afford = jax.vmap(
        lambda r: can_afford_recipe(state, player_idx, r).astype(
            jnp.float32
        )
    )(jnp.arange(NUM_RECIPES))

    scalars = jnp.array(
        [
            pos[0] / params.map_width,
            pos[1] / params.map_height,
            state.player_directions[player_idx] / NUM_ACTIONS,
            state.timestep / params.max_timesteps,
            state.selected_slots[player_idx] / NUM_INVENTORY_SLOTS,
            state.craft_progress[player_idx] / _MAX_CRAFT_TICKS,
        ],
        dtype=jnp.float32,
    )
    scalars = jnp.concatenate([scalars, afford])
    inv_items = state.inventory_items[player_idx].astype(jnp.float32) / _INV_ITEM_NORM
    inv_counts = state.inventory_counts[player_idx].astype(jnp.float32) / MAX_STACK_SIZE
    # Research state: per-technology [unlocked, progress/cost].
    research_unlocked = state.research_unlocked.astype(jnp.float32)
    research_progress = state.research_progress.astype(jnp.float32) / RESEARCH_COST
    return jnp.concatenate(
        [scalars, inv_items, inv_counts, research_unlocked, research_progress]
    )


def global_array(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
) -> jax.Array:
    """Full-map flat observation for one player.

    Concatenates the normalized map (row-major) with the player's
    position, facing direction, timestep fraction, and inventory.
    JAX-native and JIT-compatible.

    Args:
        state: Current environment state.
        params: Environment parameters.
        player_idx: Index of the observing player.

    Returns:
        Float32 array of shape
        ``(map_h * map_w + 7 + 2 * NUM_INVENTORY_SLOTS,)``.
    """
    flat_map = state.map.flatten().astype(jnp.float32) / _MAP_NORM
    return jnp.concatenate([flat_map, _player_scalars(state, params, player_idx)])


def local_array(
    state: EnvState,
    params: EnvParams,
    player_idx: int | jax.Array,
    radius: int = 10,
) -> jax.Array:
    """Local windowed observation centered on one player.

    Extracts a ``(2*radius+1) x (2*radius+1)`` patch around the player
    from three spatial channels: block type, machine type, and remaining
    block resources.  Out-of-bounds tiles are filled with their natural
    sentinel values (``BlockType.OUT_OF_BOUNDS``, ``MachineType.NONE``,
    and 0 resources respectively).  Player scalars are appended.

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
        Float32 array of shape
        ``(3 * (2*radius+1)**2 + 7 + 2 * NUM_INVENTORY_SLOTS,)``.
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
    padded_resources = jnp.pad(
        state.block_resources,
        pw,
        mode="constant",
        constant_values=0,
    ).astype(jnp.float32) / float(BLOCK_MAX_RESOURCES)

    # Original player (row, col) = (pos[1], pos[0]) is the start index in
    # padded space for a centered window of size (2*radius+1).
    pos = state.player_positions[player_idx]
    start = (pos[1], pos[0])
    slice_shape = (size, size)

    window_map = jax.lax.dynamic_slice(padded_map, start, slice_shape)
    window_machines = jax.lax.dynamic_slice(padded_machines, start, slice_shape)
    window_resources = jax.lax.dynamic_slice(padded_resources, start, slice_shape)

    spatial = jnp.concatenate(
        [window_map.ravel(), window_machines.ravel(), window_resources.ravel()]
    )
    return jnp.concatenate([spatial, _player_scalars(state, params, player_idx)])


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
