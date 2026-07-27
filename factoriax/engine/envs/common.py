"""Shared six-patch geometry and state predicates for 16x16 scenarios.

The easy-rocket map family — EasyRocket-v1 and the miner-curriculum
scenarios — shares one world geometry: a 16x16 dirt map with six
non-overlapping 2x2 ore patches (one per ore block), placed by PRNG
clear of the centre spawn area and the outer ring. This module owns
that geometry (host-side sampling helpers plus the jittable
:func:`six_patch_terrain` generator) and the pure-JAX state predicates
scenarios compose into achievement conditions, so scenario modules
never import from each other.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from factoriax.engine.constants import BlockType, Machine
from factoriax.engine.state import EnvParams, EnvState

MAP_SIZE: int = 16
PATCH_SIZE: int = 2
ORE_RESOURCES_PER_TILE: int = 3000
MAX_SAMPLE_ATTEMPTS: int = 1000

# --- Centre spawn area (2x2; room for multi-agent later). ---
SPAWN_AREA_SIZE: int = 2
SPAWN_AREA_MIN: int = (MAP_SIZE - SPAWN_AREA_SIZE) // 2  # 7
SPAWN_AREA_MAX: int = SPAWN_AREA_MIN + SPAWN_AREA_SIZE - 1  # 8

# --- Guaranteed-dirt rings flanking the play area. ---
OUTER_RING_WIDTH: int = 1
INNER_RING_WIDTH: int = 1

# No-patch zone = spawn area expanded by the inner ring on every side.
INNER_ZONE_MIN: int = SPAWN_AREA_MIN - INNER_RING_WIDTH  # 6
INNER_ZONE_MAX: int = SPAWN_AREA_MAX + INNER_RING_WIDTH  # 9

# Patch corners must keep the 2x2 patch fully inside the inner play area
# (i.e. clear of the outer ring on every side).
MIN_CORNER: int = OUTER_RING_WIDTH  # 1
MAX_CORNER: int = MAP_SIZE - PATCH_SIZE - OUTER_RING_WIDTH  # 13

# Active-agent spawn cell. One of the four cells of the 2x2 spawn area;
# future multi-agent setups fill the other three.
SPAWN: tuple[int, int] = (SPAWN_AREA_MAX, SPAWN_AREA_MAX)  # (8, 8)

PATCH_BLOCKS: tuple[BlockType, ...] = (
    BlockType.IRON,
    BlockType.COPPER,
    BlockType.TIN,
    BlockType.SILICON,
    BlockType.COAL,
    BlockType.LIMESTONE,
)


def patch_touches_inner_zone(px: int, py: int) -> bool:
    """True if a 2x2 patch at ``(px, py)`` overlaps the spawn area or its
    inner-ring buffer.

    Parameters
    ----------
    px, py :
        Top-left corner of the patch.
    """
    for dy in range(PATCH_SIZE):
        for dx in range(PATCH_SIZE):
            tx, ty = px + dx, py + dy
            if (
                INNER_ZONE_MIN <= tx <= INNER_ZONE_MAX
                and INNER_ZONE_MIN <= ty <= INNER_ZONE_MAX
            ):
                return True
    return False


def patches_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True if 2x2 patches with corners ``a`` and ``b`` share any tile."""
    return abs(a[0] - b[0]) < PATCH_SIZE and abs(a[1] - b[1]) < PATCH_SIZE


def sample_patch_corner(
    key: jax.Array, placed: list[tuple[int, int]]
) -> tuple[int, int]:
    """Host-side rejection sampler for one valid, non-overlapping patch corner.

    Parameters
    ----------
    key :
        PRNG key consumed by the rejection loop.
    placed :
        Corners already taken; the sample avoids overlap with all of them.

    Raises
    ------
    RuntimeError
        If no valid corner is found in :data:`MAX_SAMPLE_ATTEMPTS` draws.
    """
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        key, subkey = jax.random.split(key)
        coords = jax.random.randint(
            subkey, shape=(2,), minval=MIN_CORNER, maxval=MAX_CORNER + 1
        )
        corner = (int(coords[0]), int(coords[1]))
        if patch_touches_inner_zone(*corner):
            continue
        if any(patches_overlap(corner, other) for other in placed):
            continue
        return corner
    raise RuntimeError(
        f"sample_patch_corner: failed to place a patch in "
        f"{MAX_SAMPLE_ATTEMPTS} attempts."
    )


#: Every 2x2 patch corner inside the outer ring that does not touch the
#: inner spawn+ring zone, computed once. The JAX generator draws
#: non-overlapping patches from this fixed set, so the avoidance rules
#: match the host sampler exactly.
VALID_PATCH_CORNERS: np.ndarray = np.array(
    [
        (cx, cy)
        for cx in range(MIN_CORNER, MAX_CORNER + 1)
        for cy in range(MIN_CORNER, MAX_CORNER + 1)
        if not patch_touches_inner_zone(cx, cy)
    ],
    dtype=np.int32,
)


def six_patch_terrain(key: jax.Array, params: EnvParams) -> jax.Array:
    """Build a dirt map with one non-overlapping 2x2 patch per ore block.

    Jittable, vmappable placement: shuffle the valid (spawn-avoiding)
    corners with ``key`` and greedily take the first
    ``len(PATCH_BLOCKS)`` that do not overlap an already-placed patch.
    With ~200 candidates and six patches this always succeeds, so no
    rejection-failure branch is needed.

    Parameters
    ----------
    key :
        PRNG key deciding the layout.
    params :
        Unused; present for the ``TerrainFn`` signature.
    """
    corners = jnp.asarray(VALID_PATCH_CORNERS)
    shuffled = corners[jax.random.permutation(key, corners.shape[0])]
    n_patches = len(PATCH_BLOCKS)
    placed0 = jnp.full((n_patches, 2), -PATCH_SIZE, dtype=jnp.int32)

    def place(
        carry: tuple[jax.Array, jax.Array], cand: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        placed, count = carry
        dx = jnp.abs(placed[:, 0] - cand[0])
        dy = jnp.abs(placed[:, 1] - cand[1])
        filled = jnp.arange(n_patches) < count
        overlaps = jnp.any(filled & (dx < PATCH_SIZE) & (dy < PATCH_SIZE))
        do_place = (count < n_patches) & ~overlaps
        slot = jnp.minimum(count, n_patches - 1)
        placed = placed.at[slot].set(jnp.where(do_place, cand, placed[slot]))
        return (placed, count + do_place.astype(jnp.int32)), None

    (placed, _count), _ = jax.lax.scan(place, (placed0, jnp.int32(0)), shuffled)

    world = jnp.full((MAP_SIZE, MAP_SIZE), jnp.int8(BlockType.DIRT))
    for i, block in enumerate(PATCH_BLOCKS):
        patch = jnp.full((PATCH_SIZE, PATCH_SIZE), jnp.int8(int(block)))
        world = jax.lax.dynamic_update_slice(world, patch, (placed[i, 1], placed[i, 0]))
    return world


# ---------------------------------------------------------------------------
# State predicates — pure-JAX building blocks for achievement conditions
# ---------------------------------------------------------------------------


def blocks_under_active_miners(state: EnvState) -> tuple[jax.Array, jax.Array]:
    """Active-miner mask and the block type under each entity slot.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``active_mask`` is True for slots that hold an active miner.
        ``block_at_pos`` is the block under the entity's
        ``(ent_y, ent_x)`` tile, computed with clamped indices so
        inactive slots stay JIT-safe.
    """
    active = (state.ent_type == int(Machine.MINER)) & (state.ent_y >= 0)
    safe_y = jnp.maximum(state.ent_y, 0)
    safe_x = jnp.maximum(state.ent_x, 0)
    blocks = state.map[safe_y, safe_x]
    return active, blocks


def producing_miners(state: EnvState) -> tuple[jax.Array, jax.Array]:
    """Producing-miner mask and the block type under each entity slot.

    Like :func:`blocks_under_active_miners` but the mask also requires a
    non-empty output buffer, so a slot counts only once its miner has
    actually mined ore rather than merely being placed on an ore tile.
    """
    active, blocks = blocks_under_active_miners(state)
    producing = active & (state.ent_buf_count > 0)
    return producing, blocks
