"""The shared six-patch geometry and state predicates of the 16x16 scenarios.

EasyRocket-v1 and the miner-curriculum scenarios share one world geometry: a
16x16 dirt map with six 2x2 ore patches that do not overlap, one for each ore
block. A PRNG places them, and they stay clear of the centre spawn area and of
the outer ring.

This module owns that geometry. It holds the sampling helpers that run on the
host, and the jittable generator :func:`six_patch_terrain`. It also holds the
pure-JAX state predicates that a scenario joins into an achievement condition.
No scenario module therefore imports from another scenario module.
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

# --- The 2x2 spawn area at the centre. It has room for more agents. ---
SPAWN_AREA_SIZE: int = 2
SPAWN_AREA_MIN: int = (MAP_SIZE - SPAWN_AREA_SIZE) // 2  # 7
SPAWN_AREA_MAX: int = SPAWN_AREA_MIN + SPAWN_AREA_SIZE - 1  # 8

# --- The dirt rings on each side of the play area. ---
OUTER_RING_WIDTH: int = 1
INNER_RING_WIDTH: int = 1

# The zone that holds no patch: the spawn area, plus the inner ring on each
# side of it.
INNER_ZONE_MIN: int = SPAWN_AREA_MIN - INNER_RING_WIDTH  # 6
INNER_ZONE_MAX: int = SPAWN_AREA_MAX + INNER_RING_WIDTH  # 9

# A patch corner must keep the whole 2x2 patch inside the inner play area,
# which means clear of the outer ring on every side.
MIN_CORNER: int = OUTER_RING_WIDTH  # 1
MAX_CORNER: int = MAP_SIZE - PATCH_SIZE - OUTER_RING_WIDTH  # 13

# The spawn cell of the active agent. It is one of the four cells of the 2x2
# spawn area. A later multi-agent setup fills the other three.
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
    """Report whether a 2x2 patch lands on the spawn zone.

    The zone covers the 2x2 spawn area and the one-cell ring around it, so a
    player never starts on ore.

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
    """Report whether two 2x2 patches share a tile."""
    return abs(a[0] - b[0]) < PATCH_SIZE and abs(a[1] - b[1]) < PATCH_SIZE


def sample_patch_corner(
    key: jax.Array, placed: list[tuple[int, int]]
) -> tuple[int, int]:
    """Draw one valid patch corner on the host that overlaps no other patch.

    The function draws a corner and tests it. If the corner fails a test, the
    function draws again.

    Parameters
    ----------
    key :
        PRNG key for the draws.
    placed :
        Corners that other patches already hold. The result overlaps none of
        them.

    Raises
    ------
    RuntimeError
        If :data:`MAX_SAMPLE_ATTEMPTS` draws give no valid corner.
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


#: Every 2x2 patch corner inside the outer ring that does not touch the inner
#: spawn zone or its ring. This module computes the list one time. The JAX
#: generator draws its patches from this fixed set, so its rules match the
#: rules of the host sampler exactly.
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
    """Build a dirt map with one 2x2 patch for each ore block, with no overlap.

    The placement is jittable and vmappable. The function shuffles the valid
    corners, which already avoid the spawn zone, with ``key``. It then takes
    the first ``len(PATCH_BLOCKS)`` corners that overlap no patch already
    placed. There are about 200 candidates and six patches, so this always
    succeeds and needs no branch for a failure.

    Parameters
    ----------
    key :
        PRNG key that decides the layout.
    params :
        The function does not read this argument. It is present for the
        ``TerrainFn`` signature.
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
# State predicates: the pure-JAX parts of an achievement condition
# ---------------------------------------------------------------------------


def blocks_under_active_miners(state: EnvState) -> tuple[jax.Array, jax.Array]:
    """Return the active-miner mask and the block under each entity slot.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``active_mask`` is True for a slot that holds an active miner.
        ``block_at_pos`` is the block on the ``(ent_y, ent_x)`` tile of the
        entity. The function clamps the indices, so a free slot stays safe
        under JIT.
    """
    active = (state.ent_type == int(Machine.MINER)) & (state.ent_y >= 0)
    safe_y = jnp.maximum(state.ent_y, 0)
    safe_x = jnp.maximum(state.ent_x, 0)
    blocks = state.map[safe_y, safe_x]
    return active, blocks


def producing_miners(state: EnvState) -> tuple[jax.Array, jax.Array]:
    """Return the producing-miner mask and the block under each entity slot.

    This function is like :func:`blocks_under_active_miners`, but the mask also
    needs items in the output buffer. A slot therefore counts only after its
    miner mined ore, and not when a player only placed it on an ore tile.
    """
    active, blocks = blocks_under_active_miners(state)
    producing = active & (state.ent_buf_count > 0)
    return producing, blocks
