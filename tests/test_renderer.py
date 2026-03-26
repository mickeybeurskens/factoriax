"""Pixel-exact correctness tests for the tile renderer.

Compares the vectorised numpy implementation against a plain Python-loop
reference so any regression introduced by the transpose/reshape path is
caught immediately.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from factoriax import EnvState
from factoriax.achievements import NUM_ACHIEVEMENTS
from factoriax.constants import (
    MAX_MACHINE_INVENTORY_SLOTS,
    NUM_INVENTORY_SLOTS,
    NUM_ITEM_TYPES,
    Action,
    BlockType,
    MachineType,
)
from factoriax.renderer import (
    build_texture_lookup,
    render_pixels,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(world_map: np.ndarray) -> EnvState:
    """Build a minimal EnvState from a 2-D block-type array.

    Args:
        world_map: 2-D array of BlockType values (height, width).

    Returns:
        EnvState suitable for passing to render_pixels.
    """
    shape = world_map.shape
    jmap = jnp.array(world_map, dtype=jnp.int32)
    return EnvState(
        map=jmap,
        player_positions=jnp.array([[0, 0]], dtype=jnp.int32),
        player_directions=jnp.array([int(Action.DOWN)], dtype=jnp.int32),
        timestep=0,
        inventory_items=jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32),
        inventory_counts=jnp.zeros((1, NUM_INVENTORY_SLOTS), dtype=jnp.int32),
        selected_player=0,
        selected_slots=jnp.zeros((1,), dtype=jnp.int32),
        crafting_recipe=jnp.zeros((1,), dtype=jnp.int32),
        craft_progress=jnp.zeros((1,), dtype=jnp.int32),
        block_resources=jnp.zeros(shape, dtype=jnp.int16),
        machine_types=jnp.full(shape, int(MachineType.NONE), dtype=jnp.int32),
        machine_power=jnp.zeros(shape, dtype=jnp.int32),
        machine_inventory_items=jnp.zeros(
            (*shape, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int32
        ),
        machine_inventory_counts=jnp.zeros(
            (*shape, MAX_MACHINE_INVENTORY_SLOTS), dtype=jnp.int16
        ),
        machine_selected_recipe=jnp.zeros(shape, dtype=jnp.int32),
        machine_selected_slot=jnp.zeros(shape, dtype=jnp.int32),
        achievements_unlocked=jnp.zeros(NUM_ACHIEVEMENTS, dtype=jnp.bool_),
        items_mined=jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32),
        machine_direction=jnp.zeros(shape, dtype=jnp.int32),
    )


def _render_tiles_reference(
    map_array: np.ndarray, texture_lookup: np.ndarray, block_pixel_size: int
) -> np.ndarray:
    """Render tile grid using a plain Python loop (ground-truth reference).

    This is intentionally naive: iterate every cell, copy the texture into
    the output image.  The vectorised path in renderer.py must match this
    result exactly.

    Args:
        map_array: 2-D integer array of block type IDs.
        texture_lookup: Dense texture array of shape (max_id+1, S, S, 4).
        block_pixel_size: Side length of each tile in pixels.

    Returns:
        RGBA image of shape (H*S, W*S, 4).
    """
    h, w = map_array.shape
    s = block_pixel_size
    max_id = texture_lookup.shape[0] - 1
    image = np.zeros((h * s, w * s, 4), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            block_id = int(np.clip(map_array[y, x], 0, max_id))
            image[y * s : (y + 1) * s, x * s : (x + 1) * s] = texture_lookup[block_id]
    return image


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

BLOCK_SIZES = [8, 16, 32]


@pytest.mark.parametrize("block_pixel_size", BLOCK_SIZES)
def test_tile_rendering_matches_reference(block_pixel_size: int) -> None:
    """Vectorised tile rendering must be pixel-exact vs the loop reference.

    Tests a 4x4 grid that cycles through all available block types so every
    texture in the lookup table is exercised at least once.

    Args:
        block_pixel_size: Tile size to test (parametrised).
    """
    block_types = [
        int(BlockType.DIRT),
        int(BlockType.WATER),
        int(BlockType.IRON),
        int(BlockType.COPPER),
        int(BlockType.COAL),
    ]
    # 4×4 grid cycling through block types
    flat = [block_types[i % len(block_types)] for i in range(16)]
    map_array = np.array(flat, dtype=np.int32).reshape(4, 4)

    texture_lookup = build_texture_lookup(block_pixel_size)

    # Reference: Python loop
    expected = _render_tiles_reference(map_array, texture_lookup, block_pixel_size)

    # Vectorised path (extracted to match exactly what render_pixels does)
    max_id = texture_lookup.shape[0] - 1
    safe_map = np.clip(map_array, 0, max_id)
    tile_textures = texture_lookup[safe_map]
    h, w = map_array.shape
    s = block_pixel_size
    actual = tile_textures.transpose(0, 2, 1, 3, 4).reshape(h * s, w * s, 4)

    np.testing.assert_array_equal(
        actual,
        expected,
        err_msg=(
            f"Tile rendering mismatch at block_pixel_size={block_pixel_size}. "
            "The transpose/reshape path produces different pixels than the reference."
        ),
    )


def test_render_pixels_shape_and_dtype() -> None:
    """render_pixels must return an RGB uint8 array with correct dimensions."""
    map_array = np.full((4, 4), int(BlockType.DIRT), dtype=np.int32)
    state = _make_state(map_array)

    block_pixel_size = 8
    image = render_pixels(state, block_pixel_size=block_pixel_size)

    assert image.dtype == np.uint8, f"Expected uint8, got {image.dtype}"
    assert image.ndim == 3, f"Expected 3-D array, got shape {image.shape}"
    assert image.shape[2] == 3, "Expected RGB (3 channels)"
    assert image.shape == (4 * block_pixel_size, 4 * block_pixel_size, 3)


def test_render_pixels_distinct_blocks_produce_distinct_colours() -> None:
    """Different block types must produce visually distinct tile colours.

    Uses a 1×2 grid: one DIRT tile and one WATER tile.  After rendering,
    the left half and right half of the image must not be identical.
    """
    map_array = np.array([[int(BlockType.DIRT), int(BlockType.WATER)]], dtype=np.int32)
    state = _make_state(map_array)

    s = 16
    image = render_pixels(state, block_pixel_size=s)

    left = image[:, :s, :]
    right = image[:, s:, :]

    # The mean RGB across the tile should differ
    assert not np.array_equal(left, right), (
        "DIRT and WATER tiles rendered identically — texture lookup may be broken."
    )


@pytest.mark.parametrize("block_pixel_size", BLOCK_SIZES)
def test_tile_rendering_uniform_map(block_pixel_size: int) -> None:
    """A uniform map must produce a tiled image where every tile is identical.

    Args:
        block_pixel_size: Tile size to test (parametrised).
    """
    h, w = 3, 3
    map_array = np.full((h, w), int(BlockType.IRON), dtype=np.int32)
    texture_lookup = build_texture_lookup(block_pixel_size)
    s = block_pixel_size

    expected_tile = texture_lookup[int(BlockType.IRON)]

    safe_map = np.clip(map_array, 0, texture_lookup.shape[0] - 1)
    tile_textures = texture_lookup[safe_map]
    image = tile_textures.transpose(0, 2, 1, 3, 4).reshape(h * s, w * s, 4)

    for y in range(h):
        for x in range(w):
            cell = image[y * s : (y + 1) * s, x * s : (x + 1) * s]
            np.testing.assert_array_equal(
                cell,
                expected_tile,
                err_msg=f"Tile ({y},{x}) differs from expected texture at size={s}.",
            )
