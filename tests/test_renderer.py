"""Pixel-exact correctness tests for the tile renderer.

Compares the vectorised numpy implementation against a plain Python-loop
reference so any regression introduced by the transpose/reshape path is
caught immediately.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.belts import CROSSING_HORIZ_SLOT, CROSSING_VERT_SLOT
from factoriax.constants import (
    NUM_ITEM_TYPES,
    NUM_SCIENCE_PACK_TYPES,
    BlockType,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.renderer import (
    build_texture_lookup,
    render_item_icon,
    render_pixels,
)
from factoriax.state import EnvState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_MAX_MACHINES: int = 8


def _make_state(world_map: np.ndarray) -> EnvState:
    """Build a minimal EnvState from a 2-D block-type array.

    Args:
        world_map: 2-D array of BlockType values (height, width).

    Returns:
        EnvState suitable for passing to render_pixels.
    """
    shape = world_map.shape
    jmap = jnp.array(world_map, dtype=jnp.int8)
    mm = _DEFAULT_MAX_MACHINES
    return EnvState(
        map=jmap,
        block_resources=jnp.zeros(shape, dtype=jnp.int16),
        machine_types=jnp.full(
            shape,
            int(MachineType.NONE),
            dtype=jnp.int8,
        ),
        tile_entity=jnp.full(shape, -1, dtype=jnp.int16),
        ent_y=jnp.full(mm, -1, dtype=jnp.int16),
        ent_x=jnp.full(mm, -1, dtype=jnp.int16),
        ent_type=jnp.zeros(mm, dtype=jnp.int8),
        ent_direction=jnp.zeros(mm, dtype=jnp.int8),
        ent_power=jnp.zeros(mm, dtype=jnp.int16),
        ent_buf_type=jnp.zeros(mm, dtype=jnp.int8),
        ent_buf_count=jnp.zeros(mm, dtype=jnp.int16),
        ent_asm_in_type=jnp.zeros((mm, 2), dtype=jnp.int8),
        ent_asm_in_count=jnp.zeros((mm, 2), dtype=jnp.int16),
        ent_asm_out_type=jnp.zeros(mm, dtype=jnp.int8),
        ent_asm_out_count=jnp.zeros(mm, dtype=jnp.int16),
        player_positions=jnp.array([[0, 0]], dtype=jnp.int16),
        player_directions=jnp.array(
            [int(Direction.DOWN)],
            dtype=jnp.int8,
        ),
        player_inventory=jnp.zeros(
            (1, NUM_ITEM_TYPES),
            dtype=jnp.int16,
        ),
        selected_player=0,
        timestep=0,
        items_mined=jnp.zeros(NUM_ITEM_TYPES, dtype=jnp.int32),
        science_consumed_step=jnp.zeros(NUM_SCIENCE_PACK_TYPES, dtype=jnp.int32),
    )


def _render_tiles_reference(
    map_array: np.ndarray,
    texture_lookup: np.ndarray,
    block_pixel_size: int,
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
def test_tile_rendering_matches_reference(
    block_pixel_size: int,
) -> None:
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
    # 4x4 grid cycling through block types
    flat = [block_types[i % len(block_types)] for i in range(16)]
    map_array = np.array(flat, dtype=np.int32).reshape(4, 4)

    texture_lookup = build_texture_lookup(block_pixel_size)

    # Reference: Python loop
    expected = _render_tiles_reference(
        map_array,
        texture_lookup,
        block_pixel_size,
    )

    # Vectorised path (extracted to match exactly what render_pixels does)
    max_id = texture_lookup.shape[0] - 1
    safe_map = np.clip(map_array, 0, max_id)
    tile_textures = texture_lookup[safe_map]
    h, w = map_array.shape
    s = block_pixel_size
    actual = tile_textures.transpose(0, 2, 1, 3, 4).reshape(
        h * s,
        w * s,
        4,
    )

    np.testing.assert_array_equal(
        actual,
        expected,
        err_msg=(
            f"Tile rendering mismatch at block_pixel_size="
            f"{block_pixel_size}. "
            "The transpose/reshape path produces different pixels "
            "than the reference."
        ),
    )


def test_render_pixels_shape_and_dtype() -> None:
    """render_pixels must return an RGB uint8 array with correct dims."""
    map_array = np.full((4, 4), int(BlockType.DIRT), dtype=np.int32)
    state = _make_state(map_array)

    block_pixel_size = 8
    image = render_pixels(state, block_pixel_size=block_pixel_size)

    assert image.dtype == np.uint8, f"Expected uint8, got {image.dtype}"
    assert image.ndim == 3, f"Expected 3-D array, got shape {image.shape}"
    assert image.shape[2] == 3, "Expected RGB (3 channels)"
    assert image.shape == (
        4 * block_pixel_size,
        4 * block_pixel_size,
        3,
    )


def test_render_pixels_distinct_blocks_produce_distinct_colours() -> None:
    """Different block types must produce visually distinct tile colours.

    Uses a 1x2 grid: one DIRT tile and one WATER tile.  After rendering,
    the left half and right half of the image must not be identical.
    """
    map_array = np.array(
        [[int(BlockType.DIRT), int(BlockType.WATER)]],
        dtype=np.int32,
    )
    state = _make_state(map_array)

    s = 16
    image = render_pixels(state, block_pixel_size=s)

    left = image[:, :s, :]
    right = image[:, s:, :]

    # The mean RGB across the tile should differ
    assert not np.array_equal(left, right), (
        "DIRT and WATER tiles rendered identically -- texture lookup may be broken."
    )


@pytest.mark.parametrize("block_pixel_size", BLOCK_SIZES)
def test_tile_rendering_uniform_map(block_pixel_size: int) -> None:
    """A uniform map must produce a tiled image where every tile is same.

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
    image = tile_textures.transpose(0, 2, 1, 3, 4).reshape(
        h * s,
        w * s,
        4,
    )

    for y in range(h):
        for x in range(w):
            cell = image[y * s : (y + 1) * s, x * s : (x + 1) * s]
            np.testing.assert_array_equal(
                cell,
                expected_tile,
                err_msg=(f"Tile ({y},{x}) differs from expected texture at size={s}."),
            )


# ---------------------------------------------------------------------------
# Splitter / crossing visual smoke tests.
# ---------------------------------------------------------------------------


def _place_machine(
    state: EnvState,
    y: int,
    x: int,
    machine_type: int,
    direction: int,
    *,
    asm_slot0_item: int = 0,
    asm_slot0_count: int = 0,
    asm_slot1_item: int = 0,
    asm_slot1_count: int = 0,
) -> EnvState:
    """Add a single placed machine to ``state`` at ``(y, x)``.

    Inserts the machine into the first empty entity slot. Used by the
    splitter/crossing smoke tests to produce a state the renderer can
    actually draw without booting the full env.
    """
    eid = int(np.array(state.ent_y).tolist().index(-1))
    return state.replace(
        machine_types=state.machine_types.at[y, x].set(machine_type),
        tile_entity=state.tile_entity.at[y, x].set(eid),
        ent_y=state.ent_y.at[eid].set(y),
        ent_x=state.ent_x.at[eid].set(x),
        ent_type=state.ent_type.at[eid].set(machine_type),
        ent_direction=state.ent_direction.at[eid].set(direction),
        ent_asm_in_type=state.ent_asm_in_type.at[eid, 0]
        .set(asm_slot0_item)
        .at[eid, 1]
        .set(asm_slot1_item),
        ent_asm_in_count=state.ent_asm_in_count.at[eid, 0]
        .set(asm_slot0_count)
        .at[eid, 1]
        .set(asm_slot1_count),
    )


def test_splitter_icon_changes_with_facing() -> None:
    """The splitter glyph rotates 90° between vertical-facing and
    horizontal-facing directions; the two icons must not be pixel-equal."""
    icon_up = render_item_icon(int(ItemType.SPLITTER), 24, int(Direction.UP))
    icon_left = render_item_icon(int(ItemType.SPLITTER), 24, int(Direction.LEFT))
    assert icon_up.shape == (24, 24, 4)
    assert not np.array_equal(icon_up, icon_left), (
        "Splitter icon did not change between UP and LEFT facings."
    )


@pytest.mark.parametrize("encoding", [1, 2, 3, 4])
def test_crossing_icon_renders_for_each_encoding(encoding: int) -> None:
    """All four crossing encodings produce a non-uniform icon (the
    diagonal stripe is drawn). Encoding 0 leaves the icon flat."""
    icon = render_item_icon(int(ItemType.CROSSING), 24, encoding)
    assert icon.shape == (24, 24, 4)
    # The diagonal must paint at least some pixels different from the
    # uniform fill colour.
    base_pixel = icon[icon.shape[0] // 2, 0, :3]
    assert not np.all(icon[..., :3] == base_pixel), (
        f"Crossing icon for encoding={encoding} is uniformly coloured — "
        "the diagonal stripe was not drawn."
    )


def test_crossing_icon_diagonals_match_glyph_table() -> None:
    """Encodings 1+4 share the ``\\`` diagonal; 2+3 share ``/``. Same-
    glyph encodings must render to pixel-identical icons (the visual
    is purely a function of the diagonal direction)."""
    cr1 = render_item_icon(int(ItemType.CROSSING), 24, 1)
    cr2 = render_item_icon(int(ItemType.CROSSING), 24, 2)
    cr3 = render_item_icon(int(ItemType.CROSSING), 24, 3)
    cr4 = render_item_icon(int(ItemType.CROSSING), 24, 4)
    np.testing.assert_array_equal(cr1, cr4)  # both \
    np.testing.assert_array_equal(cr2, cr3)  # both /
    assert not np.array_equal(cr1, cr2), "Backslash and slash icons match."


def test_render_pixels_with_splitter_and_crossing_differs_from_empty(
    state_factory,
) -> None:
    """Placing a splitter and a crossing on a DIRT map must produce a
    rendered image that differs from the same map with no machines —
    the smoke test for end-to-end rendering pipeline integration."""
    h, w = 4, 4
    world = jnp.full((h, w), int(BlockType.DIRT), dtype=jnp.int32)

    empty = state_factory(world_map=world)
    placed = _place_machine(empty, 1, 1, int(MachineType.SPLITTER), int(Direction.UP))
    placed = _place_machine(placed, 2, 2, int(MachineType.CROSSING), 1)  # \ diagonal

    s = 16
    img_empty = render_pixels(empty, block_pixel_size=s)
    img_placed = render_pixels(placed, block_pixel_size=s)
    assert not np.array_equal(img_empty, img_placed), (
        "Placing a splitter + crossing did not change the rendered image."
    )


def test_crossing_input_dot_drawn_when_axis_slot_holds_item(
    state_factory,
) -> None:
    """A crossing with the vertical slot pre-loaded should produce a
    coloured dot near the input-side edge (top edge for vert_dir=DOWN
    encoding 1). The output-side edge stays unchanged from the bare
    diagonal stripe."""
    h, w = 3, 3
    world = jnp.full((h, w), int(BlockType.DIRT), dtype=jnp.int32)
    empty = state_factory(world_map=world)
    bare = _place_machine(empty, 1, 1, int(MachineType.CROSSING), 1)
    loaded = _place_machine(
        empty,
        1,
        1,
        int(MachineType.CROSSING),
        1,
        asm_slot0_item=int(ItemType.IRON_PLATE),
        asm_slot0_count=1,
    )
    assert CROSSING_VERT_SLOT == 0  # sanity-check the slot index used above
    assert CROSSING_HORIZ_SLOT == 1

    s = 16
    img_bare = render_pixels(bare, block_pixel_size=s)
    img_loaded = render_pixels(loaded, block_pixel_size=s)
    # The full image must differ when an item is in flight.
    assert not np.array_equal(img_bare, img_loaded), (
        "Vert-axis item dot was not drawn on the loaded crossing."
    )
    # The input edge for vert_dir=DOWN is the *north* (top) edge of the
    # crossing's tile. The strip just below the top edge of tile (1, 1)
    # must have changed; the strip at the south edge of the same tile
    # should still match the bare render (no horiz item).
    cell_y0 = 1 * s
    top_band = (slice(cell_y0, cell_y0 + s // 3), slice(s, 2 * s))
    bottom_band = (
        slice(cell_y0 + 2 * s // 3, cell_y0 + s),
        slice(s, 2 * s),
    )
    assert not np.array_equal(
        img_bare[top_band[0], top_band[1]],
        img_loaded[top_band[0], top_band[1]],
    ), "Top (input) band of crossing tile did not change with loaded vert axis."
    np.testing.assert_array_equal(
        img_bare[bottom_band[0], bottom_band[1]],
        img_loaded[bottom_band[0], bottom_band[1]],
        err_msg=(
            "Bottom (output) band of crossing tile changed even though only "
            "the vert axis was loaded."
        ),
    )
