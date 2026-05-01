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


def test_crossing_does_not_draw_item_overlay(state_factory) -> None:
    """Crossings render only their diagonal sprite — the in-transit
    items in ``ent_asm_in`` slots are intentionally hidden so the
    sprite acts as a roof over both perpendicular flows. A loaded
    crossing must produce the *same* pixel image as a bare one."""
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
        asm_slot1_item=int(ItemType.COPPER_PLATE),
        asm_slot1_count=1,
    )
    assert CROSSING_VERT_SLOT == 0  # sanity-check the slot index used above
    assert CROSSING_HORIZ_SLOT == 1

    s = 16
    img_bare = render_pixels(bare, block_pixel_size=s)
    img_loaded = render_pixels(loaded, block_pixel_size=s)
    np.testing.assert_array_equal(
        img_bare,
        img_loaded,
        err_msg=(
            "Crossing renders changed with loaded ent_asm_in slots — the "
            "overlay is supposed to be omitted so the crossing sprite "
            "covers both items like a roof."
        ),
    )


def _place_belt_with_buffer(
    state: EnvState,
    y: int,
    x: int,
    machine_type: int,
    direction: int,
    *,
    buf_item: int,
    buf_count: int,
) -> EnvState:
    """Like _place_machine, but writes ent_buf_type / ent_buf_count
    instead of the assembler-input slots. Used by belt / pallet /
    splitter cargo-overlay tests."""
    eid = int(np.array(state.ent_y).tolist().index(-1))
    return state.replace(
        machine_types=state.machine_types.at[y, x].set(machine_type),
        tile_entity=state.tile_entity.at[y, x].set(eid),
        ent_y=state.ent_y.at[eid].set(y),
        ent_x=state.ent_x.at[eid].set(x),
        ent_type=state.ent_type.at[eid].set(machine_type),
        ent_direction=state.ent_direction.at[eid].set(direction),
        ent_buf_type=state.ent_buf_type.at[eid].set(buf_item),
        ent_buf_count=state.ent_buf_count.at[eid].set(buf_count),
    )


def _count_changed_pixels(bare: np.ndarray, loaded: np.ndarray) -> int:
    """Number of pixels (any channel changed) between two RGB images."""
    diff = np.any(bare != loaded, axis=-1)
    return int(diff.sum())


def test_belt_cargo_dot_is_smaller_than_quarter_tile(state_factory) -> None:
    """The new cargo dot is sized to ~1/8 of the tile per axis (was
    1/4). The total changed-pixel count of a single loaded belt's
    overlay must therefore be well under 1/16 of the tile area —
    use a generous bound that still rejects the old 1/4-axis sizing.
    """
    h, w = 3, 3
    world = jnp.full((h, w), int(BlockType.DIRT), dtype=jnp.int32)
    empty = state_factory(world_map=world)
    bare = _place_belt_with_buffer(
        empty,
        1,
        1,
        int(MachineType.CONVEYOR_BELT),
        int(Direction.RIGHT),
        buf_item=0,
        buf_count=0,
    )
    loaded = _place_belt_with_buffer(
        empty,
        1,
        1,
        int(MachineType.CONVEYOR_BELT),
        int(Direction.RIGHT),
        buf_item=int(ItemType.IRON_PLATE),
        buf_count=1,
    )
    s = 32
    img_bare = render_pixels(bare, block_pixel_size=s)
    img_loaded = render_pixels(loaded, block_pixel_size=s)
    changed = _count_changed_pixels(img_bare, img_loaded)
    # Old sizing: 1/4 axis dot + 1/3-of-dot border → outer ≈ 12 px,
    # so ~144 changed pixels. New sizing: 1/8 axis dot ≈ 4 px + 1 px
    # border → outer = 6 px, ~36 changed pixels. The bound 64 rejects
    # the old sizing while leaving comfortable slack.
    assert changed > 0, "Cargo overlay didn't draw at all."
    assert changed <= 64, (
        f"Cargo overlay touched {changed} pixels at block_pixel_size=32; "
        f"expected <= 64 for the new 1/8-axis dot. Did the size shrink?"
    )


def _splitter_overlay_runs(
    state_factory,
    buf_count: int,
) -> int:
    """Render a splitter at (1, 1) with the given buffer count; return
    the number of contiguous *runs* of changed pixels along the row
    through the tile centre. One run = one dot; two runs = the
    splitter pair fired both halves."""
    h, w = 3, 3
    world = jnp.full((h, w), int(BlockType.DIRT), dtype=jnp.int32)
    empty = state_factory(world_map=world)
    bare = _place_belt_with_buffer(
        empty,
        1,
        1,
        int(MachineType.SPLITTER),
        int(Direction.UP),
        buf_item=0,
        buf_count=0,
    )
    loaded = _place_belt_with_buffer(
        empty,
        1,
        1,
        int(MachineType.SPLITTER),
        int(Direction.UP),
        buf_item=int(ItemType.IRON_PLATE),
        buf_count=buf_count,
    )
    s = 32
    img_bare = render_pixels(bare, block_pixel_size=s)
    img_loaded = render_pixels(loaded, block_pixel_size=s)
    # Centre row of the splitter tile: (y * s + s/2) within the (1, 1)
    # tile is row index s + s // 2.
    centre_y = s + s // 2
    bare_row = img_bare[centre_y, s : 2 * s]
    loaded_row = img_loaded[centre_y, s : 2 * s]
    changed = np.any(bare_row != loaded_row, axis=-1)
    # Count contiguous runs of True.
    runs = 0
    in_run = False
    for v in changed:
        if v and not in_run:
            runs += 1
            in_run = True
        elif not v:
            in_run = False
    return runs


def test_splitter_with_buf_count_2_shows_two_dots(state_factory) -> None:
    """A splitter holding 2 items renders two horizontally-separated
    dots — the centre row through the tile must show two distinct
    runs of changed pixels."""
    runs = _splitter_overlay_runs(state_factory, buf_count=2)
    assert runs == 2, (
        f"Splitter with buf_count=2 produced {runs} dot run(s) on the centre "
        f"row; expected exactly 2."
    )


def test_splitter_with_buf_count_1_shows_one_dot(state_factory) -> None:
    """A splitter holding 1 item renders only the first of the pair —
    the centre row must show exactly one run of changed pixels."""
    runs = _splitter_overlay_runs(state_factory, buf_count=1)
    assert runs == 1, (
        f"Splitter with buf_count=1 produced {runs} dot run(s) on the centre "
        f"row; expected exactly 1."
    )
