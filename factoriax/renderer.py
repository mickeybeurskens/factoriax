"""Pixel rendering for the FactoriaX environment."""

import numpy as np

from factoriax.constants import (
    BLOCK_PIXEL_SIZE,
    ITEM_COLORS,
    NUM_INVENTORY_SLOTS,
    BlockType,
    load_all_textures,
)
from factoriax.state import EnvState

INVENTORY_BAR_HEIGHT = 24


def create_default_textures() -> dict[int, np.ndarray]:
    """Create simple default textures if asset files don't exist.

    Returns:
        Dictionary mapping BlockType values to RGBA texture arrays
    """
    size = BLOCK_PIXEL_SIZE
    textures: dict[int, np.ndarray] = {}

    dirt = np.zeros((size, size, 4), dtype=np.uint8)
    dirt[:, :, 0] = 139
    dirt[:, :, 1] = 90
    dirt[:, :, 2] = 43
    dirt[:, :, 3] = 255
    textures[int(BlockType.DIRT)] = dirt

    water = np.zeros((size, size, 4), dtype=np.uint8)
    water[:, :, 0] = 64
    water[:, :, 1] = 164
    water[:, :, 2] = 223
    water[:, :, 3] = 255
    textures[int(BlockType.WATER)] = water

    iron = np.zeros((size, size, 4), dtype=np.uint8)
    iron[:, :, 0] = 192
    iron[:, :, 1] = 192
    iron[:, :, 2] = 192
    iron[:, :, 3] = 255
    textures[int(BlockType.IRON)] = iron

    copper = np.zeros((size, size, 4), dtype=np.uint8)
    copper[:, :, 0] = 184
    copper[:, :, 1] = 115
    copper[:, :, 2] = 51
    copper[:, :, 3] = 255
    textures[int(BlockType.COPPER)] = copper

    coal = np.zeros((size, size, 4), dtype=np.uint8)
    coal[:, :, 0] = 54
    coal[:, :, 1] = 54
    coal[:, :, 2] = 54
    coal[:, :, 3] = 255
    textures[int(BlockType.COAL)] = coal

    return textures


def create_player_texture() -> np.ndarray:
    """Create a simple player texture.

    Returns:
        RGBA numpy array of shape (BLOCK_PIXEL_SIZE, BLOCK_PIXEL_SIZE, 4)
    """
    size = BLOCK_PIXEL_SIZE
    player = np.zeros((size, size, 4), dtype=np.uint8)
    center = size // 2
    radius = size // 3
    for y in range(size):
        for x in range(size):
            dist = ((x - center) ** 2 + (y - center) ** 2) ** 0.5
            if dist <= radius:
                player[y, x] = [255, 100, 100, 255]
    return player


def get_textures() -> dict[int, np.ndarray]:
    """Load textures from files, falling back to defaults if not found.

    Returns:
        Dictionary mapping BlockType values to RGBA texture arrays
    """
    try:
        return load_all_textures()
    except FileNotFoundError:
        return create_default_textures()


def render_inventory_bar(state: EnvState, width: int) -> np.ndarray:
    """Render inventory bar showing all 10 slots.

    Args:
        state: Current environment state containing inventory data
        width: Width of the bar in pixels (should match map render width)

    Returns:
        RGB numpy array of shape (INVENTORY_BAR_HEIGHT, width, 3)
    """
    bar = np.zeros((INVENTORY_BAR_HEIGHT, width, 3), dtype=np.uint8)
    bar[:, :] = (40, 40, 40)

    slot_width = width // NUM_INVENTORY_SLOTS
    slot_size = min(slot_width - 4, INVENTORY_BAR_HEIGHT - 4)

    inventory_items = np.array(state.inventory_items)
    inventory_counts = np.array(state.inventory_counts)

    for slot_idx in range(NUM_INVENTORY_SLOTS):
        x_center = slot_idx * slot_width + slot_width // 2
        x_start = x_center - slot_size // 2
        y_start = (INVENTORY_BAR_HEIGHT - slot_size) // 2

        bar[y_start : y_start + slot_size, x_start : x_start + slot_size] = (60, 60, 60)

        item_type = int(inventory_items[slot_idx])
        count = int(inventory_counts[slot_idx])

        if item_type != 0 and count > 0:
            pad = 2
            color = ITEM_COLORS.get(item_type, (128, 128, 128))
            bar[
                y_start + pad : y_start + slot_size - pad,
                x_start + pad : x_start + slot_size - pad,
            ] = color

    return bar


def render_pixels(
    state: EnvState, block_pixel_size: int = BLOCK_PIXEL_SIZE
) -> np.ndarray:
    """Render the environment state as an RGB pixel image.

    Args:
        state: Current environment state
        block_pixel_size: Size of each block in pixels

    Returns:
        RGB numpy array of the rendered scene
    """
    textures = get_textures()
    player_texture = create_player_texture()

    map_array = np.array(state.map)
    map_height, map_width = map_array.shape
    img_height = map_height * block_pixel_size
    img_width = map_width * block_pixel_size
    image = np.zeros((img_height, img_width, 4), dtype=np.uint8)

    for y in range(map_height):
        for x in range(map_width):
            block_type = int(map_array[y, x])
            if block_type in textures:
                texture = textures[block_type]
            else:
                texture = textures[int(BlockType.DIRT)]

            y_start = y * block_pixel_size
            x_start = x * block_pixel_size
            image[
                y_start : y_start + block_pixel_size,
                x_start : x_start + block_pixel_size,
            ] = texture

    player_pos = np.array(state.player_position)
    px, py = int(player_pos[0]), int(player_pos[1])
    py_start = py * block_pixel_size
    px_start = px * block_pixel_size
    _alpha_blend_inplace(
        image,
        player_texture,
        py_start,
        px_start,
        block_pixel_size,
    )

    image_rgb = image[:, :, :3]
    inv_bar = render_inventory_bar(state, img_width)
    return np.vstack([image_rgb, inv_bar])


def _alpha_blend_inplace(
    background: np.ndarray,
    foreground: np.ndarray,
    y_start: int,
    x_start: int,
    size: int,
) -> None:
    """Blend a foreground texture onto the background using alpha compositing.

    Args:
        background: RGBA background image to modify in place
        foreground: RGBA foreground texture
        y_start: Y coordinate of top-left corner
        x_start: X coordinate of top-left corner
        size: Size of the foreground texture
    """
    fg_alpha = foreground[:, :, 3:4].astype(np.float32) / 255.0
    bg_region = background[y_start : y_start + size, x_start : x_start + size]
    blended = foreground[:, :, :3].astype(np.float32) * fg_alpha + bg_region[
        :, :, :3
    ].astype(np.float32) * (1 - fg_alpha)
    bg_region[:, :, :3] = blended.astype(np.uint8)
    bg_region[:, :, 3] = 255
