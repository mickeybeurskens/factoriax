"""JAX parallel renderer benchmark for FactoriaX.

This script compares two rendering approaches for FactoriaX environments:

  1. CPU renderer (existing): NumPy-based `render_pixels()` called once per
     environment in a Python loop. This is the current approach used for
     evaluation videos and screenshots.

  2. JAX renderer (new): A pure-JAX renderer that compiles to XLA and can
     be vmapped across a batch of environment states. Every operation is a
     JAX array op, so the entire render pipeline runs on GPU in parallel.

The benchmark sweeps from 1 to 2048 parallel environments, steps each for
200 timesteps, and renders every state. It measures wall-clock time for the
render calls only (not the env stepping), giving a direct comparison of
rendering throughput.

The JAX renderer is intentionally simplified compared to the full
`render_pixels()`. It renders terrain tiles, machine overlays, and player
sprites, which covers the core compositing patterns. The goal is to prove
the viability of the approach, not to be pixel-perfect.

Usage:
    uv run python scripts/jax_render_benchmark.py
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use("Agg")

from factoriax.constants import (
    DIRECTIONS,
    ITEM_COLORS,
    MAX_MACHINE_INVENTORY_SLOTS,
    NUM_INVENTORY_SLOTS,
    NUM_ITEM_TYPES,
    BlockType,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.recipes import (
    NUM_RECIPES,
    RECIPE_INPUT_COUNTS,
    RECIPE_INPUT_ITEMS,
    RECIPE_OUTPUTS,
    MAX_RECIPE_INPUTS,
)
from factoriax.renderer import render_pixels
from factoriax.state import EnvParams, EnvState

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAP_SIZE = 16  # 16x16 tile grid, small enough for fast iteration
TILE_PX = 8  # pixels per tile side, gives 128x128 images
NUM_STEPS = 200  # timesteps to render per benchmark run
# Geometrically spaced batch sizes from 1 to 2048. Powers of two are natural
# for GPU workloads because warp/wavefront sizes are powers of two, and
# non-power-of-two batches waste partial warps.
MAX_BATCH_POWER = 13  # Try up to 2^13 = 8192 before giving up.


# ===================================================================
# SECTION 1: Texture Atlas Construction
#
# The key insight for a JAX renderer is that we need all visual assets
# as JAX arrays *before* JIT compilation. NumPy renderers can call
# Python functions, load PNGs, use dicts, etc. inside the render loop.
# JAX cannot: everything must be a static array that XLA can trace.
#
# We solve this by building "texture atlases" -- 3D arrays where the
# first axis is indexed by the enum value (BlockType, MachineType, etc.)
# and the remaining axes are the pixel data. At render time, we just do
# atlas[state.map] which is a pure gather operation.
# ===================================================================


def _solid_tile(r: int, g: int, b: int, size: int) -> np.ndarray:
    """Create a solid-colored square tile.

    Args:
        r: Red channel (0-255).
        g: Green channel (0-255).
        b: Blue channel (0-255).
        size: Tile side length in pixels.

    Returns:
        uint8 array of shape (size, size, 3).
    """
    tile = np.empty((size, size, 3), dtype=np.uint8)
    tile[:, :] = (r, g, b)
    return tile


def _circle_tile(
    r: int, g: int, b: int, bg: tuple[int, int, int], size: int
) -> np.ndarray:
    """Create a tile with a filled circle on a background color.

    Used for player and biter sprites. The circle is centered and has
    radius = size//3, matching the existing renderer's proportions.

    Args:
        r: Circle red channel.
        g: Circle green channel.
        b: Circle blue channel.
        bg: Background (r, g, b) tuple. Use (0, 0, 0) for sprites that
            will be alpha-composited (we skip alpha here and use
            jnp.where compositing instead).
        size: Tile side length in pixels.

    Returns:
        uint8 array of shape (size, size, 3).
    """
    tile = np.full((size, size, 3), bg, dtype=np.uint8)
    center = size / 2.0
    radius_sq = (size / 3.0) ** 2
    ys, xs = np.mgrid[:size, :size]
    dist_sq = (xs - center + 0.5) ** 2 + (ys - center + 0.5) ** 2
    mask = dist_sq <= radius_sq
    tile[mask] = (r, g, b)
    return tile


def build_block_atlas(tile_px: int) -> jnp.ndarray:
    """Build a texture atlas for terrain block types.

    The atlas is indexed by BlockType integer values. Each entry is a
    (tile_px, tile_px, 3) RGB tile. Unknown/invalid block types map to
    the dirt texture (brown).

    This runs once at startup on CPU, then the result lives as a
    device-resident JAX array for the lifetime of the benchmark.

    Args:
        tile_px: Tile side length in pixels.

    Returns:
        JAX array of shape (num_block_types, tile_px, tile_px, 3).
    """
    # Map each BlockType to an RGB color. These match the defaults in
    # renderer.py's create_default_textures().
    colors = {
        BlockType.INVALID: (139, 90, 43),  # brown fallback
        BlockType.OUT_OF_BOUNDS: (30, 30, 30),
        BlockType.DIRT: (139, 90, 43),
        BlockType.WATER: (64, 164, 223),
        BlockType.IRON: (192, 192, 192),
        BlockType.COPPER: (184, 115, 51),
        BlockType.COAL: (54, 54, 54),
        BlockType.NEST: (90, 40, 60),
    }
    num_types = max(colors.keys()) + 1
    atlas = np.zeros((num_types, tile_px, tile_px, 3), dtype=np.uint8)
    for block_id, (r, g, b) in colors.items():
        atlas[int(block_id)] = _solid_tile(r, g, b, tile_px)
    return jnp.array(atlas)


def build_machine_atlas(tile_px: int) -> jnp.ndarray:
    """Build a texture atlas for machine overlays.

    Same idea as the block atlas but for machines. NONE maps to all-zeros
    (transparent in our compositing scheme). Each machine type gets a
    solid color square matching the ITEM_COLORS from constants.py.

    Args:
        tile_px: Tile side length in pixels.

    Returns:
        JAX array of shape (num_machine_types, tile_px, tile_px, 3).
    """
    colors = {
        MachineType.NONE: (0, 0, 0),  # "transparent" -- we mask this out
        MachineType.MINER: (0, 200, 0),
        MachineType.CHEST: (210, 190, 50),
        MachineType.ASSEMBLER: (160, 80, 200),
        MachineType.CONVEYOR_BELT: (220, 180, 50),
        MachineType.ARM: (80, 120, 200),
        MachineType.ROCKET: (240, 240, 240),
    }
    num_types = max(colors.keys()) + 1
    atlas = np.zeros((num_types, tile_px, tile_px, 3), dtype=np.uint8)
    for machine_id, (r, g, b) in colors.items():
        atlas[int(machine_id)] = _solid_tile(r, g, b, tile_px)
    return jnp.array(atlas)


def build_player_atlas(tile_px: int) -> jnp.ndarray:
    """Build a texture atlas for player sprites.

    Returns a single player sprite (red circle on black background).
    In a full implementation you'd have (num_players, num_directions)
    entries. For the benchmark, one sprite is enough to prove the
    compositing works.

    Args:
        tile_px: Tile side length in pixels.

    Returns:
        JAX array of shape (tile_px, tile_px, 3).
    """
    return jnp.array(_circle_tile(255, 100, 100, (0, 0, 0), tile_px))


# Inventory strip dimensions. These are independent of tile_px so the
# inventory stays readable even at small tile sizes.
INV_HEIGHT = 16  # pixels tall
DIGIT_H = 5  # pixel rows per digit glyph
DIGIT_W = 3  # pixel columns per digit glyph
SLOT_BG = np.array([40, 40, 40], dtype=np.uint8)
SLOT_BG_SELECTED = np.array([80, 80, 80], dtype=np.uint8)
ICON_SIZE = 6  # pixels per side for the item color swatch
COUNT_COLOR = np.array([220, 220, 220], dtype=np.uint8)


def build_item_color_atlas() -> jnp.ndarray:
    """Build an RGB color lookup indexed by ItemType.

    Args: None.

    Returns:
        JAX uint8 array of shape (NUM_ITEM_TYPES, 3).
    """
    atlas = np.zeros((NUM_ITEM_TYPES, 3), dtype=np.uint8)
    for item_id, rgb in ITEM_COLORS.items():
        atlas[int(item_id)] = rgb
    return jnp.array(atlas)


def build_digit_atlas() -> jnp.ndarray:
    """Build a 3x5 bitmap font atlas for digits 0-9.

    Each digit is a binary mask of shape (DIGIT_H, DIGIT_W). A 1 means
    the pixel is "on" (will be colored with COUNT_COLOR), 0 is transparent.
    The glyphs are intentionally blocky at 3x5 -- just enough to be
    readable at small sizes.

    Returns:
        JAX bool array of shape (10, DIGIT_H, DIGIT_W).
    """
    # Each string is 5 rows of 3 chars. '#' = on, '.' = off.
    glyphs = {
        0: "###" "#.#" "#.#" "#.#" "###",
        1: ".#." "##." ".#." ".#." "###",
        2: "###" "..#" "###" "#.." "###",
        3: "###" "..#" "###" "..#" "###",
        4: "#.#" "#.#" "###" "..#" "..#",
        5: "###" "#.." "###" "..#" "###",
        6: "###" "#.." "###" "#.#" "###",
        7: "###" "..#" "..#" "..#" "..#",
        8: "###" "#.#" "###" "#.#" "###",
        9: "###" "#.#" "###" "..#" "###",
    }
    atlas = np.zeros((10, DIGIT_H, DIGIT_W), dtype=np.bool_)
    for digit, chars in glyphs.items():
        for i, ch in enumerate(chars):
            atlas[digit, i // DIGIT_W, i % DIGIT_W] = ch == "#"
    return jnp.array(atlas)


# ===================================================================
# SECTION 2: Pure-JAX Renderer
#
# The renderer is a single pure function: state in, pixels out.
# No Python control flow, no NumPy, no side effects. This means JAX
# can trace it, compile it to XLA, and vmap it across a batch dimension.
#
# The rendering happens in three compositing layers, back to front:
#   Layer 1: Terrain tiles (gathered from block atlas via state.map)
#   Layer 2: Machine overlays (gathered from machine atlas, masked)
#   Layer 3: Player sprites (scatter-written at player positions)
#
# Each layer uses only array indexing, reshaping, and jnp.where for
# compositing. No alpha blending with floats needed because we use
# binary masks (machine present / not present, player here / not here).
# ===================================================================


def render_jax(
    state: EnvState,
    block_atlas: jnp.ndarray,
    machine_atlas: jnp.ndarray,
    player_sprite: jnp.ndarray,
) -> jnp.ndarray:
    """Render an environment state to an RGB image using pure JAX ops.

    This function is designed to be JIT-compiled and vmapped. Every
    operation is a JAX primitive (gather, scatter, where, reshape),
    so XLA can fuse them into efficient GPU kernels.

    The tile pixel size is inferred from the block_atlas shape rather
    than passed as an argument. This is important because JAX needs
    shapes to be compile-time constants for reshape operations, and
    atlas shapes are always static (known at trace time).

    Args:
        state: A single FactoriaX EnvState.
        block_atlas: Pre-built block texture atlas, shape
            (num_block_types, tile_px, tile_px, 3).
        machine_atlas: Pre-built machine texture atlas, shape
            (num_machine_types, tile_px, tile_px, 3).
        player_sprite: Player sprite, shape (tile_px, tile_px, 3).

    Returns:
        RGB uint8 image of shape (H * tile_px, W * tile_px, 3).
    """
    # Infer tile_px from the atlas. Atlas shapes are static (concrete at
    # trace time) because they're fixed-size arrays, not vmapped over.
    # This avoids passing tile_px as a traced integer, which would make
    # reshape fail since JAX needs concrete shape dimensions.
    tile_px = block_atlas.shape[1]
    map_h, map_w = state.map.shape

    # --- Layer 1: Terrain ---
    # state.map has shape (H, W) with integer BlockType values.
    # block_atlas[state.map] gathers one (tile_px, tile_px, 3) texture
    # per tile, producing shape (H, W, tile_px, tile_px, 3).
    # We then rearrange axes to interleave the tile pixels into a
    # contiguous image: (H, tile_px, W, tile_px, 3) -> (H*tile_px, W*tile_px, 3).
    safe_map = jnp.clip(state.map, 0, block_atlas.shape[0] - 1)
    tile_textures = block_atlas[safe_map]  # (H, W, tile_px, tile_px, 3)
    image = tile_textures.transpose(0, 2, 1, 3, 4).reshape(
        map_h * tile_px, map_w * tile_px, 3
    )

    # --- Layer 2: Machine overlays ---
    # Same gather pattern, but we only composite where a machine exists.
    # Machines are stored on the same (H, W) grid as terrain. We build
    # the full machine image, then use jnp.where with a per-pixel mask
    # to blend it onto the terrain.
    safe_machines = jnp.clip(state.machine_types, 0, machine_atlas.shape[0] - 1)
    machine_textures = machine_atlas[safe_machines]  # (H, W, tile_px, tile_px, 3)
    machine_image = machine_textures.transpose(0, 2, 1, 3, 4).reshape(
        map_h * tile_px, map_w * tile_px, 3
    )
    # Build the per-pixel mask: True where a machine is present.
    # state.machine_types is (H, W). We need (H*tile_px, W*tile_px, 1)
    # for broadcasting against the 3-channel image.
    has_machine = (state.machine_types != int(MachineType.NONE))  # (H, W)
    # Repeat each tile's boolean to cover its pixel extent.
    # jnp.repeat is fine here because the shapes are static (known at
    # trace time from map_h, map_w, tile_px).
    mask = jnp.repeat(jnp.repeat(has_machine, tile_px, axis=0), tile_px, axis=1)
    mask = mask[:, :, None]  # (H*tile_px, W*tile_px, 1) for broadcast
    image = jnp.where(mask, machine_image, image)

    # --- Layer 3: Player sprites ---
    # Players are sparse (typically 1-2), not grid-aligned in the same
    # way as tiles. We use dynamic_update_slice to "stamp" the sprite
    # at each player's pixel position. Since the number of players is
    # fixed at trace time, we unroll with fori_loop.
    #
    # fori_loop is the JAX equivalent of a for-loop that compiles to
    # a fixed-iteration XLA while loop. It's necessary because
    # dynamic_update_slice needs runtime indices (the player position),
    # which rules out static array indexing.
    num_players = state.player_positions.shape[0]

    def _stamp_player(i: int, img: jnp.ndarray) -> jnp.ndarray:
        """Composite one player sprite onto the image.

        Args:
            i: Player index.
            img: Current image being built up.

        Returns:
            Image with the player sprite stamped at the player's position.
        """
        px = state.player_positions[i, 0]  # x = column
        py = state.player_positions[i, 1]  # y = row
        # Convert tile coords to pixel coords.
        px_pixel = px * tile_px
        py_pixel = py * tile_px
        # dynamic_update_slice writes a small array into a larger one
        # at runtime-determined offsets. It compiles to an efficient
        # scatter on GPU.
        return jax.lax.dynamic_update_slice(
            img, player_sprite, (py_pixel, px_pixel, 0)
        )

    image = jax.lax.fori_loop(0, num_players, _stamp_player, image)

    return image.astype(jnp.uint8)


def render_inventory_strip(
    state: EnvState,
    item_colors: jnp.ndarray,
    digit_atlas: jnp.ndarray,
    img_width: int,
) -> jnp.ndarray:
    """Render the player-0 inventory as a horizontal strip in pure JAX.

    The strip is INV_HEIGHT pixels tall and img_width pixels wide. It
    contains NUM_INVENTORY_SLOTS evenly-spaced cells, each showing:
      - A colored item swatch (ICON_SIZE x ICON_SIZE).
      - A 2-digit stack count rendered from the digit atlas.
    The selected slot gets a brighter background.

    All operations are static-shape JAX primitives so this function
    composes with jit and vmap.

    Args:
        state: Single (non-batched) EnvState.
        item_colors: Item color atlas, shape (NUM_ITEM_TYPES, 3).
        digit_atlas: Digit bitmap atlas, shape (10, DIGIT_H, DIGIT_W).
        img_width: Width of the strip in pixels (must match map render).

    Returns:
        uint8 RGB array of shape (INV_HEIGHT, img_width, 3).
    """
    # Start with a dark background.
    strip = jnp.full((INV_HEIGHT, img_width, 3), SLOT_BG, dtype=jnp.uint8)

    slot_width = img_width // NUM_INVENTORY_SLOTS
    items = state.inventory_items[0]   # (NUM_INVENTORY_SLOTS,)
    counts = state.inventory_counts[0]  # (NUM_INVENTORY_SLOTS,)
    selected = state.selected_slots[0]

    def _draw_slot(slot_idx: int, img: jnp.ndarray) -> jnp.ndarray:
        """Draw one inventory slot into the strip.

        Args:
            slot_idx: Slot index (0 to NUM_INVENTORY_SLOTS-1).
            img: Strip image being built.

        Returns:
            Updated strip image.
        """
        x_start = slot_idx * slot_width

        # Highlight selected slot background.
        is_selected = slot_idx == selected
        bg = jnp.where(is_selected, SLOT_BG_SELECTED, SLOT_BG)
        slot_bg = jnp.broadcast_to(bg, (INV_HEIGHT, slot_width, 3))
        img = jax.lax.dynamic_update_slice(
            img, slot_bg.astype(jnp.uint8), (0, x_start, 0)
        )

        # Item color swatch: ICON_SIZE x ICON_SIZE, vertically centered.
        item_type = items[slot_idx]
        count = counts[slot_idx]
        color = item_colors[item_type]  # (3,)
        has_item = (item_type > 0) & (count > 0)

        swatch_y = (INV_HEIGHT - ICON_SIZE) // 2
        swatch_x = x_start + 1
        swatch = jnp.broadcast_to(color, (ICON_SIZE, ICON_SIZE, 3))
        # Only stamp if slot is non-empty. Multiply by has_item to zero
        # out the swatch for empty slots (keeps the dark background).
        swatch = swatch * has_item.astype(jnp.uint8)
        img = jax.lax.dynamic_update_slice(
            img, swatch, (swatch_y, swatch_x, 0)
        )

        # 2-digit count. Max stack is 64, so tens and ones suffice.
        safe_count = jnp.clip(count, 0, 99)
        tens = safe_count // 10
        ones = safe_count % 10

        digit_y = (INV_HEIGHT - DIGIT_H) // 2
        digit_x = swatch_x + ICON_SIZE + 1

        # Helper: read a patch from img using dynamic_slice (traced-safe),
        # blend the digit glyph onto it, write it back.
        def _stamp_digit(
            img: jnp.ndarray,
            digit_val: jnp.ndarray,
            dx: jnp.ndarray,
            show: jnp.ndarray,
        ) -> jnp.ndarray:
            """Stamp a single digit glyph onto the strip.

            Args:
                img: Strip image.
                digit_val: Digit 0-9 to render.
                dx: X position in the strip.
                show: Boolean, whether to actually draw.

            Returns:
                Updated strip image.
            """
            mask = digit_atlas[digit_val]  # (DIGIT_H, DIGIT_W)
            bg = jax.lax.dynamic_slice(
                img, (digit_y, dx, 0), (DIGIT_H, DIGIT_W, 3)
            )
            blended = jnp.where(mask[:, :, None], COUNT_COLOR, bg)
            result = jnp.where(show, blended, bg)
            return jax.lax.dynamic_update_slice(
                img, result.astype(jnp.uint8), (digit_y, dx, 0)
            )

        # Tens digit: only shown when count >= 10.
        show_tens = has_item & (safe_count >= 10)
        img = _stamp_digit(img, tens, digit_x, show_tens)

        # Ones digit: always shown when slot is non-empty.
        ones_x = digit_x + DIGIT_W + 1
        img = _stamp_digit(img, ones, ones_x, has_item)

        return img

    strip = jax.lax.fori_loop(0, NUM_INVENTORY_SLOTS, _draw_slot, strip)
    return strip


def render_jax_with_inventory(
    state: EnvState,
    block_atlas: jnp.ndarray,
    machine_atlas: jnp.ndarray,
    player_sprite: jnp.ndarray,
    item_colors: jnp.ndarray,
    digit_atlas: jnp.ndarray,
) -> jnp.ndarray:
    """Render map + inventory strip, concatenated vertically.

    Calls render_jax for the map, then render_inventory_strip, then
    stacks them so the inventory appears below the map at the same
    width. The output height is map_h * tile_px + INV_HEIGHT.

    Args:
        state: Single (non-batched) EnvState.
        block_atlas: Block texture atlas.
        machine_atlas: Machine texture atlas.
        player_sprite: Player sprite.
        item_colors: Item color atlas, shape (NUM_ITEM_TYPES, 3).
        digit_atlas: Digit bitmap atlas, shape (10, DIGIT_H, DIGIT_W).

    Returns:
        uint8 RGB array of shape (map_h * tile_px + INV_HEIGHT,
        map_w * tile_px, 3).
    """
    map_img = render_jax(state, block_atlas, machine_atlas, player_sprite)
    img_width = map_img.shape[1]
    inv_strip = render_inventory_strip(state, item_colors, digit_atlas, img_width)
    return jnp.concatenate([map_img, inv_strip], axis=0)


# ===================================================================
# SECTION 2b: Full HUD Renderer (4-quadrant info panel)
#
# The HUD is a panel the same size as the map, split into 4 quadrants:
#   Q1 (top-left):  Tile inspector -- info about the facing tile
#   Q2 (top-right): Machine inventory -- slots of the facing machine
#   Q3 (bot-left):  Player inventory -- 2x5 grid with items/counts
#   Q4 (bot-right): Crafting menu -- 5 recipes with affordability
#
# The full frame is: map (top) + HUD (bottom), same width.
# ===================================================================

HUD_BG = jnp.array([30, 30, 30], dtype=jnp.uint8)
HUD_BORDER = jnp.array([60, 60, 60], dtype=jnp.uint8)
HUD_LABEL = jnp.array([160, 160, 160], dtype=jnp.uint8)
HUD_GREEN = jnp.array([60, 200, 60], dtype=jnp.uint8)
HUD_RED = jnp.array([200, 60, 60], dtype=jnp.uint8)


def _get_facing_tile(state: EnvState) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the tile coordinates the player is facing.

    Args:
        state: Single EnvState.

    Returns:
        Tuple (fx, fy) as scalar int32 arrays, clamped to map bounds.
    """
    px = state.player_positions[0, 0]
    py = state.player_positions[0, 1]
    d = state.player_directions[0]
    dx = DIRECTIONS[d, 0]
    dy = DIRECTIONS[d, 1]
    map_h, map_w = state.map.shape
    fx = jnp.clip(px + dx, 0, map_w - 1)
    fy = jnp.clip(py + dy, 0, map_h - 1)
    return fx, fy


def _stamp_number(
    img: jnp.ndarray,
    digit_atlas: jnp.ndarray,
    value: jnp.ndarray,
    max_digits: int,
    y: int,
    x: jnp.ndarray,
    color: jnp.ndarray,
) -> jnp.ndarray:
    """Stamp a right-aligned integer into an image.

    Renders up to max_digits digits of value. Leading zeros are hidden.
    Each digit is DIGIT_H x DIGIT_W pixels with 1px spacing.

    Args:
        img: Image to draw on.
        digit_atlas: (10, DIGIT_H, DIGIT_W) bool atlas.
        value: Non-negative integer to render.
        max_digits: Maximum number of digit positions.
        y: Top row of the number.
        x: Left column of the first (leftmost) digit position.
        color: RGB color for the digits.

    Returns:
        Updated image.
    """
    safe_val = jnp.clip(value, 0, 10**max_digits - 1)

    def _stamp_one(i: int, carry: jnp.ndarray) -> jnp.ndarray:
        """Stamp digit position i (0 = leftmost).

        Args:
            i: Digit position from left.
            carry: Image being built.

        Returns:
            Updated image.
        """
        # Extract the i-th digit from the left.
        divisor = jnp.int32(10) ** jnp.int32(max_digits - 1 - i)
        digit_val = (safe_val // divisor) % 10
        # Only show if there are significant digits at this position.
        show = safe_val >= divisor
        mask = digit_atlas[digit_val]  # (DIGIT_H, DIGIT_W)
        dx = x + i * (DIGIT_W + 1)
        bg = jax.lax.dynamic_slice(
            carry, (y, dx, 0), (DIGIT_H, DIGIT_W, 3)
        )
        blended = jnp.where(mask[:, :, None], color, bg)
        result = jnp.where(show, blended, bg)
        return jax.lax.dynamic_update_slice(
            carry, result.astype(jnp.uint8), (y, dx, 0)
        )

    return jax.lax.fori_loop(0, max_digits, _stamp_one, img)


def render_q1_inspector(
    state: EnvState,
    block_atlas: jnp.ndarray,
    machine_atlas: jnp.ndarray,
    item_colors: jnp.ndarray,
    digit_atlas: jnp.ndarray,
    quad_h: int,
    quad_w: int,
) -> jnp.ndarray:
    """Render Q1: tile inspector showing info about the facing tile.

    Shows block type (color swatch + resource count), machine info
    (type swatch, direction arrow, health, working status), and biter
    presence with HP.

    Args:
        state: Single EnvState.
        block_atlas: Block texture atlas.
        machine_atlas: Machine texture atlas.
        item_colors: Item color atlas.
        digit_atlas: Digit bitmap atlas.
        quad_h: Quadrant height in pixels.
        quad_w: Quadrant width in pixels.

    Returns:
        uint8 RGB array of shape (quad_h, quad_w, 3).
    """
    img = jnp.full((quad_h, quad_w, 3), HUD_BG, dtype=jnp.uint8)
    fx, fy = _get_facing_tile(state)

    # Row 1 (y=2): Block type swatch + resource count
    block_type = state.map[fy, fx]
    safe_bt = jnp.clip(block_type, 0, block_atlas.shape[0] - 1)
    block_color = block_atlas[safe_bt, 0, 0, :]  # sample one pixel
    swatch = jnp.broadcast_to(block_color, (ICON_SIZE, ICON_SIZE, 3))
    img = jax.lax.dynamic_update_slice(img, swatch, (2, 2, 0))

    resources = state.block_resources[fy, fx]
    img = _stamp_number(
        img, digit_atlas, resources, 4, 2, jnp.int32(ICON_SIZE + 4),
        COUNT_COLOR,
    )

    # Row 2 (y=14): Machine type swatch + direction + status
    mt = state.machine_types[fy, fx]
    safe_mt = jnp.clip(mt, 0, machine_atlas.shape[0] - 1)
    m_color = machine_atlas[safe_mt, 0, 0, :]
    has_machine = mt != int(MachineType.NONE)
    m_swatch = jnp.broadcast_to(m_color, (ICON_SIZE, ICON_SIZE, 3))
    m_swatch = m_swatch * has_machine.astype(jnp.uint8)
    img = jax.lax.dynamic_update_slice(img, m_swatch, (14, 2, 0))

    # Direction indicator: small 3x3 colored dot offset by direction
    m_dir = state.machine_direction[fy, fx]
    dir_dx = DIRECTIONS[jnp.clip(m_dir, 0, 4), 0]
    dir_dy = DIRECTIONS[jnp.clip(m_dir, 0, 4), 1]
    arrow_dot = jnp.full((3, 3, 3), HUD_LABEL, dtype=jnp.uint8)
    arrow_dot = arrow_dot * has_machine.astype(jnp.uint8)
    arrow_y = 15 + dir_dy * 2
    arrow_x = ICON_SIZE + 4 + dir_dx * 2
    img = jax.lax.dynamic_update_slice(
        img, arrow_dot, (arrow_y, arrow_x, 0)
    )

    # Working status dot
    is_working = has_machine & (state.machine_power[fy, fx] > 0)
    status_color = jnp.where(is_working, HUD_GREEN, HUD_RED)
    status_dot = jnp.broadcast_to(status_color, (3, 3, 3))
    status_dot = status_dot * has_machine.astype(jnp.uint8)
    img = jax.lax.dynamic_update_slice(img, status_dot, (14, ICON_SIZE + 12, 0))

    # Machine health
    health = state.machine_health[fy, fx]
    img = _stamp_number(
        img, digit_atlas, health, 3, 14, jnp.int32(ICON_SIZE + 18),
        COUNT_COLOR,
    )

    # Row 3 (y=26): Biter check -- scan for any biter at (fx, fy)
    biter_at = (
        (state.biter_positions[:, 0] == fx)
        & (state.biter_positions[:, 1] == fy)
        & (state.biter_health > 0)
    )
    any_biter = jnp.any(biter_at)
    # Get the first matching biter's HP (or 0 if none)
    biter_idx = jnp.argmax(biter_at)  # first True index
    biter_hp = jnp.where(any_biter, state.biter_health[biter_idx], 0)

    biter_dot = jnp.broadcast_to(HUD_RED, (ICON_SIZE, ICON_SIZE, 3))
    biter_dot = biter_dot * any_biter.astype(jnp.uint8)
    img = jax.lax.dynamic_update_slice(img, biter_dot, (26, 2, 0))
    img = _stamp_number(
        img, digit_atlas, biter_hp, 2, 26, jnp.int32(ICON_SIZE + 4),
        HUD_RED,
    )

    return img


def render_q2_machine_inv(
    state: EnvState,
    item_colors: jnp.ndarray,
    digit_atlas: jnp.ndarray,
    quad_h: int,
    quad_w: int,
) -> jnp.ndarray:
    """Render Q2: machine inventory of the facing tile.

    Shows up to 8 machine slots in a 4-row x 2-column grid. Each slot
    has a colored item swatch and a 2-digit count. If no machine is
    present, the quadrant is blank.

    Args:
        state: Single EnvState.
        item_colors: Item color atlas.
        digit_atlas: Digit bitmap atlas.
        quad_h: Quadrant height in pixels.
        quad_w: Quadrant width in pixels.

    Returns:
        uint8 RGB array of shape (quad_h, quad_w, 3).
    """
    img = jnp.full((quad_h, quad_w, 3), HUD_BG, dtype=jnp.uint8)
    fx, fy = _get_facing_tile(state)
    mt = state.machine_types[fy, fx]
    has_machine = mt != int(MachineType.NONE)

    inv_items = state.machine_inventory_items[fy, fx, :]   # (8,)
    inv_counts = state.machine_inventory_counts[fy, fx, :]  # (8,)

    cell_w = quad_w // 2
    cell_h = quad_h // 4

    def _draw_mslot(slot: int, img: jnp.ndarray) -> jnp.ndarray:
        """Draw one machine inventory slot.

        Args:
            slot: Slot index 0-7.
            img: Image being built.

        Returns:
            Updated image.
        """
        col = slot % 2
        row = slot // 2
        cx = col * cell_w + 2
        cy = row * cell_h + 2

        item = inv_items[slot]
        count = inv_counts[slot]
        color = item_colors[jnp.clip(item, 0, NUM_ITEM_TYPES - 1)]
        has_item = has_machine & (item > 0) & (count > 0)

        swatch = jnp.broadcast_to(color, (ICON_SIZE, ICON_SIZE, 3))
        swatch = swatch * has_item.astype(jnp.uint8)
        img = jax.lax.dynamic_update_slice(img, swatch, (cy, cx, 0))

        img = _stamp_number(
            img, digit_atlas,
            jnp.where(has_item, count, 0),
            2, cy, jnp.int32(cx + ICON_SIZE + 2), COUNT_COLOR,
        )
        return img

    return jax.lax.fori_loop(0, MAX_MACHINE_INVENTORY_SLOTS, _draw_mslot, img)


def render_q3_inventory(
    state: EnvState,
    item_colors: jnp.ndarray,
    digit_atlas: jnp.ndarray,
    quad_h: int,
    quad_w: int,
) -> jnp.ndarray:
    """Render Q3: player inventory as a 2-row x 5-column grid.

    Each cell shows a colored item swatch and a 2-digit count. The
    selected slot has a brighter background.

    Args:
        state: Single EnvState.
        item_colors: Item color atlas.
        digit_atlas: Digit bitmap atlas.
        quad_h: Quadrant height in pixels.
        quad_w: Quadrant width in pixels.

    Returns:
        uint8 RGB array of shape (quad_h, quad_w, 3).
    """
    img = jnp.full((quad_h, quad_w, 3), HUD_BG, dtype=jnp.uint8)
    items = state.inventory_items[0]   # (10,)
    counts = state.inventory_counts[0]  # (10,)
    selected = state.selected_slots[0]

    cell_w = quad_w // 5
    cell_h = quad_h // 2

    def _draw_islot(slot: int, img: jnp.ndarray) -> jnp.ndarray:
        """Draw one player inventory slot.

        Args:
            slot: Slot index 0-9.
            img: Image being built.

        Returns:
            Updated image.
        """
        col = slot % 5
        row = slot // 5
        cx = col * cell_w
        cy = row * cell_h

        # Selected slot highlight
        is_sel = slot == selected
        bg_color = jnp.where(is_sel, SLOT_BG_SELECTED, SLOT_BG)
        cell_bg = jnp.broadcast_to(
            bg_color, (cell_h, cell_w, 3)
        ).astype(jnp.uint8)
        img = jax.lax.dynamic_update_slice(img, cell_bg, (cy, cx, 0))

        # Item swatch
        item = items[slot]
        count = counts[slot]
        color = item_colors[jnp.clip(item, 0, NUM_ITEM_TYPES - 1)]
        has_item = (item > 0) & (count > 0)

        swatch = jnp.broadcast_to(color, (ICON_SIZE, ICON_SIZE, 3))
        swatch = swatch * has_item.astype(jnp.uint8)
        img = jax.lax.dynamic_update_slice(
            img, swatch, (cy + 2, cx + 1, 0)
        )

        # Count digits below the swatch
        img = _stamp_number(
            img, digit_atlas,
            jnp.where(has_item, count, 0),
            2, cy + ICON_SIZE + 4, jnp.int32(cx + 1), COUNT_COLOR,
        )
        return img

    return jax.lax.fori_loop(0, NUM_INVENTORY_SLOTS, _draw_islot, img)


def render_q4_crafting(
    state: EnvState,
    item_colors: jnp.ndarray,
    digit_atlas: jnp.ndarray,
    quad_h: int,
    quad_w: int,
) -> jnp.ndarray:
    """Render Q4: player crafting menu.

    Shows the 5 player recipes. Each row has the output item swatch,
    input item swatches, and a green/red affordability dot. The
    currently-crafting recipe (if any) gets a progress indicator.

    Args:
        state: Single EnvState.
        item_colors: Item color atlas.
        digit_atlas: Digit bitmap atlas.
        quad_h: Quadrant height in pixels.
        quad_w: Quadrant width in pixels.

    Returns:
        uint8 RGB array of shape (quad_h, quad_w, 3).
    """
    img = jnp.full((quad_h, quad_w, 3), HUD_BG, dtype=jnp.uint8)
    inv_items = state.inventory_items[0]   # (10,)
    inv_counts = state.inventory_counts[0]  # (10,)
    craft_recipe = state.crafting_recipe[0]
    craft_progress = state.craft_progress[0]

    row_h = quad_h // NUM_RECIPES

    def _draw_recipe(r: int, img: jnp.ndarray) -> jnp.ndarray:
        """Draw one crafting recipe row.

        Args:
            r: Recipe index 0-4.
            img: Image being built.

        Returns:
            Updated image.
        """
        ry = r * row_h

        # Output item swatch
        out_item = RECIPE_OUTPUTS[r]
        out_color = item_colors[jnp.clip(out_item, 0, NUM_ITEM_TYPES - 1)]
        out_swatch = jnp.broadcast_to(
            out_color, (ICON_SIZE, ICON_SIZE, 3)
        )
        img = jax.lax.dynamic_update_slice(
            img, out_swatch, (ry + 1, 2, 0)
        )

        # Input item swatches (up to MAX_RECIPE_INPUTS)
        def _draw_input(j: int, img: jnp.ndarray) -> jnp.ndarray:
            in_item = RECIPE_INPUT_ITEMS[r, j]
            in_color = item_colors[
                jnp.clip(in_item, 0, NUM_ITEM_TYPES - 1)
            ]
            has_input = in_item > 0
            in_sw = jnp.broadcast_to(in_color, (4, 4, 3))
            in_sw = in_sw * has_input.astype(jnp.uint8)
            ix = ICON_SIZE + 5 + j * 6
            img = jax.lax.dynamic_update_slice(
                img, in_sw, (ry + 2, ix, 0)
            )
            return img

        img = jax.lax.fori_loop(0, MAX_RECIPE_INPUTS, _draw_input, img)

        # Affordability check: does the player have all inputs?
        def _check_input(j: int, affordable: jnp.ndarray) -> jnp.ndarray:
            needed_item = RECIPE_INPUT_ITEMS[r, j]
            needed_count = RECIPE_INPUT_COUNTS[r, j]
            # Find this item in inventory and sum counts.
            has_enough = jnp.where(
                needed_item == 0,
                True,
                jnp.sum(
                    jnp.where(inv_items == needed_item, inv_counts, 0)
                ) >= needed_count,
            )
            return affordable & has_enough

        can_afford = jax.lax.fori_loop(
            0, MAX_RECIPE_INPUTS, _check_input, jnp.bool_(True)
        )

        dot_color = jnp.where(can_afford, HUD_GREEN, HUD_RED)
        dot = jnp.broadcast_to(dot_color, (3, 3, 3))
        img = jax.lax.dynamic_update_slice(
            img, dot.astype(jnp.uint8),
            (ry + 2, quad_w - 6, 0),
        )

        # Crafting-in-progress highlight
        is_crafting = (craft_progress > 0) & (craft_recipe == r)
        bar_w = jnp.where(is_crafting, 4, 0).astype(jnp.int32)
        bar = jnp.broadcast_to(HUD_GREEN, (2, 4, 3))
        bar = bar * is_crafting.astype(jnp.uint8)
        img = jax.lax.dynamic_update_slice(
            img, bar.astype(jnp.uint8),
            (ry + row_h - 3, 2, 0),
        )

        return img

    return jax.lax.fori_loop(0, NUM_RECIPES, _draw_recipe, img)


def render_full_hud(
    state: EnvState,
    block_atlas: jnp.ndarray,
    machine_atlas: jnp.ndarray,
    player_sprite: jnp.ndarray,
    item_colors: jnp.ndarray,
    digit_atlas: jnp.ndarray,
) -> jnp.ndarray:
    """Render the complete frame: map on top, 4-quadrant HUD on bottom.

    The HUD is the same pixel dimensions as the map, split into four
    equal quadrants. The total output height is 2x the map height.

    Args:
        state: Single (non-batched) EnvState.
        block_atlas: Block texture atlas.
        machine_atlas: Machine texture atlas.
        player_sprite: Player sprite.
        item_colors: Item color atlas.
        digit_atlas: Digit bitmap atlas.

    Returns:
        uint8 RGB array of shape (2 * map_h * tile_px, map_w * tile_px, 3).
    """
    map_img = render_jax(state, block_atlas, machine_atlas, player_sprite)
    map_h, map_w = map_img.shape[0], map_img.shape[1]
    quad_h = map_h // 2
    quad_w = map_w // 2

    q1 = render_q1_inspector(
        state, block_atlas, machine_atlas, item_colors, digit_atlas,
        quad_h, quad_w,
    )
    q2 = render_q2_machine_inv(
        state, item_colors, digit_atlas, quad_h, quad_w,
    )
    q3 = render_q3_inventory(
        state, item_colors, digit_atlas, quad_h, quad_w,
    )
    q4 = render_q4_crafting(
        state, item_colors, digit_atlas, quad_h, quad_w,
    )

    # Draw 1px border between quadrants.
    border_h = jnp.full((1, map_w, 3), HUD_BORDER, dtype=jnp.uint8)
    border_v = jnp.full((quad_h, 1, 3), HUD_BORDER, dtype=jnp.uint8)

    top_row = jnp.concatenate([q1, border_v, q2], axis=1)
    # Trim 1px overflow from the vertical border to match map_w.
    top_row = top_row[:, :map_w, :]
    bot_row = jnp.concatenate([q3, border_v, q4], axis=1)
    bot_row = bot_row[:, :map_w, :]

    hud = jnp.concatenate([top_row, border_h, bot_row], axis=0)
    # Trim 1px overflow from horizontal border to match map_h.
    hud = hud[:map_h, :, :]

    return jnp.concatenate([map_img, hud], axis=0)


# ===================================================================
# SECTION 3: Batched Environment Setup
#
# FactoriaX environments are JAX pytrees (FLAX struct.dataclass).
# To run N environments in parallel, we vmap over the batch dimension.
# Each leaf array in the state gets an extra leading axis of size N.
#
# We use generate_state (procedural, JIT-compatible) to create each
# env independently from a different RNG key, then stack them into
# a single batched pytree.
# ===================================================================


def make_batched_envs(
    n: int, params: EnvParams
) -> tuple[jnp.ndarray, EnvState]:
    """Create N parallel environment states via vmapped reset.

    Each environment gets a unique random seed, producing different
    terrain layouts. The returned state is a pytree where every leaf
    has shape (N, ...) -- the standard layout for vmapped operations.

    Args:
        n: Number of parallel environments.
        params: Shared environment parameters (map size, etc.).

    Returns:
        Tuple of (rng_keys, batched_states) where rng_keys has shape
        (N, 2) and can be used for subsequent vmapped steps.
    """
    env = FactoriaXEnv()
    keys = jax.random.split(jax.random.key(42), n)
    # vmap reset_env over the key axis, broadcasting params.
    _, states = jax.vmap(env.reset_env, in_axes=(0, None))(keys, params)
    return keys, states


# ===================================================================
# SECTION 4: Benchmark Harness
#
# The benchmark measures rendering throughput in two modes:
#
# CPU mode: For each timestep, loop over N envs in Python and call
#   render_pixels(state_i). This is what you'd do today if you wanted
#   frames during training.
#
# JAX mode: For each timestep, call vmap(render_jax)(batched_state).
#   The entire batch is rendered in a single GPU kernel launch.
#
# We time only the render calls, not the env stepping, to isolate
# rendering performance. The env stepping uses random actions so the
# states evolve and we're not just re-rendering the same frame.
#
# The first call in each mode is a warmup (JIT compilation for JAX,
# cache warmup for CPU) and is excluded from timing.
# ===================================================================


def extract_single_state(batched_state: EnvState, idx: int) -> EnvState:
    """Extract a single environment state from a batched pytree.

    jax.tree.map slices each leaf at index `idx` along axis 0,
    reconstructing a single-env EnvState from the batched version.

    Args:
        batched_state: Batched state with shape (N, ...) leaves.
        idx: Index of the environment to extract.

    Returns:
        Single-env EnvState with original (non-batched) shapes.
    """
    return jax.tree.map(lambda x: x[idx], batched_state)


def benchmark_cpu_render(
    states_over_time: list[EnvState],
    n_envs: int,
    block_pixel_size: int,
) -> float:
    """Benchmark the CPU renderer over a sequence of batched states.

    For each timestep, extracts each individual env state from the batch
    and calls the NumPy-based render_pixels(). This is the serial baseline.

    Args:
        states_over_time: List of batched EnvState, one per timestep.
        n_envs: Number of environments in each batch.
        block_pixel_size: Pixel size per tile for render_pixels.

    Returns:
        Wall-clock seconds for all render calls (excluding warmup).
    """
    # Warmup: render one frame to populate texture caches.
    s0 = extract_single_state(states_over_time[0], 0)
    render_pixels(s0, block_pixel_size=block_pixel_size)

    t0 = time.perf_counter()
    for batched_state in states_over_time:
        for i in range(n_envs):
            single = extract_single_state(batched_state, i)
            render_pixels(single, block_pixel_size=block_pixel_size)
    t1 = time.perf_counter()
    return t1 - t0


def benchmark_jax_render(
    states_over_time: list[EnvState],
    block_atlas: jnp.ndarray,
    machine_atlas: jnp.ndarray,
    player_sprite: jnp.ndarray,
) -> float:
    """Benchmark the JAX renderer over a sequence of batched states.

    Calls vmap(render_jax) once per timestep. The first call triggers
    JIT compilation and is excluded from timing. All subsequent calls
    hit the compiled kernel directly.

    After the timed loop, we call jax.block_until_ready() on the last
    result to ensure all GPU work has actually completed. JAX dispatches
    asynchronously by default, so without this the timer would only
    measure kernel *launch* time, not execution time.

    Args:
        states_over_time: List of batched EnvState, one per timestep.
        block_atlas: Block texture atlas on device.
        machine_atlas: Machine texture atlas on device.
        player_sprite: Player sprite on device.

    Returns:
        Wall-clock seconds for all render calls (excluding JIT warmup).
    """
    # Build the vmapped render function. in_axes=(0, None, None, None)
    # means: vmap over the first arg (state batch axis), broadcast the rest.
    vmap_render = jax.jit(
        jax.vmap(render_jax, in_axes=(0, None, None, None))
    )

    # Warmup: trigger JIT compilation. This can take several seconds for
    # large batch sizes, but we don't count it.
    result = vmap_render(
        states_over_time[0], block_atlas, machine_atlas, player_sprite
    )
    jax.block_until_ready(result)

    t0 = time.perf_counter()
    for batched_state in states_over_time:
        result = vmap_render(
            batched_state, block_atlas, machine_atlas, player_sprite
        )
    # Block until the GPU finishes the last batch. Without this, we'd
    # only measure how fast Python can *enqueue* work, not how fast the
    # GPU can *finish* it.
    jax.block_until_ready(result)
    t1 = time.perf_counter()
    return t1 - t0


def collect_states(
    n_envs: int, params: EnvParams, num_steps: int
) -> list[EnvState]:
    """Step N environments for num_steps with random actions, saving states.

    This pre-computes all the states we'll render so that the benchmark
    loop only measures rendering, not env stepping. Actions are random
    to produce varied states (machines won't appear without crafting,
    but terrain and player positions will change).

    Args:
        n_envs: Number of parallel environments.
        params: Environment parameters.
        num_steps: Number of timesteps to simulate.

    Returns:
        List of batched EnvState, length num_steps.
    """
    env = FactoriaXEnv()
    keys, states = make_batched_envs(n_envs, params)

    # JIT-compile the vmapped step.
    vmap_step = jax.jit(
        jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
    )

    rng = jax.random.key(0)
    collected = [states]

    for _ in range(num_steps - 1):
        rng, key_act, key_step = jax.random.split(rng, 3)
        actions = jax.random.randint(
            key_act, (n_envs,), 0, params.NUM_ACTIONS
        )
        step_keys = jax.random.split(key_step, n_envs)
        _, states, _, _, _ = vmap_step(step_keys, states, actions, params)
        collected.append(states)

    # Make sure all states are materialized on device before benchmarking.
    jax.block_until_ready(jax.tree.leaves(collected[-1]))
    return collected


def benchmark_jax_render_inline(
    n_envs: int,
    params: EnvParams,
    num_steps: int,
    block_atlas: jnp.ndarray,
    machine_atlas: jnp.ndarray,
    player_sprite: jnp.ndarray,
) -> float:
    """Benchmark JAX rendering with inline stepping (no state storage).

    Steps and renders one timestep at a time, keeping only the current
    state in VRAM. This avoids OOM at large batch sizes. The step cost
    is included in the timing, but it's small relative to the render
    cost at large batches. Compare against the pre-collected approach
    at overlapping batch sizes to quantify the overhead.

    Args:
        n_envs: Number of parallel environments.
        params: Environment parameters.
        num_steps: Number of timesteps.
        block_atlas: Block texture atlas on device.
        machine_atlas: Machine texture atlas on device.
        player_sprite: Player sprite on device.

    Returns:
        Wall-clock seconds for all step+render calls (excluding warmup).
    """
    env = FactoriaXEnv()
    _, states = make_batched_envs(n_envs, params)

    vmap_step = jax.jit(
        jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
    )
    vmap_render = jax.jit(
        jax.vmap(render_jax, in_axes=(0, None, None, None))
    )

    # Warmup both kernels.
    rng = jax.random.key(99)
    rng, k_a, k_s = jax.random.split(rng, 3)
    actions = jax.random.randint(k_a, (n_envs,), 0, params.NUM_ACTIONS)
    step_keys = jax.random.split(k_s, n_envs)
    _, states, _, _, _ = vmap_step(step_keys, states, actions, params)
    result = vmap_render(states, block_atlas, machine_atlas, player_sprite)
    jax.block_until_ready(result)

    # Timed loop: step then render, discard frames immediately.
    t0 = time.perf_counter()
    for _ in range(num_steps):
        rng, k_a, k_s = jax.random.split(rng, 3)
        actions = jax.random.randint(k_a, (n_envs,), 0, params.NUM_ACTIONS)
        step_keys = jax.random.split(k_s, n_envs)
        _, states, _, _, _ = vmap_step(step_keys, states, actions, params)
        result = vmap_render(states, block_atlas, machine_atlas, player_sprite)
    jax.block_until_ready(result)
    t1 = time.perf_counter()
    return t1 - t0


def benchmark_jax_render_inventory_inline(
    n_envs: int,
    params: EnvParams,
    num_steps: int,
    block_atlas: jnp.ndarray,
    machine_atlas: jnp.ndarray,
    player_sprite: jnp.ndarray,
    item_colors: jnp.ndarray,
    digit_atlas: jnp.ndarray,
) -> float:
    """Benchmark JAX map+inventory rendering with inline stepping.

    Same as benchmark_jax_render_inline but uses render_jax_with_inventory
    which appends the inventory strip below the map each frame.

    Args:
        n_envs: Number of parallel environments.
        params: Environment parameters.
        num_steps: Number of timesteps.
        block_atlas: Block texture atlas on device.
        machine_atlas: Machine texture atlas on device.
        player_sprite: Player sprite on device.
        item_colors: Item color atlas on device.
        digit_atlas: Digit bitmap atlas on device.

    Returns:
        Wall-clock seconds for all step+render calls (excluding warmup).
    """
    env = FactoriaXEnv()
    _, states = make_batched_envs(n_envs, params)

    vmap_step = jax.jit(
        jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
    )
    vmap_render = jax.jit(
        jax.vmap(
            render_jax_with_inventory,
            in_axes=(0, None, None, None, None, None),
        )
    )

    # Warmup.
    rng = jax.random.key(88)
    rng, k_a, k_s = jax.random.split(rng, 3)
    actions = jax.random.randint(k_a, (n_envs,), 0, params.NUM_ACTIONS)
    step_keys = jax.random.split(k_s, n_envs)
    _, states, _, _, _ = vmap_step(step_keys, states, actions, params)
    result = vmap_render(
        states, block_atlas, machine_atlas, player_sprite,
        item_colors, digit_atlas,
    )
    jax.block_until_ready(result)

    t0 = time.perf_counter()
    for _ in range(num_steps):
        rng, k_a, k_s = jax.random.split(rng, 3)
        actions = jax.random.randint(k_a, (n_envs,), 0, params.NUM_ACTIONS)
        step_keys = jax.random.split(k_s, n_envs)
        _, states, _, _, _ = vmap_step(step_keys, states, actions, params)
        result = vmap_render(
            states, block_atlas, machine_atlas, player_sprite,
            item_colors, digit_atlas,
        )
    jax.block_until_ready(result)
    t1 = time.perf_counter()
    return t1 - t0


def try_inline_step_render_inventory(
    n_envs: int,
    params: EnvParams,
    num_steps: int,
    block_atlas: jnp.ndarray,
    machine_atlas: jnp.ndarray,
    player_sprite: jnp.ndarray,
    item_colors: jnp.ndarray,
    digit_atlas: jnp.ndarray,
) -> float | None:
    """Try benchmark_jax_render_inventory_inline, returning None on OOM.

    Args:
        n_envs: Number of parallel environments.
        params: Environment parameters.
        num_steps: Number of timesteps.
        block_atlas: Block texture atlas on device.
        machine_atlas: Machine texture atlas on device.
        player_sprite: Player sprite on device.
        item_colors: Item color atlas on device.
        digit_atlas: Digit bitmap atlas on device.

    Returns:
        Wall-clock seconds, or None on OOM.
    """
    try:
        return benchmark_jax_render_inventory_inline(
            n_envs, params, num_steps,
            block_atlas, machine_atlas, player_sprite,
            item_colors, digit_atlas,
        )
    except (RuntimeError, jax.errors.JaxRuntimeError):
        return None


def benchmark_jax_render_hud_inline(
    n_envs: int,
    params: EnvParams,
    num_steps: int,
    block_atlas: jnp.ndarray,
    machine_atlas: jnp.ndarray,
    player_sprite: jnp.ndarray,
    item_colors: jnp.ndarray,
    digit_atlas: jnp.ndarray,
) -> float:
    """Benchmark JAX full HUD rendering with inline stepping.

    Same pattern as the inventory benchmark but renders the complete
    4-quadrant HUD (inspector, machine inv, player inv, crafting).

    Args:
        n_envs: Number of parallel environments.
        params: Environment parameters.
        num_steps: Number of timesteps.
        block_atlas: Block texture atlas on device.
        machine_atlas: Machine texture atlas on device.
        player_sprite: Player sprite on device.
        item_colors: Item color atlas on device.
        digit_atlas: Digit bitmap atlas on device.

    Returns:
        Wall-clock seconds for all step+render calls (excluding warmup).
    """
    env = FactoriaXEnv()
    _, states = make_batched_envs(n_envs, params)

    vmap_step = jax.jit(
        jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
    )
    vmap_render = jax.jit(
        jax.vmap(
            render_full_hud,
            in_axes=(0, None, None, None, None, None),
        )
    )

    # Warmup.
    rng = jax.random.key(66)
    rng, k_a, k_s = jax.random.split(rng, 3)
    actions = jax.random.randint(k_a, (n_envs,), 0, params.NUM_ACTIONS)
    step_keys = jax.random.split(k_s, n_envs)
    _, states, _, _, _ = vmap_step(step_keys, states, actions, params)
    result = vmap_render(
        states, block_atlas, machine_atlas, player_sprite,
        item_colors, digit_atlas,
    )
    jax.block_until_ready(result)

    t0 = time.perf_counter()
    for _ in range(num_steps):
        rng, k_a, k_s = jax.random.split(rng, 3)
        actions = jax.random.randint(k_a, (n_envs,), 0, params.NUM_ACTIONS)
        step_keys = jax.random.split(k_s, n_envs)
        _, states, _, _, _ = vmap_step(step_keys, states, actions, params)
        result = vmap_render(
            states, block_atlas, machine_atlas, player_sprite,
            item_colors, digit_atlas,
        )
    jax.block_until_ready(result)
    t1 = time.perf_counter()
    return t1 - t0


def try_inline_step_render_hud(
    n_envs: int,
    params: EnvParams,
    num_steps: int,
    block_atlas: jnp.ndarray,
    machine_atlas: jnp.ndarray,
    player_sprite: jnp.ndarray,
    item_colors: jnp.ndarray,
    digit_atlas: jnp.ndarray,
) -> float | None:
    """Try benchmark_jax_render_hud_inline, returning None on OOM.

    Args:
        n_envs: Number of parallel environments.
        params: Environment parameters.
        num_steps: Number of timesteps.
        block_atlas: Block texture atlas on device.
        machine_atlas: Machine texture atlas on device.
        player_sprite: Player sprite on device.
        item_colors: Item color atlas on device.
        digit_atlas: Digit bitmap atlas on device.

    Returns:
        Wall-clock seconds, or None on OOM.
    """
    try:
        return benchmark_jax_render_hud_inline(
            n_envs, params, num_steps,
            block_atlas, machine_atlas, player_sprite,
            item_colors, digit_atlas,
        )
    except (RuntimeError, jax.errors.JaxRuntimeError):
        return None


def benchmark_vector_step(
    n_envs: int,
    params: EnvParams,
    num_steps: int,
) -> float:
    """Benchmark vmapped env stepping only (no rendering).

    Measures the pure simulation throughput as a ceiling for what
    rendering could achieve if it were free.

    Args:
        n_envs: Number of parallel environments.
        params: Environment parameters.
        num_steps: Number of timesteps.

    Returns:
        Wall-clock seconds for all step calls (excluding warmup).
    """
    env = FactoriaXEnv()
    _, states = make_batched_envs(n_envs, params)

    vmap_step = jax.jit(
        jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
    )

    # Warmup.
    rng = jax.random.key(77)
    rng, k_a, k_s = jax.random.split(rng, 3)
    actions = jax.random.randint(k_a, (n_envs,), 0, params.NUM_ACTIONS)
    step_keys = jax.random.split(k_s, n_envs)
    _, states, _, _, _ = vmap_step(step_keys, states, actions, params)
    jax.block_until_ready(jax.tree.leaves(states))

    t0 = time.perf_counter()
    for _ in range(num_steps):
        rng, k_a, k_s = jax.random.split(rng, 3)
        actions = jax.random.randint(k_a, (n_envs,), 0, params.NUM_ACTIONS)
        step_keys = jax.random.split(k_s, n_envs)
        _, states, _, _, _ = vmap_step(step_keys, states, actions, params)
    jax.block_until_ready(jax.tree.leaves(states))
    t1 = time.perf_counter()
    return t1 - t0



def benchmark_cpu_step_render(
    params: EnvParams,
    num_steps: int,
    tile_px: int,
    log_fn: object = None,
) -> float:
    """Benchmark single-env step + NumPy render on the CPU backend.

    Runs step_env JIT-compiled on the CPU device for one environment,
    then renders with the NumPy renderer each frame. This is the true
    serial CPU baseline.

    The CPU XLA compilation of step_env is slow (~5-15 min depending
    on hardware) because the function is large. Progress is printed
    via log_fn so it's not a silent hang.

    Args:
        params: Environment parameters.
        num_steps: Number of timesteps.
        tile_px: Tile pixel size for render_pixels.
        log_fn: Optional logging function for progress messages.

    Returns:
        Wall-clock seconds for all step+render calls (excluding warmup).
    """
    _log = log_fn or (lambda *a, **kw: None)

    cpu_device = jax.devices("cpu")[0]
    env = FactoriaXEnv()

    with jax.default_device(cpu_device):
        key = jax.random.key(55)
        _, state = env.reset_env(key, params)
        step_fn = jax.jit(env.step_env, device=cpu_device)

        # Warmup: triggers CPU XLA compilation (slow).
        _log(" compiling on CPU...", end="", flush=True)
        rng = jax.random.key(56)
        rng, k_a, k_s = jax.random.split(rng, 3)
        action = jax.random.randint(k_a, (), 0, params.NUM_ACTIONS)
        _, state, _, _, _ = step_fn(k_s, state, action, params)
        jax.block_until_ready(jax.tree.leaves(state))
        render_pixels(state, block_pixel_size=tile_px)
        _log(" done.", end="", flush=True)

        t0 = time.perf_counter()
        for _ in range(num_steps):
            rng, k_a, k_s = jax.random.split(rng, 3)
            action = jax.random.randint(k_a, (), 0, params.NUM_ACTIONS)
            _, state, _, _, _ = step_fn(k_s, state, action, params)
            jax.block_until_ready(jax.tree.leaves(state))
            render_pixels(state, block_pixel_size=tile_px)
        t1 = time.perf_counter()

    return t1 - t0


def try_inline_step_render(
    n_envs: int,
    params: EnvParams,
    num_steps: int,
    block_atlas: jnp.ndarray,
    machine_atlas: jnp.ndarray,
    player_sprite: jnp.ndarray,
) -> float | None:
    """Try benchmark_jax_render_inline, returning None on OOM.

    Wraps the inline benchmark in a try/except so the auto-scaling
    loop can probe increasingly large batch sizes until the GPU runs
    out of memory. JAX raises RuntimeError or JaxRuntimeError on OOM
    during kernel execution (not during tracing), so we catch both.

    Args:
        n_envs: Number of parallel environments.
        params: Environment parameters.
        num_steps: Number of timesteps.
        block_atlas: Block texture atlas on device.
        machine_atlas: Machine texture atlas on device.
        player_sprite: Player sprite on device.

    Returns:
        Wall-clock seconds, or None if the batch size caused OOM.
    """
    try:
        return benchmark_jax_render_inline(
            n_envs, params, num_steps,
            block_atlas, machine_atlas, player_sprite,
        )
    except (RuntimeError, jax.errors.JaxRuntimeError):
        return None


def try_vector_step(
    n_envs: int,
    params: EnvParams,
    num_steps: int,
) -> float | None:
    """Try benchmark_vector_step, returning None on OOM.

    Args:
        n_envs: Number of parallel environments.
        params: Environment parameters.
        num_steps: Number of timesteps.

    Returns:
        Wall-clock seconds, or None if the batch size caused OOM.
    """
    try:
        return benchmark_vector_step(n_envs, params, num_steps)
    except (RuntimeError, jax.errors.JaxRuntimeError):
        return None


def discover_scaling(
    run_fn: callable,
    label: str,
    log_fn: callable,
) -> dict[int, float]:
    """Double batch size until OOM, recording fps at each step.

    Starts at batch=1 and doubles each iteration (1, 2, 4, 8, ...),
    calling run_fn(n_envs) for each. If run_fn returns None (OOM),
    stops and returns the results collected so far.

    Args:
        run_fn: Callable(n_envs) -> seconds | None.
        label: Human label for log output (e.g. "step+render").
        log_fn: Logging function with same signature as print.

    Returns:
        Dict mapping batch_size -> fps for all successful runs.
    """
    results: dict[int, float] = {}
    header = f"{'n_envs':>8} | {'time (s)':>10} | {'fps':>12}"
    log_fn(f"--- {label} ---")
    log_fn(header)
    log_fn("-" * len(header))

    for power in range(MAX_BATCH_POWER + 1):
        n = 1 << power  # 2^power
        elapsed = run_fn(n)
        if elapsed is None:
            log_fn(f"{n:>8} | {'OOM':>10} |")
            break
        total_frames = n * NUM_STEPS
        fps = total_frames / elapsed if elapsed > 0 else float("inf")
        results[n] = fps
        log_fn(f"{n:>8} | {elapsed:>10.3f} | {fps:>12,.0f}")

    return results


def save_sample_images(
    params: EnvParams,
    block_atlas: jnp.ndarray,
    machine_atlas: jnp.ndarray,
    player_sprite: jnp.ndarray,
    item_colors: jnp.ndarray,
    digit_atlas: jnp.ndarray,
    output_dir: Path,
) -> None:
    """Save sample rendered images for visual comparison.

    Produces:
      - cpu_render.png: single env rendered by the NumPy CPU renderer.
      - jax_batch4.png: 4 envs with inventory, stitched 2x2 grid.

    Args:
        params: Environment parameters.
        block_atlas: Block texture atlas.
        machine_atlas: Machine texture atlas.
        player_sprite: Player sprite.
        item_colors: Item color atlas.
        digit_atlas: Digit bitmap atlas.
        output_dir: Directory to write images into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_px = int(block_atlas.shape[1])

    # --- CPU render: single environment ---
    _, single_states = make_batched_envs(1, params)
    state = extract_single_state(single_states, 0)
    cpu_img = render_pixels(state, block_pixel_size=tile_px)
    Image.fromarray(cpu_img[:, :, :3]).save(output_dir / "cpu_render.png")

    # --- JAX render with inventory: batch of 4, stitched 2x2 ---
    _, batch_states = make_batched_envs(4, params)
    vmap_render = jax.jit(
        jax.vmap(
            render_jax_with_inventory,
            in_axes=(0, None, None, None, None, None),
        )
    )
    batch_imgs = np.array(
        vmap_render(
            batch_states, block_atlas, machine_atlas, player_sprite,
            item_colors, digit_atlas,
        )
    )
    top_row = np.concatenate([batch_imgs[0], batch_imgs[1]], axis=1)
    bot_row = np.concatenate([batch_imgs[2], batch_imgs[3]], axis=1)
    grid = np.concatenate([top_row, bot_row], axis=0)
    Image.fromarray(grid).save(output_dir / "jax_batch4.png")

    # --- Full HUD render: single env with items in inventory ---
    # Give the player some items so the HUD has content to show.
    from factoriax.levels import LevelBuilder, build_state

    level = (
        LevelBuilder(params.map_width, params.map_height)
        .fill_rect(2, 2, 3, 3, BlockType.IRON, resources=500)
        .fill_rect(10, 2, 3, 3, BlockType.COPPER, resources=300)
        .fill_rect(2, 10, 3, 3, BlockType.COAL, resources=200)
        .place_machine(6, 6, int(MachineType.MINER), int(Direction.DOWN))
        .set_player_position(6, 5)  # facing the miner
        .build("hud_sample")
    )
    level.player_inventory = [
        (int(ItemType.IRON), 42),
        (int(ItemType.COPPER), 7),
        (int(ItemType.COAL), 1),
        (int(ItemType.MINER), 3),
        (int(ItemType.CHEST), 12),
    ]
    hud_state = build_state(level, params)
    hud_img = np.array(
        jax.jit(render_full_hud)(
            hud_state, block_atlas, machine_atlas, player_sprite,
            item_colors, digit_atlas,
        )
    )
    Image.fromarray(hud_img).save(output_dir / "jax_full_hud.png")

    print(f"Sample images saved to {output_dir}/")


def save_throughput_chart(
    cpu_fps: float,
    step_render_fps: dict[int, float],
    step_render_hud_fps: dict[int, float],
    vector_fps: dict[int, float],
    output_dir: Path,
) -> None:
    """Save the throughput chart with three GPU series.

    Grouped bars per batch size:
      - Step only (green): simulation ceiling.
      - Step + map render (blue): map-only JAX render.
      - Step + full HUD (orange): map + 4-quadrant HUD.

    A horizontal dashed line shows the CPU step+render baseline.

    Args:
        cpu_fps: CPU step+render throughput (fps), single env.
        step_render_fps: batch_size -> step + map render fps.
        step_render_hud_fps: batch_size -> step + full HUD fps.
        vector_fps: batch_size -> step-only fps.
        output_dir: Directory to write the chart into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_sizes = sorted(
        set(step_render_fps)
        & set(vector_fps)
        & set(step_render_hud_fps)
    )
    if not all_sizes:
        return

    labels = [str(n) for n in all_sizes]
    x = np.arange(len(labels))
    width = 0.25

    v_vals = np.array([vector_fps[n] for n in all_sizes])
    sr_vals = np.array([step_render_fps[n] for n in all_sizes])
    hud_vals = np.array([step_render_hud_fps[n] for n in all_sizes])

    fig, ax = plt.subplots(figsize=(14, 5))

    series = [
        (x - width, v_vals, "#5fbf5f", "Step only"),
        (x, sr_vals, "#5f8fd4", "Step + map render"),
        (x + width, hud_vals, "#d4a05f", "Step + full HUD"),
    ]
    for pos, vals, color, label in series:
        ax.bar(
            pos, vals, width, color=color,
            edgecolor="white", linewidth=0.5, label=label,
        )

    ax.axhline(
        cpu_fps, color="#d45f5f", linestyle="--", linewidth=1.5,
        label=f"CPU step+render ({cpu_fps:,.0f} fps)",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Batch size (envs)", fontsize=12)
    ax.set_ylabel("Steps per second", fontsize=12)
    ax.set_title(
        f"Throughput: step vs map render vs full HUD  "
        f"({MAP_SIZE}x{MAP_SIZE} map, {TILE_PX}px tiles, {NUM_STEPS} steps)",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    path = output_dir / "throughput_step_render.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Throughput chart saved to {path}")


def main() -> None:
    """Run the full benchmark suite."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path("scripts/batch_results") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "benchmark.log"
    log_lines: list[str] = []

    def log(msg: str = "", end: str = "\n", flush: bool = False) -> None:
        """Print to stdout and append to log buffer.

        Args:
            msg: Message to print.
            end: Line ending.
            flush: Whether to flush stdout.
        """
        print(msg, end=end, flush=flush)
        log_lines.append(msg + end)

    log("FactoriaX JAX Renderer Benchmark")
    log("=" * 60)
    log(f"Timestamp: {timestamp}")
    log(f"Map size: {MAP_SIZE}x{MAP_SIZE}, Tile pixels: {TILE_PX}px")
    log(f"Image size: {MAP_SIZE * TILE_PX}x{MAP_SIZE * TILE_PX}px")
    log(f"Full HUD: {MAP_SIZE * TILE_PX}x{2 * MAP_SIZE * TILE_PX}px")
    log(f"Steps per run: {NUM_STEPS}")
    log(f"Max batch power: 2^{MAX_BATCH_POWER} = {1 << MAX_BATCH_POWER}")
    log(f"JAX devices: {jax.devices()}")
    log(f"Output dir: {output_dir}")
    log()

    params = EnvParams(
        map_width=MAP_SIZE,
        map_height=MAP_SIZE,
        num_players=1,
        max_timesteps=NUM_STEPS,
        nest_probability=0.0,
        max_biters=1,
    )

    block_atlas = build_block_atlas(TILE_PX)
    machine_atlas = build_machine_atlas(TILE_PX)
    player_sprite = build_player_atlas(TILE_PX)
    item_colors = build_item_color_atlas()
    digit_atlas = build_digit_atlas()

    # Save sample images (includes HUD render with items).
    save_sample_images(
        params, block_atlas, machine_atlas, player_sprite,
        item_colors, digit_atlas, output_dir,
    )

    # ---- CPU baseline: step + render (single env, serial) ----
    log("Measuring CPU step+render baseline (1 env)...", end="", flush=True)
    cpu_sr_time = benchmark_cpu_step_render(params, NUM_STEPS, TILE_PX, log)
    cpu_sr_fps = NUM_STEPS / cpu_sr_time
    log(f" {cpu_sr_fps:,.0f} fps ({cpu_sr_time:.3f}s)")
    log()

    # ---- Series 1: step + map render ----
    step_render_fps = discover_scaling(
        run_fn=lambda n: try_inline_step_render(
            n, params, NUM_STEPS,
            block_atlas, machine_atlas, player_sprite,
        ),
        label="Step + Map Render",
        log_fn=log,
    )
    log()

    # ---- Series 2: step + full HUD render ----
    step_render_hud_fps = discover_scaling(
        run_fn=lambda n: try_inline_step_render_hud(
            n, params, NUM_STEPS,
            block_atlas, machine_atlas, player_sprite,
            item_colors, digit_atlas,
        ),
        label="Step + Full HUD Render",
        log_fn=log,
    )
    log()

    # ---- Series 3: step only ----
    vector_fps = discover_scaling(
        run_fn=lambda n: try_vector_step(n, params, NUM_STEPS),
        label="Step Only (no render)",
        log_fn=log,
    )

    # ---- Save outputs ----
    save_throughput_chart(
        cpu_sr_fps, step_render_fps,
        step_render_hud_fps, vector_fps, output_dir,
    )

    log()
    log(f"All results saved to {output_dir}/")
    log_path.write_text("".join(log_lines))


if __name__ == "__main__":
    main()
