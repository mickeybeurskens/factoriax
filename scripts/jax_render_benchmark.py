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
    ITEM_COLORS,
    NUM_INVENTORY_SLOTS,
    NUM_ITEM_TYPES,
    BlockType,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.envs.factoriax_env import FactoriaXEnv
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


def benchmark_cpu_step(
    params: EnvParams,
    num_steps: int,
) -> float:
    """Benchmark env stepping on CPU with a single environment.

    Runs step_env on the CPU JAX backend for one environment, giving
    the serial per-step cost. This is the true CPU simulation baseline
    to compare against GPU-batched stepping.

    Args:
        params: Environment parameters.
        num_steps: Number of timesteps.

    Returns:
        Wall-clock seconds for all step calls (excluding warmup).
    """
    cpu_device = jax.devices("cpu")[0]
    env = FactoriaXEnv()

    with jax.default_device(cpu_device):
        key = jax.random.key(55)
        _, state = env.reset_env(key, params)
        step_fn = jax.jit(env.step_env, device=cpu_device)

        # Warmup: JIT compile on CPU.
        rng = jax.random.key(56)
        rng, k_a, k_s = jax.random.split(rng, 3)
        action = jax.random.randint(k_a, (), 0, params.NUM_ACTIONS)
        _, state, _, _, _ = step_fn(k_s, state, action, params)
        jax.block_until_ready(jax.tree.leaves(state))

        t0 = time.perf_counter()
        for _ in range(num_steps):
            rng, k_a, k_s = jax.random.split(rng, 3)
            action = jax.random.randint(k_a, (), 0, params.NUM_ACTIONS)
            _, state, _, _, _ = step_fn(k_s, state, action, params)
        jax.block_until_ready(jax.tree.leaves(state))
        t1 = time.perf_counter()

    return t1 - t0


def benchmark_cpu_step_render(
    params: EnvParams,
    num_steps: int,
    tile_px: int,
) -> float:
    """Benchmark stepping + NumPy rendering on CPU for a single env.

    This is the true serial baseline: one env, CPU JAX step, then
    NumPy render_pixels each iteration. The number you'd get if you
    ran evaluation today without any GPU rendering.

    Args:
        params: Environment parameters.
        num_steps: Number of timesteps.
        tile_px: Tile pixel size for render_pixels.

    Returns:
        Wall-clock seconds for all step+render calls (excluding warmup).
    """
    cpu_device = jax.devices("cpu")[0]
    env = FactoriaXEnv()

    with jax.default_device(cpu_device):
        key = jax.random.key(55)
        _, state = env.reset_env(key, params)
        step_fn = jax.jit(env.step_env, device=cpu_device)

        # Warmup.
        rng = jax.random.key(56)
        rng, k_a, k_s = jax.random.split(rng, 3)
        action = jax.random.randint(k_a, (), 0, params.NUM_ACTIONS)
        _, state, _, _, _ = step_fn(k_s, state, action, params)
        jax.block_until_ready(jax.tree.leaves(state))
        render_pixels(state, block_pixel_size=tile_px)

        t0 = time.perf_counter()
        for _ in range(num_steps):
            rng, k_a, k_s = jax.random.split(rng, 3)
            action = jax.random.randint(k_a, (), 0, params.NUM_ACTIONS)
            _, state, _, _, _ = step_fn(k_s, state, action, params)
            render_pixels(state, block_pixel_size=tile_px)
        jax.block_until_ready(jax.tree.leaves(state))
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

    print(f"Sample images saved to {output_dir}/")


def save_throughput_chart(
    cpu_fps: float,
    step_render_fps: dict[int, float],
    step_render_inv_fps: dict[int, float],
    vector_fps: dict[int, float],
    output_dir: Path,
) -> None:
    """Save the throughput breakdown chart with three GPU series.

    For batch sizes common to all three series, shows grouped bars:
      - Step only (green): simulation ceiling.
      - Step + map render (blue): map-only JAX render.
      - Step + map + inventory render (purple): full render with UI.

    A horizontal dashed line shows the CPU step+render baseline.

    Args:
        cpu_fps: CPU step+render throughput (fps), single env.
        step_render_fps: batch_size -> step + map render fps.
        step_render_inv_fps: batch_size -> step + map + inventory fps.
        vector_fps: batch_size -> step-only fps.
        output_dir: Directory to write the chart into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_sizes = sorted(
        set(step_render_fps) & set(vector_fps) & set(step_render_inv_fps)
    )
    if not all_sizes:
        return

    labels = [str(n) for n in all_sizes]
    x = np.arange(len(labels))
    width = 0.25

    v_vals = np.array([vector_fps[n] for n in all_sizes])
    sr_vals = np.array([step_render_fps[n] for n in all_sizes])
    sri_vals = np.array([step_render_inv_fps[n] for n in all_sizes])

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.bar(
        x - width, v_vals, width, color="#5fbf5f",
        edgecolor="white", linewidth=0.5, label="Step only",
    )
    ax.bar(
        x, sr_vals, width, color="#5f8fd4",
        edgecolor="white", linewidth=0.5, label="Step + map render",
    )
    ax.bar(
        x + width, sri_vals, width, color="#8f5fd4",
        edgecolor="white", linewidth=0.5, label="Step + map + inventory",
    )

    ax.axhline(
        cpu_fps, color="#d45f5f", linestyle="--", linewidth=1.5,
        label=f"CPU step+render ({cpu_fps:,.0f} fps)",
    )

    # Value labels.
    for i in range(len(all_sizes)):
        for val, offset in [
            (v_vals[i], -width), (sr_vals[i], 0), (sri_vals[i], width)
        ]:
            if val > 0:
                ax.text(
                    x[i] + offset, val * 1.05, f"{val:,.0f}",
                    ha="center", va="bottom", fontsize=6, rotation=45,
                )

    ax.set_yscale("log")
    ax.set_xlabel("Batch size (envs)", fontsize=12)
    ax.set_ylabel("Steps per second", fontsize=12)
    ax.set_title(
        f"Throughput: step vs map render vs inventory render  "
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
    log(f"Inventory strip: {INV_HEIGHT}px tall")
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

    # Save sample images.
    save_sample_images(
        params, block_atlas, machine_atlas, player_sprite,
        item_colors, digit_atlas, output_dir,
    )

    # ---- CPU baseline: step + render (single env, serial) ----
    log("Measuring CPU step+render baseline (1 env)...", end="", flush=True)
    cpu_sr_time = benchmark_cpu_step_render(params, NUM_STEPS, TILE_PX)
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

    # ---- Series 2: step + map + inventory render ----
    step_render_inv_fps = discover_scaling(
        run_fn=lambda n: try_inline_step_render_inventory(
            n, params, NUM_STEPS,
            block_atlas, machine_atlas, player_sprite,
            item_colors, digit_atlas,
        ),
        label="Step + Map + Inventory Render",
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
        cpu_sr_fps, step_render_fps, step_render_inv_fps,
        vector_fps, output_dir,
    )

    log()
    log(f"All results saved to {output_dir}/")
    log_path.write_text("".join(log_lines))


if __name__ == "__main__":
    main()
