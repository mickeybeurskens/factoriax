"""Pure-JAX pixel renderer for FactoriaX.

A GPU-accelerated renderer that compiles to XLA and can be vmapped
across batched environment states. Every operation is a JAX primitive
(gather, scatter, where, reshape), so the entire render pipeline runs
in parallel on GPU.

The renderer works in layers, composited back-to-front:
  1. Terrain tiles (gathered from a block texture atlas via state.map)
  2. Machine overlays (gathered from machine atlas, masked where present)
  3. Player sprites (scatter-written at player positions)

For the full HUD, four additional quadrants are rendered below the map:
  Q1: Tile inspector (block type, resources, machine info, biters)
  Q2: Machine inventory (slots of the facing machine)
  Q3: Player inventory (2x5 grid with items and counts)
  Q4: Crafting menu (5 recipes with affordability indicators)

Sprites for blocks, machines, and the player come from the committed
sprite atlas at ``factoriax/assets/atlas.png``. Per-tile-size atlases
are derived by slicing the relevant atlas row and downsampling each
cell from 32×32 to ``tile_px`` via nearest-neighbour. The atlas is
RGBA: block cells are fully opaque, while machine and player cells
carry per-pixel alpha so :func:`render_map` can blend them onto the
layer underneath instead of overwriting it. Replacing the PNG (with
the same layout, see ``factoriax/assets/atlas.layout.md``) swaps in
new art without touching this code.

Directional machines (conveyor belts, miners, arms, splitters,
crossings) and the player sprite each carry four cells in the atlas,
one per :class:`~factoriax.constants.Direction`. ``render_map``
gathers by ``(direction, machine_type)`` for the machine layer and
by ``player_directions[i]`` for each player so placed objects show
their orientation.

Item-color and digit-glyph atlases are still procedural — they're
scalar lookups and bool masks rather than RGB sprites, so they don't
fit the (cells, 32, 32, 4) shape and don't need to. The atlas's
``items`` and ``digits`` rows are present for completeness but
unused by this renderer.

Usage::

    renderer = JaxRenderer(tile_px=8)
    img = renderer.render_map(state)       # map only
    img = renderer.render_hud(state)       # map + 4-quadrant HUD
    imgs = renderer.vmap_render_hud(batched_states)  # batched
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np

from factoriax.constants import (
    DIRECTIONS,
    ITEM_COLORS,
    NUM_ITEM_TYPES,
    BlockType,
    MachineType,
)
from factoriax.recipes import (
    MAX_RECIPE_INPUTS,
    NUM_RECIPES,
    RECIPE_INPUT_COUNTS,
    RECIPE_INPUT_ITEMS,
    RECIPE_OUTPUTS,
)
from factoriax.state import EnvState

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

DEFAULT_TILE_PX: int = 8

# Inventory / HUD element sizes
INV_HEIGHT: int = 16
DIGIT_H: int = 5
DIGIT_W: int = 3
ICON_SIZE: int = 6

# Colors (as JAX arrays for use inside traced functions)
SLOT_BG = np.array([40, 40, 40], dtype=np.uint8)
COUNT_COLOR = np.array([220, 220, 220], dtype=np.uint8)
HUD_BG = jnp.array([30, 30, 30], dtype=jnp.uint8)
HUD_BORDER = jnp.array([60, 60, 60], dtype=jnp.uint8)
HUD_LABEL = jnp.array([160, 160, 160], dtype=jnp.uint8)
HUD_GREEN = jnp.array([60, 200, 60], dtype=jnp.uint8)
HUD_RED = jnp.array([200, 60, 60], dtype=jnp.uint8)


# ---------------------------------------------------------------------------
# Texture atlas construction
#
# Sprites for blocks, machines, and the player come from the committed
# sprite atlas at ``factoriax/assets/atlas.png``. Per-tile-size atlases
# are derived by slicing the atlas row for the category and downsampling
# each 32×32 cell to ``tile_px`` via nearest-neighbour. The atlas
# itself is built by ``scripts/build_atlas.py``; the layout is pinned
# in ``factoriax/assets/atlas.layout.md`` and verified by
# ``tests/test_atlas_fresh.py``.
#
# Item-color and digit-glyph atlases stay procedural — see the module
# docstring for why. The atlas's items/digits rows are unused here.
# ---------------------------------------------------------------------------

_ATLAS_PATH: Path = Path(__file__).resolve().parent / "assets" / "atlas.png"
_ATLAS_CELL_PX: int = 32

# Row indices in the sprite atlas (must match factoriax/assets/atlas.layout.md).
_ATLAS_ROW_BLOCKS: int = 0
# Machines occupy four rows, one per Direction value (LEFT=1, RIGHT=2,
# UP=3, DOWN=4). Atlas index = direction - 1, so the layout is:
#   row 1: LEFT, row 2: RIGHT, row 3: UP, row 4: DOWN.
_ATLAS_ROW_MACHINES_BASE: int = 1
_ATLAS_NUM_DIRECTIONS: int = 4
_ATLAS_ROW_ITEMS: int = 5
_ATLAS_ROW_MISC: int = 6
# Misc row column layout: col 0 = biter, then 8 players × 4 directions
# packed as (player_idx, direction) starting at col 1. Player p's
# direction-d cell sits at col 1 + p * 4 + (d - 1). Eight players is
# the cap because PLAYER_COLORS in factoriax/ui/icons.py defines a
# distinct palette per slot up to 8; players beyond that wrap modulo
# 8 (matching the editor and play HUD's color-recycling behavior).
_ATLAS_MISC_BITER: int = 0
_ATLAS_MISC_PLAYER_BASE: int = 1
_ATLAS_NUM_PLAYERS: int = 8
_ATLAS_ROW_DIGITS: int = 7


@functools.cache
def _load_atlas_image() -> np.ndarray:
    """Read the sprite atlas PNG once and cache the resulting array.

    Returns:
        uint8 RGBA array of shape (rows * 32, cols * 32, 4) where rows
        and cols are defined in ``factoriax/assets/atlas.layout.md``.
        If the source PNG is RGB the alpha channel is filled with 255
        so downstream gathers can assume RGBA uniformly.
    """
    img: np.ndarray = np.asarray(iio.imread(_ATLAS_PATH))
    if img.ndim == 3 and img.shape[-1] == 3:
        alpha = np.full(img.shape[:2] + (1,), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=-1)
    return img


def _atlas_row_cells(row: int, n_cols: int) -> np.ndarray:
    """Slice a row of the atlas and return its cells as a stack.

    Args:
        row: Row index into the atlas grid.
        n_cols: Number of cells (from column 0) to extract.

    Returns:
        uint8 array of shape (n_cols, 32, 32, 4).
    """
    atlas = _load_atlas_image()
    s = _ATLAS_CELL_PX
    y0 = row * s
    cells = np.empty((n_cols, s, s, 4), dtype=np.uint8)
    for col in range(n_cols):
        x0 = col * s
        cells[col] = atlas[y0 : y0 + s, x0 : x0 + s, :]
    return cells


def _atlas_cell(row: int, col: int) -> np.ndarray:
    """Slice a single cell out of the atlas.

    Returns:
        uint8 array of shape (32, 32, 4).
    """
    atlas = _load_atlas_image()
    s = _ATLAS_CELL_PX
    y0, x0 = row * s, col * s
    return atlas[y0 : y0 + s, x0 : x0 + s, :].copy()


def _downsample(cells: np.ndarray, target_px: int) -> np.ndarray:
    """Nearest-neighbour resample square cells to ``target_px``.

    Accepts either a single cell of shape (S, S, 3) or a stack of
    cells of shape (N, S, S, 3). Returns the same rank with the
    spatial dims rescaled.
    """
    src_px = cells.shape[-2]
    if src_px == target_px:
        return cells
    idx = (np.arange(target_px) * src_px // target_px).astype(np.int64)
    if cells.ndim == 3:
        return cells[np.ix_(idx, idx)]
    return cells[:, idx][:, :, idx]


def build_block_atlas(tile_px: int) -> jnp.ndarray:
    """Build a texture atlas for terrain block types from the sprite atlas.

    Args:
        tile_px: Tile side length in pixels.

    Returns:
        JAX array of shape ``(num_block_types, tile_px, tile_px, 4)``.
    """
    n_cells = max(int(b) for b in BlockType) + 1
    cells = _atlas_row_cells(_ATLAS_ROW_BLOCKS, n_cells)
    return jnp.array(_downsample(cells, tile_px))


def build_machine_atlas(tile_px: int) -> jnp.ndarray:
    """Build a direction-indexed machine atlas from the sprite atlas.

    The four rows starting at :data:`_ATLAS_ROW_MACHINES_BASE` hold one
    machine variant per :class:`~factoriax.constants.Direction`, in the
    order LEFT, RIGHT, UP, DOWN. Non-directional machines are simply
    duplicated across all four rows so a uniform gather works at render
    time.

    Args:
        tile_px: Tile side length in pixels.

    Returns:
        JAX array of shape ``(4, num_machine_types, tile_px, tile_px, 4)``.
        Index 0 is direction LEFT, 3 is direction DOWN.
    """
    n_machines = max(int(m) for m in MachineType) + 1
    rows = []
    for d in range(_ATLAS_NUM_DIRECTIONS):
        cells = _atlas_row_cells(_ATLAS_ROW_MACHINES_BASE + d, n_machines)
        rows.append(_downsample(cells, tile_px))
    return jnp.array(np.stack(rows, axis=0))


def build_player_sprite(tile_px: int) -> jnp.ndarray:
    """Build the per-player, directional player sprite stack.

    The misc row packs eight players × four directions starting at
    column :data:`_ATLAS_MISC_PLAYER_BASE`. Player ``p`` facing
    direction ``d`` (1..4) lives at column
    ``_ATLAS_MISC_PLAYER_BASE + p * _ATLAS_NUM_DIRECTIONS + (d - 1)``.

    Args:
        tile_px: Tile side length in pixels.

    Returns:
        JAX array of shape
        ``(_ATLAS_NUM_PLAYERS, _ATLAS_NUM_DIRECTIONS, tile_px, tile_px, 4)``.
        Players 0..7 each have four directional cells in the order
        LEFT, RIGHT, UP, DOWN.
    """
    sprites: list[list[np.ndarray]] = []
    for p in range(_ATLAS_NUM_PLAYERS):
        per_player: list[np.ndarray] = []
        for d in range(_ATLAS_NUM_DIRECTIONS):
            col = _ATLAS_MISC_PLAYER_BASE + p * _ATLAS_NUM_DIRECTIONS + d
            cell = _atlas_cell(_ATLAS_ROW_MISC, col)
            per_player.append(_downsample(cell, tile_px))
        sprites.append(per_player)
    return jnp.array(np.stack([np.stack(rows, axis=0) for rows in sprites], axis=0))


# ---------------------------------------------------------------------------
# Numpy-side atlas helpers
#
# The editor canvas and other pygame surfaces need RGBA numpy arrays
# rather than the device-resident JAX arrays the renderer uses. These
# helpers slice the same atlas the JAX builders consume, so swapping
# the atlas PNG re-skins both surfaces at once. Results are cached per
# requested size so a frame loop pays the slice + downsample cost
# exactly once per tile size.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def block_textures_rgba(size: int) -> np.ndarray:
    """Return per-block RGBA textures sliced from the sprite atlas.

    The shape mirrors what the editor canvas needs for its vectorised
    advanced-index blit: ``(num_block_types, size, size, 4)``. Block
    cells in the atlas are fully opaque, so the alpha channel is
    always 255 here — terrain is always the ground truth layer.

    Args:
        size: Tile side length in pixels. Downsampling from the atlas's
            32×32 base is nearest-neighbour.

    Returns:
        uint8 RGBA array of shape ``(num_block_types, size, size, 4)``.
    """
    n_cells = max(int(b) for b in BlockType) + 1
    cells = _atlas_row_cells(_ATLAS_ROW_BLOCKS, n_cells)
    return _downsample(cells, size).astype(np.uint8, copy=False)


@functools.lru_cache(maxsize=64)
def machine_icon_rgba(machine_type: int, size: int, direction: int) -> np.ndarray:
    """Return the RGBA machine sprite for a placed machine.

    Sliced from the atlas's directional machine rows; the editor
    canvas uses this in place of
    :func:`factoriax.ui.icons.render_item_icon` so both surfaces
    pull from the same source of truth. Direction values outside
    ``[1, 4]`` (e.g. an unset machine direction) are clipped to the
    LEFT row to keep the gather well-defined.

    Args:
        machine_type: ``MachineType`` integer.
        size: Side length in pixels for the returned sprite.
        direction: ``Direction`` integer (1=LEFT, 2=RIGHT, 3=UP, 4=DOWN).

    Returns:
        uint8 RGBA array of shape ``(size, size, 4)``.
    """
    direction_idx = max(0, min(_ATLAS_NUM_DIRECTIONS - 1, direction - 1))
    row = _ATLAS_ROW_MACHINES_BASE + direction_idx
    cell = _atlas_cell(row, machine_type)
    return _downsample(cell, size).astype(np.uint8, copy=False)


@functools.lru_cache(maxsize=8)
def biter_icon_rgba(size: int) -> np.ndarray:
    """Return the RGBA biter sprite from the misc row's biter cell.

    Args:
        size: Side length in pixels for the returned sprite.

    Returns:
        uint8 RGBA array of shape ``(size, size, 4)``.
    """
    cell = _atlas_cell(_ATLAS_ROW_MISC, _ATLAS_MISC_BITER)
    return _downsample(cell, size).astype(np.uint8, copy=False)


@functools.lru_cache(maxsize=64)
def player_icon_rgba(player_idx: int, size: int, direction: int) -> np.ndarray:
    """Return the RGBA player sprite for the given slot and facing.

    Players beyond :data:`_ATLAS_NUM_PLAYERS` wrap modulo the palette
    size, matching the renderer's runtime behaviour and the editor's
    legacy player-color recycling.

    Args:
        player_idx: Player slot (0-based).
        size: Side length in pixels for the returned sprite.
        direction: ``Direction`` integer (1=LEFT, 2=RIGHT, 3=UP, 4=DOWN).

    Returns:
        uint8 RGBA array of shape ``(size, size, 4)``.
    """
    slot = player_idx % _ATLAS_NUM_PLAYERS
    direction_idx = max(0, min(_ATLAS_NUM_DIRECTIONS - 1, direction - 1))
    col = _ATLAS_MISC_PLAYER_BASE + slot * _ATLAS_NUM_DIRECTIONS + direction_idx
    cell = _atlas_cell(_ATLAS_ROW_MISC, col)
    return _downsample(cell, size).astype(np.uint8, copy=False)


def build_item_color_atlas() -> jnp.ndarray:
    """Build an RGB color lookup indexed by ItemType.

    Returns:
        JAX uint8 array of shape (NUM_ITEM_TYPES, 3).
    """
    atlas = np.zeros((NUM_ITEM_TYPES, 3), dtype=np.uint8)
    for item_id, rgb in ITEM_COLORS.items():
        atlas[int(item_id)] = rgb
    return jnp.array(atlas)


def build_digit_atlas() -> jnp.ndarray:
    """Build a 3x5 bitmap font atlas for digits 0-9.

    Returns:
        JAX bool array of shape (10, DIGIT_H, DIGIT_W).
    """
    glyphs = {
        0: "####.##.##.####",
        1: ".#.##..#..#.###",
        2: "###..#####..###",
        3: "###..####..####",
        4: "#.##.####..#..#",
        5: "####..###..####",
        6: "####..####.####",
        7: "###..#..#..#..#",
        8: "####.#####.####",
        9: "####.####..####",
    }
    atlas = np.zeros((10, DIGIT_H, DIGIT_W), dtype=np.bool_)
    for digit, chars in glyphs.items():
        for i, ch in enumerate(chars):
            atlas[digit, i // DIGIT_W, i % DIGIT_W] = ch == "#"
    return jnp.array(atlas)


# ---------------------------------------------------------------------------
# Rendering helpers (pure JAX, vmappable)
# ---------------------------------------------------------------------------


def _get_facing_tile(state: EnvState) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the tile coordinates player 0 is facing.

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
    color: np.ndarray | jnp.ndarray,
) -> jnp.ndarray:
    """Stamp a left-aligned integer into an image, hiding leading zeros.

    Args:
        img: Image to draw on.
        digit_atlas: (10, DIGIT_H, DIGIT_W) bool atlas.
        value: Non-negative integer to render.
        max_digits: Maximum number of digit positions.
        y: Top row of the number.
        x: Left column of the first digit position.
        color: RGB color for the digits.

    Returns:
        Updated image.
    """
    safe_val = jnp.clip(value, 0, 10**max_digits - 1)

    def _stamp_one(i: int, carry: jnp.ndarray) -> jnp.ndarray:
        divisor = jnp.int32(10) ** jnp.int32(max_digits - 1 - i)
        digit_val = (safe_val // divisor) % 10
        show = safe_val >= divisor
        mask = digit_atlas[digit_val]
        dx = x + i * (DIGIT_W + 1)
        bg = jax.lax.dynamic_slice(carry, (y, dx, 0), (DIGIT_H, DIGIT_W, 3))
        blended = jnp.where(mask[:, :, None], color, bg)
        result = jnp.where(show, blended, bg)
        return jax.lax.dynamic_update_slice(carry, result.astype(jnp.uint8), (y, dx, 0))

    result: jnp.ndarray = jax.lax.fori_loop(0, max_digits, _stamp_one, img)
    return result


# ---------------------------------------------------------------------------
# Map renderer
# ---------------------------------------------------------------------------


def _alpha_composite(below: jnp.ndarray, above_rgba: jnp.ndarray) -> jnp.ndarray:
    """Per-pixel ``over`` blend of an RGBA layer onto an RGB layer.

    Args:
        below: uint8 RGB array of shape ``(..., 3)`` (the background).
        above_rgba: uint8 RGBA array of shape ``(..., 4)``. The alpha
            channel acts as the per-pixel blend mask; ``alpha=0`` falls
            through to ``below``, ``alpha=255`` overwrites it.

    Returns:
        uint8 RGB array of the same leading shape as ``below``.
    """
    rgb_above = above_rgba[..., :3].astype(jnp.float32)
    alpha = above_rgba[..., 3:4].astype(jnp.float32) / 255.0
    rgb_below = below.astype(jnp.float32)
    out = rgb_above * alpha + rgb_below * (1.0 - alpha)
    return jnp.clip(out, 0.0, 255.0).astype(jnp.uint8)


def render_map(
    state: EnvState,
    block_atlas: jnp.ndarray,
    machine_atlas: jnp.ndarray,
    player_sprite: jnp.ndarray,
) -> jnp.ndarray:
    """Render the map: terrain + machines + players.

    Pure JAX, JIT-compilable, vmappable. Tile pixel size is inferred
    from the block_atlas shape. Layers composite back-to-front using
    each layer's alpha channel: terrain is fully opaque; machine and
    player cells reveal what's underneath wherever ``alpha < 255``.

    Machine and player sprites are direction-indexed: the right cell
    is selected per tile from ``state.ent_direction[tile_entity]`` and
    per player from ``state.player_directions[i]``. The atlas builders
    duplicate non-directional machines across all four direction rows
    so this gather is uniform.

    Args:
        state: Single (non-batched) EnvState.
        block_atlas: Shape ``(num_block_types, tile_px, tile_px, 4)``.
        machine_atlas: Shape ``(4, num_machine_types, tile_px, tile_px, 4)``
            indexed by ``(direction - 1, machine_type)``.
        player_sprite: Shape
            ``(_ATLAS_NUM_PLAYERS, 4, tile_px, tile_px, 4)`` indexed by
            ``(player_idx % _ATLAS_NUM_PLAYERS, direction - 1)``.

    Returns:
        uint8 RGB image of shape ``(H * tile_px, W * tile_px, 3)``.
    """
    tile_px = block_atlas.shape[1]
    map_h, map_w = state.map.shape

    # Layer 1: Terrain. Block cells are fully opaque, so we drop alpha
    # to keep the working image RGB and let the blend helper produce
    # RGB outputs.
    safe_map = jnp.clip(state.map, 0, block_atlas.shape[0] - 1)
    tile_textures = block_atlas[safe_map][..., :3]
    image = tile_textures.transpose(0, 2, 1, 3, 4).reshape(
        map_h * tile_px, map_w * tile_px, 3
    )

    # Layer 2: Machine overlays. Direction comes from the entity at
    # each tile; tiles with no entity (tile_entity == -1) read from
    # entity 0 as a fallback, but the NONE machine cell carries
    # alpha=0 so those reads are fully discarded by the blend.
    safe_ent_idx = jnp.maximum(state.tile_entity, 0)
    direction = state.ent_direction[safe_ent_idx]
    direction_idx = jnp.clip(
        direction.astype(jnp.int32) - 1, 0, _ATLAS_NUM_DIRECTIONS - 1
    )
    safe_machines = jnp.clip(state.machine_types, 0, machine_atlas.shape[1] - 1)
    machine_textures = machine_atlas[direction_idx, safe_machines]
    machine_image = machine_textures.transpose(0, 2, 1, 3, 4).reshape(
        map_h * tile_px, map_w * tile_px, 4
    )
    image = _alpha_composite(image, machine_image)

    # Layer 3: Player sprites. Each player picks its own directional
    # cell; the 32x32 sprite is composited over the underlying region
    # so transparent margins reveal the belt/ore/machine the player
    # stands on.
    num_players = state.player_positions.shape[0]

    def _stamp_player(i: int, img: jnp.ndarray) -> jnp.ndarray:
        px = state.player_positions[i, 0].astype(jnp.int32)
        py = state.player_positions[i, 1].astype(jnp.int32)
        pdir = state.player_directions[i].astype(jnp.int32)
        # Player slot is i mod _ATLAS_NUM_PLAYERS; players beyond the
        # palette wrap (matches the editor + play HUD).
        slot = i % _ATLAS_NUM_PLAYERS
        sprite = player_sprite[
            slot,
            jnp.clip(pdir - 1, 0, _ATLAS_NUM_DIRECTIONS - 1),
        ]
        region = jax.lax.dynamic_slice(
            img, (py * tile_px, px * tile_px, jnp.int32(0)), (tile_px, tile_px, 3)
        )
        blended = _alpha_composite(region, sprite)
        return jax.lax.dynamic_update_slice(
            img, blended, (py * tile_px, px * tile_px, jnp.int32(0))
        )

    composited: jnp.ndarray = jax.lax.fori_loop(0, num_players, _stamp_player, image)
    return composited.astype(jnp.uint8)


# ---------------------------------------------------------------------------
# Inventory strip renderer
# ---------------------------------------------------------------------------


def render_inventory_strip(
    state: EnvState,
    item_colors: jnp.ndarray,
    digit_atlas: jnp.ndarray,
    img_width: int,
) -> jnp.ndarray:
    """Render player-0 inventory as a horizontal strip.

    Args:
        state: Single EnvState.
        item_colors: Shape (NUM_ITEM_TYPES, 3).
        digit_atlas: Shape (10, DIGIT_H, DIGIT_W).
        img_width: Strip width in pixels.

    Returns:
        uint8 RGB array of shape (INV_HEIGHT, img_width, 3).
    """
    strip = jnp.full((INV_HEIGHT, img_width, 3), SLOT_BG, dtype=jnp.uint8)
    slot_width = img_width // NUM_ITEM_TYPES
    inv = state.player_inventory[0]

    def _draw_slot(slot_idx: int, img: jnp.ndarray) -> jnp.ndarray:
        x_start = slot_idx * slot_width
        slot_bg = jnp.broadcast_to(SLOT_BG, (INV_HEIGHT, slot_width, 3)).astype(
            jnp.uint8
        )
        img = jax.lax.dynamic_update_slice(img, slot_bg, (0, x_start, 0))

        item_type = slot_idx
        count = inv[slot_idx]
        color = item_colors[item_type]
        has_item = (item_type > 0) & (count > 0)

        swatch_y = (INV_HEIGHT - ICON_SIZE) // 2
        swatch_x = x_start + 1
        swatch = jnp.broadcast_to(color, (ICON_SIZE, ICON_SIZE, 3))
        swatch = swatch * has_item.astype(jnp.uint8)
        img = jax.lax.dynamic_update_slice(img, swatch, (swatch_y, swatch_x, 0))

        safe_count = jnp.clip(count, 0, 99)
        tens = safe_count // 10
        ones = safe_count % 10
        digit_y = (INV_HEIGHT - DIGIT_H) // 2
        digit_x = swatch_x + ICON_SIZE + 1

        def _stamp_digit(
            img: jnp.ndarray,
            digit_val: jnp.ndarray,
            dx: int | jnp.ndarray,
            show: int | jnp.ndarray,
        ) -> jnp.ndarray:
            d_mask = digit_atlas[digit_val]
            bg_patch = jax.lax.dynamic_slice(
                img, (digit_y, dx, 0), (DIGIT_H, DIGIT_W, 3)
            )
            blended = jnp.where(d_mask[:, :, None], COUNT_COLOR, bg_patch)
            result = jnp.where(show, blended, bg_patch)
            return jax.lax.dynamic_update_slice(
                img, result.astype(jnp.uint8), (digit_y, dx, 0)
            )

        show_tens = has_item & (safe_count >= 10)
        img = _stamp_digit(img, tens, digit_x, show_tens)
        ones_x = digit_x + DIGIT_W + 1
        img = _stamp_digit(img, ones, ones_x, has_item)
        return img

    result: jnp.ndarray = jax.lax.fori_loop(0, NUM_ITEM_TYPES, _draw_slot, strip)
    return result


def render_map_with_inventory(
    state: EnvState,
    block_atlas: jnp.ndarray,
    machine_atlas: jnp.ndarray,
    player_sprite: jnp.ndarray,
    item_colors: jnp.ndarray,
    digit_atlas: jnp.ndarray,
) -> jnp.ndarray:
    """Render map + inventory strip, concatenated vertically.

    Args:
        state: Single (non-batched) EnvState.
        block_atlas: Block texture atlas.
        machine_atlas: Machine texture atlas.
        player_sprite: Player sprite.
        item_colors: Item color atlas.
        digit_atlas: Digit bitmap atlas.

    Returns:
        uint8 RGB array of shape (map_h*tile_px + INV_HEIGHT, map_w*tile_px, 3).
    """
    map_img = render_map(state, block_atlas, machine_atlas, player_sprite)
    img_width = map_img.shape[1]
    inv_strip = render_inventory_strip(state, item_colors, digit_atlas, img_width)
    return jnp.concatenate([map_img, inv_strip], axis=0)


# ---------------------------------------------------------------------------
# HUD quadrant renderers
# ---------------------------------------------------------------------------


def render_q1_inspector(
    state: EnvState,
    block_atlas: jnp.ndarray,
    machine_atlas: jnp.ndarray,
    item_colors: jnp.ndarray,
    digit_atlas: jnp.ndarray,
    quad_h: int,
    quad_w: int,
) -> jnp.ndarray:
    """Q1: tile inspector -- block, resources, machine.

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

    # Block type swatch + resource count
    block_type = state.map[fy, fx]
    safe_bt = jnp.clip(block_type, 0, block_atlas.shape[0] - 1)
    block_color = block_atlas[safe_bt, 0, 0, :]
    swatch = jnp.broadcast_to(block_color, (ICON_SIZE, ICON_SIZE, 3))
    img = jax.lax.dynamic_update_slice(img, swatch, (2, 2, 0))
    resources = state.block_resources[fy, fx]
    img = _stamp_number(
        img,
        digit_atlas,
        resources,
        4,
        2,
        jnp.int32(ICON_SIZE + 4),
        COUNT_COLOR,
    )

    # Machine type swatch + direction + status
    mt = state.machine_types[fy, fx]
    safe_mt = jnp.clip(mt, 0, machine_atlas.shape[0] - 1)
    m_color = machine_atlas[safe_mt, 0, 0, :]
    has_machine = mt != int(MachineType.NONE)
    m_swatch = jnp.broadcast_to(m_color, (ICON_SIZE, ICON_SIZE, 3))
    m_swatch = m_swatch * has_machine.astype(jnp.uint8)
    img = jax.lax.dynamic_update_slice(img, m_swatch, (14, 2, 0))

    # NOTE: machine_direction was removed from EnvState; this renderer is
    # being rewritten in Phase A item 3 (single render path consolidation).
    m_dir = state.machine_direction[fy, fx]  # type: ignore[attr-defined]
    dir_dx = DIRECTIONS[jnp.clip(m_dir, 0, 4), 0]
    dir_dy = DIRECTIONS[jnp.clip(m_dir, 0, 4), 1]
    arrow_dot = jnp.full((3, 3, 3), HUD_LABEL, dtype=jnp.uint8)
    arrow_dot = arrow_dot * has_machine.astype(jnp.uint8)
    img = jax.lax.dynamic_update_slice(
        img, arrow_dot, (15 + dir_dy * 2, ICON_SIZE + 4 + dir_dx * 2, 0)
    )

    # NOTE: machine_power was removed from EnvState; rewrite in Phase A item 3.
    is_working = has_machine & (state.machine_power[fy, fx] > 0)  # type: ignore[attr-defined]
    status_color = jnp.where(is_working, HUD_GREEN, HUD_RED)
    status_dot = jnp.broadcast_to(status_color, (3, 3, 3))
    status_dot = status_dot * has_machine.astype(jnp.uint8)
    img = jax.lax.dynamic_update_slice(img, status_dot, (14, ICON_SIZE + 12, 0))

    return img


def render_q2_machine_inv(
    state: EnvState,
    item_colors: jnp.ndarray,
    digit_atlas: jnp.ndarray,
    quad_h: int,
    quad_w: int,
) -> jnp.ndarray:
    """Q2: machine inventory -- grid of item type counts.

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
    # NOTE: machine_inventory was removed from EnvState; rewrite in Phase A item 3.
    inv = state.machine_inventory[fy, fx, :]  # type: ignore[attr-defined]
    cols = 5
    cell_w = quad_w // cols
    rows = (NUM_ITEM_TYPES + cols - 1) // cols
    cell_h = quad_h // rows

    def _draw_mslot(slot: int, img: jnp.ndarray) -> jnp.ndarray:
        col = slot % cols
        row = slot // cols
        cx = col * cell_w + 2
        cy = row * cell_h + 2
        count = inv[slot]
        color = item_colors[slot]
        has_item = has_machine & (slot > 0) & (count > 0)
        swatch = jnp.broadcast_to(color, (ICON_SIZE, ICON_SIZE, 3))
        swatch = swatch * has_item.astype(jnp.uint8)
        img = jax.lax.dynamic_update_slice(img, swatch, (cy, cx, 0))
        img = _stamp_number(
            img,
            digit_atlas,
            jnp.where(has_item, count, 0),
            2,
            cy,
            jnp.int32(cx + ICON_SIZE + 2),
            COUNT_COLOR,
        )
        return img

    result: jnp.ndarray = jax.lax.fori_loop(0, NUM_ITEM_TYPES, _draw_mslot, img)
    return result


def render_q3_inventory(
    state: EnvState,
    item_colors: jnp.ndarray,
    digit_atlas: jnp.ndarray,
    quad_h: int,
    quad_w: int,
) -> jnp.ndarray:
    """Q3: player inventory -- grid of item type counts.

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
    inv = state.player_inventory[0]
    cols = 5
    cell_w = quad_w // cols
    rows = (NUM_ITEM_TYPES + cols - 1) // cols
    cell_h = quad_h // rows

    def _draw_islot(slot: int, img: jnp.ndarray) -> jnp.ndarray:
        col = slot % cols
        row = slot // cols
        cx = col * cell_w
        cy = row * cell_h
        cell_bg = jnp.broadcast_to(SLOT_BG, (cell_h, cell_w, 3)).astype(jnp.uint8)
        img = jax.lax.dynamic_update_slice(img, cell_bg, (cy, cx, 0))
        count = inv[slot]
        color = item_colors[slot]
        has_item = (slot > 0) & (count > 0)
        swatch = jnp.broadcast_to(color, (ICON_SIZE, ICON_SIZE, 3))
        swatch = swatch * has_item.astype(jnp.uint8)
        img = jax.lax.dynamic_update_slice(img, swatch, (cy + 2, cx + 1, 0))
        img = _stamp_number(
            img,
            digit_atlas,
            jnp.where(has_item, count, 0),
            2,
            cy + ICON_SIZE + 4,
            jnp.int32(cx + 1),
            COUNT_COLOR,
        )
        return img

    result: jnp.ndarray = jax.lax.fori_loop(0, NUM_ITEM_TYPES, _draw_islot, img)
    return result


def render_q4_crafting(
    state: EnvState,
    item_colors: jnp.ndarray,
    digit_atlas: jnp.ndarray,
    quad_h: int,
    quad_w: int,
) -> jnp.ndarray:
    """Q4: crafting menu -- 5 recipes with affordability.

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
    inv = state.player_inventory[0]
    # NOTE: crafting_recipe and craft_progress were removed from EnvState;
    # rewrite in Phase A item 3 (single render path consolidation).
    craft_recipe = state.crafting_recipe[0]  # type: ignore[attr-defined]
    craft_progress = state.craft_progress[0]  # type: ignore[attr-defined]
    row_h = quad_h // NUM_RECIPES

    def _draw_recipe(r: int, img: jnp.ndarray) -> jnp.ndarray:
        ry = r * row_h
        out_item = RECIPE_OUTPUTS[r]
        out_color = item_colors[jnp.clip(out_item, 0, NUM_ITEM_TYPES - 1)]
        out_swatch = jnp.broadcast_to(out_color, (ICON_SIZE, ICON_SIZE, 3))
        img = jax.lax.dynamic_update_slice(img, out_swatch, (ry + 1, 2, 0))

        def _draw_input(j: int, img: jnp.ndarray) -> jnp.ndarray:
            in_item = RECIPE_INPUT_ITEMS[r, j]
            in_color = item_colors[jnp.clip(in_item, 0, NUM_ITEM_TYPES - 1)]
            has_input = in_item > 0
            in_sw = jnp.broadcast_to(in_color, (4, 4, 3))
            in_sw = in_sw * has_input.astype(jnp.uint8)
            ix = ICON_SIZE + 5 + j * 6
            return jax.lax.dynamic_update_slice(img, in_sw, (ry + 2, ix, 0))

        img = jax.lax.fori_loop(0, MAX_RECIPE_INPUTS, _draw_input, img)

        def _check_input(j: int, affordable: jnp.ndarray) -> jnp.ndarray:
            needed_item = RECIPE_INPUT_ITEMS[r, j]
            needed_count = RECIPE_INPUT_COUNTS[r, j]
            has_enough = jnp.where(
                needed_item == 0,
                True,
                inv[needed_item] >= needed_count,
            )
            return affordable & has_enough

        can_afford = jax.lax.fori_loop(
            0, MAX_RECIPE_INPUTS, _check_input, jnp.bool_(True)
        )
        dot_color = jnp.where(can_afford, HUD_GREEN, HUD_RED)
        dot = jnp.broadcast_to(dot_color, (3, 3, 3))
        img = jax.lax.dynamic_update_slice(
            img, dot.astype(jnp.uint8), (ry + 2, quad_w - 6, 0)
        )

        is_crafting = (craft_progress > 0) & (craft_recipe == r)
        bar = jnp.broadcast_to(HUD_GREEN, (2, 4, 3))
        bar = bar * is_crafting.astype(jnp.uint8)
        img = jax.lax.dynamic_update_slice(
            img, bar.astype(jnp.uint8), (ry + row_h - 3, 2, 0)
        )
        return img

    result: jnp.ndarray = jax.lax.fori_loop(0, NUM_RECIPES, _draw_recipe, img)
    return result


# ---------------------------------------------------------------------------
# Full HUD compositor
# ---------------------------------------------------------------------------


def render_hud(
    state: EnvState,
    block_atlas: jnp.ndarray,
    machine_atlas: jnp.ndarray,
    player_sprite: jnp.ndarray,
    item_colors: jnp.ndarray,
    digit_atlas: jnp.ndarray,
) -> jnp.ndarray:
    """Render the complete frame: map on top, 4-quadrant HUD on bottom.

    The HUD has the same pixel dimensions as the map. Total output
    height is 2x the map height.

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
    map_img = render_map(state, block_atlas, machine_atlas, player_sprite)
    map_h, map_w = map_img.shape[0], map_img.shape[1]
    quad_h = map_h // 2
    quad_w = map_w // 2

    q1 = render_q1_inspector(
        state,
        block_atlas,
        machine_atlas,
        item_colors,
        digit_atlas,
        quad_h,
        quad_w,
    )
    q2 = render_q2_machine_inv(
        state,
        item_colors,
        digit_atlas,
        quad_h,
        quad_w,
    )
    q3 = render_q3_inventory(
        state,
        item_colors,
        digit_atlas,
        quad_h,
        quad_w,
    )
    q4 = render_q4_crafting(
        state,
        item_colors,
        digit_atlas,
        quad_h,
        quad_w,
    )

    border_h = jnp.full((1, map_w, 3), HUD_BORDER, dtype=jnp.uint8)
    border_v = jnp.full((quad_h, 1, 3), HUD_BORDER, dtype=jnp.uint8)

    top_row = jnp.concatenate([q1, border_v, q2], axis=1)[:, :map_w, :]
    bot_row = jnp.concatenate([q3, border_v, q4], axis=1)[:, :map_w, :]
    hud = jnp.concatenate([top_row, border_h, bot_row], axis=0)[:map_h, :, :]

    return jnp.concatenate([map_img, hud], axis=0)


# ---------------------------------------------------------------------------
# JaxRenderer class -- holds atlases, provides convenience methods
# ---------------------------------------------------------------------------


class JaxRenderer:
    """Stateful wrapper around the pure-JAX rendering functions.

    Builds all texture atlases once at init, then provides methods
    that only need the EnvState. The atlases are device-resident JAX
    arrays for the lifetime of the renderer.

    Example::

        renderer = JaxRenderer(tile_px=8)
        img = renderer.render_map(state)           # single state
        imgs = renderer.vmap_render_hud(batched)   # batched states

    Args:
        tile_px: Tile side length in pixels.
    """

    def __init__(self, tile_px: int = DEFAULT_TILE_PX) -> None:
        self.tile_px = tile_px
        self.block_atlas = build_block_atlas(tile_px)
        self.machine_atlas = build_machine_atlas(tile_px)
        self.player_sprite = build_player_sprite(tile_px)
        self.item_colors = build_item_color_atlas()
        self.digit_atlas = build_digit_atlas()

    def render_map_single(self, state: EnvState) -> jnp.ndarray:
        """Render map for a single state (not jitted).

        Args:
            state: Single EnvState.

        Returns:
            uint8 RGB image.
        """
        return render_map(
            state,
            self.block_atlas,
            self.machine_atlas,
            self.player_sprite,
        )

    def render_hud_single(self, state: EnvState) -> jnp.ndarray:
        """Render map + HUD for a single state (not jitted).

        Args:
            state: Single EnvState.

        Returns:
            uint8 RGB image.
        """
        return render_hud(
            state,
            self.block_atlas,
            self.machine_atlas,
            self.player_sprite,
            self.item_colors,
            self.digit_atlas,
        )

    @functools.cached_property
    def _jit_render_map(self) -> Callable[..., jnp.ndarray]:
        """JIT-compiled map renderer."""
        return jax.jit(render_map)

    @functools.cached_property
    def _jit_render_hud(self) -> Callable[..., jnp.ndarray]:
        """JIT-compiled HUD renderer."""
        return jax.jit(render_hud)

    @functools.cached_property
    def _vmap_render_map(self) -> Callable[..., jnp.ndarray]:
        """JIT+vmapped map renderer."""
        return jax.jit(jax.vmap(render_map, in_axes=(0, None, None, None)))

    @functools.cached_property
    def _vmap_render_hud(self) -> Callable[..., jnp.ndarray]:
        """JIT+vmapped HUD renderer."""
        return jax.jit(jax.vmap(render_hud, in_axes=(0, None, None, None, None, None)))

    def jit_render_map(self, state: EnvState) -> jnp.ndarray:
        """JIT-compiled map render for a single state.

        Args:
            state: Single EnvState.

        Returns:
            uint8 RGB image.
        """
        result: jnp.ndarray = self._jit_render_map(
            state,
            self.block_atlas,
            self.machine_atlas,
            self.player_sprite,
        )
        return result

    def jit_render_hud(self, state: EnvState) -> jnp.ndarray:
        """JIT-compiled HUD render for a single state.

        Args:
            state: Single EnvState.

        Returns:
            uint8 RGB image.
        """
        result: jnp.ndarray = self._jit_render_hud(
            state,
            self.block_atlas,
            self.machine_atlas,
            self.player_sprite,
            self.item_colors,
            self.digit_atlas,
        )
        return result

    def vmap_render_map(self, batched_state: EnvState) -> jnp.ndarray:
        """Batched map render (jit + vmap).

        Args:
            batched_state: Batched EnvState with leading batch dimension.

        Returns:
            uint8 RGB images with shape (batch, H, W, 3).
        """
        result: jnp.ndarray = self._vmap_render_map(
            batched_state,
            self.block_atlas,
            self.machine_atlas,
            self.player_sprite,
        )
        return result

    def vmap_render_hud(self, batched_state: EnvState) -> jnp.ndarray:
        """Batched HUD render (jit + vmap).

        Args:
            batched_state: Batched EnvState with leading batch dimension.

        Returns:
            uint8 RGB images with shape (batch, 2*H, W, 3).
        """
        result: jnp.ndarray = self._vmap_render_hud(
            batched_state,
            self.block_atlas,
            self.machine_atlas,
            self.player_sprite,
            self.item_colors,
            self.digit_atlas,
        )
        return result
