"""Pure-JAX pixel renderer for FactoriaX.

A GPU-accelerated renderer that compiles to XLA and can be vmapped
across batched environment states. Every operation is a JAX primitive
(gather, scatter, where, reshape), so the entire render pipeline runs
in parallel on GPU.

The renderer works in layers, composited back-to-front:
  1. Terrain tiles (gathered from a block texture atlas via ``state.map``).
  2. Machine overlays (gathered by ``(direction, machine_type)``,
     alpha-blended onto terrain via the atlas's alpha channel).
  3. Player sprites (one cell per ``(player_idx, direction)``,
     alpha-blended onto the underlying region).

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
one per :class:`~factoriax.engine.constants.Direction`. ``render_map``
gathers by ``(direction, machine_type)`` for the machine layer and
by ``player_directions[i]`` for each player so placed objects show
their orientation.

The HUD lives in the editor and play UI, not here.

Usage::

    renderer = JaxRenderer(tile_px=8)
    img = renderer.jit_render_map(state)              # single state
    imgs = renderer.vmap_render_map(batched_states)   # batched
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np

from factoriax.engine.constants import (
    BlockType,
    Machine,
)
from factoriax.engine.state import EnvState

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

DEFAULT_TILE_PX: int = 8

# Digit-glyph constants — the bitmap font produced by build_digit_atlas
# is consumed by scripts/build_atlas.py to fill the atlas's digits row.
DIGIT_H: int = 5
DIGIT_W: int = 3


# ---------------------------------------------------------------------------
# Texture atlas construction
#
# The atlas is built by ``scripts/build_atlas.py``; the layout is pinned in
# ``factoriax/assets/atlas.layout.md`` and verified by
# ``tests/test_atlas_fresh.py``. Item-color and digit-glyph atlases stay
# procedural; the atlas's items/digits rows are unused here.
# ---------------------------------------------------------------------------

_ATLAS_PATH: Path = Path(__file__).resolve().parent.parent / "assets" / "atlas.png"
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
    machine variant per :class:`~factoriax.engine.constants.Direction`, in the
    order LEFT, RIGHT, UP, DOWN. Non-directional machines are simply
    duplicated across all four rows so a uniform gather works at render
    time.

    Args:
        tile_px: Tile side length in pixels.

    Returns:
        JAX array of shape ``(4, num_machine_types, tile_px, tile_px, 4)``.
        Index 0 is direction LEFT, 3 is direction DOWN.
    """
    n_machines = max(int(m) for m in Machine) + 1
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
    :func:`factoriax.playground.ui.icons.render_item_icon` so both surfaces
    pull from the same source of truth. Direction values outside
    ``[1, 4]`` (e.g. an unset machine direction) are clipped to the
    LEFT row to keep the gather well-defined.

    Args:
        machine_type: ``Machine`` integer.
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
# JaxRenderer class -- holds atlases, provides convenience methods
# ---------------------------------------------------------------------------


class JaxRenderer:
    """Stateful wrapper around the pure-JAX rendering functions.

    Builds the per-tile-size sprite atlases once at construction,
    then exposes plain, JIT-compiled, and vmapped variants of
    :func:`render_map`. The atlases are device-resident JAX arrays
    for the lifetime of the renderer.

    Example::

        renderer = JaxRenderer(tile_px=8)
        img = renderer.jit_render_map(state)              # single state
        imgs = renderer.vmap_render_map(batched_states)   # batched

    Args:
        tile_px: Tile side length in pixels.
    """

    def __init__(self, tile_px: int = DEFAULT_TILE_PX) -> None:
        self.tile_px = tile_px
        self.block_atlas = build_block_atlas(tile_px)
        self.machine_atlas = build_machine_atlas(tile_px)
        self.player_sprite = build_player_sprite(tile_px)

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

    @functools.cached_property
    def _jit_render_map(self) -> Callable[..., jnp.ndarray]:
        """JIT-compiled map renderer."""
        return jax.jit(render_map)

    @functools.cached_property
    def _vmap_render_map(self) -> Callable[..., jnp.ndarray]:
        """JIT+vmapped map renderer."""
        return jax.jit(jax.vmap(render_map, in_axes=(0, None, None, None)))

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
