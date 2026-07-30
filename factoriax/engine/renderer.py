"""Draw an :class:`~factoriax.engine.state.EnvState` as a pixel image.

:func:`render_map` is traceable, so it runs under ``jax.jit`` and vmaps over a
batch of states. It composites three layers back to front:

1. Terrain, gathered from a block atlas by ``state.map``.
2. Machines, gathered by ``(direction, machine_type)`` and blended onto the
   terrain through the atlas alpha channel.
3. Players, one cell per ``(player_idx, direction)``, blended onto the tile
   each player stands on.

Sprites come from ``factoriax/assets/atlas.png``, a grid of 32x32 RGBA cells
whose layout is pinned in ``factoriax/assets/atlas.layout.md`` and checked by
``tests/test_atlas_fresh.py``. Block cells are opaque, so terrain always
paints. Machine and player cells carry transparent regions, so a player
standing on a belt no longer hides it. The row and column constants below are
hardcoded rather than read from the sidecar ``atlas.json``, so a layout change
has to be mirrored here by hand.

Each atlas is built for one ``tile_px`` by slicing a row and resampling every
cell from 32x32 with nearest-neighbour. Conveyor belts, miners, arms,
splitters, and crossings carry one cell per
:class:`~factoriax.engine.constants.Direction`; every other machine repeats
its one sprite across the four direction rows so the gather needs no branch.

The module serves two consumers. :class:`JaxRenderer` holds device-resident
atlases for the JAX path. The ``*_rgba`` helpers return host-side numpy arrays
for the pygame surfaces in the editor and play UI, sliced from the same PNG so
one art change re-skins both.

The HUD is drawn by the editor and play UI, not here.

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

#: Tile side length in pixels used when a caller names no other size. A map of
#: ``H`` by ``W`` tiles renders to ``(H * 8, W * 8, 3)``.
DEFAULT_TILE_PX: int = 8

#: Rows in a digit glyph from :func:`build_digit_atlas`.
DIGIT_H: int = 5
#: Columns in a digit glyph from :func:`build_digit_atlas`.
DIGIT_W: int = 3


# ---------------------------------------------------------------------------
# Atlas layout
#
# Mirrors factoriax/assets/atlas.layout.md, which build_atlas.py writes and
# tests/test_atlas_fresh.py enforces. Change both together.
# ---------------------------------------------------------------------------

_ATLAS_PATH: Path = Path(__file__).resolve().parent.parent / "assets" / "atlas.png"
_ATLAS_CELL_PX: int = 32

_ATLAS_ROW_BLOCKS: int = 0
# Rows 1 to 4 hold one machine set per Direction, in value order: LEFT=1,
# RIGHT=2, UP=3, DOWN=4. Row index is direction - 1.
_ATLAS_ROW_MACHINES_BASE: int = 1
_ATLAS_NUM_DIRECTIONS: int = 4
_ATLAS_ROW_MISC: int = 6
# Misc row: col 0 is the biter, then 8 players x 4 directions from col 1, so
# player p facing d sits at col 1 + p * 4 + (d - 1).
_ATLAS_MISC_BITER: int = 0
_ATLAS_MISC_PLAYER_BASE: int = 1
# Eight player sprites are baked into the atlas, so an index past the eighth
# wraps to the first color rather than failing.
_ATLAS_NUM_PLAYERS: int = 8


@functools.cache
def _load_atlas_image() -> np.ndarray:
    """Read the sprite atlas PNG and cache it for the process lifetime.

    Returns
    -------
    np.ndarray
        Shape ``(rows * 32, cols * 32, 4)``, uint8. An RGB source PNG gains a
        fully opaque alpha channel, so the result always has four channels.
        Every caller shares the one cached array and must not write to it.
        Because the read happens once, replacing the PNG on disk has no effect
        until the process restarts.

    Raises
    ------
    FileNotFoundError
        The atlas is missing. Run ``factoriax/assets/build_atlas.py``.
    """
    img: np.ndarray = np.asarray(iio.imread(_ATLAS_PATH))
    if img.ndim == 3 and img.shape[-1] == 3:
        alpha = np.full(img.shape[:2] + (1,), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=-1)
    return img


def _atlas_row_cells(row: int, n_cols: int) -> np.ndarray:
    """Cut the leading cells out of one atlas row.

    Parameters
    ----------
    row
        Row index into the atlas grid.
    n_cols
        Number of cells to take, counting from column 0.

    Returns
    -------
    np.ndarray
        Shape ``(n_cols, 32, 32, 4)``, uint8. A fresh array, safe to mutate.

    Raises
    ------
    ValueError
        ``n_cols`` runs past the right edge of the atlas. The overhanging cell
        comes back empty and fails to broadcast, so a row that has grown an
        enum member without a rebuilt atlas fails loudly here.
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
    """Copy one cell out of the atlas.

    Parameters
    ----------
    row
        Row index into the atlas grid.
    col
        Column index into that row.

    Returns
    -------
    np.ndarray
        Shape ``(32, 32, 4)``, uint8. A copy, so the caller may mutate it
        without touching the cached atlas. An out-of-range ``row`` or ``col``
        returns an undersized or empty array instead of raising, because numpy
        clips a slice rather than rejecting it.
    """
    atlas = _load_atlas_image()
    s = _ATLAS_CELL_PX
    y0, x0 = row * s, col * s
    return atlas[y0 : y0 + s, x0 : x0 + s, :].copy()


def _downsample(cells: np.ndarray, target_px: int) -> np.ndarray:
    """Resample square cells to ``target_px`` by nearest neighbour.

    Handles both directions: a ``target_px`` above the source size repeats
    rows and columns, which is what a zoomed-in view needs. Nearest neighbour
    is deliberate. Averaging neighbours would bleed a transparent margin into
    the sprite edge and soften the pixel-art look.

    Parameters
    ----------
    cells
        One cell of shape ``(S, S, C)`` or a stack of shape ``(N, S, S, C)``.
        ``C`` is 4 for every atlas slice.
    target_px
        Side length to resample to, in pixels. Must be positive.

    Returns
    -------
    np.ndarray
        Same rank and dtype as ``cells`` with the two spatial axes at
        ``target_px``. When the size already matches, ``cells`` comes back
        unchanged rather than copied, so the caller shares its storage.
    """
    src_px = cells.shape[-2]
    if src_px == target_px:
        return cells
    idx = (np.arange(target_px) * src_px // target_px).astype(np.int64)
    if cells.ndim == 3:
        return cells[np.ix_(idx, idx)]
    return cells[:, idx][:, :, idx]


def build_block_atlas(tile_px: int) -> jnp.ndarray:
    """Build the terrain layer atlas that :func:`render_map` gathers from.

    Parameters
    ----------
    tile_px
        Tile side length in pixels.

    Returns
    -------
    jnp.ndarray
        Shape ``(num_block_types, tile_px, tile_px, 4)``, uint8, indexed by
        :class:`~factoriax.engine.constants.BlockType` value. Every block cell
        is opaque, so the alpha channel is 255 throughout and
        :func:`render_map` discards it. The row is sized to the highest enum
        value plus one, so a gap in the numbering leaves a magenta cell that
        the atlas builder wrote as a missing-sprite marker.
    """
    n_cells = max(int(b) for b in BlockType) + 1
    cells = _atlas_row_cells(_ATLAS_ROW_BLOCKS, n_cells)
    return jnp.array(_downsample(cells, tile_px))


def build_machine_atlas(tile_px: int) -> jnp.ndarray:
    """Build the machine layer atlas, one sprite set per facing.

    Parameters
    ----------
    tile_px
        Tile side length in pixels.

    Returns
    -------
    jnp.ndarray
        Shape ``(4, num_machine_types, tile_px, tile_px, 4)``, uint8, indexed
        by ``(direction - 1, machine_type)`` with the direction axis in value
        order LEFT, RIGHT, UP, DOWN. A machine that draws the same either way
        repeats its one sprite across the four rows, so the render-time gather
        needs no test for whether a kind is directional. Alpha is meaningful
        here: ``Machine.NONE`` is fully transparent, which is what lets
        :func:`render_map` blend an empty tile away instead of masking it.
    """
    n_machines = max(int(m) for m in Machine) + 1
    rows = []
    for d in range(_ATLAS_NUM_DIRECTIONS):
        cells = _atlas_row_cells(_ATLAS_ROW_MACHINES_BASE + d, n_machines)
        rows.append(_downsample(cells, tile_px))
    return jnp.array(np.stack(rows, axis=0))


def build_player_sprite(tile_px: int) -> jnp.ndarray:
    """Build the player layer atlas, one sprite per slot and facing.

    Parameters
    ----------
    tile_px
        Tile side length in pixels.

    Returns
    -------
    jnp.ndarray
        Shape ``(8, 4, tile_px, tile_px, 4)``, uint8, indexed by
        ``(player_slot, direction - 1)``. Eight slots is the whole stack: a
        scenario with more players than that reuses earlier colors, since the
        atlas bakes no ninth palette. Cells carry transparent margins so the
        tile under a player stays visible.
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
# Pygame surfaces need host-side RGBA arrays, not the device arrays render_map
# gathers from. Every result is cached, so a frame loop pays the slice and
# resample once per distinct argument set and callers must treat the array as
# read-only.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def block_textures_rgba(size: int) -> np.ndarray:
    """Return every terrain texture as one stack the editor can index into.

    Parameters
    ----------
    size
        Tile side length in pixels.

    Returns
    -------
    np.ndarray
        Shape ``(num_block_types, size, size, 4)``, uint8, indexed by
        :class:`~factoriax.engine.constants.BlockType` value. The single stack
        is what lets the editor canvas blit a whole viewport with one advanced
        index rather than a loop over tiles. Alpha is 255 everywhere, because
        terrain is the layer everything else paints over. Cached and shared
        between callers, so treat it as read-only.
    """
    n_cells = max(int(b) for b in BlockType) + 1
    cells = _atlas_row_cells(_ATLAS_ROW_BLOCKS, n_cells)
    return _downsample(cells, size).astype(np.uint8, copy=False)


@functools.lru_cache(maxsize=64)
def machine_icon_rgba(machine_type: int, size: int, direction: int) -> np.ndarray:
    """Return one machine sprite for a pygame surface to blit.

    The editor canvas calls this instead of
    :func:`factoriax.playground.ui.icons.render_item_icon` so the map it draws
    and the map :func:`render_map` draws come from the same art.

    Parameters
    ----------
    machine_type
        :class:`~factoriax.engine.constants.Machine` value. A value past the
        end of the row returns an empty array rather than raising.
    size
        Side length in pixels for the returned sprite.
    direction
        :class:`~factoriax.engine.constants.Direction` value. Anything outside
        1 to 4, including the 0 an unplaced or unset entity carries, clamps to
        the LEFT sprite.

    Returns
    -------
    np.ndarray
        Shape ``(size, size, 4)``, uint8. ``Machine.NONE`` yields a fully
        transparent cell. Cached and shared between callers, so treat it as
        read-only.
    """
    direction_idx = max(0, min(_ATLAS_NUM_DIRECTIONS - 1, direction - 1))
    row = _ATLAS_ROW_MACHINES_BASE + direction_idx
    cell = _atlas_cell(row, machine_type)
    return _downsample(cell, size).astype(np.uint8, copy=False)


@functools.lru_cache(maxsize=8)
def biter_icon_rgba(size: int) -> np.ndarray:
    """Return the biter sprite for a pygame surface to blit.

    A biter is level data, held in
    :attr:`~factoriax.engine.levels.Level.biter_positions` rather than in
    :class:`~factoriax.engine.state.EnvState`, so :func:`render_map` never
    draws one and only the editor reads this cell.

    Parameters
    ----------
    size
        Side length in pixels for the returned sprite.

    Returns
    -------
    np.ndarray
        Shape ``(size, size, 4)``, uint8, with a transparent margin. Cached
        and shared between callers, so treat it as read-only.
    """
    cell = _atlas_cell(_ATLAS_ROW_MISC, _ATLAS_MISC_BITER)
    return _downsample(cell, size).astype(np.uint8, copy=False)


@functools.lru_cache(maxsize=64)
def player_icon_rgba(player_idx: int, size: int, direction: int) -> np.ndarray:
    """Return one player sprite for a pygame surface to blit.

    Parameters
    ----------
    player_idx
        Player index. Eight sprites are baked, so an index past the eighth
        wraps to the first color. :func:`render_map` wraps on the same count,
        so a sprite drawn here matches the one drawn on the map.
    size
        Side length in pixels for the returned sprite.
    direction
        :class:`~factoriax.engine.constants.Direction` value. Anything outside
        1 to 4 clamps to the LEFT sprite.

    Returns
    -------
    np.ndarray
        Shape ``(size, size, 4)``, uint8, with a transparent margin. Cached
        and shared between callers, so treat it as read-only.
    """
    slot = player_idx % _ATLAS_NUM_PLAYERS
    direction_idx = max(0, min(_ATLAS_NUM_DIRECTIONS - 1, direction - 1))
    col = _ATLAS_MISC_PLAYER_BASE + slot * _ATLAS_NUM_DIRECTIONS + direction_idx
    cell = _atlas_cell(_ATLAS_ROW_MISC, col)
    return _downsample(cell, size).astype(np.uint8, copy=False)


def build_digit_atlas() -> jnp.ndarray:
    """Build the bitmap font for the digits 0 to 9.

    Nothing here draws digits. ``factoriax/assets/build_atlas.py`` is the one
    caller, which bakes these glyphs into the atlas digits row at build time.

    Returns
    -------
    jnp.ndarray
        Shape ``(10, DIGIT_H, DIGIT_W)``, bool, indexed by the digit itself.
        True marks an inked pixel and carries no color, so a caller picks one.
    """
    # Each glyph is DIGIT_H * DIGIT_W characters read row-major, '#' for ink.
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
    """Blend an RGBA layer over an RGB layer, pixel by pixel.

    Parameters
    ----------
    below
        Background, uint8 of shape ``(..., 3)``.
    above_rgba
        Foreground, uint8 of shape ``(..., 4)`` broadcastable against
        ``below``. Alpha is the blend weight: 0 leaves the background
        untouched, 255 replaces it, and values between mix the two.

    Returns
    -------
    jnp.ndarray
        Shape ``(..., 3)``, uint8. The result carries no alpha, so a stack of
        layers has to composite bottom up. Mixing happens in float32 on
        straight, non-premultiplied color and rounds toward zero on the way
        back to uint8, which can darken a blended edge by one count.
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
    """Draw terrain, machines, and players into one image.

    Reads only the state fields it draws, allocates a new image, and mutates
    nothing, so it traces cleanly under ``jax.jit`` and vmaps over a batch of
    states. Tile size comes from ``block_atlas``, not from an argument.

    Out-of-range input is clamped rather than rejected, because a traced
    gather cannot raise. A block or machine id past the end of its atlas draws
    the last cell, a facing outside 1 to 4 draws the LEFT cell, and a player
    standing off the map is drawn at the nearest edge. None of these are
    reported, so a scenario that ships a bad id renders a wrong sprite in
    silence.

    Parameters
    ----------
    state
        One unbatched state. Machines are read from ``machine_types`` and
        ``tile_entity``, which index by ``[row, column]``, while
        ``player_positions`` holds ``(x, y)`` in the opposite order.
    block_atlas
        Terrain sprites from :func:`build_block_atlas`.
    machine_atlas
        Machine sprites from :func:`build_machine_atlas`.
    player_sprite
        Player sprites from :func:`build_player_sprite`. Only its leading axis
        length caps how many distinct player colors appear.

    Returns
    -------
    jnp.ndarray
        Shape ``(H * tile_px, W * tile_px, 3)``, uint8, with ``H`` and ``W``
        taken from ``state.map``. Row 0 is the top of the map. The result is
        opaque: alpha is consumed while compositing and not returned, so this
        is a finished picture, not a layer to blend further.
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

    # Layer 3: Player sprites, one tile each, blended so the belt or ore the
    # player stands on stays visible through the sprite's transparent margin.
    num_players = state.player_positions.shape[0]

    def _stamp_player(i: int, img: jnp.ndarray) -> jnp.ndarray:
        """Blend player ``i`` onto ``img`` and return the updated image.

        Players are drawn in index order, so the higher index wins where two
        share a tile.
        """
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
    """Hold the sprite atlases for one tile size and render states with them.

    Building an atlas reads and resamples the PNG, so a caller that renders
    every step wants one renderer kept alive rather than one per frame. The
    atlases move to the device at construction and stay there. Compilation is
    deferred instead: the jit and vmap wrappers are built on first use, and
    each traces again whenever the shape of a state changes.

    One tile size per instance. Rendering at a second size means a second
    renderer.

    Example::

        renderer = JaxRenderer(tile_px=8)
        img = renderer.jit_render_map(state)              # single state
        imgs = renderer.vmap_render_map(batched_states)   # batched

    Parameters
    ----------
    tile_px
        Tile side length in pixels. Sizes other than 32 resample the atlas by
        nearest neighbour, so a size that does not divide 32 evenly drops or
        repeats rows of pixels.
    """

    def __init__(self, tile_px: int = DEFAULT_TILE_PX) -> None:
        self.tile_px = tile_px
        self.block_atlas = build_block_atlas(tile_px)
        self.machine_atlas = build_machine_atlas(tile_px)
        self.player_sprite = build_player_sprite(tile_px)

    def render_map_single(self, state: EnvState) -> jnp.ndarray:
        """Render one state without compiling first.

        Every primitive dispatches on its own, which is slower per call than
        :meth:`jit_render_map` but skips the compile. Useful for a one-off
        frame or for stepping through the layers in a debugger.

        Parameters
        ----------
        state
            One unbatched state.

        Returns
        -------
        jnp.ndarray
            Image as described in :func:`render_map`.
        """
        return render_map(
            state,
            self.block_atlas,
            self.machine_atlas,
            self.player_sprite,
        )

    @functools.cached_property
    def _jit_render_map(self) -> Callable[..., jnp.ndarray]:
        """Compile :func:`render_map` on first access and reuse it after."""
        return jax.jit(render_map)

    @functools.cached_property
    def _vmap_render_map(self) -> Callable[..., jnp.ndarray]:
        """Compile the batched renderer, mapping over the state axis only.

        The three atlases are broadcast rather than mapped, so one copy serves
        the whole batch.
        """
        return jax.jit(jax.vmap(render_map, in_axes=(0, None, None, None)))

    def jit_render_map(self, state: EnvState) -> jnp.ndarray:
        """Render one state through the compiled renderer.

        The first call for a given map size and player count pays the compile;
        later calls of that shape reuse it. This is the method a play loop or
        video recorder should use.

        Parameters
        ----------
        state
            One unbatched state.

        Returns
        -------
        jnp.ndarray
            Image as described in :func:`render_map`. It lives on the device,
            so a pygame or imageio consumer needs ``np.asarray`` first, which
            is where the render actually blocks.
        """
        result: jnp.ndarray = self._jit_render_map(
            state,
            self.block_atlas,
            self.machine_atlas,
            self.player_sprite,
        )
        return result

    def vmap_render_map(self, batched_state: EnvState) -> jnp.ndarray:
        """Render a batch of states in one call.

        Parameters
        ----------
        batched_state
            A state whose every field carries a leading batch axis, as a
            vmapped environment produces. Passing an unbatched state raises
            during tracing rather than rendering one image.

        Returns
        -------
        jnp.ndarray
            Shape ``(batch, H * tile_px, W * tile_px, 3)``, uint8. Every image
            is held at once, so a large batch at a large tile size can exhaust
            device memory well before the states themselves would.
        """
        result: jnp.ndarray = self._vmap_render_map(
            batched_state,
            self.block_atlas,
            self.machine_atlas,
            self.player_sprite,
        )
        return result
