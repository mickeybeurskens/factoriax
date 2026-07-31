"""Draw an :class:`~factoriax.engine.state.EnvState` as a pixel image.

JAX can trace :func:`render_map`, so it runs under ``jax.jit`` and vmaps over a
batch of states. It draws three layers, from back to front:

1. The terrain, gathered from a block atlas by ``state.map``.
2. The machines, gathered by ``(direction, machine_type)`` and blended onto the
   terrain through the alpha channel of the atlas.
3. The players, one cell for each ``(player_idx, direction)``, blended onto the
   tile of each player.

The sprites come from ``factoriax/assets/atlas.png``, a grid of 32x32 RGBA
cells. ``factoriax/assets/atlas.layout.md`` fixes the layout, and
``tests/test_atlas_fresh.py`` tests it. A block cell is opaque, so the terrain
always paints. A machine cell and a player cell hold transparent regions, so a
player on a belt does not hide the belt.

CAUTION: The row and column constants below are written into this file. They do
not come from the ``atlas.json`` file next to the PNG. Copy every layout change
into this file by hand.

Each atlas is built for one ``tile_px``. The builder slices a row and resamples
every cell from 32x32 with nearest-neighbour. A conveyor belt, a miner, an arm,
a splitter, and a crossing each carry one cell for each
:class:`~factoriax.engine.constants.Direction`. Every other machine repeats its
one sprite across the four direction rows, so the gather needs no branch.

The module serves two readers. :class:`JaxRenderer` holds the atlases on the
device for the JAX path. The ``*_rgba`` helpers return numpy arrays on the host
for the pygame surfaces in the editor and the play UI. Both come from the same
PNG, so one change to the art changes both.

The editor and the play UI draw the HUD. This module does not.

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

#: Tile side length in pixels when a caller gives no other size. A map of ``H``
#: by ``W`` tiles renders to ``(H * 8, W * 8, 3)``.
DEFAULT_TILE_PX: int = 8

#: Rows in a digit glyph from :func:`build_digit_atlas`.
DIGIT_H: int = 5
#: Columns in a digit glyph from :func:`build_digit_atlas`.
DIGIT_W: int = 3


# ---------------------------------------------------------------------------
# Atlas layout
#
# These match factoriax/assets/atlas.layout.md, which build_atlas.py writes and
# tests/test_atlas_fresh.py enforces. Change both files together.
# ---------------------------------------------------------------------------

_ATLAS_PATH: Path = Path(__file__).resolve().parent.parent / "assets" / "atlas.png"
_ATLAS_CELL_PX: int = 32

_ATLAS_ROW_BLOCKS: int = 0
# Rows 1 to 4 hold one machine set for each Direction, in value order: LEFT=1,
# RIGHT=2, UP=3, DOWN=4. The row index is direction - 1.
_ATLAS_ROW_MACHINES_BASE: int = 1
_ATLAS_NUM_DIRECTIONS: int = 4
_ATLAS_ROW_MISC: int = 6
# Misc row: 8 players x 4 directions from col 1, so player p with facing d sits
# at col 1 + p * 4 + (d - 1). Col 0 holds a sprite that nothing reads today.
_ATLAS_MISC_PLAYER_BASE: int = 1
# The atlas holds eight player sprites, so an index past the eighth returns to
# the first colour and does not fail.
_ATLAS_NUM_PLAYERS: int = 8


@functools.cache
def _load_atlas_image() -> np.ndarray:
    """Read the sprite atlas PNG and cache it for the process lifetime.

    Returns
    -------
    np.ndarray
        Shape ``(rows * 32, cols * 32, 4)``, uint8. An RGB source PNG gets a
        fully opaque alpha channel, so the result always has four channels.
        Every caller shares the one cached array and must not write to it. The
        read happens one time, so a new PNG on disk changes nothing until the
        process starts again.

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
        Shape ``(n_cols, 32, 32, 4)``, uint8. This is a new array, and a caller
        can write to it.

    Raises
    ------
    ValueError
        If ``n_cols`` runs past the right edge of the atlas. The cell past the
        edge comes back empty and fails to broadcast. A row that gained an enum
        member without a new atlas build therefore fails here.
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
        Shape ``(32, 32, 4)``, uint8. This is a copy, so a caller can write to
        it and leave the cached atlas unchanged. A ``row`` or ``col`` outside
        the atlas returns a small or empty array and raises nothing, because
        numpy clips a slice and does not refuse it.
    """
    atlas = _load_atlas_image()
    s = _ATLAS_CELL_PX
    y0, x0 = row * s, col * s
    return atlas[y0 : y0 + s, x0 : x0 + s, :].copy()


def _downsample(cells: np.ndarray, target_px: int) -> np.ndarray:
    """Resample square cells to ``target_px`` by nearest neighbour.

    The function works in both directions. A ``target_px`` larger than the
    source size repeats rows and columns, which a zoomed-in view needs. The
    nearest-neighbour method is deliberate. An average of the neighbours pulls
    a transparent margin into the edge of the sprite and softens the pixel art.

    Parameters
    ----------
    cells
        One cell of shape ``(S, S, C)``, or a stack of shape ``(N, S, S, C)``.
        ``C`` is 4 for every atlas slice.
    target_px
        Side length to resample to, in pixels. The value must be positive.

    Returns
    -------
    np.ndarray
        The same rank and dtype as ``cells``, with the two spatial axes at
        ``target_px``. When the size already matches, the function returns
        ``cells`` itself and makes no copy, so the caller shares its storage.
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
        Shape ``(num_block_types, tile_px, tile_px, 4)``, uint8. A
        :class:`~factoriax.engine.constants.BlockType` value indexes it. Every
        block cell is opaque, so the alpha channel is 255 everywhere and
        :func:`render_map` drops it. The row length is the highest enum value
        plus one. A gap in the numbering therefore leaves a magenta cell, which
        the atlas builder wrote as a marker for a missing sprite.
    """
    n_cells = max(int(b) for b in BlockType) + 1
    cells = _atlas_row_cells(_ATLAS_ROW_BLOCKS, n_cells)
    return jnp.array(_downsample(cells, tile_px))


def build_machine_atlas(tile_px: int) -> jnp.ndarray:
    """Build the machine layer atlas, with one sprite set for each facing.

    Parameters
    ----------
    tile_px
        Tile side length in pixels.

    Returns
    -------
    jnp.ndarray
        Shape ``(4, num_machine_types, tile_px, tile_px, 4)``, uint8. The pair
        ``(direction - 1, machine_type)`` indexes it, and the direction axis is
        in value order LEFT, RIGHT, UP, DOWN. A machine that looks the same in
        every direction repeats its one sprite across the four rows, so the
        gather at render time needs no test for a directional kind. The alpha
        channel carries meaning here. ``Machine.NONE`` is fully transparent,
        and that lets :func:`render_map` blend an empty tile away instead of a
        mask.
    """
    n_machines = max(int(m) for m in Machine) + 1
    rows = []
    for d in range(_ATLAS_NUM_DIRECTIONS):
        cells = _atlas_row_cells(_ATLAS_ROW_MACHINES_BASE + d, n_machines)
        rows.append(_downsample(cells, tile_px))
    return jnp.array(np.stack(rows, axis=0))


def build_player_sprite(tile_px: int) -> jnp.ndarray:
    """Build the player layer atlas, with one sprite for each slot and facing.

    Parameters
    ----------
    tile_px
        Tile side length in pixels.

    Returns
    -------
    jnp.ndarray
        Shape ``(8, 4, tile_px, tile_px, 4)``, uint8. The pair
        ``(player_slot, direction - 1)`` indexes it. There are eight slots in
        total. A scenario with more players than eight reuses the earlier
        colours, because the atlas holds no ninth palette. Each cell carries a
        transparent margin, so the tile under a player stays visible.
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
# A pygame surface needs an RGBA array on the host, not the device array that
# render_map gathers from. A cache holds every result, so a frame loop pays for
# the slice and the resample one time for each set of arguments. A caller must
# not write to the array.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def block_textures_rgba(size: int) -> np.ndarray:
    """Return every terrain texture as one stack that the editor can index.

    Parameters
    ----------
    size
        Tile side length in pixels.

    Returns
    -------
    np.ndarray
        Shape ``(num_block_types, size, size, 4)``, uint8. A
        :class:`~factoriax.engine.constants.BlockType` value indexes it. The
        one stack lets the editor canvas blit a whole viewport with one
        advanced index, and not with a loop over the tiles. The alpha channel
        is 255 everywhere, because the terrain is the layer that every other
        layer paints over. A cache holds this array and every caller shares it,
        so do not write to it.
    """
    n_cells = max(int(b) for b in BlockType) + 1
    cells = _atlas_row_cells(_ATLAS_ROW_BLOCKS, n_cells)
    return _downsample(cells, size).astype(np.uint8, copy=False)


@functools.lru_cache(maxsize=64)
def machine_icon_rgba(machine_type: int, size: int, direction: int) -> np.ndarray:
    """Return one machine sprite for a pygame surface to blit.

    The editor canvas calls this function instead of
    :func:`factoriax.playground.ui.icons.render_item_icon`. The map that the
    editor draws and the map that :func:`render_map` draws therefore come from
    the same art.

    Parameters
    ----------
    machine_type
        :class:`~factoriax.engine.constants.Machine` value. A value past the
        end of the row returns an empty array and raises nothing.
    size
        Side length of the returned sprite, in pixels.
    direction
        :class:`~factoriax.engine.constants.Direction` value. A value outside 1
        to 4 clamps to the LEFT sprite. This includes the 0 of an entity that
        is not on the map, and of an entity with no direction.

    Returns
    -------
    np.ndarray
        Shape ``(size, size, 4)``, uint8. ``Machine.NONE`` gives a fully
        transparent cell. A cache holds this array and every caller shares it,
        so do not write to it.
    """
    direction_idx = max(0, min(_ATLAS_NUM_DIRECTIONS - 1, direction - 1))
    row = _ATLAS_ROW_MACHINES_BASE + direction_idx
    cell = _atlas_cell(row, machine_type)
    return _downsample(cell, size).astype(np.uint8, copy=False)


@functools.lru_cache(maxsize=64)
def player_icon_rgba(player_idx: int, size: int, direction: int) -> np.ndarray:
    """Return one player sprite for a pygame surface to blit.

    Parameters
    ----------
    player_idx
        Player index. The atlas holds eight sprites, so an index past the
        eighth returns to the first colour. :func:`render_map` uses the same
        count, so a sprite here matches the sprite on the map.
    size
        Side length of the returned sprite, in pixels.
    direction
        :class:`~factoriax.engine.constants.Direction` value. A value outside 1
        to 4 clamps to the LEFT sprite.

    Returns
    -------
    np.ndarray
        Shape ``(size, size, 4)``, uint8, with a transparent margin. A cache
        holds this array and every caller shares it, so do not write to it.
    """
    slot = player_idx % _ATLAS_NUM_PLAYERS
    direction_idx = max(0, min(_ATLAS_NUM_DIRECTIONS - 1, direction - 1))
    col = _ATLAS_MISC_PLAYER_BASE + slot * _ATLAS_NUM_DIRECTIONS + direction_idx
    cell = _atlas_cell(_ATLAS_ROW_MISC, col)
    return _downsample(cell, size).astype(np.uint8, copy=False)


def build_digit_atlas() -> jnp.ndarray:
    """Build the bitmap font for the digits 0 to 9.

    Nothing in this module draws digits. ``factoriax/assets/build_atlas.py`` is
    the one caller. It writes these glyphs into the digits row of the atlas at
    build time.

    Returns
    -------
    jnp.ndarray
        Shape ``(10, DIGIT_H, DIGIT_W)``, bool. The digit itself indexes it.
        True marks an inked pixel and carries no colour, so the caller selects
        one.
    """
    # Each glyph is DIGIT_H * DIGIT_W characters, read row by row. '#' is ink.
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
        Foreground, uint8 of shape ``(..., 4)``, which must broadcast against
        ``below``. The alpha channel is the blend weight. A value of 0 leaves
        the background unchanged, 255 replaces it, and a value between the two
        mixes them.

    Returns
    -------
    jnp.ndarray
        Shape ``(..., 3)``, uint8. The result carries no alpha, so a stack of
        layers must blend from the bottom up. The mix runs in float32 on
        straight, non-premultiplied colour, and the result rounds towards zero
        on the way back to uint8. A blended edge can therefore lose one count
        of brightness.
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

    The function reads only the state fields that it draws. It allocates a new
    image and changes nothing, so it traces under ``jax.jit`` and vmaps over a
    batch of states. The tile size comes from ``block_atlas``, not from an
    argument.

    The function clamps input outside the valid range and does not refuse it,
    because a traced gather cannot raise. A block id or a machine id past the
    end of its atlas draws the last cell. A facing outside 1 to 4 draws the
    LEFT cell. A player outside the map appears at the nearest edge.

    CAUTION: The function reports none of these cases. A scenario with a bad id
    therefore renders the wrong sprite and gives no error.

    Parameters
    ----------
    state
        One unbatched state. The machines come from ``machine_types`` and
        ``tile_entity``, which both index by ``[row, column]``.
        ``player_positions`` holds ``(x, y)``, in the other order.
    block_atlas
        Terrain sprites from :func:`build_block_atlas`.
    machine_atlas
        Machine sprites from :func:`build_machine_atlas`.
    player_sprite
        Player sprites from :func:`build_player_sprite`. The length of its
        first axis limits the number of different player colours.

    Returns
    -------
    jnp.ndarray
        Shape ``(H * tile_px, W * tile_px, 3)``, uint8, with ``H`` and ``W``
        from ``state.map``. Row 0 is the top of the map. The result is opaque.
        The blend consumes the alpha channel and does not return it, so this is
        a finished picture and not a layer for a further blend.
    """
    tile_px = block_atlas.shape[1]
    map_h, map_w = state.map.shape

    # Layer 1: the terrain. A block cell is fully opaque, so drop the alpha
    # channel. The working image stays RGB, and the blend helper returns RGB.
    safe_map = jnp.clip(state.map, 0, block_atlas.shape[0] - 1)
    tile_textures = block_atlas[safe_map][..., :3]
    image = tile_textures.transpose(0, 2, 1, 3, 4).reshape(
        map_h * tile_px, map_w * tile_px, 3
    )

    # Layer 2: the machines. The direction comes from the entity on each tile.
    # A tile with no entity holds tile_entity == -1 and reads entity 0 instead.
    # The NONE machine cell carries alpha=0, so the blend drops those reads.
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

    # Layer 3: the player sprites, one tile each. The blend keeps the belt or
    # the ore under a player visible through the transparent margin.
    num_players = state.player_positions.shape[0]

    def _stamp_player(i: int, img: jnp.ndarray) -> jnp.ndarray:
        """Blend player ``i`` onto ``img`` and return the new image.

        The function draws the players in index order, so the higher index
        wins on a tile that two players share.
        """
        px = state.player_positions[i, 0].astype(jnp.int32)
        py = state.player_positions[i, 1].astype(jnp.int32)
        pdir = state.player_directions[i].astype(jnp.int32)
        # The player slot is i mod _ATLAS_NUM_PLAYERS. A player past the
        # palette returns to the first colour, as in the editor and play HUD.
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
# JaxRenderer: holds the atlases and offers the render methods
# ---------------------------------------------------------------------------


class JaxRenderer:
    """Hold the sprite atlases for one tile size and render states with them.

    An atlas build reads and resamples the PNG. A caller that renders every
    step must therefore keep one renderer alive, and must not build one for
    each frame. The atlases move to the device at construction and stay there.
    The compile happens later. The jit wrapper and the vmap wrapper build on
    first use, and each one traces again when the shape of a state changes.

    One instance serves one tile size. A second size needs a second renderer.

    Example::

        renderer = JaxRenderer(tile_px=8)
        img = renderer.jit_render_map(state)              # single state
        imgs = renderer.vmap_render_map(batched_states)   # batched

    Parameters
    ----------
    tile_px
        Tile side length in pixels. Any size except 32 resamples the atlas by
        nearest neighbour. A size that does not divide 32 exactly therefore
        drops or repeats rows of pixels.
    """

    def __init__(self, tile_px: int = DEFAULT_TILE_PX) -> None:
        self.tile_px = tile_px
        self.block_atlas = build_block_atlas(tile_px)
        self.machine_atlas = build_machine_atlas(tile_px)
        self.player_sprite = build_player_sprite(tile_px)

    def render_map_single(self, state: EnvState) -> jnp.ndarray:
        """Render one state with no compile first.

        Each primitive dispatches on its own. This is slower for each call than
        :meth:`jit_render_map`, but it pays for no compile. Use it for a single
        frame, or to step through the layers in a debugger.

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
        """Compile :func:`render_map` on the first read, and reuse it after."""
        return jax.jit(render_map)

    @functools.cached_property
    def _vmap_render_map(self) -> Callable[..., jnp.ndarray]:
        """Compile the batched renderer, which maps over the state axis only.

        The three atlases broadcast and do not map, so one copy of each serves
        the whole batch.
        """
        return jax.jit(jax.vmap(render_map, in_axes=(0, None, None, None)))

    def jit_render_map(self, state: EnvState) -> jnp.ndarray:
        """Render one state through the compiled renderer.

        The first call for a map size and player count pays for the compile,
        and a later call with that shape reuses it. A play loop and a video
        recorder must use this method.

        Parameters
        ----------
        state
            One unbatched state.

        Returns
        -------
        jnp.ndarray
            The image that :func:`render_map` describes. It lives on the
            device, so a pygame or imageio reader must call ``np.asarray``
            first. That call is where the render blocks.
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
            A state in which every field carries a leading batch axis, as a
            vmapped environment produces. An unbatched state raises during the
            trace and renders no image.

        Returns
        -------
        jnp.ndarray
            Shape ``(batch, H * tile_px, W * tile_px, 3)``, uint8. The device
            holds every image at the same time. A large batch at a large tile
            size can therefore fill the device memory long before the states
            themselves do.
        """
        result: jnp.ndarray = self._vmap_render_map(
            batched_state,
            self.block_atlas,
            self.machine_atlas,
            self.player_sprite,
        )
        return result
