"""Viewport and tile rendering for the level editor canvas.

Renders only the visible slice of the map. Block and machine sprites
come from the same sprite atlas the JAX renderer reads, so the editor
canvas is pixel-identical to the play view down to the swapped art on
``factoriax/assets/atlas.png``. Per-cell helpers in
:mod:`factoriax.engine.jax_renderer` (``block_textures_rgba``,
``machine_icon_rgba``) provide RGBA numpy arrays at the requested
tile size.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING

import numpy as np
import pygame

from factoriax.engine.constants import Machine
from factoriax.engine.jax_renderer import (
    biter_icon_rgba,
    block_textures_rgba,
    machine_icon_rgba,
)

if TYPE_CHECKING:
    from factoriax.editor.state import EditorState

TILE_SIZES = (16, 24, 32, 48)
_GRID_COLOR = (80, 80, 80, 100)
_CURSOR_COLOR = (255, 255, 255, 120)
_SELECT_COLOR = (100, 200, 255, 100)


@dataclasses.dataclass
class Viewport:
    """Camera state for the scrollable/zoomable canvas.

    Attributes:
        camera_x: Leftmost visible tile column (fractional for smooth pan).
        camera_y: Topmost visible tile row.
        tile_size: Current zoom level in pixels per tile.
        canvas_w: Width of the canvas area in pixels.
        canvas_h: Height of the canvas area in pixels.
    """

    camera_x: float = 0.0
    camera_y: float = 0.0
    tile_size: int = 32
    canvas_w: int = 400
    canvas_h: int = 400


def screen_to_tile(vp: Viewport, sx: int, sy: int) -> tuple[int, int]:
    """Convert canvas-relative pixel coordinates to tile coordinates.

    Args:
        vp: Current viewport state.
        sx: X pixel offset from the canvas left edge.
        sy: Y pixel offset from the canvas top edge.

    Returns:
        ``(tile_x, tile_y)`` integer tile coordinates.
    """
    tx = int(vp.camera_x + sx / vp.tile_size)
    ty = int(vp.camera_y + sy / vp.tile_size)
    return tx, ty


def tile_to_screen(vp: Viewport, tx: int, ty: int) -> tuple[int, int]:
    """Convert tile coordinates to canvas-relative pixel coordinates.

    Args:
        vp: Current viewport state.
        tx: Tile column.
        ty: Tile row.

    Returns:
        ``(px, py)`` pixel offset from the canvas top-left.
    """
    px = int((tx - vp.camera_x) * vp.tile_size)
    py = int((ty - vp.camera_y) * vp.tile_size)
    return px, py


def clamp_camera(vp: Viewport, map_w: int, map_h: int) -> None:
    """Clamp camera so visible area stays within map bounds.

    Args:
        vp: Viewport (mutated in place).
        map_w: Map width in tiles.
        map_h: Map height in tiles.
    """
    max_x = max(0.0, map_w - vp.canvas_w / vp.tile_size)
    max_y = max(0.0, map_h - vp.canvas_h / vp.tile_size)
    vp.camera_x = max(0.0, min(vp.camera_x, max_x))
    vp.camera_y = max(0.0, min(vp.camera_y, max_y))


def pan(vp: Viewport, dx: float, dy: float, map_w: int, map_h: int) -> None:
    """Pan the camera by a tile-space offset, clamped to map bounds.

    Args:
        vp: Viewport (mutated in place).
        dx: Horizontal offset in tiles (positive = right).
        dy: Vertical offset in tiles (positive = down).
        map_w: Map width in tiles.
        map_h: Map height in tiles.
    """
    vp.camera_x += dx
    vp.camera_y += dy
    clamp_camera(vp, map_w, map_h)


def zoom(
    vp: Viewport,
    direction: int,
    mouse_sx: int,
    mouse_sy: int,
    map_w: int,
    map_h: int,
) -> None:
    """Zoom in or out, keeping the tile under the mouse fixed.

    Cycles through :data:`TILE_SIZES` in the given direction.

    Args:
        vp: Viewport (mutated in place).
        direction: ``+1`` to zoom in, ``-1`` to zoom out.
        mouse_sx: Mouse x relative to canvas left edge.
        mouse_sy: Mouse y relative to canvas top edge.
        map_w: Map width in tiles.
        map_h: Map height in tiles.
    """
    idx = TILE_SIZES.index(vp.tile_size) if vp.tile_size in TILE_SIZES else 2
    new_idx = max(0, min(len(TILE_SIZES) - 1, idx + direction))
    new_size = TILE_SIZES[new_idx]
    if new_size == vp.tile_size:
        return

    tile_x = vp.camera_x + mouse_sx / vp.tile_size
    tile_y = vp.camera_y + mouse_sy / vp.tile_size

    vp.tile_size = new_size
    vp.camera_x = tile_x - mouse_sx / new_size
    vp.camera_y = tile_y - mouse_sy / new_size
    clamp_camera(vp, map_w, map_h)


def render_canvas(
    state: object,
    vp: Viewport,
    cursor_tile: tuple[int, int] | None,
    selection_rect: tuple[int, int, int, int] | None = None,
    show_resources: bool = False,
) -> np.ndarray:
    """Render the visible canvas area as an RGBA image.

    Draws terrain tiles via the vectorized texture lookup, overlays
    machines, grid lines, and cursor / selection highlights.  When
    *show_resources* is ``True``, the resource count is drawn on each
    tile that has resources.

    Args:
        state: :class:`~factoriax.editor.state.EditorState` instance.
        vp: Current viewport.
        cursor_tile: ``(tx, ty)`` of the tile under the mouse, or ``None``.
        selection_rect: ``(x0, y0, x1, y1)`` tile coordinates of the
            fill-rect selection, or ``None``.
        show_resources: Whether to draw resource amounts on tiles.

    Returns:
        RGBA uint8 array of shape ``(canvas_h, canvas_w, 4)``.
    """
    es: EditorState = state  # type: ignore[assignment]
    canvas = np.zeros((vp.canvas_h, vp.canvas_w, 4), dtype=np.uint8)

    col0 = max(0, int(vp.camera_x))
    row0 = max(0, int(vp.camera_y))
    col1 = min(es.map_width, int(vp.camera_x + vp.canvas_w / vp.tile_size) + 1)
    row1 = min(es.map_height, int(vp.camera_y + vp.canvas_h / vp.tile_size) + 1)

    if col1 <= col0 or row1 <= row0:
        return canvas

    lookup = block_textures_rgba(vp.tile_size)
    block_slice = es.block_map[row0:row1, col0:col1]
    max_id = lookup.shape[0] - 1
    safe = np.clip(block_slice, 0, max_id)
    tile_textures = lookup[safe]

    rows = row1 - row0
    cols = col1 - col0
    ts = vp.tile_size
    tiles_image = tile_textures.transpose(0, 2, 1, 3, 4).reshape(
        rows * ts, cols * ts, 4
    )

    px0 = int((col0 - vp.camera_x) * ts)
    py0 = int((row0 - vp.camera_y) * ts)

    _blit_clipped(canvas, tiles_image, py0, px0)

    machine_slice = es.machine_types[row0:row1, col0:col1]
    direction_slice = es.machine_directions[row0:row1, col0:col1]
    mys, mxs = np.nonzero(machine_slice != int(Machine.NONE))

    machine_size = int(ts * 0.6)
    offset = (ts - machine_size) // 2

    for my, mx in zip(mys, mxs):
        mt = int(machine_slice[my, mx])
        direction = int(direction_slice[my, mx])
        icon = machine_icon_rgba(mt, machine_size, direction)
        iy = int(py0 + my * ts + offset)
        ix = int(px0 + mx * ts + offset)
        _blit_clipped(canvas, icon, iy, ix)

    inv_items = es.machine_inventory_items[row0:row1, col0:col1]
    has_inv = np.any(inv_items != 0, axis=2)
    iys, ixs = np.nonzero(has_inv)
    for iy_np, ix_np in zip(iys, ixs):
        iy = int(iy_np)
        ix = int(ix_np)
        dot_y = py0 + iy * ts + 2
        dot_x = px0 + ix * ts + ts - 6
        dot_sz = max(2, ts // 8)
        dy0c = max(0, dot_y)
        dy1c = min(canvas.shape[0], dot_y + dot_sz)
        dx0c = max(0, dot_x)
        dx1c = min(canvas.shape[1], dot_x + dot_sz)
        if dy1c > dy0c and dx1c > dx0c:
            canvas[dy0c:dy1c, dx0c:dx1c] = (255, 200, 60, 255)

    if show_resources and ts >= 16:
        resource_slice = es.block_resources[row0:row1, col0:col1]
        rys, rxs = np.nonzero(resource_slice > 0)
        font = _get_resource_font(ts)
        for ry, rx in zip(rys, rxs):
            amount = int(resource_slice[ry, rx])
            label = _render_resource_label(font, amount)
            lh, lw = label.shape[:2]
            ly = py0 + ry * ts + (ts - lh) // 2
            lx = px0 + rx * ts + (ts - lw) // 2
            _blit_alpha(canvas, label, ly, lx)

    for r in range(rows + 1):
        gy = py0 + r * ts
        if 0 <= gy < vp.canvas_h:
            gx0 = max(0, px0)
            gx1 = min(vp.canvas_w, px0 + cols * ts)
            if gx1 > gx0:
                canvas[gy, gx0:gx1] = _GRID_COLOR

    for c in range(cols + 1):
        gx = px0 + c * ts
        if 0 <= gx < vp.canvas_w:
            gy0 = max(0, py0)
            gy1 = min(vp.canvas_h, py0 + rows * ts)
            if gy1 > gy0:
                canvas[gy0:gy1, gx] = _GRID_COLOR

    _render_entities(canvas, es, vp)

    if cursor_tile is not None:
        cx, cy = cursor_tile
        if 0 <= cx < es.map_width and 0 <= cy < es.map_height:
            _highlight_tile(canvas, vp, cx, cy, _CURSOR_COLOR)

    if selection_rect is not None:
        sx0, sy0, sx1, sy1 = selection_rect
        sel_lx = int(min(sx0, sx1))
        sel_rx = int(max(sx0, sx1))
        sel_ly = int(min(sy0, sy1))
        sel_ry = int(max(sy0, sy1))
        for ty in range(max(0, sel_ly), min(es.map_height, sel_ry + 1)):
            for tx in range(max(0, sel_lx), min(es.map_width, sel_rx + 1)):
                _highlight_tile(canvas, vp, tx, ty, _SELECT_COLOR)

    return canvas


@functools.lru_cache(maxsize=72)
def _cached_player_start_icon(player_idx: int, size: int) -> np.ndarray:
    """Cached player start icon for the editor canvas."""
    from factoriax.ui.icons import create_player_start_icon

    return create_player_start_icon(player_idx, size)


def _render_entities(canvas: np.ndarray, es: EditorState, vp: Viewport) -> None:
    """Draw player start markers and biters on the canvas.

    Biters are drawn first so player markers appear on top when
    they overlap. Biter sprites come from the atlas (same source the
    play renderer uses); player start markers stay procedural since
    they're editor-only spawn-point indicators with no in-game
    counterpart.

    Args:
        canvas: RGBA canvas array (mutated in place).
        es: Current editor state.
        vp: Current viewport.
    """
    ts = vp.tile_size
    biter_sprite = biter_icon_rgba(ts)
    for bx, by in es.biter_positions:
        sx, sy = tile_to_screen(vp, bx, by)
        _blit_alpha(canvas, biter_sprite, sy, sx)
    for idx, (px, py) in es.player_positions.items():
        sx, sy = tile_to_screen(vp, px, py)
        _blit_alpha(canvas, _cached_player_start_icon(idx, ts), sy, sx)


def _highlight_tile(
    canvas: np.ndarray,
    vp: Viewport,
    tx: int,
    ty: int,
    color: tuple[int, int, int, int],
) -> None:
    """Draw a translucent highlight over a single tile.

    Args:
        canvas: RGBA canvas array (mutated in place).
        vp: Current viewport.
        tx: Tile column.
        ty: Tile row.
        color: RGBA highlight colour.
    """
    px, py = tile_to_screen(vp, tx, ty)
    ts = vp.tile_size
    y0 = max(0, py)
    y1 = min(canvas.shape[0], py + ts)
    x0 = max(0, px)
    x1 = min(canvas.shape[1], px + ts)
    if y1 <= y0 or x1 <= x0:
        return
    alpha = color[3] / 255.0
    region = canvas[y0:y1, x0:x1]
    fg = np.array(color[:3], dtype=np.float32)
    blended = (fg * alpha + region[:, :, :3].astype(np.float32) * (1 - alpha)).astype(
        np.uint8
    )
    region[:, :, :3] = blended
    region[:, :, 3] = np.maximum(region[:, :, 3], color[3])


def _blit_clipped(dst: np.ndarray, src: np.ndarray, y: int, x: int) -> None:
    """Copy *src* onto *dst* at ``(y, x)`` with boundary clipping.

    Args:
        dst: Destination RGBA array (mutated in place).
        src: Source RGBA array.
        y: Top row in destination.
        x: Left column in destination.
    """
    dh, dw = dst.shape[:2]
    sh, sw = src.shape[:2]

    sy0 = max(0, -y)
    sx0 = max(0, -x)
    dy0 = max(0, y)
    dx0 = max(0, x)
    dy1 = min(dh, y + sh)
    dx1 = min(dw, x + sw)

    if dy1 <= dy0 or dx1 <= dx0:
        return

    ch = dy1 - dy0
    cw = dx1 - dx0
    dst[dy0:dy1, dx0:dx1] = src[sy0 : sy0 + ch, sx0 : sx0 + cw]


def _get_resource_font(tile_size: int) -> pygame.font.Font:
    """Return a font sized for resource labels at the given tile size.

    Args:
        tile_size: Current tile pixel size.

    Returns:
        A pygame font instance.
    """
    from factoriax.ui.fonts import get_pixel_font

    font_size = max(8, tile_size * 2 // 5)
    return get_pixel_font(font_size)


@functools.lru_cache(maxsize=256)
def _render_resource_label(font: pygame.font.Font, amount: int) -> np.ndarray:
    """Render a resource amount as a small RGBA text label.

    Results are cached per ``(font, amount)`` pair so repeated
    amounts on the same zoom level don't re-render.

    Args:
        font: Pygame font for rendering.
        amount: Resource count to display.

    Returns:
        RGBA uint8 array of shape ``(H, W, 4)``.
    """
    color = (255, 255, 255)
    surface = font.render(str(amount), False, color)
    w, h = surface.get_size()
    rgb = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
    result = np.zeros((h, w, 4), dtype=np.uint8)
    result[:, :, :3] = rgb
    result[:, :, 3] = np.where(np.any(rgb != 0, axis=2), 220, 0)
    return result


def _blit_alpha(dst: np.ndarray, src: np.ndarray, y: int, x: int) -> None:
    """Alpha-composite *src* onto *dst* with boundary clipping.

    Args:
        dst: Destination RGBA array (mutated in place).
        src: Source RGBA array.
        y: Top row in destination.
        x: Left column in destination.
    """
    dh, dw = dst.shape[:2]
    sh, sw = src.shape[:2]

    sy0 = max(0, -y)
    sx0 = max(0, -x)
    dy0 = max(0, y)
    dx0 = max(0, x)
    dy1 = min(dh, y + sh)
    dx1 = min(dw, x + sw)

    if dy1 <= dy0 or dx1 <= dx0:
        return

    ch = dy1 - dy0
    cw = dx1 - dx0
    crop = src[sy0 : sy0 + ch, sx0 : sx0 + cw]
    region = dst[dy0:dy1, dx0:dx1]
    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
    region[:, :, :3] = (
        crop[:, :, :3].astype(np.float32) * alpha
        + region[:, :, :3].astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)
    region[:, :, 3] = np.maximum(region[:, :, 3], crop[:, :, 3])
