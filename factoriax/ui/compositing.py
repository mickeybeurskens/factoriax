"""Image compositing utilities for numpy-array-based UI rendering."""

from __future__ import annotations

import numpy as np

from factoriax.ui.theme import SCROLLBAR_BG, SCROLLBAR_THUMB, SCROLLBAR_W


def composite_rgba_over_rgb(background: np.ndarray, overlay: np.ndarray) -> None:
    """Composite an RGBA overlay onto an RGB background in-place.

    Only blends pixels within the bounding box of non-transparent
    overlay content, skipping the float arithmetic for the large
    fully-transparent regions that surround a centered menu panel.

    Args:
        background: RGB image array of shape (H, W, 3), modified
            in place.
        overlay: RGBA image array of shape (H, W, 4).
    """
    alpha_chan = overlay[:, :, 3]
    row_has_alpha = np.any(alpha_chan > 0, axis=1)
    if not np.any(row_has_alpha):
        return
    col_has_alpha = np.any(alpha_chan > 0, axis=0)

    r0 = int(np.argmax(row_has_alpha))
    r1 = len(row_has_alpha) - int(np.argmax(row_has_alpha[::-1]))
    c0 = int(np.argmax(col_has_alpha))
    c1 = len(col_has_alpha) - int(np.argmax(col_has_alpha[::-1]))

    a = overlay[r0:r1, c0:c1, 3:4].astype(np.float32) / 255.0
    fg = overlay[r0:r1, c0:c1, :3].astype(np.float32)
    bg = background[r0:r1, c0:c1].astype(np.float32)
    background[r0:r1, c0:c1] = (fg * a + bg * (1 - a)).astype(np.uint8)


def blit_rgba(
    overlay: np.ndarray,
    src: np.ndarray,
    y: int,
    x: int,
) -> None:
    """Alpha-composite *src* onto *overlay* at (y, x), clipping to bounds.

    Args:
        overlay: Destination RGBA array of shape (H, W, 4); modified in place.
        src: Source RGBA array of shape (h, w, 4).
        y: Top row in *overlay*.
        x: Left column in *overlay*.
    """
    oh, ow = overlay.shape[:2]
    sh, sw = src.shape[:2]

    src_y0 = max(0, -y)
    src_x0 = max(0, -x)
    dst_y0, dst_x0 = max(0, y), max(0, x)
    dst_y1 = min(oh, y + sh)
    dst_x1 = min(ow, x + sw)

    if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
        return

    crop_h = dst_y1 - dst_y0
    crop_w = dst_x1 - dst_x0
    src_crop = src[src_y0 : src_y0 + crop_h, src_x0 : src_x0 + crop_w]
    dst = overlay[dst_y0:dst_y1, dst_x0:dst_x1]

    alpha = src_crop[:, :, 3:4].astype(np.float32) / 255.0
    dst[:, :, :3] = (
        src_crop[:, :, :3].astype(np.float32) * alpha
        + dst[:, :, :3].astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)
    dst[:, :, 3] = np.maximum(dst[:, :, 3], src_crop[:, :, 3])


def clip_scroll_offset(offset: int, content_h: int, viewport_h: int) -> int:
    """Clamp a scroll offset to the valid range.

    Args:
        offset: Proposed scroll offset in pixels.
        content_h: Total height of the scrollable content in pixels.
        viewport_h: Height of the visible viewport in pixels.

    Returns:
        Clamped offset in ``[0, max(0, content_h - viewport_h)]``.
    """
    return max(0, min(offset, max(0, content_h - viewport_h)))


def blit_scroll_view(
    overlay: np.ndarray,
    content: np.ndarray,
    vp_x: int,
    vp_y: int,
    vp_w: int,
    vp_h: int,
    scroll_offset: int,
) -> None:
    """Composite a scrollable content canvas into a viewport on *overlay*.

    When the content is taller than the viewport a scrollbar is drawn
    along the right edge.

    Args:
        overlay: Destination RGBA array; modified in place.
        content: Full content RGBA canvas of shape ``(content_h, vp_w, 4)``.
        vp_x: Left edge of the viewport in overlay coordinates.
        vp_y: Top edge of the viewport in overlay coordinates.
        vp_w: Viewport width in pixels (includes scrollbar when shown).
        vp_h: Viewport height in pixels.
        scroll_offset: Number of content pixels scrolled off the top.
    """
    content_h = content.shape[0]
    needs_bar = content_h > vp_h
    render_w = vp_w - (SCROLLBAR_W if needs_bar else 0)

    visible = content[scroll_offset : scroll_offset + vp_h, :render_w]
    blit_rgba(overlay, visible, vp_y, vp_x)

    if needs_bar:
        bar_x = vp_x + vp_w - SCROLLBAR_W
        overlay[vp_y : vp_y + vp_h, bar_x : bar_x + SCROLLBAR_W] = SCROLLBAR_BG
        thumb_h = max(12, vp_h * vp_h // content_h)
        max_scroll = content_h - vp_h
        thumb_y = vp_y + int((vp_h - thumb_h) * scroll_offset / max(1, max_scroll))
        overlay[thumb_y : thumb_y + thumb_h, bar_x : bar_x + SCROLLBAR_W] = (
            SCROLLBAR_THUMB
        )
