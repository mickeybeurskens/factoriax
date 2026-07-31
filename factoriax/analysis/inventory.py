"""Draw a player inventory as a standalone RGB panel.

The panel holds one row for each item type except ``EMPTY``. A row
shows a sprite, a label, and a count. A row with a count of zero is
dimmed and shows a ``-``. The reader can therefore see what the player
holds and what the player lacks.

The sprites come from
:func:`factoriax.playground.ui.icons.render_item_icon`, which is the
same source the map renderer, the hotbar, and the editor use. One item
therefore looks the same everywhere.

The module uses NumPy and :mod:`factoriax.playground.ui` only. It does
not import the debugger. The live debugger HUD, the replay viewer, and
the PPO eval video all call it.
"""

from __future__ import annotations

import numpy as np

from factoriax.engine.constants import ItemType
from factoriax.playground.ui.compositing import composite_rgba_over_rgb
from factoriax.playground.ui.fonts import get_pixel_font, render_text_rgba
from factoriax.playground.ui.icons import render_item_icon

__all__ = ["INVENTORY_ITEMS", "render_inventory_panel"]


# Sprite size cache: rendering each icon costs ~1ms, so memoise per
# (item_type, size). Keyed by (item_type, size) → RGBA array.
_ICON_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _icon_rgba(item_type: int, size: int) -> np.ndarray:
    """Return the icon for one item and size, from the cache.

    Parameters
    ----------
    item_type :
        The ``ItemType`` value to draw.
    size :
        The side length of the icon in pixels. Icons are square.

    Returns
    -------
    numpy.ndarray
        RGBA uint8 of shape ``(size, size, 4)``. The array belongs to
        the cache, so a caller that changes it must copy it first.

    Notes
    -----
    The cache never drops an entry. It holds one array for each pair
    of item and size seen in the process. That count is bounded by the
    item count times the number of panel sizes in use.
    """
    key = (item_type, size)
    cached = _ICON_CACHE.get(key)
    if cached is None:
        cached = render_item_icon(item_type, size)
        _ICON_CACHE[key] = cached
    return cached


# Every non-EMPTY ItemType, ordered by enum value so the layout is stable
# across releases. Callers can rely on this tuple being a layout contract.
INVENTORY_ITEMS: tuple[ItemType, ...] = tuple(
    it for it in ItemType if it != ItemType.EMPTY
)


def _item_label(item: ItemType) -> str:
    """Return the display label of an item, at most 14 characters.

    Parameters
    ----------
    item :
        The item to name.

    Returns
    -------
    str
        The enum name in title case with the underscores replaced by
        spaces. A name longer than 14 characters is cut to 13 and
        given a trailing period, so the result is 14 characters.
    """
    name = item.name.replace("_", " ").title()
    return name if len(name) <= 14 else name[:13] + "."


def _inventory_slot_positions(
    width: int,
    height: int,
    num_items: int,
    *,
    row_top: int,
    min_row_h: int = 10,
    max_row_h: int = 16,
    pad_x: int = 6,
) -> list[tuple[int, int, int, int]]:
    """Compute ``(x, y, col_w, row_h)`` for each inventory item.

    Items fill one column at a time. The function takes the smallest
    column count that lets every row reach ``min_row_h``. One column
    that overflows therefore becomes two, two become three, and so on.
    Row height then grows to fill the space left, up to ``max_row_h``.

    A panel too small even for ``min_row_h`` gets shorter rows instead
    of fewer items. The labels can then overlap, but every item keeps
    a slot.

    Parameters
    ----------
    width :
        Panel width in pixels.
    height :
        Panel height in pixels.
    num_items :
        How many rows to place.
    row_top :
        The y of the first row, under the title bar.
    min_row_h :
        The shortest row that still fits the label font.
    max_row_h :
        The tallest row. A cap keeps rows readable in an oversized
        panel instead of stretching a few rows over the whole height.
    pad_x :
        The padding at each side, in pixels.

    Returns
    -------
    list
        One ``(x, y, col_w, row_h)`` for each item, in the order given.
        The first ``ceil(num_items / num_cols)`` entries fill column 0,
        the next batch fills column 1, and so on. A ``num_items`` of 0
        or less gives an empty list.

    Notes
    -----
    The returned rows all fit inside ``height`` only when ``height`` is
    at least ``row_top`` plus 3. Below that, ``y + row_h`` can exceed
    ``height`` and :func:`render_inventory_panel` drops those rows.
    """
    if num_items <= 0:
        return []
    avail = max(1, height - row_top - 2)
    # Smallest column count that lets every row be at least min_row_h tall.
    rows_at_min = max(1, avail // min_row_h)
    num_cols = max(1, (num_items + rows_at_min - 1) // rows_at_min)
    rows_per_col = (num_items + num_cols - 1) // num_cols
    row_h = min(max_row_h, max(min_row_h, avail // rows_per_col))
    # Last-resort: shrink rows below the nominal minimum if the panel is
    # truly tiny. The label font may overlap, but every item still gets
    # a numbered slot.
    if row_h * rows_per_col > avail:
        row_h = max(1, avail // rows_per_col)
    col_w = max(1, (width - (num_cols + 1) * pad_x) // num_cols)
    positions: list[tuple[int, int, int, int]] = []
    for i in range(num_items):
        col_idx = i // rows_per_col
        row_i = i % rows_per_col
        col_x = pad_x + col_idx * (col_w + pad_x)
        positions.append((col_x, row_top + row_i * row_h, col_w, row_h))
    return positions


def render_inventory_panel(
    inventory: np.ndarray,
    width: int,
    height: int,
    *,
    title: str = "Inventory",
) -> np.ndarray:
    """Render the player inventory as an RGB panel.

    Every item in :data:`INVENTORY_ITEMS` gets its own row, at any
    panel size. One condition applies: the panel must be tall enough
    for one row under the title bar. With the default title, that
    means a ``height`` of 22 or more. Below that, the rows do not fit and the
    function draws none of them, which leaves an almost empty panel.

    A count of zero still gets a row. That row is dimmed and shows a
    ``-`` in place of the number, so the reader can see which items
    the player does not hold.

    Parameters
    ----------
    inventory :
        Per-item counts in a 1-D array, indexed by ``ItemType`` value.
        An array shorter than the item count is allowed. Items past
        its end read as 0 instead of raising.
    width :
        Output width in pixels.
    height :
        Output height in pixels.
    title :
        The heading drawn at the top left.

    Returns
    -------
    numpy.ndarray
        RGB uint8 of shape ``(height, width, 3)``. The panel has a
        dark background, so a caller can composite it without a mask.

    Notes
    -----
    A label too wide for its column is retried at two smaller fonts,
    and dropped if it still does not fit. The icon and the count are
    drawn whether or not the label fits.
    """
    img = np.full((height, width, 3), (30, 30, 35), dtype=np.uint8)
    font = get_pixel_font(11)
    title_font = get_pixel_font(12)

    # Title bar.
    title_rgba = render_text_rgba(title, title_font, (210, 210, 200))
    th = title_rgba.shape[0]
    pad_x = 6
    top_pad = 4
    if top_pad + th <= height and pad_x + title_rgba.shape[1] <= width:
        composite_rgba_over_rgb(
            img[top_pad : top_pad + th, pad_x : pad_x + title_rgba.shape[1]],
            title_rgba,
        )

    row_top = top_pad + th + 4
    positions = _inventory_slot_positions(
        width,
        height,
        len(INVENTORY_ITEMS),
        row_top=row_top,
        pad_x=pad_x,
    )

    for item, (slot_x, y, col_w, row_h) in zip(INVENTORY_ITEMS, positions, strict=True):
        if y + row_h > height:
            continue
        swatch_sz = max(6, row_h - 4)
        count = int(inventory[int(item)]) if int(item) < inventory.shape[0] else 0
        active = count > 0

        # Sprite icon — same art as the GPU map renderer's atlas.
        sx = slot_x
        sy = y + (row_h - swatch_sz) // 2
        icon_rgba = _icon_rgba(int(item), swatch_sz).copy()
        if not active:
            # Dim the alpha so inactive items recede; preserves the
            # silhouette without recolouring.
            icon_rgba[..., 3] = (icon_rgba[..., 3].astype(np.uint16) // 3).astype(
                np.uint8
            )
        composite_rgba_over_rgb(
            img[sy : sy + swatch_sz, sx : sx + swatch_sz], icon_rgba
        )

        # Label.
        label = _item_label(item)
        label_color = (220, 220, 210) if active else (110, 110, 110)
        col_right = slot_x + col_w
        lx = sx + swatch_sz + 5
        avail_w = col_right - 20 - lx  # reserve 20 px for the count on the right
        label_rgba = render_text_rgba(label, font, label_color)
        lh, lw = label_rgba.shape[:2]
        if lw > avail_w:
            for fallback_size in (9, 7):
                small_font = get_pixel_font(fallback_size)
                label_rgba = render_text_rgba(label, small_font, label_color)
                lh, lw = label_rgba.shape[:2]
                if lw <= avail_w:
                    break
        ly = y + (row_h - lh) // 2
        if lw <= avail_w and 0 <= ly and ly + lh <= height:
            composite_rgba_over_rgb(img[ly : ly + lh, lx : lx + lw], label_rgba)

        # Count, right-aligned inside this slot's column.
        if active:
            count_str = str(count)
            count_color = (130, 220, 150)
        else:
            count_str = "-"
            count_color = (90, 90, 95)
        count_rgba = render_text_rgba(count_str, font, count_color)
        ch, cw = count_rgba.shape[:2]
        cx = col_right - cw
        cy = y + (row_h - ch) // 2
        if cx >= 0 and 0 <= cy and cy + ch <= height:
            composite_rgba_over_rgb(img[cy : cy + ch, cx : cx + cw], count_rgba)

    return img
