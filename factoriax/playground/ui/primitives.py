"""Core UI primitives: click regions, hit testing, and panel drawing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from factoriax.playground.ui import theme as _theme


@dataclass(frozen=True, slots=True)
class ClickRegion:
    """Represents a clickable rectangular area in the UI.

    Coordinates are in base render resolution (before window scaling).
    """

    x: int
    y: int
    w: int
    h: int
    action: str
    param: int


def hit_test_regions(regions: list[ClickRegion], x: int, y: int) -> ClickRegion | None:
    """Find the first click region containing the given point.

    Parameters
    ----------
    regions :
        List of click regions to test.
    x :
        X coordinate in base resolution.
    y :
        Y coordinate in base resolution.

    Returns
    -------
        The first matching ClickRegion, or None if no hit.
    """
    for region in regions:
        if region.x <= x < region.x + region.w and region.y <= y < region.y + region.h:
            return region
    return None


def draw_panel(
    overlay: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    bg: tuple[int, int, int, int] | None = None,
    border: tuple[int, int, int, int] | None = None,
    border_px: int | None = None,
) -> None:
    """Draw a rectangular panel: solid background with a uniform border.

    Modifies *overlay* in place.

    Parameters
    ----------
    overlay :
        Destination RGBA array of shape (H, W, 4).
    x :
        Left column of the panel.
    y :
        Top row of the panel.
    w :
        Panel width in pixels.
    h :
        Panel height in pixels.
    bg :
        Background RGBA colour.
    border :
        Border RGBA colour.
    border_px :
        Border thickness in pixels.

    * :

    int :

    int] | None :
    """
    if bg is None:
        bg = _theme.PANEL_BG
    if border is None:
        border = _theme.BORDER
    if border_px is None:
        border_px = _theme.BORDER_PX
    overlay[y : y + h, x : x + w] = bg
    for i in range(border_px):
        overlay[y + i, x : x + w] = border
        overlay[y + h - 1 - i, x : x + w] = border
        overlay[y : y + h, x + i] = border
        overlay[y : y + h, x + w - 1 - i] = border
