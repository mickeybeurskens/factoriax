"""Cached pixel-font loader and text rendering for pygame applications."""

from __future__ import annotations

import numpy as np
import pygame

# Comma-separated preference list for pygame.font.SysFont.  Terminus is a
# 1:1 pixel bitmap font common on Linux; the rest are fallbacks.
_PIXEL_FONT_PREFERENCE = "terminus,fixedsys excelsior,courier new,monospace,courier"

_font_cache: dict[int, pygame.font.Font] = {}
# Cache for rendered text arrays, keyed by (font identity, text, color).
_text_rgba_cache: dict[tuple[int, str, tuple[int, int, int]], np.ndarray] = {}


def get_pixel_font(size: int) -> pygame.font.Font:
    """Return a pixel-style monospace font at the requested size.

    Results are cached so font objects are created at most once per unique
    size, regardless of how many frames are rendered. Re-initialises
    :mod:`pygame.font` automatically if it has been quit (e.g. after
    ``pygame.quit()`` is called by an interactive play session), clearing
    stale font and text-render caches so callers always receive a valid
    ``Font`` object.

    Parameters
    ----------
    size :
        Desired font height in pixels.

    Returns
    -------
    pygame.font.Font
    """
    if not pygame.font.get_init():
        pygame.font.init()
        _font_cache.clear()
        _text_rgba_cache.clear()
    if size not in _font_cache:
        font = pygame.font.SysFont(_PIXEL_FONT_PREFERENCE, size)
        _font_cache[size] = (
            font
            if font is not None
            else pygame.font.Font(pygame.font.get_default_font(), size)
        )
    return _font_cache[size]


def render_text_rgba(
    text: str,
    font: pygame.font.Font,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Render text to an RGBA array with a fully transparent background.

    Results are cached by ``(id(font), text, color)`` so that identical
    text drawn on consecutive frames is rasterised at most once.

    Parameters
    ----------
    text :
        String to render.
    font :
        pygame Font to use.
    color :
        RGB glyph colour.
    int] :

    Returns
    -------

        RGBA uint8 array of shape ``(h, w, 4)``.
    """
    key = (id(font), text, color)
    cached = _text_rgba_cache.get(key)
    if cached is not None:
        return cached

    surface = font.render(text, False, color)
    w, h = surface.get_size()
    rgb = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
    alpha = np.where((rgb == (0, 0, 0)).all(axis=-1), np.uint8(0), np.uint8(255))
    result = np.zeros((h, w, 4), dtype=np.uint8)
    result[:, :, :3] = rgb
    result[:, :, 3] = alpha

    _text_rgba_cache[key] = result
    return result
