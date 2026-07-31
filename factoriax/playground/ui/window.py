"""Window sizing utilities for pygame applications."""

from __future__ import annotations

import pygame

_monitor_size: tuple[int, int] | None = None


def _get_monitor_size() -> tuple[int, int]:
    """Return the monitor resolution, cached on first call.

    ``pygame.display.Info()`` reports the monitor size before any
    display mode is set, but returns the *window* size afterwards.
    This function captures the true monitor dimensions once and
    reuses them for all subsequent calls.

    Returns
    -------
    type
        ``(width, height)`` of the primary monitor in pixels.
    """
    global _monitor_size  # noqa: PLW0603
    if _monitor_size is None:
        info = pygame.display.Info()
        _monitor_size = (info.current_w, info.current_h)
    return _monitor_size


def calculate_window_size(
    base_width: int, base_height: int, scale_factor: float = 0.8
) -> tuple[int, int]:
    """Calculate window size using integer scaling for crisp pixel art.

    Uses the largest integer scale factor that fits within *scale_factor*
    (default 80%) of the screen.

    Parameters
    ----------
    base_width :
        Base render width in pixels.
    base_height :
        Base render height in pixels.
    scale_factor :
        Fraction of screen to use (0.0 to 1.0).

    Returns
    -------
        ``(window_width, window_height)`` in pixels.
    """
    monitor_w, monitor_h = _get_monitor_size()
    max_width = int(monitor_w * scale_factor)
    max_height = int(monitor_h * scale_factor)

    max_scale_w = max_width // base_width
    max_scale_h = max_height // base_height
    scale = max(1, min(max_scale_w, max_scale_h))

    return base_width * scale, base_height * scale


def auto_ui_scale(base_size: int = 1024) -> int:
    """Pick the highest UI scale where the canvas fits the monitor.

    Returns the largest integer ``s`` in ``{3, 2, 1}`` such that
    ``base_size * s`` fits within 80% of the monitor on both axes.

    Parameters
    ----------
    base_size :
        Logical base resolution (default 1024).

    Returns
    -------
        Integer scale factor (1, 2, or 3).
    """
    monitor_w, monitor_h = _get_monitor_size()
    limit_w = int(monitor_w * 0.8)
    limit_h = int(monitor_h * 0.8)
    for scale in (3, 2, 1):
        if base_size * scale <= limit_w and base_size * scale <= limit_h:
            return scale
    return 1
