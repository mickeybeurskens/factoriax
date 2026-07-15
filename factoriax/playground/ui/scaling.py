"""Fixed-resolution canvas with integer scaling to a pygame window.

All menu and UI drawing happens on a fixed-size pygame surface. At
the end of each frame the canvas is integer-scaled and centered on
the actual display surface with black letterboxing. This ensures
pixel-perfect layout regardless of the window size.
"""

from __future__ import annotations

import pygame


class ScaledCanvas:
    """A fixed-size pygame surface that integer-scales to the window."""

    def __init__(
        self,
        base_size: int,
        ui_scale: int,
        window: pygame.Surface,
    ) -> None:
        """Create a canvas and compute the initial scale for the window.

        Parameters
        ----------
            base_size: Logical base resolution (e.g. 1024).
            ui_scale: Theme scale factor (1, 2, or 3).
            window: The pygame display surface.
        """
        self._width = base_size * ui_scale
        self._height = base_size * ui_scale
        self._surface = pygame.Surface((self._width, self._height))
        self._scale = 1
        self._ox = 0
        self._oy = 0
        self.handle_resize(*window.get_size())

    @property
    def surface(self) -> pygame.Surface:
        """The fixed-size drawing target."""
        return self._surface

    @property
    def width(self) -> int:
        """Canvas width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Canvas height in pixels."""
        return self._height

    def handle_resize(self, window_w: int, window_h: int) -> None:
        """Recompute scale and offsets after a window resize event.

        Parameters
        ----------
        window_w :
            New window width in pixels.
        window_h :
            New window height in pixels.
        """
        self._scale = max(
            1,
            min(window_w // self._width, window_h // self._height),
        )
        self._ox = (window_w - self._width * self._scale) // 2
        self._oy = (window_h - self._height * self._scale) // 2

    def to_canvas(self, window_x: int, window_y: int) -> tuple[int, int]:
        """Transform window pixel coordinates to canvas coordinates.

        Parameters
        ----------
        window_x :
            X position in window pixels.
        window_y :
            Y position in window pixels.
        """
        return (
            (window_x - self._ox) // self._scale,
            (window_y - self._oy) // self._scale,
        )

    def present(self, screen: pygame.Surface) -> None:
        """Scale the canvas and blit it centered onto the display surface.

        Fills the screen with black for letterboxing, integer-scales the
        canvas, and blits it centered.

        Parameters
        ----------
        screen :
            The pygame display surface.
        """
        scaled = pygame.transform.scale(
            self._surface,
            (self._width * self._scale, self._height * self._scale),
        )
        screen.fill((0, 0, 0))
        screen.blit(scaled, (self._ox, self._oy))
        pygame.display.flip()
