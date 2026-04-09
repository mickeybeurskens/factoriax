"""Simple file browser dialog for the debugger.

Renders as an RGBA overlay. Scans the working directory for files
matching a given glob pattern. Keyboard-driven: Up/Down to select,
Enter to confirm, Escape to cancel, type to filter.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pygame

from factoriax.ui.compositing import blit_rgba
from factoriax.ui.fonts import get_pixel_font, render_text_rgba

_BG: tuple[int, int, int, int] = (28, 28, 28, 240)
_BORDER: tuple[int, int, int, int] = (190, 165, 55, 255)
_TEXT_COLOR: tuple[int, int, int] = (220, 215, 180)
_LABEL_COLOR: tuple[int, int, int] = (140, 140, 130)
_FIELD_BG: tuple[int, int, int, int] = (45, 45, 45, 255)
_SELECTED_BG: tuple[int, int, int, int] = (60, 80, 60, 255)
_ROW_BG: tuple[int, int, int, int] = (35, 35, 35, 255)

_DLG_W = 300
_DLG_H = 280
_LIST_ROWS = 8


def _scan_files(pattern: str) -> list[str]:
    """Recursively find files matching *pattern* under cwd.

    Args:
        pattern: Glob pattern (e.g. ``"**/*.npz"``).

    Returns:
        Sorted list of relative path strings.
    """
    return sorted(str(p) for p in Path(".").glob(pattern) if p.is_file())


@dataclasses.dataclass
class FileBrowserDialog:
    """Keyboard-driven file browser overlay.

    Attributes:
        title: Dialog title.
        pattern: Glob pattern for file discovery.
        files: Discovered file paths.
        filter_text: User-typed text to narrow the list.
        selected_index: Currently highlighted row.
        scroll_offset: First visible row.
    """

    title: str = "Load File"
    pattern: str = "**/*.npz"
    files: list[str] = dataclasses.field(default_factory=list)
    filter_text: str = ""
    selected_index: int = 0
    scroll_offset: int = 0

    def __post_init__(self) -> None:
        """Scan for matching files."""
        self.files = _scan_files(self.pattern)
        if self.files:
            self.selected_index = 0

    @property
    def _filtered(self) -> list[str]:
        """Files matching the current filter text."""
        if not self.filter_text:
            return self.files
        ft = self.filter_text.lower()
        return [f for f in self.files if ft in f.lower()]

    def handle_event(self, event: pygame.event.Event) -> str | None:
        """Process a pygame event.

        Returns:
            ``"ok"`` on Enter, ``"cancel"`` on Escape, ``None`` otherwise.
        """
        if event.type != pygame.KEYDOWN:
            return None
        if event.key == pygame.K_ESCAPE:
            return "cancel"
        if event.key == pygame.K_RETURN:
            return "ok"
        if event.key == pygame.K_UP:
            self.selected_index = max(0, self.selected_index - 1)
            self._clamp_scroll()
        elif event.key == pygame.K_DOWN:
            filtered = self._filtered
            self.selected_index = min(len(filtered) - 1, self.selected_index + 1)
            self._clamp_scroll()
        elif event.key == pygame.K_BACKSPACE:
            self.filter_text = self.filter_text[:-1]
            self.selected_index = 0
            self.scroll_offset = 0
        elif event.unicode and event.unicode.isprintable():
            self.filter_text += event.unicode
            self.selected_index = 0
            self.scroll_offset = 0
        return None

    def _clamp_scroll(self) -> None:
        """Keep the selection visible."""
        if self.selected_index < self.scroll_offset:
            self.scroll_offset = self.selected_index
        elif self.selected_index >= self.scroll_offset + _LIST_ROWS:
            self.scroll_offset = self.selected_index - _LIST_ROWS + 1

    def get_path(self) -> str | None:
        """Return the selected file path, or None.

        Returns:
            File path string, or None if nothing selected.
        """
        filtered = self._filtered
        if 0 <= self.selected_index < len(filtered):
            return filtered[self.selected_index]
        return None

    def render(self, base_w: int, base_h: int) -> np.ndarray:
        """Render the dialog as an RGBA overlay.

        Args:
            base_w: Base frame width.
            base_h: Base frame height.

        Returns:
            RGBA uint8 array of shape ``(base_h, base_w, 4)``.
        """
        overlay = np.zeros((base_h, base_w, 4), dtype=np.uint8)
        overlay[:, :] = (0, 0, 0, 140)

        w, h = _DLG_W, _DLG_H
        dx = (base_w - w) // 2
        dy = (base_h - h) // 2
        overlay[dy : dy + h, dx : dx + w] = _BG
        for i in range(2):
            overlay[dy + i, dx : dx + w] = _BORDER
            overlay[dy + h - 1 - i, dx : dx + w] = _BORDER
            overlay[dy : dy + h, dx + i] = _BORDER
            overlay[dy : dy + h, dx + w - 1 - i] = _BORDER

        font = get_pixel_font(12)
        small = get_pixel_font(10)

        # Title.
        title = render_text_rgba(self.title, font, _TEXT_COLOR)
        blit_rgba(overlay, title, dy + 6, dx + (w - title.shape[1]) // 2)

        # Filter field.
        fy = dy + 24
        lbl = render_text_rgba("Filter:", small, _LABEL_COLOR)
        blit_rgba(overlay, lbl, fy, dx + 8)
        fy += 14
        overlay[fy : fy + 16, dx + 8 : dx + w - 8] = _FIELD_BG
        ft = render_text_rgba(self.filter_text + "_", small, _TEXT_COLOR)
        blit_rgba(overlay, ft, fy + 2, dx + 12)

        # File list.
        list_y = fy + 22
        filtered = self._filtered
        visible = filtered[self.scroll_offset : self.scroll_offset + _LIST_ROWS]
        row_h = 16
        for i, fname in enumerate(visible):
            abs_idx = self.scroll_offset + i
            ry = list_y + i * row_h
            bg = _SELECTED_BG if abs_idx == self.selected_index else _ROW_BG
            overlay[ry : ry + row_h - 1, dx + 8 : dx + w - 8] = bg
            # Truncate long paths to fit.
            display = fname if len(fname) < 38 else "..." + fname[-35:]
            ftxt = render_text_rgba(display, small, _TEXT_COLOR)
            blit_rgba(overlay, ftxt, ry + 1, dx + 12)

        if not filtered:
            empty = render_text_rgba("(no files found)", small, _LABEL_COLOR)
            blit_rgba(overlay, empty, list_y + 2, dx + 12)

        # Hints.
        hint = render_text_rgba(
            "Up/Down: select  Enter: OK  Esc: cancel",
            small,
            _LABEL_COLOR,
        )
        blit_rgba(overlay, hint, dy + h - 14, dx + (w - hint.shape[1]) // 2)
        return overlay
