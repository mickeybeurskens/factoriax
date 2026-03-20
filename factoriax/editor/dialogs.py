"""Dialogs for the level editor: new-level picker and file open/save.

The new-level dialog is rendered as an RGBA overlay (similar to the
game's pause menu).  Save/load use :mod:`tkinter.filedialog` for
native OS file pickers, avoiding the need for a custom file browser.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pygame

from factoriax.play.ui import get_pixel_font

_BG = (22, 22, 22, 228)
_BORDER = (190, 165, 55, 255)
_FIELD_BG = (40, 40, 40, 255)
_FIELD_ACTIVE = (55, 55, 65, 255)
_TEXT_COLOR = (200, 200, 200)
_LABEL_COLOR = (160, 160, 160)

_DIALOG_W = 240
_DIALOG_H = 180


@dataclasses.dataclass
class NewLevelDialog:
    """In-editor dialog for specifying a new level's dimensions and name.

    Rendered as an RGBA overlay centred on the screen.  Handles its own
    keyboard events and returns ``"ok"`` or ``"cancel"`` when done.

    Attributes:
        width_text: Editable string for the map width.
        height_text: Editable string for the map height.
        name_text: Editable string for the level name.
        active_field: Index of the focused text field (0-2).
    """

    width_text: str = "15"
    height_text: str = "15"
    name_text: str = "untitled"
    active_field: int = 0

    def handle_event(self, event: pygame.event.Event) -> str | None:
        """Process a pygame event and return a result if the dialog closes.

        Args:
            event: A ``pygame.KEYDOWN`` event.

        Returns:
            ``"ok"`` when Enter is pressed, ``"cancel"`` on Escape,
            or ``None`` if the dialog stays open.
        """
        if event.type != pygame.KEYDOWN:
            return None
        if event.key == pygame.K_ESCAPE:
            return "cancel"
        if event.key == pygame.K_RETURN:
            return "ok"
        if event.key == pygame.K_TAB:
            self.active_field = (self.active_field + 1) % 3
            return None
        if event.key == pygame.K_BACKSPACE:
            self._delete_char()
            return None
        if event.unicode and event.unicode.isprintable():
            self._insert_char(event.unicode)
        return None

    def _delete_char(self) -> None:
        """Remove the last character from the active field."""
        if self.active_field == 0:
            self.width_text = self.width_text[:-1]
        elif self.active_field == 1:
            self.height_text = self.height_text[:-1]
        else:
            self.name_text = self.name_text[:-1]

    def _insert_char(self, ch: str) -> None:
        """Append a character to the active field.

        Args:
            ch: Character to insert.
        """
        if self.active_field == 0:
            if ch.isdigit():
                self.width_text += ch
        elif self.active_field == 1:
            if ch.isdigit():
                self.height_text += ch
        else:
            self.name_text += ch

    def render(self, base_w: int, base_h: int) -> np.ndarray:
        """Render the dialog as an RGBA overlay.

        Args:
            base_w: Base window width.
            base_h: Base window height.

        Returns:
            RGBA uint8 array of shape ``(base_h, base_w, 4)``.
        """
        overlay = np.zeros((base_h, base_w, 4), dtype=np.uint8)
        overlay[:, :] = (0, 0, 0, 140)

        dx = (base_w - _DIALOG_W) // 2
        dy = (base_h - _DIALOG_H) // 2
        overlay[dy : dy + _DIALOG_H, dx : dx + _DIALOG_W] = _BG
        for i in range(2):
            overlay[dy + i, dx : dx + _DIALOG_W] = _BORDER
            overlay[dy + _DIALOG_H - 1 - i, dx : dx + _DIALOG_W] = _BORDER
            overlay[dy : dy + _DIALOG_H, dx + i] = _BORDER
            overlay[dy : dy + _DIALOG_H, dx + _DIALOG_W - 1 - i] = _BORDER

        font = get_pixel_font(14)
        title = _render_text_rgba("New Level", font, _TEXT_COLOR)
        _blit_rgba(overlay, title, dy + 8, dx + (_DIALOG_W - title.shape[1]) // 2)

        labels = ["Width:", "Height:", "Name:"]
        values = [self.width_text, self.height_text, self.name_text]
        field_y = dy + 36

        for i, (label, value) in enumerate(zip(labels, values)):
            lbl = _render_text_rgba(label, font, _LABEL_COLOR)
            _blit_rgba(overlay, lbl, field_y, dx + 12)

            field_bg = _FIELD_ACTIVE if i == self.active_field else _FIELD_BG
            fx = dx + 80
            fw = _DIALOG_W - 92
            fh = 20
            overlay[field_y : field_y + fh, fx : fx + fw] = field_bg

            val = _render_text_rgba(
                value + ("_" if i == self.active_field else ""), font, _TEXT_COLOR
            )
            _blit_rgba(overlay, val, field_y + 2, fx + 4)
            field_y += 30

        hint = _render_text_rgba(
            "Tab: next field  Enter: OK  Esc: cancel", get_pixel_font(10), _LABEL_COLOR
        )
        _blit_rgba(
            overlay, hint, dy + _DIALOG_H - 20, dx + (_DIALOG_W - hint.shape[1]) // 2
        )

        return overlay

    def get_values(self) -> tuple[int, int, str]:
        """Parse the dialog fields into typed values.

        Returns:
            ``(width, height, name)`` with fallback defaults for empty
            or invalid inputs.
        """
        try:
            w = max(3, int(self.width_text))
        except ValueError:
            w = 15
        try:
            h = max(3, int(self.height_text))
        except ValueError:
            h = 15
        name = self.name_text.strip() or "untitled"
        return w, h, name


@dataclasses.dataclass
class NumberInputDialog:
    """Small overlay for typing a single integer value.

    Renders as a compact RGBA overlay centred on the screen.  Accepts
    digits, backspace, Enter to confirm, and Escape to cancel.

    Attributes:
        label: Prompt shown above the input field.
        text: Current input text.
        min_value: Minimum allowed value (clamped on confirm).
        max_value: Maximum allowed value (clamped on confirm).
        default: Fallback value when the input is empty or invalid.
    """

    label: str = "Value:"
    text: str = ""
    min_value: int = 0
    max_value: int = 100
    default: int = 0

    def handle_event(self, event: pygame.event.Event) -> str | None:
        """Process a pygame event.

        Args:
            event: A ``pygame.KEYDOWN`` event.

        Returns:
            ``"ok"`` on Enter, ``"cancel"`` on Escape, or ``None``.
        """
        if event.type != pygame.KEYDOWN:
            return None
        if event.key == pygame.K_ESCAPE:
            return "cancel"
        if event.key == pygame.K_RETURN:
            return "ok"
        if event.key == pygame.K_BACKSPACE:
            self.text = self.text[:-1]
            return None
        if event.unicode and event.unicode.isdigit():
            self.text += event.unicode
        return None

    def get_value(self) -> int:
        """Parse the text field into a clamped integer.

        Returns:
            Integer between ``min_value`` and ``max_value``.
        """
        try:
            val = int(self.text)
        except ValueError:
            val = self.default
        return max(self.min_value, min(self.max_value, val))

    def render(self, base_w: int, base_h: int) -> np.ndarray:
        """Render the dialog as an RGBA overlay.

        Args:
            base_w: Base window width.
            base_h: Base window height.

        Returns:
            RGBA uint8 array of shape ``(base_h, base_w, 4)``.
        """
        w, h = 180, 80
        overlay = np.zeros((base_h, base_w, 4), dtype=np.uint8)
        overlay[:, :] = (0, 0, 0, 140)

        dx = (base_w - w) // 2
        dy = (base_h - h) // 2
        overlay[dy : dy + h, dx : dx + w] = _BG
        for i in range(2):
            overlay[dy + i, dx : dx + w] = _BORDER
            overlay[dy + h - 1 - i, dx : dx + w] = _BORDER
            overlay[dy : dy + h, dx + i] = _BORDER
            overlay[dy : dy + h, dx + w - 1 - i] = _BORDER

        font = get_pixel_font(14)
        lbl = _render_text_rgba(self.label, font, _LABEL_COLOR)
        _blit_rgba(overlay, lbl, dy + 10, dx + 12)

        fy = dy + 34
        overlay[fy : fy + 22, dx + 12 : dx + w - 12] = _FIELD_ACTIVE
        val = _render_text_rgba(
            self.text + "_",
            font,
            _TEXT_COLOR,
        )
        _blit_rgba(overlay, val, fy + 3, dx + 16)

        hint = _render_text_rgba(
            "Enter: OK  Esc: cancel",
            get_pixel_font(10),
            _LABEL_COLOR,
        )
        _blit_rgba(
            overlay,
            hint,
            dy + h - 16,
            dx + (w - hint.shape[1]) // 2,
        )
        return overlay


def ask_save_path(initial_name: str) -> Path | None:
    """Open a native file-save dialog and return the chosen path.

    Uses :mod:`tkinter.filedialog` so no custom file browser is needed.

    Args:
        initial_name: Suggested filename (without extension).

    Returns:
        Chosen :class:`Path`, or ``None`` if the user cancelled.
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    path = filedialog.asksaveasfilename(
        title="Save Level",
        initialfile=f"{initial_name}.json",
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(path) if path else None


def ask_load_path() -> Path | None:
    """Open a native file-open dialog and return the chosen path.

    Returns:
        Chosen :class:`Path`, or ``None`` if the user cancelled.
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Load Level",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(path) if path else None


_HELP_LINES = [
    "-- Drawing --",
    "Left click/drag  Paint tile or machine",
    "Right click/drag  Erase (layer-aware)",
    "X  Eraser tool (clears both layers)",
    "B  Paint tool    F  Fill rect tool",
    "1-5  Terrain: Dirt Water Iron Copper Coal",
    "6-9,0  Machine: Miner Chest Belt Arm Asm",
    "R  Rotate machine (or rotate under cursor)",
    "",
    "-- Resources --",
    "T  Toggle exact / range mode",
    "V  Toggle resource overlay",
    "[ ]  Adjust resource value",
    "Click value in toolbar to type amount",
    "",
    "-- View --",
    "Scroll wheel  Zoom",
    "Middle drag / Arrows  Pan",
    "",
    "-- Map --",
    "Ctrl+Arrows  Add/remove row or column",
    "Ctrl+N  New    Ctrl+O  Load    Ctrl+S  Save",
    "F5  Play-test    Esc  Quit",
    "",
    "Press any key to close",
]


def render_help_overlay(base_w: int, base_h: int) -> np.ndarray:
    """Render a controls reference overlay.

    Args:
        base_w: Base window width.
        base_h: Base window height.

    Returns:
        RGBA uint8 array of shape ``(base_h, base_w, 4)``.
    """
    overlay = np.zeros((base_h, base_w, 4), dtype=np.uint8)
    overlay[:, :] = (0, 0, 0, 180)

    font = get_pixel_font(12)
    line_h = 16
    total_h = len(_HELP_LINES) * line_h + 24
    panel_w = min(360, base_w - 20)
    dx = (base_w - panel_w) // 2
    dy = max(4, (base_h - total_h) // 2)
    panel_h = min(total_h, base_h - 8)

    overlay[dy : dy + panel_h, dx : dx + panel_w] = _BG
    for i in range(2):
        overlay[dy + i, dx : dx + panel_w] = _BORDER
        overlay[dy + panel_h - 1 - i, dx : dx + panel_w] = _BORDER
        overlay[dy : dy + panel_h, dx + i] = _BORDER
        overlay[dy : dy + panel_h, dx + panel_w - 1 - i] = _BORDER

    title = _render_text_rgba("Editor Controls", get_pixel_font(14), _TEXT_COLOR)
    _blit_rgba(overlay, title, dy + 6, dx + (panel_w - title.shape[1]) // 2)

    y = dy + 24
    for line in _HELP_LINES:
        if not line:
            y += line_h // 2
            continue
        if line.startswith("--"):
            txt = _render_text_rgba(line, font, (190, 165, 55))
        else:
            txt = _render_text_rgba(line, font, _LABEL_COLOR)
        _blit_rgba(overlay, txt, y, dx + 12)
        y += line_h

    return overlay


def _render_text_rgba(
    text: str,
    font: pygame.font.Font,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Render text to RGBA with transparent background.

    Args:
        text: String to render.
        font: Pygame font.
        color: RGB text colour.

    Returns:
        RGBA uint8 array of shape ``(H, W, 4)``.
    """
    surface = font.render(text, False, color)
    w, h = surface.get_size()
    rgb = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
    result = np.zeros((h, w, 4), dtype=np.uint8)
    result[:, :, :3] = rgb
    result[:, :, 3] = np.where(np.any(rgb != 0, axis=2), 255, 0)
    return result


def _blit_rgba(overlay: np.ndarray, src: np.ndarray, y: int, x: int) -> None:
    """Alpha-composite *src* onto *overlay* with clipping.

    Args:
        overlay: Destination RGBA array (mutated in place).
        src: Source RGBA array.
        y: Top row.
        x: Left column.
    """
    oh, ow = overlay.shape[:2]
    sh, sw = src.shape[:2]
    sy0 = max(0, -y)
    sx0 = max(0, -x)
    dy0 = max(0, y)
    dx0 = max(0, x)
    dy1 = min(oh, y + sh)
    dx1 = min(ow, x + sw)
    if dy1 <= dy0 or dx1 <= dx0:
        return
    ch = dy1 - dy0
    cw = dx1 - dx0
    crop = src[sy0 : sy0 + ch, sx0 : sx0 + cw]
    dst = overlay[dy0:dy1, dx0:dx1]
    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
    dst[:, :, :3] = (
        crop[:, :, :3].astype(np.float32) * alpha
        + dst[:, :, :3].astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)
    dst[:, :, 3] = np.maximum(dst[:, :, 3], crop[:, :, 3])
