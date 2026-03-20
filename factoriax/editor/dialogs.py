"""Dialogs for the level editor: new-level picker and file open/save.

The new-level dialog is rendered as an RGBA overlay (similar to the
game's pause menu).  Save/load write to a ``levels/`` directory,
avoiding any dependency on tkinter or native OS file pickers.
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


LEVELS_DIR = Path("levels")

_FILE_DLG_W = 280
_FILE_DLG_H = 260
_FILE_LIST_ROWS = 6
_FILE_SELECTED_BG: tuple[int, int, int, int] = (60, 80, 60, 255)
_FILE_ROW_BG: tuple[int, int, int, int] = (35, 35, 35, 255)


@dataclasses.dataclass
class FileDialog:
    """In-editor dialog for choosing or typing a filename.

    Works for both save and load. Shows a text field for the filename
    and a scrollable list of existing ``.json`` files in ``levels/``.
    Press W/S to navigate the list, Enter to confirm, Escape to cancel.

    Attributes:
        mode: ``"save"`` or ``"load"``.
        filename_text: Editable filename (without extension).
        files: List of existing level filenames (stems only).
        selected_index: Currently highlighted file in the list.
        scroll_offset: First visible row in the file list.
    """

    mode: str = "save"
    filename_text: str = ""
    files: list[str] = dataclasses.field(default_factory=list)
    selected_index: int = -1
    scroll_offset: int = 0

    def __post_init__(self) -> None:
        """Scan the levels directory for existing files."""
        self.files = _list_level_files()
        if self.mode == "load" and self.files:
            self.selected_index = 0
            self.filename_text = self.files[0]

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
        if event.key == pygame.K_UP:
            self._move_selection(-1)
            return None
        if event.key == pygame.K_DOWN:
            self._move_selection(1)
            return None
        if event.key == pygame.K_BACKSPACE:
            self.filename_text = self.filename_text[:-1]
            self.selected_index = -1
            return None
        if event.unicode and event.unicode.isprintable():
            self.filename_text += event.unicode
            self.selected_index = -1
        return None

    def _move_selection(self, delta: int) -> None:
        """Move the file list selection by *delta* rows."""
        if not self.files:
            return
        new_idx = self.selected_index + delta
        new_idx = max(0, min(len(self.files) - 1, new_idx))
        self.selected_index = new_idx
        self.filename_text = self.files[new_idx]
        if new_idx < self.scroll_offset:
            self.scroll_offset = new_idx
        elif new_idx >= self.scroll_offset + _FILE_LIST_ROWS:
            self.scroll_offset = new_idx - _FILE_LIST_ROWS + 1

    def get_path(self) -> Path | None:
        """Return the chosen path, or ``None`` for empty input.

        Returns:
            :class:`Path` in the levels directory, or ``None``.
        """
        name = self.filename_text.strip()
        if not name:
            return None
        LEVELS_DIR.mkdir(parents=True, exist_ok=True)
        if not name.endswith(".json"):
            name += ".json"
        return LEVELS_DIR / name

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

        w, h = _FILE_DLG_W, _FILE_DLG_H
        dx = (base_w - w) // 2
        dy = (base_h - h) // 2
        overlay[dy : dy + h, dx : dx + w] = _BG
        for i in range(2):
            overlay[dy + i, dx : dx + w] = _BORDER
            overlay[dy + h - 1 - i, dx : dx + w] = _BORDER
            overlay[dy : dy + h, dx + i] = _BORDER
            overlay[dy : dy + h, dx + w - 1 - i] = _BORDER

        font = get_pixel_font(14)
        small = get_pixel_font(10)

        title_text = "Save Level" if self.mode == "save" else "Load Level"
        title = _render_text_rgba(title_text, font, _TEXT_COLOR)
        _blit_rgba(overlay, title, dy + 8, dx + (w - title.shape[1]) // 2)

        # Filename field
        lbl = _render_text_rgba("Filename:", font, _LABEL_COLOR)
        _blit_rgba(overlay, lbl, dy + 32, dx + 12)
        fy = dy + 50
        overlay[fy : fy + 22, dx + 12 : dx + w - 12] = _FIELD_ACTIVE
        val = _render_text_rgba(
            self.filename_text + "_", font, _TEXT_COLOR,
        )
        _blit_rgba(overlay, val, fy + 3, dx + 16)

        # File list
        list_y = fy + 30
        list_label = _render_text_rgba("Existing files:", small, _LABEL_COLOR)
        _blit_rgba(overlay, list_label, list_y, dx + 12)
        list_y += 16
        row_h = 18
        visible = self.files[
            self.scroll_offset : self.scroll_offset + _FILE_LIST_ROWS
        ]
        for i, fname in enumerate(visible):
            abs_idx = self.scroll_offset + i
            ry = list_y + i * row_h
            bg = (
                _FILE_SELECTED_BG
                if abs_idx == self.selected_index
                else _FILE_ROW_BG
            )
            overlay[ry : ry + row_h - 1, dx + 12 : dx + w - 12] = bg
            ftxt = _render_text_rgba(fname, small, _TEXT_COLOR)
            _blit_rgba(overlay, ftxt, ry + 2, dx + 16)

        if not self.files:
            empty = _render_text_rgba("(no files)", small, _LABEL_COLOR)
            _blit_rgba(overlay, empty, list_y + 2, dx + 16)

        hint = _render_text_rgba(
            "Up/Down: select  Enter: OK  Esc: cancel",
            small,
            _LABEL_COLOR,
        )
        _blit_rgba(
            overlay, hint, dy + h - 16, dx + (w - hint.shape[1]) // 2,
        )
        return overlay


def _list_level_files() -> list[str]:
    """Return sorted list of level filenames (stems) in the levels dir.

    Returns:
        List of filename stems, sorted alphabetically.
    """
    if not LEVELS_DIR.is_dir():
        return []
    return sorted(p.stem for p in LEVELS_DIR.glob("*.json"))


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
