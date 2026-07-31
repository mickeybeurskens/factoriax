"""The dialogs of the level editor.

There are three dialogs. The new level dialog asks for the size and the name.
The file dialog opens and saves a level. The machine inspector shows the
contents of one machine, and lets the user change them.

Each dialog draws an RGBA overlay at the center of the screen, and reads its
own keyboard events.

The file dialog lists the files of the ``levels/`` directory. It is therefore
free of tkinter, and free of the file picker of the operating system.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pygame

from factoriax.engine.constants import (
    ItemType,
    SlotRole,
)
from factoriax.engine.tables import MACHINE_MAX_STACK
from factoriax.playground.editor.slot_display import (
    MACHINE_NUM_SLOTS,
    MACHINE_SLOT_ROLES,
    SLOT_ROLE_COLORS,
    SLOT_ROLE_LABELS,
)
from factoriax.playground.ui.fonts import get_pixel_font
from factoriax.playground.ui.icons import render_item_icon
from factoriax.playground.ui.labels import MACHINE_TYPE_NAMES

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
    """Dialog that asks for the size and the name of a new level.

    :meth:`handle_event` returns ``"ok"`` when the user accepts the dialog,
    and ``"cancel"`` when the user leaves it. It returns ``None`` while the
    dialog is open.

    Attributes
    ----------
    width_text, height_text
        Size fields, as the text that the user typed.
    name_text
        Name field, as the text that the user typed.
    active_field
        Index of the field that has the keyboard focus.
    """

    width_text: str = "15"
    height_text: str = "15"
    name_text: str = "untitled"
    active_field: int = 0

    def handle_event(self, event: pygame.event.Event) -> str | None:
        """Read one event, and report whether the dialog closes.

        Parameters
        ----------
        event
            Event to read. The dialog reads a ``pygame.KEYDOWN`` event only.

        Returns
        -------
        str or None
            ``"ok"`` for Enter, and ``"cancel"`` for Escape. It is ``None``
            while the dialog stays open.
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
        """Add one character to the field that has the focus.

        A size field takes a digit only. The name field takes any character.

        Parameters
        ----------
        ch
            Character to add.
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

        Parameters
        ----------
        base_w :
            Base window width.
        base_h :
            Base window height.
        base_w: int :

        base_h: int :


        Returns
        -------
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
        """Return the size and the name that the user typed.

        An empty field, or a field that holds no number, gives the default
        value of that field.

        Returns
        -------
        tuple[int, int, str]
            Width, height, and name of the new level.
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
    """Dialog that asks for one number.

    The dialog takes a digit and a backspace. Enter accepts the value, and
    Escape leaves the dialog.

    Attributes
    ----------
    label
        Text that the dialog shows in front of the field.
    text
        Field, as the text that the user typed.
    min_value, max_value
        Lowest and highest value that :meth:`get_value` returns.
    default
        Value that :meth:`get_value` returns for an empty field.
    """

    label: str = "Value:"
    text: str = ""
    min_value: int = 0
    max_value: int = 100
    default: int = 0

    def handle_event(self, event: pygame.event.Event) -> str | None:
        """Process a pygame event.

        Parameters
        ----------
        event :
            A ``pygame.KEYDOWN`` event.
        event: pygame.event.Event :


        Returns
        -------
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
        """Return the number that the user typed.

        An empty field gives ``default``. A value outside the limits moves to
        the nearest limit.

        Returns
        -------
        int
            A value from ``min_value`` to ``max_value``.
        """
        try:
            val = int(self.text)
        except ValueError:
            val = self.default
        return max(self.min_value, min(self.max_value, val))

    def render(self, base_w: int, base_h: int) -> np.ndarray:
        """Render the dialog as an RGBA overlay.

        Parameters
        ----------
        base_w :
            Base window width.
        base_h :
            Base window height.
        base_w: int :

        base_h: int :


        Returns
        -------
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
    """Dialog that asks for a file name, to save a level or to open one.

    The dialog holds a field for the name, and a list of the ``.json`` files
    of the ``levels/`` directory. W and S move through the list. Enter accepts
    the name, and Escape leaves the dialog.

    Attributes
    ----------
    mode
        ``"save"`` or ``"load"``. It sets the title and the button text.
    filename_text
        Name field, as the text that the user typed.
    files
        Names of the files in the ``levels/`` directory.
    selected_index
        Index of the file that the list shows as selected.
    scroll_offset
        Number of files that the list moved past the top edge.
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

        Parameters
        ----------
        event :
            A ``pygame.KEYDOWN`` event.
        event: pygame.event.Event :


        Returns
        -------
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
        """Move the selection through the file list.

        The selection stops at the first and the last file. The list then
        moves, so the selected file stays in view. The name field takes the
        name of the selected file.

        Parameters
        ----------
        delta
            Number of rows to move. A positive value moves down.
        """
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
        """Return the path of the file that the user selected.

        The function adds the ``.json`` suffix to a name without one. If the
        ``levels/`` directory is absent, the function makes it.

        Returns
        -------
        Path or None
            Path in the ``levels/`` directory. It is ``None`` when the name
            field is empty.
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

        Parameters
        ----------
        base_w :
            Base window width.
        base_h :
            Base window height.
        base_w: int :

        base_h: int :


        Returns
        -------
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
            self.filename_text + "_",
            font,
            _TEXT_COLOR,
        )
        _blit_rgba(overlay, val, fy + 3, dx + 16)

        # File list
        list_y = fy + 30
        list_label = _render_text_rgba("Existing files:", small, _LABEL_COLOR)
        _blit_rgba(overlay, list_label, list_y, dx + 12)
        list_y += 16
        row_h = 18
        visible = self.files[self.scroll_offset : self.scroll_offset + _FILE_LIST_ROWS]
        for i, fname in enumerate(visible):
            abs_idx = self.scroll_offset + i
            ry = list_y + i * row_h
            bg = _FILE_SELECTED_BG if abs_idx == self.selected_index else _FILE_ROW_BG
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
            overlay,
            hint,
            dy + h - 16,
            dx + (w - hint.shape[1]) // 2,
        )
        return overlay


def _list_level_files() -> list[str]:
    """Return the name of each level in the ``levels/`` directory.

    Returns
    -------
    list[str]
        File name of each level, without the ``.json`` suffix, in alphabetic
        order. An absent directory gives an empty list.
    """
    if not LEVELS_DIR.is_dir():
        return []
    return sorted(p.stem for p in LEVELS_DIR.glob("*.json"))


# ---------------------------------------------------------------------------
# Item names and valid items per slot role
# ---------------------------------------------------------------------------

# Names that differ from the title-cased enum name, because a dialog row is
# narrow and the full name does not fit.
_ITEM_NAME_OVERRIDES: dict[int, str] = {
    int(ItemType.EMPTY): "(empty)",
    int(ItemType.CONVEYOR_BELT): "Belt",
    int(ItemType.TIER1_SCIENCE_PACK): "T1 Sci",
    int(ItemType.TIER2_SCIENCE_PACK): "T2 Sci",
    int(ItemType.TIER3_SCIENCE_PACK): "T3 Sci",
}

#: Name that a dialog shows for each item. The enum supplies the entries, so a
#: new item gets a name with no edit here.
_ITEM_NAMES: dict[int, str] = {
    int(it): _ITEM_NAME_OVERRIDES.get(int(it), it.name.replace("_", " ").title())
    for it in ItemType
}

# Items that each slot can hold. A machine writes its own output, so the
# editor offers the same items for every part, and lets the user fill any of
# them.
_ALL_ITEMS: list[int] = [int(it) for it in ItemType if it != ItemType.EMPTY]


def _valid_items_for_role(role: int) -> list[int]:
    """Return the list of item types valid for a given slot role.

    Parameters
    ----------
    role :
        SlotRole
    role: int :


    Returns
    -------
    type
        List of ``ItemType`` integer values (always includes EMPTY
        as the first entry for clearing).

    """
    if role == int(SlotRole.NONE):
        return []
    if role == int(SlotRole.FUEL):
        return [int(ItemType.EMPTY), int(ItemType.COAL)]
    return [int(ItemType.EMPTY)] + _ALL_ITEMS


# ---------------------------------------------------------------------------
# Machine inspector dialog
# ---------------------------------------------------------------------------

_INSP_W = 320
_INSP_SLOT_H = 24
_INSP_PICKER_ITEM_H = 18


@dataclasses.dataclass
class MachineInspectorDialog:
    """Dialog that shows the contents of one machine, and edits them.

    The dialog holds one row for each slot. A row gives the part that the slot
    has in the recipe, the item, and the count. A and D move through the rows.
    Enter opens a picker for the item, and then takes a count.

    The dialog edits its own copy of the rows. When the dialog closes,
    ``factoriax.playground.editor.main`` writes that copy back. The dialog
    therefore does not reach into the editor state.

    Attributes
    ----------
    tile_x, tile_y
        Tile of the machine.
    machine_type
        :class:`~factoriax.engine.constants.Machine` value of the machine.
    inv_items, inv_counts
        Item and count of each row.
    focused_slot
        Index of the row that A and D move.
    editing_slot
        Index of the row that the picker is open for. It is -1 while the
        picker is closed.
    editing_count
        ``True`` while the dialog takes a count for ``editing_slot``.
    count_text
        Count field, as the text that the user typed.
    picker_scroll
        Number of items that the picker moved past its top edge.
    """

    tile_x: int
    tile_y: int
    machine_type: int
    inv_items: np.ndarray
    inv_counts: np.ndarray
    focused_slot: int = 0
    editing_slot: int = -1
    editing_count: bool = False
    count_text: str = ""
    picker_scroll: int = 0

    @property
    def num_slots(self) -> int:
        """Number of active slots for this machine type."""
        return int(MACHINE_NUM_SLOTS[self.machine_type])

    @property
    def max_count(self) -> int:
        """Per-slot count cap for this machine type (its buffer capacity)."""
        return int(MACHINE_MAX_STACK[self.machine_type])

    def handle_event(self, event: pygame.event.Event) -> str | None:
        """Process a pygame event.

        Parameters
        ----------
        event :
            A pygame event (KEYDOWN or MOUSEBUTTONDOWN).
        event: pygame.event.Event :


        Returns
        -------
            ``"close"`` to dismiss, or ``None`` to stay open.

        """
        if event.type == pygame.KEYDOWN:
            return self._handle_key(event)
        return None

    def _handle_key(self, event: pygame.event.Event) -> str | None:
        """Handle keyboard input for the inspector dialog.

        Parameters
        ----------
        event :
            A pygame KEYDOWN event.
        event: pygame.event.Event :


        Returns
        -------
            ``"close"`` to dismiss, or ``None`` to stay open.

        """
        key = event.key

        if self.editing_count:
            return self._handle_count_key(key, event)

        if self.editing_slot >= 0:
            return self._handle_picker_key(key)

        if key == pygame.K_ESCAPE:
            return "close"

        if key == pygame.K_a or key == pygame.K_LEFT:
            self.focused_slot = max(0, self.focused_slot - 1)
        elif key == pygame.K_d or key == pygame.K_RIGHT:
            self.focused_slot = min(self.num_slots - 1, self.focused_slot + 1)
        elif key == pygame.K_RETURN or key == pygame.K_e:
            if self.num_slots > 0:
                role = int(MACHINE_SLOT_ROLES[self.machine_type, self.focused_slot])
                if role != int(SlotRole.NONE):
                    self.editing_slot = self.focused_slot
                    self.picker_scroll = 0
        elif key == pygame.K_c:
            if self.num_slots > 0:
                self.editing_count = True
                self.editing_slot = self.focused_slot
                current = int(self.inv_counts[self.focused_slot])
                self.count_text = str(current) if current > 0 else ""
        return None

    def _handle_picker_key(self, key: int) -> str | None:
        """Handle keys while the item picker is open.

        Parameters
        ----------
        key :
            Pygame key constant.
        key: int :


        Returns
        -------
            ``None`` (picker stays within the dialog).

        """
        role = int(MACHINE_SLOT_ROLES[self.machine_type, self.editing_slot])
        valid = _valid_items_for_role(role)
        if not valid:
            self.editing_slot = -1
            return None

        if key == pygame.K_ESCAPE:
            self.editing_slot = -1
        elif key == pygame.K_UP or key == pygame.K_w:
            self.picker_scroll = max(0, self.picker_scroll - 1)
        elif key == pygame.K_DOWN or key == pygame.K_s:
            self.picker_scroll = min(len(valid) - 1, self.picker_scroll + 1)
        elif key == pygame.K_RETURN:
            chosen = valid[self.picker_scroll]
            if chosen == int(ItemType.EMPTY):
                self.inv_items[self.editing_slot] = 0
                self.inv_counts[self.editing_slot] = 0
            else:
                self.inv_items[self.editing_slot] = chosen
                if int(self.inv_counts[self.editing_slot]) == 0:
                    self.inv_counts[self.editing_slot] = 1
            self.editing_slot = -1
        return None

    def _handle_count_key(self, key: int, event: pygame.event.Event) -> str | None:
        """Handle keys while editing a slot count.

        Parameters
        ----------
        key :
            Pygame key constant.
        event :
            Full pygame event for unicode access.
        key: int :

        event: pygame.event.Event :


        Returns
        -------
            ``None``.

        """
        if key == pygame.K_ESCAPE:
            self.editing_count = False
            self.editing_slot = -1
        elif key == pygame.K_RETURN:
            try:
                val = int(self.count_text)
            except ValueError:
                val = 0
            val = max(0, min(self.max_count, val))
            self.inv_counts[self.editing_slot] = val
            if val == 0:
                self.inv_items[self.editing_slot] = 0
            self.editing_count = False
            self.editing_slot = -1
        elif key == pygame.K_BACKSPACE:
            self.count_text = self.count_text[:-1]
        elif event.unicode and event.unicode.isdigit():
            self.count_text += event.unicode
        return None

    def render(self, base_w: int, base_h: int) -> np.ndarray:
        """Render the inspector dialog as an RGBA overlay.

        Parameters
        ----------
        base_w :
            Base window width.
        base_h :
            Base window height.
        base_w: int :

        base_h: int :


        Returns
        -------
            RGBA uint8 array of shape ``(base_h, base_w, 4)``.

        """
        overlay = np.zeros((base_h, base_w, 4), dtype=np.uint8)
        overlay[:, :] = (0, 0, 0, 140)

        ns = self.num_slots
        dlg_h = 50 + ns * _INSP_SLOT_H + 24
        dlg_w = min(_INSP_W, base_w - 20)
        dx = (base_w - dlg_w) // 2
        dy = (base_h - dlg_h) // 2

        overlay[dy : dy + dlg_h, dx : dx + dlg_w] = _BG
        for i in range(2):
            overlay[dy + i, dx : dx + dlg_w] = _BORDER
            overlay[dy + dlg_h - 1 - i, dx : dx + dlg_w] = _BORDER
            overlay[dy : dy + dlg_h, dx + i] = _BORDER
            overlay[dy : dy + dlg_h, dx + dlg_w - 1 - i] = _BORDER

        font = get_pixel_font(14)
        small = get_pixel_font(10)

        machine_name = MACHINE_TYPE_NAMES.get(self.machine_type, "Machine")
        title = _render_text_rgba(machine_name, font, _TEXT_COLOR)
        _blit_rgba(overlay, title, dy + 8, dx + (dlg_w - title.shape[1]) // 2)

        y = dy + 30

        slot_roles = MACHINE_SLOT_ROLES[self.machine_type]
        for slot_idx in range(ns):
            sy = y + slot_idx * _INSP_SLOT_H
            role = int(slot_roles[slot_idx])
            is_focused = slot_idx == self.focused_slot

            row_bg = (70, 70, 70, 255) if is_focused else (45, 45, 45, 255)
            overlay[sy : sy + _INSP_SLOT_H - 1, dx + 8 : dx + dlg_w - 8] = row_bg

            role_color = SLOT_ROLE_COLORS.get(role, (60, 60, 60))
            role_label = SLOT_ROLE_LABELS.get(role, "")
            overlay[sy : sy + _INSP_SLOT_H - 1, dx + 8 : dx + 42] = (
                *role_color,
                220,
            )
            if role_label:
                rl = _render_text_rgba(role_label, small, (230, 230, 230))
                _blit_rgba(overlay, rl, sy + 3, dx + 10)

            item_type = int(self.inv_items[slot_idx])
            count = int(self.inv_counts[slot_idx])

            if item_type != 0 and count > 0:
                icon_s = _INSP_SLOT_H - 6
                icon = render_item_icon(item_type, max(4, icon_s))
                _blit_rgba(overlay, icon, sy + 3, dx + 46)

                name = _ITEM_NAMES.get(item_type, "?")
                ntxt = _render_text_rgba(f"{name} x{count}", font, _TEXT_COLOR)
                _blit_rgba(overlay, ntxt, sy + 4, dx + 46 + icon_s + 4)
            else:
                etxt = _render_text_rgba("(empty)", small, _LABEL_COLOR)
                _blit_rgba(overlay, etxt, sy + 5, dx + 46)

        hint_y = dy + dlg_h - 20
        hint_parts = ["A/D: slot", "Enter: item", "C: count", "Esc: close"]
        hint_str = "  ".join(hint_parts)
        hint = _render_text_rgba(hint_str, small, _LABEL_COLOR)
        _blit_rgba(overlay, hint, hint_y, dx + (dlg_w - hint.shape[1]) // 2)

        if self.editing_slot >= 0 and not self.editing_count:
            self._render_item_picker(overlay, base_w, base_h)
        elif self.editing_count:
            self._render_count_editor(overlay, base_w, base_h)

        return overlay

    def _render_item_picker(
        self, overlay: np.ndarray, base_w: int, base_h: int
    ) -> None:
        """Draw the item picker over the dialog.

        The picker lists the items that the focused slot can hold. A slot that
        holds no item draws no picker.

        Parameters
        ----------
        overlay
            RGBA overlay. The function writes to it.
        base_w
            Window width in pixels.
        base_h
            Window height in pixels.
        """
        role = int(MACHINE_SLOT_ROLES[self.machine_type, self.editing_slot])
        valid = _valid_items_for_role(role)
        if not valid:
            return

        font = get_pixel_font(12)
        small = get_pixel_font(10)
        max_visible = 8
        pw = 160
        ph = 24 + min(len(valid), max_visible) * _INSP_PICKER_ITEM_H + 20
        px = (base_w - pw) // 2
        py = (base_h - ph) // 2

        overlay[py : py + ph, px : px + pw] = (30, 30, 30, 240)
        for i in range(2):
            overlay[py + i, px : px + pw] = _BORDER
            overlay[py + ph - 1 - i, px : px + pw] = _BORDER
            overlay[py : py + ph, px + i] = _BORDER
            overlay[py : py + ph, px + pw - 1 - i] = _BORDER

        title = _render_text_rgba("Select Item", font, _TEXT_COLOR)
        _blit_rgba(overlay, title, py + 4, px + (pw - title.shape[1]) // 2)

        iy = py + 22
        vis_start = max(
            0, min(self.picker_scroll - max_visible // 2, len(valid) - max_visible)
        )
        vis_start = max(0, vis_start)
        vis_end = min(len(valid), vis_start + max_visible)

        for i in range(vis_start, vis_end):
            item_id = valid[i]
            row_y = iy + (i - vis_start) * _INSP_PICKER_ITEM_H
            is_sel = i == self.picker_scroll
            bg = (60, 80, 60, 255) if is_sel else (40, 40, 40, 255)
            overlay[
                row_y : row_y + _INSP_PICKER_ITEM_H - 1,
                px + 4 : px + pw - 4,
            ] = bg

            if item_id != 0:
                icon_s = _INSP_PICKER_ITEM_H - 4
                icon = render_item_icon(item_id, max(4, icon_s))
                _blit_rgba(overlay, icon, row_y + 2, px + 8)

            name = _ITEM_NAMES.get(item_id, "?")
            ntxt = _render_text_rgba(name, small, _TEXT_COLOR)
            _blit_rgba(overlay, ntxt, row_y + 3, px + 8 + _INSP_PICKER_ITEM_H)

        hint = _render_text_rgba(
            "Up/Down  Enter: select  Esc: cancel", small, _LABEL_COLOR
        )
        _blit_rgba(overlay, hint, py + ph - 16, px + (pw - hint.shape[1]) // 2)

    def _render_count_editor(
        self, overlay: np.ndarray, base_w: int, base_h: int
    ) -> None:
        """Draw the count field over the dialog.

        The field shows the highest count that the slot holds, so the user can
        see the limit of the machine.

        Parameters
        ----------
        overlay
            RGBA overlay. The function writes to it.
        base_w
            Window width in pixels.
        base_h
            Window height in pixels.
        """
        font = get_pixel_font(14)
        small = get_pixel_font(10)
        cw, ch = 160, 60
        cx = (base_w - cw) // 2
        cy = (base_h - ch) // 2

        overlay[cy : cy + ch, cx : cx + cw] = (30, 30, 30, 240)
        for i in range(2):
            overlay[cy + i, cx : cx + cw] = _BORDER
            overlay[cy + ch - 1 - i, cx : cx + cw] = _BORDER
            overlay[cy : cy + ch, cx + i] = _BORDER
            overlay[cy : cy + ch, cx + cw - 1 - i] = _BORDER

        lbl = _render_text_rgba(f"Count (0-{self.max_count}):", small, _LABEL_COLOR)
        _blit_rgba(overlay, lbl, cy + 6, cx + 8)

        fy = cy + 22
        overlay[fy : fy + 20, cx + 8 : cx + cw - 8] = _FIELD_ACTIVE
        val = _render_text_rgba(self.count_text + "_", font, _TEXT_COLOR)
        _blit_rgba(overlay, val, fy + 2, cx + 12)

        hint = _render_text_rgba("Enter: OK  Esc: cancel", small, _LABEL_COLOR)
        _blit_rgba(overlay, hint, cy + ch - 14, cx + (cw - hint.shape[1]) // 2)


_HELP_LINES = [
    "-- Drawing --",
    "Left click/drag  Paint tile or machine",
    "Right click/drag  Erase (layer-aware)",
    "X  Eraser tool (clears both layers)",
    "B  Paint tool    F  Fill rect tool",
    "1-5  Terrain: Dirt Water Iron Copper Coal",
    "6-0  Select machine from palette",
    "R  Rotate machine (or rotate under cursor)",
    "I  Inspect machine inventory",
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

    Parameters
    ----------
    base_w :
        Base window width.
    base_h :
        Base window height.
    base_w: int :

    base_h: int :


    Returns
    -------
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

    Parameters
    ----------
    text :
        String to render.
    font :
        Pygame font.
    color :
        RGB text colour.
    text: str :

    font: pygame.font.Font :

    color: tuple[int :

    int :

    int] :


    Returns
    -------
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
    """Draw one RGBA image on another, and mix the two by the alpha channel.

    The function draws the part of the source that falls on the destination.
    A source fully outside the destination has no effect.

    Parameters
    ----------
    overlay
        Destination RGBA array. The function writes to it.
    src
        Source RGBA array.
    y
        Top row in the destination.
    x
        Left column in the destination.
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
