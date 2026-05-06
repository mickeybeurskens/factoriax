"""Play settings screen for FactoriaX.

Presents all EnvParams fields organized in labeled sections with editable
input boxes. The user clicks a field to focus it, types a new value, and
presses Enter to confirm or Escape to revert. Back returns None, Play
returns the configured EnvParams.

All drawing happens on a fixed-size :class:`ScaledCanvas` that is
integer-scaled to the window, giving pixel-perfect layout at any window
size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pygame

from factoriax.config import (
    PlayerAction,
    PlayerConfig,
    build_controller_lookup,
    build_key_lookup,
    default_controller,
    default_keyboard,
    resolve_event,
)
from factoriax.state import EnvParams
from factoriax.ui import theme as _theme
from factoriax.ui.fonts import get_pixel_font
from factoriax.ui.scaling import ScaledCanvas
from factoriax.ui.window import auto_ui_scale, calculate_window_size

logger = logging.getLogger(__name__)

# -- Colors ---------------------------------------------------------------
_BG: tuple[int, int, int] = (20, 20, 25)
_GOLD: tuple[int, int, int] = (190, 165, 55)
_SECTION_RULE: tuple[int, int, int] = (80, 70, 30)
_LABEL_COLOR: tuple[int, int, int] = (200, 200, 200)
_LABEL_DISABLED: tuple[int, int, int] = (90, 90, 90)
_INPUT_BG: tuple[int, int, int] = (35, 35, 40)
_INPUT_BORDER: tuple[int, int, int] = (80, 80, 80)
_INPUT_TEXT: tuple[int, int, int] = (220, 220, 220)
_INPUT_TEXT_DISABLED: tuple[int, int, int] = (90, 90, 90)
_BTN_BG: tuple[int, int, int] = (30, 30, 35)
_BTN_HOVER: tuple[int, int, int] = (45, 45, 50)
_BTN_TEXT: tuple[int, int, int] = (220, 215, 180)
_ERROR_BORDER: tuple[int, int, int] = (200, 50, 50)

# -- Base layout constants (multiplied by UI_SCALE at runtime) ------------
_BASE_TOP_BAR_H: int = 80
_BASE_SECTION_PAD_TOP: int = 24
_BASE_SECTION_HEADER_H: int = 44
_BASE_SECTION_RULE_H: int = 2
_BASE_SECTION_GAP: int = 16
_BASE_ROW_H: int = 48
_BASE_ROW_GAP: int = 8
_BASE_COL_GAP: int = 24
_BASE_SIDE_PAD: int = 32
_BASE_INPUT_W: int = 140
_BASE_INPUT_H: int = 40
_BASE_CHECKBOX_SIZE: int = 30
_BASE_BTN_W: int = 160
_BASE_BTN_H: int = 56
_BASE_SCROLL_STEP: int = 24
_FPS: int = 30
_ERROR_FLASH_FRAMES: int = 15


# -- Field state ----------------------------------------------------------
@dataclass
class FieldState:
    """Editable parameter field.

    Attributes:
        name: EnvParams attribute name.
        label: Human-readable display label.
        value: Current string representation of the value.
        field_type: Either ``"int"`` or ``"float"``.
        editing: Whether the field is actively being edited.
        edit_buffer: Text typed so far while editing.
        error_timer: Frames remaining for the red-flash error indicator.
    """

    name: str
    label: str
    value: str
    field_type: str
    editing: bool = False
    edit_buffer: str = ""
    error_timer: int = 0


@dataclass
class _Section:
    """A group of fields under a shared heading.

    Attributes:
        title: Section heading text.
        fields: Fields belonging to this section, laid out in 2-column
            pairs (left column at even indices, right column at odd).
    """

    title: str
    fields: list[FieldState] = field(default_factory=list)


# -- Helpers --------------------------------------------------------------


def _format_value(value: int | float, field_type: str) -> str:
    """Format a numeric value as a string suitable for display.

    Args:
        value: The raw numeric value.
        field_type: ``"int"`` or ``"float"``.

    Returns:
        Formatted string (floats get two decimal places).
    """
    if field_type == "float":
        return f"{value:.2f}"
    return str(int(value))


def _parse_value(text: str, field_type: str) -> int | float | None:
    """Parse a string into the appropriate numeric type.

    Args:
        text: User-entered string.
        field_type: ``"int"`` or ``"float"``.

    Returns:
        Parsed value, or ``None`` on failure.
    """
    try:
        if field_type == "int":
            return int(text)
        return float(text)
    except ValueError:
        return None


def _build_sections(params: EnvParams) -> list[_Section]:
    """Construct the section/field layout from an EnvParams instance.

    Args:
        params: The starting parameter values.

    Returns:
        Ordered list of sections, each containing its fields.
    """

    def _f(
        name: str,
        label: str,
        field_type: str,
    ) -> FieldState:
        raw = getattr(params, name)
        return FieldState(
            name=name,
            label=label,
            value=_format_value(raw, field_type),
            field_type=field_type,
        )

    world = _Section(
        "World",
        [
            _f("map_width", "Map Width", "int"),
            _f("map_height", "Map Height", "int"),
            _f("num_players", "Num Players", "int"),
            _f("max_timesteps", "Max Steps", "int"),
        ],
    )
    resources = _Section(
        "Resources",
        [
            _f("water_probability", "Water Prob", "float"),
            _f("base_resources", "Base Resources", "int"),
            _f("iron_probability", "Iron Prob", "float"),
            _f("copper_probability", "Copper Prob", "float"),
            _f("coal_probability", "Coal Prob", "float"),
            _f("tin_probability", "Tin Prob", "float"),
            _f("silicon_probability", "Silicon Prob", "float"),
        ],
    )
    machines = _Section(
        "Machines",
        [
            _f("max_machines", "Max Machines", "int"),
            _f("miner_mining_rate", "Mining Rate", "int"),
            _f("max_assembler_stack_size", "Assembler Stack", "int"),
        ],
    )
    return [world, resources, machines]


def _field_index_in_sections(
    sections: list[_Section],
    target: FieldState,
) -> int:
    """Return a flat index for *target* across all sections.

    Args:
        sections: The section list.
        target: The field to locate.

    Returns:
        Zero-based flat index.
    """
    idx = 0
    for sec in sections:
        for fs in sec.fields:
            if fs is target:
                return idx
            idx += 1
    return -1


def _flat_fields(sections: list[_Section]) -> list[FieldState]:
    """Return all fields across sections in layout order.

    Args:
        sections: The section list.

    Returns:
        Flat list of every FieldState.
    """
    return [fs for sec in sections for fs in sec.fields]


def _focus_field(
    sections: list[_Section],
    target: FieldState | None,
) -> None:
    """Set *target* as the only focused field, unfocusing all others.

    Args:
        sections: The section list (modified in place).
        target: The field to focus, or ``None`` to clear focus.
    """
    for sec in sections:
        for fs in sec.fields:
            if fs is target:
                fs.editing = True
                fs.edit_buffer = fs.value
            else:
                fs.editing = False
                fs.edit_buffer = ""


def _confirm_edit(fs: FieldState) -> None:
    """Validate and apply the edit buffer, or flash red on failure.

    Args:
        fs: The field being confirmed (modified in place).
    """
    parsed = _parse_value(fs.edit_buffer, fs.field_type)
    if parsed is not None:
        fs.value = _format_value(parsed, fs.field_type)
    else:
        fs.error_timer = _ERROR_FLASH_FRAMES
    fs.editing = False
    fs.edit_buffer = ""


def _cancel_edit(fs: FieldState) -> None:
    """Discard the edit buffer without applying changes.

    Args:
        fs: The field being cancelled (modified in place).
    """
    fs.editing = False
    fs.edit_buffer = ""


def _build_params(sections: list[_Section]) -> EnvParams:
    """Construct an EnvParams from the current field string values.

    Invalid strings fall back to the EnvParams default for that field.

    Args:
        sections: The populated section list.

    Returns:
        A new EnvParams instance.
    """
    defaults = EnvParams()
    kwargs: dict[str, int | float] = {}
    for sec in sections:
        for fs in sec.fields:
            parsed = _parse_value(fs.value, fs.field_type)
            if parsed is not None:
                kwargs[fs.name] = parsed
            else:
                kwargs[fs.name] = getattr(defaults, fs.name)

    return EnvParams(**kwargs)  # type: ignore[arg-type]


# -- Drawing helpers ------------------------------------------------------


def _draw_button(
    surface: pygame.Surface,
    rect: pygame.Rect,
    text: str,
    font: pygame.font.Font,
    hovered: bool,
) -> None:
    """Draw a button with gold border, optional hover highlight.

    Args:
        surface: Destination surface.
        rect: Button bounding rectangle.
        text: Button label.
        font: Font for the label.
        hovered: Whether the mouse is over the button.
    """
    bg = _BTN_HOVER if hovered else _BTN_BG
    pygame.draw.rect(surface, bg, rect)
    pygame.draw.rect(surface, _GOLD, rect, 2)
    txt_surf = font.render(text, False, _BTN_TEXT)
    tx = rect.x + (rect.width - txt_surf.get_width()) // 2
    ty = rect.y + (rect.height - txt_surf.get_height()) // 2
    surface.blit(txt_surf, (tx, ty))


def _draw_checkbox(
    surface: pygame.Surface,
    x: int,
    y: int,
    size: int,
    checked: bool,
    hovered: bool,
) -> pygame.Rect:
    """Draw a checkbox square and return its bounding rect.

    Args:
        surface: Destination surface.
        x: Left edge in pixels.
        y: Top edge in pixels.
        size: Side length of the checkbox in pixels.
        checked: Whether the box is checked.
        hovered: Whether the mouse is over the checkbox area.

    Returns:
        The bounding rectangle of the checkbox.
    """
    rect = pygame.Rect(x, y, size, size)
    bg = _BTN_HOVER if hovered else _INPUT_BG
    pygame.draw.rect(surface, bg, rect)
    pygame.draw.rect(surface, _GOLD, rect, 1)
    if checked:
        font = get_pixel_font(24 * _theme.UI_SCALE)
        x_surf = font.render("X", False, _GOLD)
        cx = rect.x + (rect.width - x_surf.get_width()) // 2
        cy = rect.y + (rect.height - x_surf.get_height()) // 2
        surface.blit(x_surf, (cx, cy))
    return rect


# -- Controls display helper ----------------------------------------------


_REBIND_ACTIONS: list[tuple[str | None, list[tuple[str, str]]]] = []
"""Grouped actions for the rebinding UI: ``(category, [(action, label)])``."""


def _init_rebind_actions() -> list[tuple[str | None, list[tuple[str, str]]]]:
    """Build the grouped action list on first access.

    Returns:
        List of ``(category_label, [(action_key, display_label)])``.
    """
    from factoriax.config import PlayerAction

    return [
        (
            "Movement",
            [
                (PlayerAction.MOVE_UP, "Move Up"),
                (PlayerAction.MOVE_DOWN, "Move Down"),
                (PlayerAction.MOVE_LEFT, "Move Left"),
                (PlayerAction.MOVE_RIGHT, "Move Right"),
            ],
        ),
        (
            "Actions",
            [
                (PlayerAction.MINE, "Mine"),
                (PlayerAction.INTERACT, "Interact"),
                (PlayerAction.ROTATE, "Rotate"),
            ],
        ),
        (
            "Menus",
            [
                (PlayerAction.OPEN_INVENTORY, "Inventory"),
                (PlayerAction.OPEN_ACHIEVEMENTS, "Achievements"),
                (PlayerAction.OPEN_MACHINE, "Inspect Machine"),
                (PlayerAction.OPEN_HELP, "Help"),
                (PlayerAction.TOGGLE_HOTBAR, "Hotbar Page"),
                (PlayerAction.CONFIRM, "Confirm"),
            ],
        ),
    ]


# Friendly display names for controller input strings.
_CONTROLLER_DISPLAY: dict[str, str] = {
    "BUTTON_0": "A / Cross",
    "BUTTON_1": "B / Circle",
    "BUTTON_2": "X / Square",
    "BUTTON_3": "Y / Triangle",
    "BUTTON_4": "LB",
    "BUTTON_5": "RB",
    "BUTTON_6": "Select",
    "BUTTON_7": "Start",
    "BUTTON_8": "L3",
    "BUTTON_9": "R3",
    "AXIS_0_NEG": "L-Stick Left",
    "AXIS_0_POS": "L-Stick Right",
    "AXIS_1_NEG": "L-Stick Up",
    "AXIS_1_POS": "L-Stick Down",
    "HAT_0_UP": "D-pad Up",
    "HAT_0_DOWN": "D-pad Down",
    "HAT_0_LEFT": "D-pad Left",
    "HAT_0_RIGHT": "D-pad Right",
}


def _format_key_display(names: list[str]) -> str:
    """Format keyboard binding names for display.

    Args:
        names: Raw binding names (e.g. ``["K_w", "K_UP"]``).

    Returns:
        Human-readable string (e.g. ``"w, UP"``).
    """
    if not names:
        return "-"
    parts: list[str] = []
    for n in names:
        # "SHIFT+K_1" -> "Shift+1", "K_w" -> "w", "K_UP" -> "UP"
        pieces = n.split("+")
        formatted = []
        for p in pieces:
            if p.startswith("K_"):
                formatted.append(p[2:])
            else:
                formatted.append(p.capitalize())
        parts.append("+".join(formatted))
    return ", ".join(parts)


def _format_controller_display(names: list[str]) -> str:
    """Format controller binding names for display.

    Args:
        names: Raw binding names (e.g. ``["BUTTON_0"]``).

    Returns:
        Human-readable string (e.g. ``"A / Cross"``).
    """
    if not names:
        return "-"
    return ", ".join(_CONTROLLER_DISPLAY.get(n, n) for n in names)


def _format_binding(names: list[str], device: str) -> str:
    """Format binding names for the active device tab.

    Args:
        names: Raw binding name list from config.
        device: ``"keyboard"`` or ``"controller"``.

    Returns:
        Human-readable display string.
    """
    if device == "keyboard":
        return _format_key_display(names)
    return _format_controller_display(names)


@dataclass
class _RebindState:
    """Tracks which tab is active and whether we're listening for input."""

    active_tab: str = "keyboard"
    listening_action: str | None = None
    focused_row: int = 0


# -- Scrollbar helper -----------------------------------------------------


def _draw_scrollbar(
    surface: pygame.Surface,
    scroll_offset: int,
    max_scroll: int,
    content_h: int,
    top_bar_h: int,
) -> None:
    """Draw a vertical scrollbar on the right edge when content overflows.

    Args:
        surface: Target surface.
        scroll_offset: Current scroll position.
        max_scroll: Maximum scroll offset.
        content_h: Total content height.
        top_bar_h: Height of the top bar (scrollbar starts below it).
    """
    if max_scroll <= 0:
        return
    sh = surface.get_height()
    bar_area_h = sh - top_bar_h
    thumb_h = max(20, int(bar_area_h * bar_area_h / content_h))
    thumb_y = top_bar_h + int(scroll_offset / max_scroll * (bar_area_h - thumb_h))
    bar_x = surface.get_width() - 8
    pygame.draw.rect(
        surface,
        (40, 40, 40),
        pygame.Rect(bar_x, top_bar_h, 8, bar_area_h),
    )
    pygame.draw.rect(
        surface,
        (140, 130, 80),
        pygame.Rect(bar_x, thumb_y, 8, thumb_h),
    )


# -- Public menus ---------------------------------------------------------


def run_settings_menu(
    screen: pygame.Surface,
    initial_params: EnvParams | None = None,
) -> EnvParams | None:
    """Show play settings and return configured EnvParams, or None to go back.

    Presents all environment parameters organized in sections. Each field
    can be clicked and edited. The Back button (or closing the window)
    returns None. The Play button builds an EnvParams from the current
    field values and returns it.

    Args:
        screen: Pygame display surface.
        initial_params: Starting parameter values. Defaults to
            ``EnvParams()`` when ``None``.

    Returns:
        EnvParams with user's choices if Play was clicked,
        None if Back was clicked or window closed.
    """
    s = _theme.UI_SCALE
    canvas = ScaledCanvas(1024, s, screen)
    sw, sh = canvas.width, canvas.height

    # Scaled layout constants.
    top_bar_h = _BASE_TOP_BAR_H * s
    section_pad_top = _BASE_SECTION_PAD_TOP * s
    section_header_h = _BASE_SECTION_HEADER_H * s
    section_rule_h = _BASE_SECTION_RULE_H * s
    section_gap = _BASE_SECTION_GAP * s
    row_h = _BASE_ROW_H * s
    row_gap = _BASE_ROW_GAP * s
    col_gap = _BASE_COL_GAP * s
    side_pad = _BASE_SIDE_PAD * s
    input_w = _BASE_INPUT_W * s
    input_h = _BASE_INPUT_H * s
    btn_w = _BASE_BTN_W * s
    btn_h = _BASE_BTN_H * s
    scroll_step = _BASE_SCROLL_STEP * s

    # Derived layout (constant because canvas size is fixed).
    content_w = sw - 2 * side_pad
    col_w = (content_w - col_gap) // 2
    label_w = col_w - input_w - 8

    back_rect = pygame.Rect(side_pad, 8 * s, btn_w, btn_h)
    play_rect = pygame.Rect(sw - side_pad - btn_w, 8 * s, btn_w, btn_h)

    clock = pygame.time.Clock()
    params = initial_params if initial_params is not None else EnvParams()
    sections = _build_sections(params)

    scroll_offset = 0
    font_header = get_pixel_font(32 * s)
    font_label = get_pixel_font(28 * s)
    font_input = get_pixel_font(28 * s)
    font_btn = get_pixel_font(28 * s)
    font_section = get_pixel_font(32 * s)

    # Measure total content height (constant).
    content_h = section_pad_top
    for sec in sections:
        content_h += section_header_h + section_rule_h + section_gap
        num_rows = (len(sec.fields) + 1) // 2
        content_h += num_rows * (row_h + row_gap)
        content_h += section_pad_top

    while True:
        scrollable_area = sh - top_bar_h
        max_scroll = max(0, content_h - scrollable_area)
        scroll_offset = max(0, min(scroll_offset, max_scroll))

        # -- Event handling -----------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

            if event.type == pygame.VIDEORESIZE:
                canvas.handle_resize(event.w, event.h)

            if event.type == pygame.MOUSEWHEEL:
                scroll_offset -= event.y * scroll_step
                scroll_offset = max(0, min(scroll_offset, max_scroll))

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = canvas.to_canvas(*event.pos)

                if back_rect.collidepoint(mx, my):
                    return None
                if play_rect.collidepoint(mx, my):
                    for fs in _flat_fields(sections):
                        if fs.editing:
                            _confirm_edit(fs)
                    return _build_params(sections)

                # Content area clicks (adjusted for scroll).
                cy = top_bar_h + section_pad_top - scroll_offset
                clicked_field: FieldState | None = None

                for sec in sections:
                    cy += section_header_h + section_rule_h + section_gap

                    for i in range(0, len(sec.fields), 2):
                        for col_idx in range(2):
                            fi = i + col_idx
                            if fi >= len(sec.fields):
                                break
                            fx = side_pad + col_idx * (col_w + col_gap) + label_w + 8
                            fy = cy + (row_h - input_h) // 2
                            hit = pygame.Rect(fx, fy, input_w, input_h)
                            if hit.collidepoint(mx, my):
                                clicked_field = sec.fields[fi]
                        cy += row_h + row_gap

                    cy += section_pad_top

                if clicked_field is not None:
                    for fs in _flat_fields(sections):
                        if fs.editing and fs is not clicked_field:
                            _confirm_edit(fs)
                    _focus_field(sections, clicked_field)
                else:
                    for fs in _flat_fields(sections):
                        if fs.editing:
                            _confirm_edit(fs)

            if event.type == pygame.KEYDOWN:
                active = next(
                    (fs for fs in _flat_fields(sections) if fs.editing),
                    None,
                )
                if active is not None:
                    if event.key == pygame.K_RETURN:
                        _confirm_edit(active)
                    elif event.key == pygame.K_ESCAPE:
                        _cancel_edit(active)
                    elif event.key == pygame.K_BACKSPACE:
                        active.edit_buffer = active.edit_buffer[:-1]
                    elif event.key == pygame.K_TAB:
                        _confirm_edit(active)
                        flat = _flat_fields(sections)
                        idx = _field_index_in_sections(sections, active)
                        mods = pygame.key.get_mods()
                        if mods & pygame.KMOD_SHIFT:
                            next_idx = (idx - 1) % len(flat)
                        else:
                            next_idx = (idx + 1) % len(flat)
                        _focus_field(sections, flat[next_idx])
                    elif event.key in (pygame.K_UP, pygame.K_DOWN):
                        _confirm_edit(active)
                        flat = _flat_fields(sections)
                        idx = _field_index_in_sections(sections, active)
                        if event.key == pygame.K_UP:
                            next_idx = (idx - 1) % len(flat)
                        else:
                            next_idx = (idx + 1) % len(flat)
                        _focus_field(sections, flat[next_idx])
                    elif event.unicode and event.unicode.isprintable():
                        active.edit_buffer += event.unicode
                else:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    if event.key == pygame.K_RETURN:
                        return _build_params(sections)
                    if event.key == pygame.K_TAB:
                        flat = _flat_fields(sections)
                        if flat:
                            _focus_field(sections, flat[0])

            # Controller/keyboard navigation via configured bindings.
            if event.type in (
                pygame.JOYHATMOTION,
                pygame.JOYBUTTONDOWN,
            ):
                _kb = build_key_lookup(default_keyboard())
                _cl = build_controller_lookup(default_controller())
                nav = resolve_event(event, _kb, _cl)
                active = next(
                    (fs for fs in _flat_fields(sections) if fs.editing),
                    None,
                )
                if PlayerAction.NAV_DOWN in nav:
                    if active is not None:
                        _confirm_edit(active)
                    flat = _flat_fields(sections)
                    if flat:
                        cur = next(
                            (i for i, f in enumerate(flat) if f.editing),
                            -1,
                        )
                        _focus_field(
                            sections,
                            flat[(cur + 1) % len(flat)],
                        )
                elif PlayerAction.NAV_UP in nav:
                    if active is not None:
                        _confirm_edit(active)
                    flat = _flat_fields(sections)
                    if flat:
                        cur = next(
                            (i for i, f in enumerate(flat) if f.editing),
                            -1,
                        )
                        _focus_field(
                            sections,
                            flat[(cur - 1) % len(flat)],
                        )
                elif PlayerAction.CONFIRM in nav:
                    if active is not None:
                        _confirm_edit(active)
                    else:
                        # No field editing → Play.
                        for fs in _flat_fields(sections):
                            if fs.editing:
                                _confirm_edit(fs)
                        return _build_params(sections)
                elif PlayerAction.BACK in nav:
                    if active is not None:
                        _cancel_edit(active)
                    else:
                        return None

        # -- Drawing ------------------------------------------------------
        surf = canvas.surface
        mouse_pos = canvas.to_canvas(*pygame.mouse.get_pos())
        surf.fill(_BG)

        # Top bar.
        _draw_button(
            surf,
            back_rect,
            "< Back",
            font_btn,
            back_rect.collidepoint(mouse_pos),
        )
        _draw_button(
            surf,
            play_rect,
            "> Play",
            font_btn,
            play_rect.collidepoint(mouse_pos),
        )

        # Title.
        title_surf = font_header.render("Play Settings", False, _GOLD)
        title_x = (sw - title_surf.get_width()) // 2
        title_y = 8 * s + (btn_h - title_surf.get_height()) // 2
        surf.blit(title_surf, (title_x, title_y))

        # Clip content area below top bar.
        content_clip = pygame.Rect(0, top_bar_h, sw, sh - top_bar_h)
        surf.set_clip(content_clip)

        cy = top_bar_h + section_pad_top - scroll_offset

        for sec in sections:
            sec_surf = font_section.render(sec.title, False, _GOLD)
            surf.blit(sec_surf, (side_pad, cy))
            cy += section_header_h
            pygame.draw.line(
                surf,
                _SECTION_RULE,
                (side_pad, cy),
                (sw - side_pad, cy),
            )
            cy += section_rule_h + section_gap

            for i in range(0, len(sec.fields), 2):
                for col_idx in range(2):
                    fi = i + col_idx
                    if fi >= len(sec.fields):
                        break

                    fs = sec.fields[fi]
                    disabled = False
                    col_x = side_pad + col_idx * (col_w + col_gap)

                    lbl_color = _LABEL_DISABLED if disabled else _LABEL_COLOR
                    lbl_surf = font_label.render(fs.label, False, lbl_color)
                    lbl_y = cy + (row_h - lbl_surf.get_height()) // 2
                    surf.blit(lbl_surf, (col_x, lbl_y))

                    fx = col_x + label_w + 8
                    fy = cy + (row_h - input_h) // 2
                    input_rect = pygame.Rect(fx, fy, input_w, input_h)
                    pygame.draw.rect(surf, _INPUT_BG, input_rect)

                    if fs.error_timer > 0:
                        border_color = _ERROR_BORDER
                        fs.error_timer -= 1
                    elif fs.editing:
                        border_color = _GOLD
                    else:
                        border_color = _INPUT_BORDER

                    if disabled:
                        border_color = _INPUT_BORDER

                    pygame.draw.rect(surf, border_color, input_rect, 1)

                    if fs.editing and not disabled:
                        display_text = fs.edit_buffer + "_"
                    else:
                        display_text = fs.value
                    txt_color = _INPUT_TEXT_DISABLED if disabled else _INPUT_TEXT
                    txt_surf = font_input.render(display_text, False, txt_color)
                    max_txt_w = input_w - 8
                    if txt_surf.get_width() > max_txt_w:
                        txt_surf = txt_surf.subsurface(
                            txt_surf.get_width() - max_txt_w,
                            0,
                            max_txt_w,
                            txt_surf.get_height(),
                        )
                    surf.blit(
                        txt_surf,
                        (
                            fx + 4,
                            fy + (input_h - txt_surf.get_height()) // 2,
                        ),
                    )

                cy += row_h + row_gap

            cy += section_pad_top

        surf.set_clip(None)
        _draw_scrollbar(surf, scroll_offset, max_scroll, content_h, top_bar_h)

        canvas.present(screen)
        clock.tick(_FPS)


_CONFIRM_TIMEOUT_MS: int = 10_000


def _confirm_scale_change(
    screen: pygame.Surface,
    new_scale_value: int,
) -> bool:
    """Apply a new UI scale and wait for user confirmation.

    Shows a centered dialog with a countdown timer. The new scale is
    applied immediately so the user can see the result. If the user
    presses Enter or clicks "Keep", the change is confirmed. If the
    timer expires or Escape is pressed, the change is reverted.

    Args:
        screen: Pygame display surface.
        new_scale_value: The ``ui_scale`` config value to try (0-3).

    Returns:
        ``True`` if the user confirmed the new scale, ``False`` if
        reverted.
    """
    new_s = new_scale_value if new_scale_value > 0 else auto_ui_scale()
    _theme.apply_scale(new_s)
    canvas_size = 1024 * new_s
    w, h = calculate_window_size(canvas_size, canvas_size)
    screen = pygame.display.set_mode((w, h))
    canvas = ScaledCanvas(1024, new_s, screen)
    clock = pygame.time.Clock()
    deadline = pygame.time.get_ticks() + _CONFIRM_TIMEOUT_MS

    font = get_pixel_font(28 * new_s)
    sw, sh = canvas.width, canvas.height
    box_w = 500 * new_s
    box_h = 140 * new_s
    box_x = (sw - box_w) // 2
    box_y = (sh - box_h) // 2
    btn_w = 140 * new_s
    btn_h = 50 * new_s
    btn_gap = 24 * new_s
    keep_x = box_x + (box_w - 2 * btn_w - btn_gap) // 2
    revert_x = keep_x + btn_w + btn_gap
    btn_y = box_y + box_h - btn_h - 16 * new_s

    while True:
        remaining = max(0, deadline - pygame.time.get_ticks())
        if remaining == 0:
            return False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.VIDEORESIZE:
                canvas.handle_resize(event.w, event.h)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return True
                if event.key == pygame.K_ESCAPE:
                    return False
            _kb = build_key_lookup(default_keyboard())
            _cl = build_controller_lookup(default_controller())
            _nav = resolve_event(event, _kb, _cl)
            if PlayerAction.CONFIRM in _nav:
                return True
            if PlayerAction.BACK in _nav:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = canvas.to_canvas(*event.pos)
                if pygame.Rect(keep_x, btn_y, btn_w, btn_h).collidepoint(
                    mx,
                    my,
                ):
                    return True
                if pygame.Rect(
                    revert_x,
                    btn_y,
                    btn_w,
                    btn_h,
                ).collidepoint(mx, my):
                    return False

        surf = canvas.surface
        surf.fill((15, 15, 20))

        # Dialog box.
        pygame.draw.rect(
            surf,
            (30, 30, 35),
            (box_x, box_y, box_w, box_h),
        )
        pygame.draw.rect(
            surf,
            _GOLD,
            (box_x, box_y, box_w, box_h),
            3,
        )

        secs = (remaining + 999) // 1000
        msg = font.render(
            f"Keep this scale? Reverting in {secs}s...",
            False,
            _LABEL_COLOR,
        )
        surf.blit(
            msg,
            (box_x + (box_w - msg.get_width()) // 2, box_y + 20 * new_s),
        )

        # Keep button.
        mouse_pos = canvas.to_canvas(*pygame.mouse.get_pos())
        keep_rect = pygame.Rect(keep_x, btn_y, btn_w, btn_h)
        keep_hover = keep_rect.collidepoint(mouse_pos)
        pygame.draw.rect(
            surf,
            (55, 55, 60) if keep_hover else (35, 35, 40),
            keep_rect,
        )
        pygame.draw.rect(surf, _GOLD, keep_rect, 2)
        kt = font.render("Keep", False, _LABEL_COLOR)
        surf.blit(
            kt,
            (
                keep_x + (btn_w - kt.get_width()) // 2,
                btn_y + (btn_h - kt.get_height()) // 2,
            ),
        )

        # Revert button.
        rev_rect = pygame.Rect(revert_x, btn_y, btn_w, btn_h)
        rev_hover = rev_rect.collidepoint(mouse_pos)
        pygame.draw.rect(
            surf,
            (55, 55, 60) if rev_hover else (35, 35, 40),
            rev_rect,
        )
        pygame.draw.rect(surf, (120, 60, 60), rev_rect, 2)
        rt = font.render("Revert", False, _LABEL_COLOR)
        surf.blit(
            rt,
            (
                revert_x + (btn_w - rt.get_width()) // 2,
                btn_y + (btn_h - rt.get_height()) // 2,
            ),
        )

        canvas.present(screen)
        clock.tick(30)


def _controls_section_height(
    n_rows: int,
    n_categories: int,
    s: int,
) -> int:
    """Compute the pixel height of the controls section.

    Args:
        n_rows: Number of binding rows.
        n_categories: Number of category sub-headers.
        s: UI scale factor.

    Returns:
        Total height in pixels.
    """
    section_header_h = _BASE_SECTION_HEADER_H * s
    section_rule_h = _BASE_SECTION_RULE_H * s
    section_gap = _BASE_SECTION_GAP * s
    row_h = _BASE_ROW_H * s
    row_gap = _BASE_ROW_GAP * s
    cat_h = 20 * s
    btn_h = _BASE_BTN_H * s
    pad = _BASE_SECTION_PAD_TOP * s
    return (
        pad
        + section_header_h
        + row_h
        + row_gap  # tab bar
        + section_rule_h
        + section_gap
        + n_categories * (cat_h + row_gap)
        + n_rows * (row_h + row_gap)
        + btn_h
        + row_gap  # reset button
        + pad
    )


def run_controls_menu(
    screen: pygame.Surface,
    config: PlayerConfig,
) -> tuple[bool, int]:
    """Show key/controller bindings with rebinding and display settings.

    Displays a tabbed view of keyboard and controller bindings. Each
    binding is clickable: click to enter listening mode, then press the
    desired key or button to rebind it. The Display section at the
    bottom provides fullscreen and UI scale toggles.

    Bindings are mutated in place on *config*. The caller should
    persist the config after this function returns.

    Args:
        screen: Pygame display surface.
        config: Full player configuration (bindings mutated in place).

    Returns:
        Tuple of ``(fullscreen, ui_scale)`` with possibly updated
        display settings.
    """
    from factoriax.config import (
        Bindings,
        controller_event_to_name,
        default_controller,
        default_keyboard,
        event_to_key_name,
    )

    global _REBIND_ACTIONS  # noqa: PLW0603
    if not _REBIND_ACTIONS:
        _REBIND_ACTIONS = _init_rebind_actions()

    s = _theme.UI_SCALE
    canvas = ScaledCanvas(1024, s, screen)
    sw, sh = canvas.width, canvas.height

    # Scaled layout constants.
    top_bar_h = _BASE_TOP_BAR_H * s
    section_pad_top = _BASE_SECTION_PAD_TOP * s
    section_header_h = _BASE_SECTION_HEADER_H * s
    section_rule_h = _BASE_SECTION_RULE_H * s
    section_gap = _BASE_SECTION_GAP * s
    row_h = _BASE_ROW_H * s
    row_gap = _BASE_ROW_GAP * s
    side_pad = _BASE_SIDE_PAD * s
    input_w = _BASE_INPUT_W * s
    input_h = _BASE_INPUT_H * s
    checkbox_size = _BASE_CHECKBOX_SIZE * s
    btn_w = _BASE_BTN_W * s
    btn_h = _BASE_BTN_H * s
    scroll_step = _BASE_SCROLL_STEP * s
    cat_h = 20 * s  # category sub-header height
    tab_gap = 24 * s

    back_rect = pygame.Rect(side_pad, 8 * s, btn_w, btn_h)
    reset_btn_w = 200 * s

    clock = pygame.time.Clock()

    fullscreen = config.fullscreen
    ui_scale = config.ui_scale
    rs = _RebindState()
    scroll_offset = 0

    font_header = get_pixel_font(32 * s)
    font_label = get_pixel_font(28 * s)
    font_input = get_pixel_font(28 * s)
    font_btn = get_pixel_font(28 * s)
    font_section = get_pixel_font(32 * s)
    font_cat = get_pixel_font(20 * s)

    # Count rows and categories for height calculation.
    n_rows = sum(len(acts) for _, acts in _REBIND_ACTIONS)
    n_cats = len(_REBIND_ACTIONS)

    controls_h = _controls_section_height(n_rows, n_cats, s)
    display_h = (
        section_header_h
        + section_rule_h
        + section_gap
        + 2 * (row_h + row_gap)
        + section_pad_top
    )
    content_h = controls_h + display_h

    def _active_bindings() -> Bindings:
        if rs.active_tab == "keyboard":
            return config.keyboard
        return config.controller

    # Flat list of action keys for focused_row indexing.
    flat_actions: list[str] = [
        action_key for _, actions in _REBIND_ACTIONS for action_key, _ in actions
    ]

    while True:
        scrollable_area = sh - top_bar_h
        max_scroll = max(0, content_h - scrollable_area)
        scroll_offset = max(0, min(scroll_offset, max_scroll))

        # -- Events ---------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return fullscreen, ui_scale

            if event.type == pygame.VIDEORESIZE:
                canvas.handle_resize(event.w, event.h)

            # --- Listening mode: capture input -----------------------
            if rs.listening_action is not None:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        rs.listening_action = None
                    elif rs.active_tab == "keyboard":
                        mods = pygame.key.get_mods()
                        name = event_to_key_name(event.key, mods)
                        bindings = _active_bindings()
                        bindings[rs.listening_action] = [name]
                        rs.listening_action = None
                elif rs.active_tab == "controller":
                    if event.type in (
                        pygame.JOYBUTTONDOWN,
                        pygame.JOYHATMOTION,
                        pygame.JOYAXISMOTION,
                    ):
                        ctrl_name = controller_event_to_name(event)
                        if ctrl_name is not None:
                            bindings = _active_bindings()
                            bindings[rs.listening_action] = [ctrl_name]
                            rs.listening_action = None
                continue  # Consume all events while listening.

            # --- Normal mode -----------------------------------------
            if event.type == pygame.MOUSEWHEEL:
                scroll_offset -= event.y * scroll_step
                scroll_offset = max(
                    0,
                    min(scroll_offset, max_scroll),
                )

            # Keyboard/controller navigation via configured bindings.
            _kb = build_key_lookup(default_keyboard())
            _cl = build_controller_lookup(default_controller())
            nav = resolve_event(event, _kb, _cl)

            _nav_up = PlayerAction.NAV_UP in nav
            _nav_down = PlayerAction.NAV_DOWN in nav
            _nav_confirm = PlayerAction.CONFIRM in nav
            _nav_back = PlayerAction.BACK in nav
            _nav_left = PlayerAction.NAV_LEFT in nav
            _nav_right = PlayerAction.NAV_RIGHT in nav

            # Escape is hardcoded (not in bindings) so handle it too.
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                _nav_back = True

            if _nav_up and flat_actions:
                rs.focused_row = (rs.focused_row - 1) % len(flat_actions)
                # Auto-scroll to keep focused row visible.
                row_y = rs.focused_row * (row_h + row_gap)
                if row_y < scroll_offset:
                    scroll_offset = row_y
                elif row_y + row_h > scroll_offset + scrollable_area:
                    scroll_offset = row_y + row_h - scrollable_area
            elif _nav_down and flat_actions:
                rs.focused_row = (rs.focused_row + 1) % len(flat_actions)
                row_y = rs.focused_row * (row_h + row_gap)
                if row_y < scroll_offset:
                    scroll_offset = row_y
                elif row_y + row_h > scroll_offset + scrollable_area:
                    scroll_offset = row_y + row_h - scrollable_area
            elif _nav_confirm and flat_actions:
                rs.listening_action = flat_actions[rs.focused_row]
            elif _nav_back:
                return fullscreen, ui_scale
            elif _nav_left:
                rs.active_tab = "keyboard"
            elif _nav_right:
                rs.active_tab = "controller"

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = canvas.to_canvas(*event.pos)

                if back_rect.collidepoint(mx, my):
                    return fullscreen, ui_scale

                # Tab buttons (recompute positions to match draw).
                tab_cy = top_bar_h + section_pad_top - scroll_offset + section_header_h
                kb_surf = font_label.render(
                    "Keyboard",
                    False,
                    _GOLD,
                )
                kb_rect = pygame.Rect(
                    side_pad,
                    tab_cy,
                    kb_surf.get_width() + 16 * s,
                    row_h,
                )
                ctrl_surf = font_label.render(
                    "Controller",
                    False,
                    _GOLD,
                )
                ctrl_rect = pygame.Rect(
                    kb_rect.right + tab_gap,
                    tab_cy,
                    ctrl_surf.get_width() + 16 * s,
                    row_h,
                )
                if kb_rect.collidepoint(mx, my):
                    rs.active_tab = "keyboard"
                elif ctrl_rect.collidepoint(mx, my):
                    rs.active_tab = "controller"

                # Binding boxes.
                bind_cy = tab_cy + row_h + row_gap + section_rule_h + section_gap
                box_x = sw - side_pad - input_w
                for cat_label, actions in _REBIND_ACTIONS:
                    if cat_label is not None:
                        bind_cy += cat_h + row_gap
                    for action_key, _label in actions:
                        box_y = bind_cy + (row_h - input_h) // 2
                        box_rect = pygame.Rect(
                            box_x,
                            box_y,
                            input_w,
                            input_h,
                        )
                        if box_rect.collidepoint(mx, my):
                            rs.listening_action = action_key
                        bind_cy += row_h + row_gap

                # Reset button.
                reset_cy = bind_cy + row_gap
                reset_rect = pygame.Rect(
                    (sw - reset_btn_w) // 2,
                    reset_cy,
                    reset_btn_w,
                    btn_h,
                )
                if reset_rect.collidepoint(mx, my):
                    if rs.active_tab == "keyboard":
                        config.keyboard = default_keyboard()
                    else:
                        config.controller = default_controller()

                # Display: fullscreen checkbox.
                disp_cy = (
                    reset_cy
                    + btn_h
                    + row_gap
                    + section_pad_top
                    + section_header_h
                    + section_rule_h
                    + section_gap
                )
                cb_x = side_pad
                cb_y = disp_cy + (row_h - checkbox_size) // 2
                cb_lbl = font_label.render(
                    "Fullscreen",
                    False,
                    _LABEL_COLOR,
                )
                cb_hit = pygame.Rect(
                    cb_x,
                    cb_y,
                    checkbox_size + 8 + cb_lbl.get_width(),
                    checkbox_size,
                )
                if cb_hit.collidepoint(mx, my):
                    fullscreen = not fullscreen
                    if fullscreen:
                        screen = pygame.display.set_mode(
                            (0, 0),
                            pygame.FULLSCREEN,
                        )
                    else:
                        cpx = 1024 * s
                        w, h = calculate_window_size(cpx, cpx)
                        screen = pygame.display.set_mode((w, h))
                    canvas.handle_resize(*screen.get_size())

                # Display: UI scale buttons.
                scale_cy = disp_cy + row_h + row_gap
                scale_lbl = font_label.render(
                    "UI Scale",
                    False,
                    _LABEL_COLOR,
                )
                sbtn_w = 60 * s
                sbtn_h2 = row_h - 4 * s
                sbtn_x0 = side_pad + scale_lbl.get_width() + 16 * s
                sbtn_y = scale_cy + (row_h - sbtn_h2) // 2
                for si in range(4):
                    bx = sbtn_x0 + si * (sbtn_w + 4 * s)
                    if pygame.Rect(
                        bx,
                        sbtn_y,
                        sbtn_w,
                        sbtn_h2,
                    ).collidepoint(mx, my):
                        if si != ui_scale:
                            confirmed = _confirm_scale_change(
                                screen,
                                si,
                            )
                            if confirmed:
                                ui_scale = si
                            cur_s = ui_scale if ui_scale > 0 else auto_ui_scale()
                            _theme.apply_scale(cur_s)
                            return fullscreen, ui_scale
                        break

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return fullscreen, ui_scale

        # -- Drawing --------------------------------------------------
        surf = canvas.surface
        mouse_pos = canvas.to_canvas(*pygame.mouse.get_pos())
        surf.fill(_BG)

        _draw_button(
            surf,
            back_rect,
            "< Back",
            font_btn,
            back_rect.collidepoint(mouse_pos),
        )

        title_surf = font_header.render("Controls", False, _GOLD)
        title_x = (sw - title_surf.get_width()) // 2
        title_y = 8 * s + (btn_h - title_surf.get_height()) // 2
        surf.blit(title_surf, (title_x, title_y))

        content_clip = pygame.Rect(0, top_bar_h, sw, sh - top_bar_h)
        surf.set_clip(content_clip)

        cy = top_bar_h + section_pad_top - scroll_offset

        # -- Controls section header ----------------------------------
        sec_surf = font_section.render("Bindings", False, _GOLD)
        surf.blit(sec_surf, (side_pad, cy))
        cy += section_header_h

        # Tab bar.
        for tab_name in ("Keyboard", "Controller"):
            tab_key = tab_name.lower()
            is_active_tab = rs.active_tab == tab_key
            color = _GOLD if is_active_tab else _LABEL_COLOR
            tab_surf = font_label.render(tab_name, False, color)
            tab_x = (
                side_pad
                if tab_name == "Keyboard"
                else (
                    side_pad
                    + font_label.render("Keyboard", False, _GOLD).get_width()
                    + 16 * s
                    + tab_gap
                )
            )
            tab_y = cy + (row_h - tab_surf.get_height()) // 2
            surf.blit(tab_surf, (tab_x, tab_y))
            if is_active_tab:
                underline_y = cy + row_h - 2 * s
                pygame.draw.line(
                    surf,
                    _GOLD,
                    (tab_x, underline_y),
                    (tab_x + tab_surf.get_width(), underline_y),
                    2,
                )
        cy += row_h + row_gap

        # Rule below tabs.
        pygame.draw.line(
            surf,
            _SECTION_RULE,
            (side_pad, cy),
            (sw - side_pad, cy),
        )
        cy += section_rule_h + section_gap

        # Binding rows.
        bindings = _active_bindings()
        box_x = sw - side_pad - input_w
        flat_idx = 0
        for cat_label, actions in _REBIND_ACTIONS:
            if cat_label is not None:
                cat_surf = font_cat.render(
                    cat_label,
                    False,
                    _GOLD,
                )
                cat_y = cy + (cat_h - cat_surf.get_height()) // 2
                surf.blit(cat_surf, (side_pad, cat_y))
                cy += cat_h + row_gap

            for action_key, label in actions:
                is_focused = flat_idx == rs.focused_row
                flat_idx += 1

                # Label.
                lbl_color = _GOLD if is_focused else _LABEL_COLOR
                lbl_surf = font_label.render(
                    label,
                    False,
                    lbl_color,
                )
                lbl_y = cy + (row_h - lbl_surf.get_height()) // 2
                surf.blit(lbl_surf, (side_pad, lbl_y))

                # Binding box.
                box_y = cy + (row_h - input_h) // 2
                box_rect = pygame.Rect(
                    box_x,
                    box_y,
                    input_w,
                    input_h,
                )
                is_listening = rs.listening_action == action_key
                is_hovered = box_rect.collidepoint(mouse_pos)

                pygame.draw.rect(
                    surf,
                    _INPUT_BG,
                    box_rect,
                )
                if is_listening:
                    border_c = _GOLD
                elif is_focused:
                    border_c = _LABEL_COLOR
                elif is_hovered:
                    border_c = _LABEL_COLOR
                else:
                    border_c = _INPUT_BORDER
                pygame.draw.rect(surf, border_c, box_rect, 1)

                if is_listening:
                    prompt = (
                        "Press key..."
                        if rs.active_tab == "keyboard"
                        else "Press input..."
                    )
                    txt_surf = font_input.render(
                        prompt,
                        False,
                        _GOLD,
                    )
                else:
                    names = bindings.get(action_key, [])
                    display = _format_binding(
                        names,
                        rs.active_tab,
                    )
                    txt_surf = font_input.render(
                        display,
                        False,
                        _INPUT_TEXT,
                    )

                # Clip text to box width.
                max_tw = input_w - 8
                if txt_surf.get_width() > max_tw:
                    txt_surf = txt_surf.subsurface(
                        txt_surf.get_width() - max_tw,
                        0,
                        max_tw,
                        txt_surf.get_height(),
                    )
                surf.blit(
                    txt_surf,
                    (
                        box_x + 4,
                        box_y + (input_h - txt_surf.get_height()) // 2,
                    ),
                )

                cy += row_h + row_gap

        # Reset button.
        cy += row_gap
        reset_rect = pygame.Rect(
            (sw - reset_btn_w) // 2,
            cy,
            reset_btn_w,
            btn_h,
        )
        _draw_button(
            surf,
            reset_rect,
            "Reset to Defaults",
            font_btn,
            reset_rect.collidepoint(mouse_pos),
        )
        cy += btn_h + row_gap + section_pad_top

        # -- Display section ------------------------------------------
        sec_surf = font_section.render("Display", False, _GOLD)
        surf.blit(sec_surf, (side_pad, cy))
        cy += section_header_h
        pygame.draw.line(
            surf,
            _SECTION_RULE,
            (side_pad, cy),
            (sw - side_pad, cy),
        )
        cy += section_rule_h + section_gap

        cb_x = side_pad
        cb_y = cy + (row_h - checkbox_size) // 2
        _draw_checkbox(
            surf,
            cb_x,
            cb_y,
            checkbox_size,
            fullscreen,
            pygame.Rect(
                cb_x,
                cb_y,
                checkbox_size,
                checkbox_size,
            ).collidepoint(mouse_pos),
        )
        lbl_surf = font_label.render(
            "Fullscreen",
            False,
            _LABEL_COLOR,
        )
        surf.blit(
            lbl_surf,
            (
                cb_x + checkbox_size + 8,
                cy + (row_h - lbl_surf.get_height()) // 2,
            ),
        )
        cy += row_h + row_gap

        # UI Scale selector.
        scale_labels = ["Auto", "1x", "2x", "3x"]
        scale_lbl = font_label.render(
            "UI Scale",
            False,
            _LABEL_COLOR,
        )
        surf.blit(
            scale_lbl,
            (
                side_pad,
                cy + (row_h - scale_lbl.get_height()) // 2,
            ),
        )
        sbtn_w = 60 * s
        sbtn_h2 = row_h - 4 * s
        sbtn_x0 = side_pad + scale_lbl.get_width() + 16 * s
        sbtn_y = cy + (row_h - sbtn_h2) // 2
        for si, sl in enumerate(scale_labels):
            bx = sbtn_x0 + si * (sbtn_w + 4 * s)
            is_active_scale = si == ui_scale
            bg = (80, 75, 50) if is_active_scale else (40, 40, 45)
            pygame.draw.rect(
                surf,
                bg,
                (bx, sbtn_y, sbtn_w, sbtn_h2),
            )
            border_c = _GOLD if is_active_scale else (70, 70, 70)
            pygame.draw.rect(
                surf,
                border_c,
                (bx, sbtn_y, sbtn_w, sbtn_h2),
                2,
            )
            st = font_label.render(sl, False, _LABEL_COLOR)
            surf.blit(
                st,
                (
                    bx + (sbtn_w - st.get_width()) // 2,
                    sbtn_y + (sbtn_h2 - st.get_height()) // 2,
                ),
            )
        cy += row_h + row_gap + section_pad_top

        surf.set_clip(None)
        _draw_scrollbar(
            surf,
            scroll_offset,
            max_scroll,
            content_h,
            top_bar_h,
        )

        canvas.present(screen)
        clock.tick(_FPS)
