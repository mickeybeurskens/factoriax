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
        ],
    )
    machines = _Section(
        "Machines",
        [
            _f("machine_max_health", "Machine Health", "int"),
            _f("power_per_coal", "Power/Coal", "int"),
            _f("miner_mining_rate", "Mining Rate", "int"),
            _f("max_assembler_stack_size", "Assembler Stack", "int"),
        ],
    )
    biters = _Section(
        "Biters",
        [
            _f("max_biters", "Max Biters", "int"),
            _f("biter_spawn_rate", "Spawn Rate", "float"),
            _f("nest_probability", "Nest Prob", "float"),
            _f("biter_tick_interval", "Tick Interval", "int"),
            _f("biter_attack_damage", "Attack Damage", "int"),
            _f("biter_health_default", "Biter Health", "int"),
            _f("scent_decay", "Scent Decay", "float"),
            _f("scent_emission", "Scent Emission", "float"),
        ],
    )
    return [world, resources, machines, biters]


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


def _build_params(
    sections: list[_Section],
    biters_enabled: bool,
) -> EnvParams:
    """Construct an EnvParams from the current field string values.

    Invalid strings fall back to the EnvParams default for that field.

    Args:
        sections: The populated section list.
        biters_enabled: Whether the biters checkbox is checked.

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

    if not biters_enabled:
        kwargs["biter_spawn_rate"] = 0.0
        kwargs["nest_probability"] = 0.0

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


def _build_controls_lines() -> list[tuple[str, str]]:
    """Build read-only key binding display rows.

    Returns:
        List of (label, key_names) tuples for each displayed action.
    """
    from factoriax.config import PlayerAction, default_keyboard

    kb = default_keyboard()
    display_names: dict[str, str] = {
        PlayerAction.MOVE_UP: "Move Up",
        PlayerAction.MOVE_DOWN: "Move Down",
        PlayerAction.MOVE_LEFT: "Move Left",
        PlayerAction.MOVE_RIGHT: "Move Right",
        PlayerAction.MINE: "Mine",
        PlayerAction.INTERACT: "Interact",
        PlayerAction.ROTATE: "Rotate",
        PlayerAction.REPAIR: "Repair",
        PlayerAction.OPEN_INVENTORY: "Inventory",
        PlayerAction.OPEN_ACHIEVEMENTS: "Achievements",
        PlayerAction.OPEN_RESEARCH: "Research",
        PlayerAction.OPEN_MACHINE: "Inspect Machine",
        PlayerAction.OPEN_HELP: "Help",
        PlayerAction.TOGGLE_HOTBAR: "Hotbar Page",
        PlayerAction.CONFIRM: "Confirm",
    }
    lines: list[tuple[str, str]] = []
    for action, label in display_names.items():
        keys = kb.get(action, [])
        key_str = ", ".join(k.replace("K_", "") for k in keys) if keys else "-"
        lines.append((label, key_str))
    return lines


def _draw_controls_section(
    surface: pygame.Surface,
    lines: list[tuple[str, str]],
    cy: int,
    label_w: int,
    font_section: pygame.font.Font,
    font_label: pygame.font.Font,
    font_value: pygame.font.Font,
    side_pad: int,
    section_header_h: int,
    section_rule_h: int,
    section_gap: int,
    row_h: int,
    row_gap: int,
    section_pad_top: int,
    sw: int,
) -> int:
    """Draw the read-only controls section and return updated y cursor.

    Args:
        surface: Target surface.
        lines: (label, key_names) pairs from :func:`_build_controls_lines`.
        cy: Current y position.
        label_w: Width allocated for the label column.
        font_section: Font for the section header.
        font_label: Font for row labels.
        font_value: Font for key name values.
        side_pad: Horizontal padding from edges.
        section_header_h: Height of section header text area.
        section_rule_h: Thickness of the horizontal rule.
        section_gap: Gap below the rule before content rows.
        row_h: Height of each content row.
        row_gap: Vertical gap between rows.
        section_pad_top: Padding after all rows in the section.
        sw: Total surface width (for rule endpoint).

    Returns:
        Updated y cursor after the section.
    """
    sec_surf = font_section.render("Controls", False, _GOLD)
    surface.blit(sec_surf, (side_pad, cy))
    cy += section_header_h
    pygame.draw.line(
        surface,
        _SECTION_RULE,
        (side_pad, cy),
        (sw - side_pad, cy),
    )
    cy += section_rule_h + section_gap
    for label, keys in lines:
        lbl_surf = font_label.render(label, False, _LABEL_COLOR)
        lbl_y = cy + (row_h - lbl_surf.get_height()) // 2
        surface.blit(lbl_surf, (side_pad, lbl_y))
        val_surf = font_value.render(keys, False, _INPUT_TEXT)
        val_x = side_pad + label_w + 8
        val_y = cy + (row_h - val_surf.get_height()) // 2
        surface.blit(val_surf, (val_x, val_y))
        cy += row_h + row_gap
    cy += section_pad_top
    return cy


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
    checkbox_size = _BASE_CHECKBOX_SIZE * s
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
    biters_enabled = params.biter_spawn_rate > 0.0
    stored_spawn_rate = _format_value(params.biter_spawn_rate, "float")

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
        if sec.title == "Biters":
            content_h += row_h + row_gap
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
                    return _build_params(sections, biters_enabled)

                # Content area clicks (adjusted for scroll).
                cy = top_bar_h + section_pad_top - scroll_offset
                clicked_field: FieldState | None = None

                for sec in sections:
                    cy += section_header_h + section_rule_h + section_gap

                    if sec.title == "Biters":
                        cb_x = side_pad
                        cb_y = cy + (row_h - checkbox_size) // 2
                        cb_label = font_label.render(
                            "Enable Biters",
                            False,
                            _LABEL_COLOR,
                        )
                        cb_hit = pygame.Rect(
                            cb_x,
                            cb_y,
                            checkbox_size + 8 + cb_label.get_width(),
                            checkbox_size,
                        )
                        if cb_hit.collidepoint(mx, my):
                            biters_enabled = not biters_enabled
                            if biters_enabled:
                                spawn_fs = next(
                                    (
                                        f
                                        for f in sec.fields
                                        if f.name == "biter_spawn_rate"
                                    ),
                                    None,
                                )
                                if spawn_fs is not None and spawn_fs.value == "0.00":
                                    spawn_fs.value = stored_spawn_rate
                        cy += row_h + row_gap

                    is_biter_section = sec.title == "Biters"
                    for i in range(0, len(sec.fields), 2):
                        for col_idx in range(2):
                            fi = i + col_idx
                            if fi >= len(sec.fields):
                                break
                            if is_biter_section and not biters_enabled:
                                continue
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
                    if event.key == pygame.K_TAB:
                        flat = _flat_fields(sections)
                        if flat:
                            _focus_field(sections, flat[0])

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
            is_biter_section = sec.title == "Biters"

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

            if is_biter_section:
                cb_x = side_pad
                cb_y = cy + (row_h - checkbox_size) // 2
                cb_rect = pygame.Rect(
                    cb_x,
                    cb_y,
                    checkbox_size,
                    checkbox_size,
                )
                _draw_checkbox(
                    surf,
                    cb_x,
                    cb_y,
                    checkbox_size,
                    biters_enabled,
                    cb_rect.collidepoint(mouse_pos),
                )
                lbl_surf = font_label.render(
                    "Enable Biters",
                    False,
                    _LABEL_COLOR,
                )
                lbl_x = cb_x + checkbox_size + 8
                lbl_y = cy + (row_h - lbl_surf.get_height()) // 2
                surf.blit(lbl_surf, (lbl_x, lbl_y))
                cy += row_h + row_gap

            for i in range(0, len(sec.fields), 2):
                for col_idx in range(2):
                    fi = i + col_idx
                    if fi >= len(sec.fields):
                        break

                    fs = sec.fields[fi]
                    disabled = is_biter_section and not biters_enabled
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
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = canvas.to_canvas(*event.pos)
                if pygame.Rect(keep_x, btn_y, btn_w, btn_h).collidepoint(
                    mx, my,
                ):
                    return True
                if pygame.Rect(
                    revert_x, btn_y, btn_w, btn_h,
                ).collidepoint(mx, my):
                    return False

        surf = canvas.surface
        surf.fill((15, 15, 20))

        # Dialog box.
        pygame.draw.rect(
            surf, (30, 30, 35), (box_x, box_y, box_w, box_h),
        )
        pygame.draw.rect(
            surf, _GOLD, (box_x, box_y, box_w, box_h), 3,
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


def run_controls_menu(
    screen: pygame.Surface,
    fullscreen: bool = False,
    ui_scale: int = 0,
) -> tuple[bool, int]:
    """Show a read-only overview of current key bindings.

    Displays all mapped player actions and their keyboard keys plus
    display settings (fullscreen toggle, UI scale selector). The only
    interactions are scrolling, toggling options, and pressing Back
    (or Escape) to return.

    Args:
        screen: Pygame display surface.
        fullscreen: Current fullscreen state (shown as a checkbox).
        ui_scale: Current UI scale (0=auto, 1/2/3=fixed).

    Returns:
        Tuple of (fullscreen, ui_scale) with possibly updated values.
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
    checkbox_size = _BASE_CHECKBOX_SIZE * s
    btn_w = _BASE_BTN_W * s
    btn_h = _BASE_BTN_H * s
    scroll_step = _BASE_SCROLL_STEP * s

    # Derived layout (constant because canvas size is fixed).
    content_w = sw - 2 * side_pad
    col_w = (content_w - col_gap) // 2
    label_w = col_w - input_w - 8

    back_rect = pygame.Rect(side_pad, 8 * s, btn_w, btn_h)

    clock = pygame.time.Clock()
    controls_lines = _build_controls_lines()

    scroll_offset = 0

    font_header = get_pixel_font(32 * s)
    font_label = get_pixel_font(28 * s)
    font_input = get_pixel_font(28 * s)
    font_btn = get_pixel_font(28 * s)
    font_section = get_pixel_font(32 * s)

    # Measure total content height (constant).
    content_h = section_pad_top
    # Controls section.
    content_h += section_header_h + section_rule_h + section_gap
    content_h += len(controls_lines) * (row_h + row_gap)
    content_h += section_pad_top
    # Display section (fullscreen checkbox + UI scale selector).
    content_h += section_header_h + section_rule_h + section_gap
    content_h += 2 * (row_h + row_gap)  # fullscreen + ui_scale rows
    content_h += section_pad_top

    while True:
        scrollable_area = sh - top_bar_h
        max_scroll = max(0, content_h - scrollable_area)
        scroll_offset = max(0, min(scroll_offset, max_scroll))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return fullscreen, ui_scale

            if event.type == pygame.VIDEORESIZE:
                canvas.handle_resize(event.w, event.h)

            if event.type == pygame.MOUSEWHEEL:
                scroll_offset -= event.y * scroll_step
                scroll_offset = max(0, min(scroll_offset, max_scroll))

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = canvas.to_canvas(*event.pos)
                if back_rect.collidepoint(mx, my):
                    return fullscreen, ui_scale

                # Hit-test the fullscreen checkbox.
                # Recompute its position to match the draw pass.
                fs_cy = top_bar_h + section_pad_top - scroll_offset
                # Skip controls section.
                fs_cy += section_header_h + section_rule_h + section_gap
                fs_cy += len(controls_lines) * (row_h + row_gap)
                fs_cy += section_pad_top
                # Display section header.
                fs_cy += section_header_h + section_rule_h + section_gap
                cb_x = side_pad
                cb_y = fs_cy + (row_h - checkbox_size) // 2
                cb_label_surf = font_label.render("Fullscreen", False, _LABEL_COLOR)
                cb_hit = pygame.Rect(
                    cb_x,
                    cb_y,
                    checkbox_size + 8 + cb_label_surf.get_width(),
                    checkbox_size,
                )
                if cb_hit.collidepoint(mx, my):
                    fullscreen = not fullscreen
                    if fullscreen:
                        screen = pygame.display.set_mode(
                            (0, 0), pygame.FULLSCREEN,
                        )
                    else:
                        canvas_px = 1024 * s
                        w, h = calculate_window_size(canvas_px, canvas_px)
                        screen = pygame.display.set_mode((w, h))
                    canvas.handle_resize(*screen.get_size())

                # Hit-test UI scale buttons (row below fullscreen).
                scale_row_cy = fs_cy + row_h + row_gap
                scale_label_surf = font_label.render(
                    "UI Scale", False, _LABEL_COLOR,
                )
                sbtn_w = 60 * s
                sbtn_h = row_h - 4 * s
                sbtn_x0 = (
                    side_pad + scale_label_surf.get_width() + 16 * s
                )
                sbtn_y = scale_row_cy + (row_h - sbtn_h) // 2
                for si in range(4):
                    bx = sbtn_x0 + si * (sbtn_w + 4 * s)
                    if pygame.Rect(
                        bx, sbtn_y, sbtn_w, sbtn_h,
                    ).collidepoint(mx, my):
                        if si != ui_scale:
                            confirmed = _confirm_scale_change(
                                screen, si,
                            )
                            if confirmed:
                                ui_scale = si
                            # Restore current menu's scale either way
                            # since _confirm_scale_change resets on
                            # revert but the caller re-enters this
                            # function on next launch.
                            cur_s = (
                                ui_scale
                                if ui_scale > 0
                                else auto_ui_scale()
                            )
                            _theme.apply_scale(cur_s)
                            # Fully re-enter: return so __main__
                            # can rebuild the window if needed.
                            return fullscreen, ui_scale
                        break

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return fullscreen, ui_scale

        # -- Drawing ------------------------------------------------------
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
        cy = _draw_controls_section(
            surf,
            controls_lines,
            cy,
            label_w,
            font_section,
            font_label,
            font_input,
            side_pad,
            section_header_h,
            section_rule_h,
            section_gap,
            row_h,
            row_gap,
            section_pad_top,
            sw,
        )

        # -- Display section with fullscreen checkbox ---------------------
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
        cb_rect = pygame.Rect(cb_x, cb_y, checkbox_size, checkbox_size)
        _draw_checkbox(
            surf,
            cb_x,
            cb_y,
            checkbox_size,
            fullscreen,
            cb_rect.collidepoint(mouse_pos),
        )
        lbl_surf = font_label.render("Fullscreen", False, _LABEL_COLOR)
        lbl_x = cb_x + checkbox_size + 8
        lbl_y = cy + (row_h - lbl_surf.get_height()) // 2
        surf.blit(lbl_surf, (lbl_x, lbl_y))
        cy += row_h + row_gap

        # UI Scale selector row.
        scale_labels = ["Auto", "1x", "2x", "3x"]
        scale_label = font_label.render("UI Scale", False, _LABEL_COLOR)
        surf.blit(
            scale_label,
            (side_pad, cy + (row_h - scale_label.get_height()) // 2),
        )
        scale_btn_w = 60 * s
        scale_btn_h = row_h - 4 * s
        scale_btn_x = side_pad + scale_label.get_width() + 16 * s
        scale_btn_y = cy + (row_h - scale_btn_h) // 2
        for si, sl in enumerate(scale_labels):
            bx = scale_btn_x + si * (scale_btn_w + 4 * s)
            is_active = si == ui_scale
            bg = (80, 75, 50) if is_active else (40, 40, 45)
            pygame.draw.rect(surf, bg, (bx, scale_btn_y, scale_btn_w, scale_btn_h))
            border_c = _GOLD if is_active else (70, 70, 70)
            pygame.draw.rect(
                surf, border_c, (bx, scale_btn_y, scale_btn_w, scale_btn_h), 2,
            )
            st = font_label.render(sl, False, _LABEL_COLOR)
            surf.blit(
                st,
                (
                    bx + (scale_btn_w - st.get_width()) // 2,
                    scale_btn_y + (scale_btn_h - st.get_height()) // 2,
                ),
            )
        cy += row_h + row_gap
        cy += section_pad_top

        surf.set_clip(None)
        _draw_scrollbar(surf, scroll_offset, max_scroll, content_h, top_bar_h)

        canvas.present(screen)
        clock.tick(_FPS)
