"""Play settings screen for FactoriaX.

Presents all EnvParams fields organized in labeled sections with editable
input boxes. The user clicks a field to focus it, types a new value, and
presses Enter to confirm or Escape to revert. Back returns None, Play
returns the configured EnvParams.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pygame

from factoriax.state import EnvParams
from factoriax.ui.fonts import get_pixel_font

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

# -- Layout constants -----------------------------------------------------
_TOP_BAR_H: int = 80
_SECTION_PAD_TOP: int = 24
_SECTION_HEADER_H: int = 44
_SECTION_RULE_H: int = 2
_SECTION_GAP: int = 16
_ROW_H: int = 48
_ROW_GAP: int = 8
_COL_GAP: int = 24
_SIDE_PAD: int = 32
_INPUT_W: int = 140
_INPUT_H: int = 40
_CHECKBOX_SIZE: int = 30
_BTN_W: int = 160
_BTN_H: int = 56
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
            _f("nest_probability", "Nest Prob", "float"),
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
    checked: bool,
    hovered: bool,
) -> pygame.Rect:
    """Draw a checkbox square and return its bounding rect.

    Args:
        surface: Destination surface.
        x: Left edge in pixels.
        y: Top edge in pixels.
        checked: Whether the box is checked.
        hovered: Whether the mouse is over the checkbox area.

    Returns:
        The bounding rectangle of the checkbox.
    """
    rect = pygame.Rect(x, y, _CHECKBOX_SIZE, _CHECKBOX_SIZE)
    bg = _BTN_HOVER if hovered else _INPUT_BG
    pygame.draw.rect(surface, bg, rect)
    pygame.draw.rect(surface, _GOLD, rect, 1)
    if checked:
        font = get_pixel_font(24)
        x_surf = font.render("X", False, _GOLD)
        cx = rect.x + (rect.width - x_surf.get_width()) // 2
        cy = rect.y + (rect.height - x_surf.get_height()) // 2
        surface.blit(x_surf, (cx, cy))
    return rect


# -- Main entry point -----------------------------------------------------


def run_settings_menu(screen: pygame.Surface) -> EnvParams | None:
    """Show settings and return configured EnvParams, or None to go back.

    Presents all environment parameters organized in sections. Each field
    can be clicked and edited. The Back button (or closing the window)
    returns None. The Play button builds an EnvParams from the current
    field values and returns it.

    Args:
        screen: Pygame display surface.

    Returns:
        EnvParams with user's choices if Play was clicked,
        None if Back was clicked or window closed.
    """
    clock = pygame.time.Clock()
    params = EnvParams()
    sections = _build_sections(params)
    biters_enabled = params.biter_spawn_rate > 0.0
    stored_spawn_rate = _format_value(params.biter_spawn_rate, "float")

    scroll_offset = 0
    font_header = get_pixel_font(32)
    font_label = get_pixel_font(28)
    font_input = get_pixel_font(28)
    font_btn = get_pixel_font(28)
    font_section = get_pixel_font(32)

    # Pre-compute rects each frame since window can resize.
    while True:
        sw, sh = screen.get_size()
        content_w = sw - 2 * _SIDE_PAD
        col_w = (content_w - _COL_GAP) // 2
        label_w = col_w - _INPUT_W - 8

        # Measure total content height.
        content_h = _SECTION_PAD_TOP
        for sec in sections:
            content_h += _SECTION_HEADER_H + _SECTION_RULE_H + _SECTION_GAP
            if sec.title == "Biters":
                content_h += _ROW_H + _ROW_GAP  # checkbox row
            num_rows = (len(sec.fields) + 1) // 2
            content_h += num_rows * (_ROW_H + _ROW_GAP)
            content_h += _SECTION_PAD_TOP

        scrollable_area = sh - _TOP_BAR_H
        max_scroll = max(0, content_h - scrollable_area)
        scroll_offset = max(0, min(scroll_offset, max_scroll))

        # -- Event handling -----------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

            if event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(
                    (event.w, event.h),
                    pygame.RESIZABLE,
                )

            if event.type == pygame.MOUSEWHEEL:
                scroll_offset -= event.y * 24
                scroll_offset = max(0, min(scroll_offset, max_scroll))

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos

                # Top bar buttons.
                back_rect = pygame.Rect(
                    _SIDE_PAD,
                    8,
                    _BTN_W,
                    _BTN_H,
                )
                play_rect = pygame.Rect(
                    sw - _SIDE_PAD - _BTN_W,
                    8,
                    _BTN_W,
                    _BTN_H,
                )
                if back_rect.collidepoint(mx, my):
                    return None
                if play_rect.collidepoint(mx, my):
                    # Confirm any active edit first.
                    for fs in _flat_fields(sections):
                        if fs.editing:
                            _confirm_edit(fs)
                    return _build_params(sections, biters_enabled)

                # Content area clicks (adjusted for scroll).
                cy = _TOP_BAR_H + _SECTION_PAD_TOP - scroll_offset
                clicked_field: FieldState | None = None

                for sec in sections:
                    cy += _SECTION_HEADER_H + _SECTION_RULE_H + _SECTION_GAP

                    if sec.title == "Biters":
                        cb_x = _SIDE_PAD
                        cb_y = cy + (_ROW_H - _CHECKBOX_SIZE) // 2
                        cb_rect = pygame.Rect(
                            cb_x,
                            cb_y,
                            _CHECKBOX_SIZE,
                            _CHECKBOX_SIZE,
                        )
                        # Extend click area to include label.
                        cb_label = font_label.render(
                            "Enable Biters",
                            False,
                            _LABEL_COLOR,
                        )
                        cb_hit = pygame.Rect(
                            cb_x,
                            cb_y,
                            _CHECKBOX_SIZE + 8 + cb_label.get_width(),
                            _CHECKBOX_SIZE,
                        )
                        if cb_hit.collidepoint(mx, my):
                            biters_enabled = not biters_enabled
                            # Restore stored spawn rate when re-enabling.
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
                        cy += _ROW_H + _ROW_GAP

                    is_biter_section = sec.title == "Biters"
                    for i in range(0, len(sec.fields), 2):
                        for col_idx in range(2):
                            fi = i + col_idx
                            if fi >= len(sec.fields):
                                break
                            if is_biter_section and not biters_enabled:
                                continue
                            fx = _SIDE_PAD + col_idx * (col_w + _COL_GAP) + label_w + 8
                            fy = cy + (_ROW_H - _INPUT_H) // 2
                            input_rect = pygame.Rect(
                                fx,
                                fy,
                                _INPUT_W,
                                _INPUT_H,
                            )
                            if input_rect.collidepoint(mx, my):
                                clicked_field = sec.fields[fi]
                        cy += _ROW_H + _ROW_GAP

                    cy += _SECTION_PAD_TOP

                # Apply focus change.
                if clicked_field is not None:
                    # Confirm previous edit if any.
                    for fs in _flat_fields(sections):
                        if fs.editing and fs is not clicked_field:
                            _confirm_edit(fs)
                    _focus_field(sections, clicked_field)
                else:
                    # Clicked outside any field: confirm active edit.
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
        mouse_pos = pygame.mouse.get_pos()
        screen.fill(_BG)

        # Top bar.
        back_rect = pygame.Rect(_SIDE_PAD, 8, _BTN_W, _BTN_H)
        play_rect = pygame.Rect(sw - _SIDE_PAD - _BTN_W, 8, _BTN_W, _BTN_H)
        _draw_button(
            screen,
            back_rect,
            "< Back",
            font_btn,
            back_rect.collidepoint(mouse_pos),
        )
        _draw_button(
            screen,
            play_rect,
            "> Play",
            font_btn,
            play_rect.collidepoint(mouse_pos),
        )

        # Title.
        title_surf = font_header.render("Play Settings", False, _GOLD)
        title_x = (sw - title_surf.get_width()) // 2
        title_y = 8 + (_BTN_H - title_surf.get_height()) // 2
        screen.blit(title_surf, (title_x, title_y))

        # Clip content area below top bar.
        content_clip = pygame.Rect(0, _TOP_BAR_H, sw, sh - _TOP_BAR_H)
        screen.set_clip(content_clip)

        cy = _TOP_BAR_H + _SECTION_PAD_TOP - scroll_offset

        for sec in sections:
            is_biter_section = sec.title == "Biters"

            # Section header.
            sec_surf = font_section.render(sec.title, False, _GOLD)
            screen.blit(sec_surf, (_SIDE_PAD, cy))
            cy += _SECTION_HEADER_H
            pygame.draw.line(
                screen,
                _SECTION_RULE,
                (_SIDE_PAD, cy),
                (sw - _SIDE_PAD, cy),
            )
            cy += _SECTION_RULE_H + _SECTION_GAP

            # Checkbox row for biters section.
            if is_biter_section:
                cb_x = _SIDE_PAD
                cb_y = cy + (_ROW_H - _CHECKBOX_SIZE) // 2
                cb_rect = pygame.Rect(
                    cb_x,
                    cb_y,
                    _CHECKBOX_SIZE,
                    _CHECKBOX_SIZE,
                )
                cb_hovered = cb_rect.collidepoint(mouse_pos)
                _draw_checkbox(
                    screen,
                    cb_x,
                    cb_y,
                    biters_enabled,
                    cb_hovered,
                )
                lbl_surf = font_label.render(
                    "Enable Biters",
                    False,
                    _LABEL_COLOR,
                )
                lbl_x = cb_x + _CHECKBOX_SIZE + 8
                lbl_y = cy + (_ROW_H - lbl_surf.get_height()) // 2
                screen.blit(lbl_surf, (lbl_x, lbl_y))
                cy += _ROW_H + _ROW_GAP

            # Field rows (2 per row).
            for i in range(0, len(sec.fields), 2):
                for col_idx in range(2):
                    fi = i + col_idx
                    if fi >= len(sec.fields):
                        break

                    fs = sec.fields[fi]
                    disabled = is_biter_section and not biters_enabled
                    col_x = _SIDE_PAD + col_idx * (col_w + _COL_GAP)

                    # Label.
                    lbl_color = _LABEL_DISABLED if disabled else _LABEL_COLOR
                    lbl_surf = font_label.render(fs.label, False, lbl_color)
                    lbl_y = cy + (_ROW_H - lbl_surf.get_height()) // 2
                    screen.blit(lbl_surf, (col_x, lbl_y))

                    # Input box.
                    fx = col_x + label_w + 8
                    fy = cy + (_ROW_H - _INPUT_H) // 2
                    input_rect = pygame.Rect(fx, fy, _INPUT_W, _INPUT_H)
                    pygame.draw.rect(screen, _INPUT_BG, input_rect)

                    if fs.error_timer > 0:
                        border_color = _ERROR_BORDER
                        fs.error_timer -= 1
                    elif fs.editing:
                        border_color = _GOLD
                    else:
                        border_color = _INPUT_BORDER

                    if disabled:
                        border_color = _INPUT_BORDER

                    pygame.draw.rect(
                        screen,
                        border_color,
                        input_rect,
                        1,
                    )

                    # Input text.
                    if fs.editing and not disabled:
                        display_text = fs.edit_buffer + "_"
                    else:
                        display_text = fs.value
                    txt_color = _INPUT_TEXT_DISABLED if disabled else _INPUT_TEXT
                    txt_surf = font_input.render(
                        display_text,
                        False,
                        txt_color,
                    )
                    # Clip text to input box width with 4px padding.
                    max_txt_w = _INPUT_W - 8
                    if txt_surf.get_width() > max_txt_w:
                        txt_surf = txt_surf.subsurface(
                            txt_surf.get_width() - max_txt_w,
                            0,
                            max_txt_w,
                            txt_surf.get_height(),
                        )
                    screen.blit(
                        txt_surf,
                        (fx + 4, fy + (_INPUT_H - txt_surf.get_height()) // 2),
                    )

                cy += _ROW_H + _ROW_GAP

            cy += _SECTION_PAD_TOP

        # Reset clip and draw scrollbar if needed.
        screen.set_clip(None)

        if max_scroll > 0:
            bar_area_h = sh - _TOP_BAR_H
            thumb_h = max(20, int(bar_area_h * bar_area_h / content_h))
            thumb_y = _TOP_BAR_H + int(
                scroll_offset / max_scroll * (bar_area_h - thumb_h)
            )
            bar_x = sw - 8
            pygame.draw.rect(
                screen,
                (40, 40, 40),
                pygame.Rect(bar_x, _TOP_BAR_H, 8, bar_area_h),
            )
            pygame.draw.rect(
                screen,
                (140, 130, 80),
                pygame.Rect(bar_x, thumb_y, 8, thumb_h),
            )

        pygame.display.flip()
        clock.tick(_FPS)
