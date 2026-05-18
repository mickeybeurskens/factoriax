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
import time
from dataclasses import dataclass, field

import pygame

from factoriax.config import (
    PlayerAction,
    PlayerConfig,
    build_controller_lookup,
    build_key_lookup,
    config_to_env_params,
    default_controller,
    default_keyboard,
    env_params_to_dict,
    resolve_event,
    save_config,
)
from factoriax.state import EnvParams
from factoriax.ui import theme as _theme
from factoriax.ui.fonts import get_pixel_font
from factoriax.ui.forms import (
    BASE_BTN_H,
    BASE_BTN_W,
    BASE_COL_GAP,
    BASE_INPUT_H,
    BASE_INPUT_W,
    BASE_ROW_GAP,
    BASE_ROW_H,
    BASE_SCROLL_STEP,
    BASE_SECTION_GAP,
    BASE_SECTION_HEADER_H,
    BASE_SECTION_PAD_TOP,
    BASE_SECTION_RULE_H,
    BASE_SIDE_PAD,
    BASE_TOP_BAR_H,
    BG,
    ERROR_BORDER,
    ERROR_FLASH_FRAMES,
    FPS,
    GOLD,
    INPUT_BG,
    INPUT_BORDER,
    INPUT_TEXT,
    INPUT_TEXT_DISABLED,
    LABEL_COLOR,
    LABEL_DISABLED,
    SECTION_RULE,
    draw_button,
    draw_scrollbar,
)
from factoriax.ui.scaling import ScaledCanvas
from factoriax.ui.window import auto_ui_scale, calculate_window_size

logger = logging.getLogger(__name__)


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
class _Button:
    """A clickable inline button rendered next to a field.

    Used for the Randomize button on the Seed field. Hit-testing and
    rendering are handled inline in :func:`run_settings_menu` so the
    button has no per-instance pygame state of its own.

    Attributes:
        label: Display text.
        target_field: Field whose ``edit_buffer`` the click writes to.
    """

    label: str
    target_field: FieldState

    def on_click(self) -> None:
        """Write a freshly generated seed into ``target_field.edit_buffer``.

        Uses :func:`time.time_ns` so two clicks in different sessions
        almost certainly produce different values. Forces the field
        into editing mode so the new value renders as the live buffer.
        """
        self.target_field.editing = True
        self.target_field.edit_buffer = str(time.time_ns())


@dataclass
class _Section:
    """A group of fields under a shared heading.

    Attributes:
        title: Section heading text.
        fields: Fields belonging to this section, laid out in 2-column
            pairs (left column at even indices, right column at odd).
        buttons: Optional inline buttons rendered in a row below the
            section's fields (single-column, left-aligned).
    """

    title: str
    fields: list[FieldState] = field(default_factory=list)
    buttons: list[_Button] = field(default_factory=list)


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


def _build_sections(config: PlayerConfig) -> list[_Section]:
    """Construct the section/field layout from a :class:`PlayerConfig`.

    The Seed field reads from ``config.seed`` and ships with a
    Randomize button that fills in a fresh ``time.time_ns()`` value.
    All other fields mirror the ``EnvParams`` mapping derived from
    ``config.env_params``.

    Args:
        config: The starting player configuration.

    Returns:
        Ordered list of sections, each containing its fields.
    """
    params = config_to_env_params(config)

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

    seed_field = FieldState(
        name="seed",
        label="Seed",
        value=str(int(config.seed)),
        field_type="int",
    )
    seed_section = _Section(
        "Seed",
        [seed_field],
        buttons=[_Button(label="Randomize", target_field=seed_field)],
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
            _f("player_mining_yield", "Player Mining Yield", "int"),
        ],
    )
    machines = _Section(
        "Machines",
        [
            _f("max_machines", "Max Machines", "int"),
            _f("miner_mining_rate", "Machine Mining Rate", "int"),
        ],
    )
    return [seed_section, world, resources, machines]


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
        fs.error_timer = ERROR_FLASH_FRAMES
    fs.editing = False
    fs.edit_buffer = ""


def _cancel_edit(fs: FieldState) -> None:
    """Discard the edit buffer without applying changes.

    Args:
        fs: The field being cancelled (modified in place).
    """
    fs.editing = False
    fs.edit_buffer = ""


def _build_config(
    sections: list[_Section],
    base: PlayerConfig,
) -> PlayerConfig:
    """Construct a :class:`PlayerConfig` from the current field values.

    Invalid strings fall back to the EnvParams default for that field
    (or to ``base.seed`` for the seed field). Bindings, display, and
    controller settings are carried through from ``base`` unchanged.

    Args:
        sections: The populated section list.
        base: The config that seeded the menu; its non-EnvParams fields
            (keyboard, controller, display, …) flow through unchanged.

    Returns:
        A new PlayerConfig reflecting the in-field values.
    """
    defaults = EnvParams()
    env_dict: dict[str, int | float] = dict(base.env_params)
    seed = int(base.seed)
    for sec in sections:
        for fs in sec.fields:
            parsed = _parse_value(fs.value, fs.field_type)
            if fs.name == "seed":
                if parsed is not None:
                    seed = int(parsed)
                continue
            if parsed is not None:
                env_dict[fs.name] = parsed
            else:
                env_dict[fs.name] = getattr(defaults, fs.name)

    return PlayerConfig(
        env_params=env_dict,
        seed=seed,
        keyboard=base.keyboard,
        controller=base.controller,
        fullscreen=base.fullscreen,
        ui_scale=base.ui_scale,
    )


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
            GOLD,
            (box_x, box_y, box_w, box_h),
            3,
        )

        secs = (remaining + 999) // 1000
        msg = font.render(
            f"Keep this scale? Reverting in {secs}s...",
            False,
            LABEL_COLOR,
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
        pygame.draw.rect(surf, GOLD, keep_rect, 2)
        kt = font.render("Keep", False, LABEL_COLOR)
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
        rt = font.render("Revert", False, LABEL_COLOR)
        surf.blit(
            rt,
            (
                revert_x + (btn_w - rt.get_width()) // 2,
                btn_y + (btn_h - rt.get_height()) // 2,
            ),
        )

        canvas.present(screen)
        clock.tick(30)


# -- Public menu ----------------------------------------------------------


def run_settings_menu(
    screen: pygame.Surface,
    initial_config: PlayerConfig | None = None,
) -> PlayerConfig:
    """Show play settings and return the resulting :class:`PlayerConfig`.

    Presents all environment parameters organized in sections, plus a
    Seed field with a Randomize button. Each field can be clicked and
    edited. Both the Play button and the Back / Escape / window-close
    paths now commit the in-field values via :func:`save_config` —
    leaving the menu always persists whatever the user typed.

    Args:
        screen: Pygame display surface.
        initial_config: Starting player config. Defaults to a fresh
            :class:`PlayerConfig` when ``None``.

    Returns:
        A :class:`PlayerConfig` mirroring the in-field values. The
        return value is the same regardless of which exit path the
        user took (Play, Back, Escape, window-close); the caller
        decides whether to launch the game.
    """
    s = _theme.UI_SCALE
    canvas = ScaledCanvas(1024, s, screen)
    sw, sh = canvas.width, canvas.height

    # Scaled layout constants.
    top_bar_h = BASE_TOP_BAR_H * s
    section_pad_top = BASE_SECTION_PAD_TOP * s
    section_header_h = BASE_SECTION_HEADER_H * s
    section_rule_h = BASE_SECTION_RULE_H * s
    section_gap = BASE_SECTION_GAP * s
    row_h = BASE_ROW_H * s
    row_gap = BASE_ROW_GAP * s
    col_gap = BASE_COL_GAP * s
    side_pad = BASE_SIDE_PAD * s
    input_w = BASE_INPUT_W * s
    input_h = BASE_INPUT_H * s
    btn_w = BASE_BTN_W * s
    btn_h = BASE_BTN_H * s
    scroll_step = BASE_SCROLL_STEP * s

    # Derived layout (constant because canvas size is fixed).
    content_w = sw - 2 * side_pad
    col_w = (content_w - col_gap) // 2
    label_w = col_w - input_w - 8

    back_rect = pygame.Rect(side_pad, 8 * s, btn_w, btn_h)
    play_rect = pygame.Rect(sw - side_pad - btn_w, 8 * s, btn_w, btn_h)

    clock = pygame.time.Clock()
    base_config = (
        initial_config
        if initial_config is not None
        else PlayerConfig(
            env_params=env_params_to_dict(EnvParams()),
            keyboard=default_keyboard(),
            controller=default_controller(),
        )
    )
    sections = _build_sections(base_config)

    def _commit_and_close() -> PlayerConfig:
        """Confirm any active edit, persist via save_config, and return."""
        for fs in _flat_fields(sections):
            if fs.editing:
                _confirm_edit(fs)
        new_config = _build_config(sections, base_config)
        try:
            save_config(new_config)
        except OSError as exc:
            logger.error("Failed to persist config from settings menu: %s", exc)
        return new_config

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
        if sec.buttons:
            content_h += row_h + row_gap
        content_h += section_pad_top

    while True:
        scrollable_area = sh - top_bar_h
        max_scroll = max(0, content_h - scrollable_area)
        scroll_offset = max(0, min(scroll_offset, max_scroll))

        # -- Event handling -----------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return _commit_and_close()

            if event.type == pygame.VIDEORESIZE:
                canvas.handle_resize(event.w, event.h)

            if event.type == pygame.MOUSEWHEEL:
                scroll_offset -= event.y * scroll_step
                scroll_offset = max(0, min(scroll_offset, max_scroll))

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = canvas.to_canvas(*event.pos)

                if back_rect.collidepoint(mx, my):
                    return _commit_and_close()
                if play_rect.collidepoint(mx, my):
                    return _commit_and_close()

                # Content area clicks (adjusted for scroll).
                cy = top_bar_h + section_pad_top - scroll_offset
                clicked_field: FieldState | None = None
                clicked_button: _Button | None = None

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

                    # Inline buttons (e.g. Randomize) render in their own
                    # row after the section's fields.
                    for b_idx, b in enumerate(sec.buttons):
                        bx = side_pad + b_idx * (btn_w + col_gap)
                        by = cy + (row_h - btn_h) // 2
                        b_rect = pygame.Rect(bx, by, btn_w, btn_h)
                        if b_rect.collidepoint(mx, my):
                            clicked_button = b
                    if sec.buttons:
                        cy += row_h + row_gap

                    cy += section_pad_top

                if clicked_button is not None:
                    clicked_button.on_click()
                elif clicked_field is not None:
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
                        return _commit_and_close()
                    if event.key == pygame.K_RETURN:
                        return _commit_and_close()
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
                        return _commit_and_close()
                elif PlayerAction.BACK in nav:
                    if active is not None:
                        _cancel_edit(active)
                    else:
                        return _commit_and_close()

        # -- Drawing ------------------------------------------------------
        surf = canvas.surface
        mouse_pos = canvas.to_canvas(*pygame.mouse.get_pos())
        surf.fill(BG)

        # Top bar.
        draw_button(
            surf,
            back_rect,
            "< Back",
            font_btn,
            back_rect.collidepoint(mouse_pos),
        )
        draw_button(
            surf,
            play_rect,
            "> Play",
            font_btn,
            play_rect.collidepoint(mouse_pos),
        )

        # Title.
        title_surf = font_header.render("Play Settings", False, GOLD)
        title_x = (sw - title_surf.get_width()) // 2
        title_y = 8 * s + (btn_h - title_surf.get_height()) // 2
        surf.blit(title_surf, (title_x, title_y))

        # Clip content area below top bar.
        content_clip = pygame.Rect(0, top_bar_h, sw, sh - top_bar_h)
        surf.set_clip(content_clip)

        cy = top_bar_h + section_pad_top - scroll_offset

        for sec in sections:
            sec_surf = font_section.render(sec.title, False, GOLD)
            surf.blit(sec_surf, (side_pad, cy))
            cy += section_header_h
            pygame.draw.line(
                surf,
                SECTION_RULE,
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

                    lbl_color = LABEL_DISABLED if disabled else LABEL_COLOR
                    lbl_surf = font_label.render(fs.label, False, lbl_color)
                    lbl_y = cy + (row_h - lbl_surf.get_height()) // 2
                    surf.blit(lbl_surf, (col_x, lbl_y))

                    fx = col_x + label_w + 8
                    fy = cy + (row_h - input_h) // 2
                    input_rect = pygame.Rect(fx, fy, input_w, input_h)
                    pygame.draw.rect(surf, INPUT_BG, input_rect)

                    if fs.error_timer > 0:
                        border_color = ERROR_BORDER
                        fs.error_timer -= 1
                    elif fs.editing:
                        border_color = GOLD
                    else:
                        border_color = INPUT_BORDER

                    if disabled:
                        border_color = INPUT_BORDER

                    pygame.draw.rect(surf, border_color, input_rect, 1)

                    if fs.editing and not disabled:
                        display_text = fs.edit_buffer + "_"
                    else:
                        display_text = fs.value
                    txt_color = INPUT_TEXT_DISABLED if disabled else INPUT_TEXT
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

            # Section's inline buttons (e.g. Randomize on Seed).
            for b_idx, b in enumerate(sec.buttons):
                bx = side_pad + b_idx * (btn_w + col_gap)
                by = cy + (row_h - btn_h) // 2
                b_rect = pygame.Rect(bx, by, btn_w, btn_h)
                draw_button(
                    surf,
                    b_rect,
                    b.label,
                    font_btn,
                    b_rect.collidepoint(mouse_pos),
                )
            if sec.buttons:
                cy += row_h + row_gap

            cy += section_pad_top

        surf.set_clip(None)
        draw_scrollbar(surf, scroll_offset, max_scroll, content_h, top_bar_h)

        canvas.present(screen)
        clock.tick(FPS)
