"""Pre-game launch screen — Play / Settings / Reset to defaults.

The screen mirrors :mod:`factoriax.menu.scenarios_menu`: a list of options on
the left, content on the right. Selecting *Settings* on the left turns the
right panel into an editable list of :class:`~factoriax.state.EnvParams`
fields; selecting *Play* commits the current values and returns control to
the play loop; selecting *Reset to defaults* replaces every field with
:meth:`EnvParams()` defaults and a fresh seed.

:func:`_confirm_scale_change` is unrelated to the launch flow; it lives here
because :mod:`factoriax.menu.controls_menu` imports it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

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
from factoriax.ui import panels
from factoriax.ui import theme as _theme
from factoriax.ui.fonts import get_pixel_font
from factoriax.ui.forms import GOLD, LABEL_COLOR
from factoriax.ui.scaling import ScaledCanvas
from factoriax.ui.window import auto_ui_scale, calculate_window_size

_BASE_TITLE_FONT: int = 48
_BASE_BUTTON_W: int = 110
_BASE_BUTTON_H: int = 26
_FPS: int = 30
_RESET_FLASH_MS: int = 1500
_SEED_MAX: int = 2**32 - 1


# ---------------------------------------------------------------------------
# Field metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SettingField:
    key: str
    label: str
    is_float: bool
    step: float
    min_value: float
    max_value: float


_SETTING_FIELDS: tuple[_SettingField, ...] = (
    _SettingField("map_width", "Map width", False, 1, 8, 64),
    _SettingField("map_height", "Map height", False, 1, 8, 64),
    _SettingField("num_players", "Players", False, 1, 1, 4),
    _SettingField("max_timesteps", "Max steps", False, 100, 100, 10_000),
    _SettingField("water_probability", "Water probability", True, 0.01, 0.0, 1.0),
    _SettingField("iron_probability", "Iron probability", True, 0.01, 0.0, 1.0),
    _SettingField("copper_probability", "Copper probability", True, 0.01, 0.0, 1.0),
    _SettingField("coal_probability", "Coal probability", True, 0.01, 0.0, 1.0),
    _SettingField("tin_probability", "Tin probability", True, 0.01, 0.0, 1.0),
    _SettingField("silicon_probability", "Silicon probability", True, 0.01, 0.0, 1.0),
    _SettingField("base_resources", "Base resources", False, 100, 100, 10_000),
    _SettingField("max_machines", "Max machines", False, 16, 0, 1024),
    _SettingField("miner_mining_rate", "Miner mining rate", False, 1, 1, 20),
    _SettingField("player_mining_yield", "Player mining yield", False, 1, 1, 10),
    _SettingField("seed", "Seed", False, 1, 0, _SEED_MAX),
)


@dataclass(frozen=True)
class _PageOption:
    action: str
    label: str
    description: str


_PAGE_OPTIONS: tuple[_PageOption, ...] = (
    _PageOption(
        action="play",
        label="Play",
        description=(
            "Launch the game with the current settings. The seed and "
            "environment parameters determine the initial world layout."
        ),
    ),
    _PageOption(
        action="settings",
        label="Settings",
        description=(
            "Adjust environment parameters and the world seed. "
            "Left/Right edits the highlighted value; Up/Down moves "
            "between fields."
        ),
    ),
    _PageOption(
        action="reset",
        label="Reset to defaults",
        description=(
            "Replace every setting with the engine's default value and "
            "pick a fresh random seed."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Value get / set / adjust
# ---------------------------------------------------------------------------


def _get_value(config: PlayerConfig, field: _SettingField) -> float:
    if field.key == "seed":
        return float(config.seed)
    raw = config.env_params.get(field.key)
    if raw is None:
        raw = getattr(EnvParams(), field.key)
    return float(raw)


def _set_value(config: PlayerConfig, field: _SettingField, value: float) -> None:
    if field.key == "seed":
        config.seed = int(value)
        return
    config.env_params[field.key] = float(value) if field.is_float else int(value)


def _clamp(value: float, field: _SettingField) -> float:
    value = max(field.min_value, min(field.max_value, value))
    if not field.is_float:
        return float(int(value))
    return round(value, 2)


def _format_value(field: _SettingField, value: float) -> str:
    if field.is_float:
        return f"{value:.2f}"
    return str(int(value))


def _reset_to_defaults(config: PlayerConfig) -> None:
    config.env_params = env_params_to_dict(EnvParams())
    config.seed = int.from_bytes(os.urandom(4), "little")


# ---------------------------------------------------------------------------
# Settings panel rendering
# ---------------------------------------------------------------------------


def _row_y(rect: pygame.Rect, idx: int) -> int:
    return rect.top + _theme.ROW_PAD + idx * (_theme.ROW_H + 4)


def _draw_setting_rows(
    surf: pygame.Surface,
    rect: pygame.Rect,
    config: PlayerConfig,
    selected_idx: int,
    panel_focused: bool,
    font: pygame.font.Font,
) -> list[pygame.Rect]:
    """Render all setting rows; return their rects for hit-testing."""
    row_rects: list[pygame.Rect] = []
    inner_x = rect.left + _theme.ROW_PAD
    inner_w = rect.width - 2 * _theme.ROW_PAD
    for i, fld in enumerate(_SETTING_FIELDS):
        y = _row_y(rect, i)
        row_rect = pygame.Rect(inner_x, y, inner_w, _theme.ROW_H)
        row_rects.append(row_rect)
        active = panel_focused and i == selected_idx
        if active:
            pygame.draw.rect(
                surf, GOLD, row_rect, border_radius=_theme.BORDER_RADIUS // 2
            )
            label_color = _theme.PAGE_BG
            value_color = _theme.PAGE_BG
        else:
            label_color = _theme.TEXT_COLOR
            value_color = _theme.TEXT_COLOR

        label_surf = font.render(fld.label, False, label_color)
        surf.blit(
            label_surf,
            (
                row_rect.x + _theme.ROW_PAD // 2,
                row_rect.y + (_theme.ROW_H - label_surf.get_height()) // 2,
            ),
        )

        value_text = _format_value(fld, _get_value(config, fld))
        decorated = f"<  {value_text}  >" if active else value_text
        value_surf = font.render(decorated, False, value_color)
        surf.blit(
            value_surf,
            (
                row_rect.right - _theme.ROW_PAD // 2 - value_surf.get_width(),
                row_rect.y + (_theme.ROW_H - value_surf.get_height()) // 2,
            ),
        )
    return row_rects


def _randomize_button_rect(rect: pygame.Rect, scale: int) -> pygame.Rect:
    """Compact button right-aligned under the last setting row."""
    btn_w = _BASE_BUTTON_W * scale
    btn_h = _BASE_BUTTON_H * scale
    btn_y = _row_y(rect, len(_SETTING_FIELDS)) + 4
    btn_x = rect.right - _theme.ROW_PAD - btn_w
    return pygame.Rect(btn_x, btn_y, btn_w, btn_h)


def _draw_randomize_button(
    surf: pygame.Surface,
    button_rect: pygame.Rect,
    font: pygame.font.Font,
    *,
    active: bool,
) -> None:
    fill = _theme.BUTTON_HOVER if active else _theme.BUTTON_FILL
    border = GOLD if active else _theme.BORDER_INACTIVE
    pygame.draw.rect(surf, fill, button_rect, border_radius=_theme.BORDER_RADIUS // 2)
    pygame.draw.rect(
        surf,
        border,
        button_rect,
        _theme.BORDER_PX,
        border_radius=_theme.BORDER_RADIUS // 2,
    )
    label = font.render("Randomize", False, _theme.TEXT_COLOR)
    surf.blit(
        label,
        (
            button_rect.x + (button_rect.w - label.get_width()) // 2,
            button_rect.y + (button_rect.h - label.get_height()) // 2,
        ),
    )


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------


_NUM_NAV_ITEMS: int = len(_SETTING_FIELDS) + 1  # fields + randomize button
_RANDOMIZE_IDX: int = len(_SETTING_FIELDS)


def run_settings_menu(
    screen: pygame.Surface,
    initial_config: PlayerConfig | None = None,
) -> PlayerConfig:
    """Show the launch screen and return the (possibly edited) config.

    Returns:
        The :class:`PlayerConfig` that the caller should hand to the play
        loop. If the user backs out, the config is returned unchanged.
        Edits are persisted via :func:`factoriax.config.save_config` before
        the function returns.
    """
    from factoriax.config import load_config

    config = initial_config if initial_config is not None else load_config()
    # Ensure env_params has every key the editor displays.
    defaults_dict = env_params_to_dict(EnvParams())
    for key, default in defaults_dict.items():
        config.env_params.setdefault(key, default)

    kb_lookup = build_key_lookup(config.keyboard or default_keyboard())
    ctrl_lookup = build_controller_lookup(config.controller or default_controller())

    s = _theme.UI_SCALE
    canvas = ScaledCanvas(1024, s, screen)
    clock = pygame.time.Clock()

    header_font = get_pixel_font(_BASE_TITLE_FONT * s)
    panel_title_font = get_pixel_font(_theme.FONT_BODY)
    body_font = get_pixel_font(_theme.FONT_BODY)
    list_font = get_pixel_font(_theme.FONT_BODY)
    setting_font = get_pixel_font(_theme.FONT_BODY)
    hint_font = get_pixel_font(_theme.FONT_HINT)
    btn_font = get_pixel_font(_theme.FONT_HINT)

    header_surf = header_font.render("PLAY", False, _theme.TEXT_COLOR)
    header_h = header_surf.get_height()
    hint_h = hint_font.get_height()

    left_idx = 0  # index into _PAGE_OPTIONS
    focus_right = False  # only meaningful when left_idx selects "settings"
    settings_idx = 0  # index into _SETTING_FIELDS + randomize button
    reset_flash_until = 0
    # Persist the user's choice on exit; Play uses this to launch.
    saved = False

    def _save_and_return() -> PlayerConfig:
        nonlocal saved
        if not saved:
            save_config(config)
            # config_to_env_params validates types; call it to surface errors early.
            config_to_env_params(config)
            saved = True
        return config

    def _adjust_setting(field: _SettingField, direction: int) -> None:
        if direction == 0:
            return
        current = _get_value(config, field)
        new_value = _clamp(current + field.step * direction, field)
        _set_value(config, field, new_value)

    def _randomize_seed() -> None:
        config.seed = int.from_bytes(os.urandom(4), "little")

    while True:
        sw, sh = canvas.width, canvas.height
        header_y = _theme.PAGE_PAD
        panels_top = header_y + header_h + _theme.PANEL_TITLE_GAP * 2
        hint_y = sh - _theme.PAGE_PAD - hint_h
        panels_bottom = hint_y - _theme.PANEL_TITLE_GAP * 2
        panels_h = panels_bottom - panels_top

        panels_x = _theme.PAGE_PAD
        panels_w = sw - 2 * _theme.PAGE_PAD
        list_w = (panels_w - _theme.PANEL_GUTTER) // 4
        desc_w = panels_w - _theme.PANEL_GUTTER - list_w
        list_rect = pygame.Rect(panels_x, panels_top, list_w, panels_h)
        desc_rect = pygame.Rect(
            panels_x + list_w + _theme.PANEL_GUTTER, panels_top, desc_w, panels_h
        )
        left_row_rects = panels.list_row_rects(list_rect, len(_PAGE_OPTIONS))

        selected_option = _PAGE_OPTIONS[left_idx]
        in_settings = selected_option.action == "settings"
        randomize_rect = (
            _randomize_button_rect(desc_rect, s)
            if in_settings
            else pygame.Rect(0, 0, 0, 0)
        )

        # ----- Events -----------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return _save_and_return()
            if event.type == pygame.VIDEORESIZE:
                canvas.handle_resize(event.w, event.h)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = canvas.to_canvas(*event.pos)
                for i, rect in enumerate(left_row_rects):
                    if rect.collidepoint(mx, my):
                        left_idx = i
                        focus_right = False
                if in_settings:
                    for i in range(len(_SETTING_FIELDS)):
                        y = _row_y(desc_rect, i)
                        row_rect = pygame.Rect(
                            desc_rect.left + _theme.ROW_PAD,
                            y,
                            desc_rect.width - 2 * _theme.ROW_PAD,
                            _theme.ROW_H,
                        )
                        if row_rect.collidepoint(mx, my):
                            focus_right = True
                            settings_idx = i
                    if randomize_rect.collidepoint(mx, my):
                        focus_right = True
                        settings_idx = _RANDOMIZE_IDX
                        _randomize_seed()
            actions = resolve_event(event, kb_lookup, ctrl_lookup)
            if PlayerAction.QUIT in actions:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
                return _save_and_return()
            if PlayerAction.BACK in actions:
                if focus_right:
                    focus_right = False
                else:
                    return _save_and_return()
                continue
            if PlayerAction.CONFIRM in actions:
                if focus_right:
                    if settings_idx == _RANDOMIZE_IDX:
                        _randomize_seed()
                    else:
                        focus_right = False
                else:
                    action = selected_option.action
                    if action == "play":
                        return _save_and_return()
                    if action == "settings":
                        focus_right = True
                        settings_idx = 0
                    elif action == "reset":
                        _reset_to_defaults(config)
                        reset_flash_until = pygame.time.get_ticks() + _RESET_FLASH_MS
                continue
            if focus_right:
                if PlayerAction.NAV_DOWN in actions:
                    settings_idx = (settings_idx + 1) % _NUM_NAV_ITEMS
                elif PlayerAction.NAV_UP in actions:
                    settings_idx = (settings_idx - 1) % _NUM_NAV_ITEMS
                elif PlayerAction.NAV_RIGHT in actions:
                    if settings_idx == _RANDOMIZE_IDX:
                        _randomize_seed()
                    else:
                        _adjust_setting(_SETTING_FIELDS[settings_idx], 1)
                elif PlayerAction.NAV_LEFT in actions:
                    if settings_idx == _RANDOMIZE_IDX:
                        _randomize_seed()
                    else:
                        _adjust_setting(_SETTING_FIELDS[settings_idx], -1)
            else:
                if PlayerAction.NAV_DOWN in actions:
                    left_idx = (left_idx + 1) % len(_PAGE_OPTIONS)
                elif PlayerAction.NAV_UP in actions:
                    left_idx = (left_idx - 1) % len(_PAGE_OPTIONS)

        # ----- Mouse hover for the left list ------------------------------
        mx, my = canvas.to_canvas(*pygame.mouse.get_pos())
        if not focus_right:
            hovered_left = next(
                (
                    i
                    for i, rect in enumerate(left_row_rects)
                    if rect.collidepoint(mx, my)
                ),
                None,
            )
            display_left_idx = hovered_left if hovered_left is not None else left_idx
        else:
            display_left_idx = left_idx

        # ----- Draw -------------------------------------------------------
        surf = canvas.surface
        surf.fill(_theme.PAGE_BG)
        surf.blit(header_surf, ((sw - header_surf.get_width()) // 2, header_y))

        panels.draw_panel(
            surf,
            list_rect,
            "Menu",
            panel_title_font,
            focused=not focus_right,
        )
        right_title = "Settings" if in_settings else "Description"
        panels.draw_panel(
            surf,
            desc_rect,
            right_title,
            panel_title_font,
            focused=focus_right,
        )
        panels.draw_list_rows(
            surf,
            list_rect,
            [o.label for o in _PAGE_OPTIONS],
            display_left_idx,
            list_font,
        )

        if in_settings:
            _draw_setting_rows(
                surf,
                desc_rect,
                config,
                settings_idx,
                focus_right,
                setting_font,
            )
            _draw_randomize_button(
                surf,
                randomize_rect,
                btn_font,
                active=focus_right and settings_idx == _RANDOMIZE_IDX,
            )
        else:
            text = selected_option.description
            if (
                selected_option.action == "reset"
                and pygame.time.get_ticks() < reset_flash_until
            ):
                text = "Defaults restored.\n\n" + text
            panels.draw_description(surf, desc_rect, text, 0, body_font)

        if focus_right:
            hint = "Up/Down field   Left/Right adjust   Enter done   Backspace done"
        else:
            hint = "Up/Down select   Enter confirm   Backspace back   Escape quit"
        panels.draw_hint_bar(surf, sw, hint_y, hint, hint_font)

        canvas.present(screen)
        clock.tick(_FPS)


# ---------------------------------------------------------------------------
# UI-scale confirmation dialog (consumed by controls_menu)
# ---------------------------------------------------------------------------


_CONFIRM_TIMEOUT_MS: int = 10_000


def _confirm_scale_change(
    screen: pygame.Surface,
    new_scale_value: int,
) -> bool:
    """Apply a new UI scale and wait for user confirmation.

    Shows a centered dialog with a countdown timer. The new scale is applied
    immediately so the user can see the result. If the user presses Enter or
    clicks "Keep", the change is confirmed. If the timer expires or Escape
    is pressed, the change is reverted.

    Args:
        screen: Pygame display surface.
        new_scale_value: The ``ui_scale`` config value to try (0-3).

    Returns:
        ``True`` if the user confirmed the new scale, ``False`` if reverted.
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
            kb = build_key_lookup(default_keyboard())
            cl = build_controller_lookup(default_controller())
            nav = resolve_event(event, kb, cl)
            if PlayerAction.CONFIRM in nav:
                return True
            if PlayerAction.BACK in nav:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = canvas.to_canvas(*event.pos)
                if pygame.Rect(keep_x, btn_y, btn_w, btn_h).collidepoint(mx, my):
                    return True
                if pygame.Rect(revert_x, btn_y, btn_w, btn_h).collidepoint(mx, my):
                    return False

        surf = canvas.surface
        surf.fill((15, 15, 20))
        pygame.draw.rect(surf, (30, 30, 35), (box_x, box_y, box_w, box_h))
        pygame.draw.rect(surf, GOLD, (box_x, box_y, box_w, box_h), 3)

        secs = (remaining + 999) // 1000
        msg = font.render(
            f"Keep this scale? Reverting in {secs}s...",
            False,
            LABEL_COLOR,
        )
        surf.blit(msg, (box_x + (box_w - msg.get_width()) // 2, box_y + 20 * new_s))

        mouse_pos = canvas.to_canvas(*pygame.mouse.get_pos())
        for label, rect in (
            ("Keep", pygame.Rect(keep_x, btn_y, btn_w, btn_h)),
            ("Revert", pygame.Rect(revert_x, btn_y, btn_w, btn_h)),
        ):
            hover = rect.collidepoint(mouse_pos)
            pygame.draw.rect(surf, (55, 55, 60) if hover else (35, 35, 40), rect)
            color = GOLD if label == "Keep" else (120, 60, 60)
            pygame.draw.rect(surf, color, rect, 2)
            text = font.render(label, False, LABEL_COLOR)
            surf.blit(
                text,
                (
                    rect.x + (rect.w - text.get_width()) // 2,
                    rect.y + (rect.h - text.get_height()) // 2,
                ),
            )

        canvas.present(screen)
        clock.tick(30)
