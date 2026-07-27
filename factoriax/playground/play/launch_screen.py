"""Pre-game launch screen — Play / Settings / Reset to defaults.

The screen mirrors :mod:`factoriax.playground.menu.main_menu`: a list of options on
the left, content on the right. Selecting *Settings* on the left turns the
right panel into an editable list of :class:`~factoriax.engine.state.EnvParams`
fields; selecting *Play* commits the current values and returns control to
the play loop; selecting *Reset to defaults* replaces every field with
:meth:`EnvParams()` defaults and a fresh seed.

:func:`_confirm_scale_change` is unrelated to the launch flow; it lives here
because :mod:`factoriax.playground.menu.controls_menu` imports it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pygame

from factoriax.engine.state import EnvParams
from factoriax.playground.config import (
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
from factoriax.playground.ui import panels
from factoriax.playground.ui import theme as _theme
from factoriax.playground.ui.fonts import get_pixel_font
from factoriax.playground.ui.forms import GOLD, LABEL_COLOR
from factoriax.playground.ui.panels import SettingField, SettingSection
from factoriax.playground.ui.scaling import ScaledCanvas
from factoriax.playground.ui.window import auto_ui_scale, calculate_window_size

_BASE_TITLE_FONT: int = 48
_BASE_BUTTON_W: int = 110
_BASE_BUTTON_H: int = 26
_FPS: int = 30
_RESET_FLASH_MS: int = 1500
_SEED_MAX: int = 2**32 - 1


_SECTIONS: tuple[SettingSection, ...] = (
    SettingSection(
        "World",
        (
            SettingField("max_timesteps", "Max steps", False, 100, 100, 10_000),
        ),
    ),
    SettingSection(
        "Resources",
        (
            SettingField("water_probability", "Water prob.", True, 0.01, 0.0, 1.0),
            SettingField("iron_probability", "Iron prob.", True, 0.01, 0.0, 1.0),
            SettingField("copper_probability", "Copper prob.", True, 0.01, 0.0, 1.0),
            SettingField("coal_probability", "Coal prob.", True, 0.01, 0.0, 1.0),
            SettingField("tin_probability", "Tin prob.", True, 0.01, 0.0, 1.0),
            SettingField("silicon_probability", "Silicon prob.", True, 0.01, 0.0, 1.0),
            SettingField("base_resources", "Base resources", False, 100, 100, 10_000),
        ),
    ),
    SettingSection(
        "Machines",
        (
            SettingField("miner_mining_rate", "Miner mining rate", False, 1, 1, 20),
            SettingField("player_mining_yield", "Player mining yield", False, 1, 1, 10),
        ),
    ),
    SettingSection(
        "Seed",
        (SettingField("seed", "Seed", False, 1, 0, _SEED_MAX),),
    ),
)

_SETTING_FIELDS: tuple[SettingField, ...] = tuple(
    f for sec in _SECTIONS for f in sec.fields
)


@dataclass(frozen=True)
class _PageOption:
    """ """
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
        label="Reset Settings",
        description=(
            "Replace every setting with the engine's default value and "
            "pick a fresh random seed."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Value get / set / adjust
# ---------------------------------------------------------------------------


def _get_value(config: PlayerConfig, field: SettingField) -> float:
    """

    Parameters
    ----------
    config: PlayerConfig :
        
    field: SettingField :
        

    Returns
    -------

    """
    if field.key == "seed":
        return float(config.seed)
    raw = config.env_params.get(field.key)
    if raw is None:
        raw = getattr(EnvParams(), field.key)
    return float(raw)


def _set_value(config: PlayerConfig, field: SettingField, value: float) -> None:
    """

    Parameters
    ----------
    config: PlayerConfig :
        
    field: SettingField :
        
    value: float :
        

    Returns
    -------

    """
    if field.key == "seed":
        config.seed = int(value)
        return
    config.env_params[field.key] = float(value) if field.is_float else int(value)


def _reset_to_defaults(config: PlayerConfig) -> None:
    """

    Parameters
    ----------
    config: PlayerConfig :
        

    Returns
    -------

    """
    config.env_params = env_params_to_dict(EnvParams())
    config.seed = int.from_bytes(os.urandom(4), "little")


def _randomize_button_rect(rect: pygame.Rect, scale: int, bottom_y: int) -> pygame.Rect:
    """Compact button right-aligned under the rendered settings block.

    Parameters
    ----------
    rect: pygame.Rect :
        
    scale: int :
        
    bottom_y: int :
        

    Returns
    -------

    """
    btn_w = _BASE_BUTTON_W * scale
    btn_h = _BASE_BUTTON_H * scale
    btn_x = rect.right - _theme.ROW_PAD - btn_w
    return pygame.Rect(btn_x, bottom_y + 8, btn_w, btn_h)


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------


_NUM_NAV_ITEMS: int = len(_SETTING_FIELDS) + 1  # fields + randomize button
_RANDOMIZE_IDX: int = len(_SETTING_FIELDS)


def run_settings_menu(
    screen: pygame.Surface,
    initial_config: PlayerConfig | None = None,
) -> PlayerConfig | None:
    """Show the launch screen and return the user's choice.

    Parameters
    ----------
    screen: pygame.Surface :
        
    initial_config: PlayerConfig | None :
         (Default value = None)

    Returns
    -------
    The
        class:`PlayerConfig` if the user pressed Enter on *Play* (the
    The
        class:`PlayerConfig` if the user pressed Enter on *Play* (the
        caller should hand it to the play loop), or ``None`` if the user
    The
        class:`PlayerConfig` if the user pressed Enter on *Play* (the
        caller should hand it to the play loop), or ``None`` if the user
        backed out via Backspace (return to main menu). Edits are persisted
    via
        func:`factoriax.playground.config.save_config` before the function returns
    via
        func:`factoriax.playground.config.save_config` before the function returns
        in either case.

    """
    from factoriax.playground.config import load_config

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
    input_source = panels.InputSourceTracker()
    # Persist the user's choice on exit; Play uses this to launch.
    saved = False

    def _persist() -> None:
        """ """
        nonlocal saved
        if not saved:
            save_config(config)
            # config_to_env_params validates types; call it to surface errors early.
            config_to_env_params(config)
            saved = True

    def _launch() -> PlayerConfig:
        """ """
        _persist()
        return config

    def _cancel() -> None:
        """ """
        _persist()
        return None

    def _adjust_setting(field: SettingField, direction: int) -> None:
        """

        Parameters
        ----------
        field: SettingField :
            
        direction: int :
            

        Returns
        -------

        """
        if direction == 0:
            return
        current = _get_value(config, field)
        _set_value(config, field, field.clamp(current + field.step * direction))

    def _randomize_seed() -> None:
        """ """
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
        heading_font = get_pixel_font(_theme.FONT_HINT)
        if in_settings:
            setting_rects, _, settings_bottom_y = panels.setting_section_layout(
                desc_rect, _SECTIONS, heading_font.get_height()
            )
            randomize_rect = _randomize_button_rect(desc_rect, s, settings_bottom_y)
        else:
            setting_rects = []
            randomize_rect = pygame.Rect(0, 0, 0, 0)

        # ----- Events -----------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                _cancel()
                return None
            if event.type == pygame.VIDEORESIZE:
                canvas.handle_resize(event.w, event.h)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = canvas.to_canvas(*event.pos)
                for i, rect in enumerate(left_row_rects):
                    if rect.collidepoint(mx, my):
                        left_idx = i
                        focus_right = False
                if in_settings:
                    for i, row_rect in enumerate(setting_rects):
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
                _cancel()
                return None
            if PlayerAction.BACK in actions:
                if focus_right:
                    focus_right = False
                else:
                    _cancel()
                    return None
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
                        return _launch()
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
                    input_source.mark_keyboard()
                elif PlayerAction.NAV_UP in actions:
                    left_idx = (left_idx - 1) % len(_PAGE_OPTIONS)
                    input_source.mark_keyboard()

        # Hover commits left_idx — mouse only wins when it's the most recent
        # input, and only when focus is on the left panel.
        input_source.tick()
        if input_source.mouse_active and not focus_right:
            mx, my = canvas.to_canvas(*pygame.mouse.get_pos())
            hovered_left = next(
                (
                    i
                    for i, rect in enumerate(left_row_rects)
                    if rect.collidepoint(mx, my)
                ),
                None,
            )
            if hovered_left is not None:
                left_idx = hovered_left

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
            left_idx,
            list_font,
        )

        if in_settings:
            current_values = [_get_value(config, f) for f in _SETTING_FIELDS]
            panels.draw_setting_sections(
                surf,
                desc_rect,
                _SECTIONS,
                current_values,
                settings_idx,
                focus_right,
                setting_font,
                heading_font,
            )
            panels.draw_button(
                surf,
                randomize_rect,
                "Randomize",
                btn_font,
                focused=focus_right and settings_idx == _RANDOMIZE_IDX,
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


def _apply_display_state(fullscreen: bool, ui_scale: int) -> pygame.Surface:
    """Reapply theme scale and recreate the pygame display surface.

    Parameters
    ----------
    fullscreen: bool :
        
    ui_scale: int :
        

    Returns
    -------

    """
    applied = ui_scale if ui_scale > 0 else auto_ui_scale()
    _theme.apply_scale(applied)
    if fullscreen:
        return pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    canvas_size = 1024 * applied
    w, h = calculate_window_size(canvas_size, canvas_size)
    return pygame.display.set_mode((w, h))


def _confirm_display_change(
    old_fullscreen: bool,
    old_ui_scale: int,
    new_fullscreen: bool,
    new_ui_scale: int,
) -> bool:
    """Apply the new (fullscreen, ui_scale) state and prompt to keep or revert.
    
    Shows a centered Keep / Revert dialog with a countdown. On revert (or
    timeout, Escape, or Backspace), restores the previous display state and
    returns ``False``. On confirm (Enter / Keep), returns ``True`` and leaves
    the new state applied.

    Parameters
    ----------
    old_fullscreen: bool :
        
    old_ui_scale: int :
        
    new_fullscreen: bool :
        
    new_ui_scale: int :
        

    Returns
    -------

    """
    screen = _apply_display_state(new_fullscreen, new_ui_scale)
    applied = new_ui_scale if new_ui_scale > 0 else auto_ui_scale()
    canvas = ScaledCanvas(1024, applied, screen)
    clock = pygame.time.Clock()
    deadline = pygame.time.get_ticks() + _CONFIRM_TIMEOUT_MS

    font = get_pixel_font(28 * applied)
    sw, sh = canvas.width, canvas.height
    box_w = 500 * applied
    box_h = 140 * applied
    box_x = (sw - box_w) // 2
    box_y = (sh - box_h) // 2
    btn_w = 140 * applied
    btn_h = 50 * applied
    btn_gap = 24 * applied
    keep_x = box_x + (box_w - 2 * btn_w - btn_gap) // 2
    revert_x = keep_x + btn_w + btn_gap
    btn_y = box_y + box_h - btn_h - 16 * applied

    def _finish(confirmed: bool) -> bool:
        """

        Parameters
        ----------
        confirmed: bool :
            

        Returns
        -------

        """
        if not confirmed:
            _apply_display_state(old_fullscreen, old_ui_scale)
        return confirmed

    kb = build_key_lookup(default_keyboard())
    cl = build_controller_lookup(default_controller())

    while True:
        remaining = max(0, deadline - pygame.time.get_ticks())
        if remaining == 0:
            return _finish(False)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return _finish(False)
            if event.type == pygame.VIDEORESIZE:
                canvas.handle_resize(event.w, event.h)
            nav = resolve_event(event, kb, cl)
            if PlayerAction.CONFIRM in nav:
                return _finish(True)
            if PlayerAction.BACK in nav or PlayerAction.QUIT in nav:
                return _finish(False)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = canvas.to_canvas(*event.pos)
                if pygame.Rect(keep_x, btn_y, btn_w, btn_h).collidepoint(mx, my):
                    return _finish(True)
                if pygame.Rect(revert_x, btn_y, btn_w, btn_h).collidepoint(mx, my):
                    return _finish(False)

        surf = canvas.surface
        surf.fill((15, 15, 20))
        pygame.draw.rect(surf, (30, 30, 35), (box_x, box_y, box_w, box_h))
        pygame.draw.rect(surf, GOLD, (box_x, box_y, box_w, box_h), 3)

        secs = (remaining + 999) // 1000
        msg = font.render(
            f"Keep these display settings? Reverting in {secs}s...",
            False,
            LABEL_COLOR,
        )
        surf.blit(msg, (box_x + (box_w - msg.get_width()) // 2, box_y + 20 * applied))

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
