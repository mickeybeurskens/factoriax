"""Settings menu — keyboard, controller, display, and reset bindings.

Tv-style two-panel layout mirroring :mod:`factoriax.menu.scenarios_menu`:
left panel lists the page (Keyboard, Controller, Display, Reset Bindings);
right panel shows the content of the selected page. Bindings are
group-headed (Movement / Actions / Menus); Display has fullscreen and UI
scale rows; Reset is an action.
"""

from __future__ import annotations

from dataclasses import dataclass

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
from factoriax.play.launch_screen import _confirm_scale_change
from factoriax.ui import panels
from factoriax.ui import theme as _theme
from factoriax.ui.fonts import get_pixel_font
from factoriax.ui.panels import LabelValueSection
from factoriax.ui.scaling import ScaledCanvas
from factoriax.ui.window import auto_ui_scale

_BASE_TITLE_FONT: int = 48
_FPS: int = 30
_UI_SCALE_LABELS: tuple[str, ...] = ("Auto", "1x", "2x", "3x")


# ---------------------------------------------------------------------------
# Page options + binding categories
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PageOption:
    action: str
    label: str
    description: str


_PAGE_OPTIONS: tuple[_PageOption, ...] = (
    _PageOption(
        action="keyboard",
        label="Keyboard",
        description=(
            "Rebind keyboard controls. Highlight a row and press Enter to "
            "listen for the next key; press the key you want to bind."
        ),
    ),
    _PageOption(
        action="controller",
        label="Controller",
        description=(
            "Rebind controller controls. Highlight a row and press Enter to "
            "listen for the next input; press the button or axis you want to "
            "bind."
        ),
    ),
    _PageOption(
        action="display",
        label="Display",
        description=(
            "Toggle fullscreen mode and choose the UI scale. Scale changes "
            "request confirmation before they stick."
        ),
    ),
    _PageOption(
        action="reset",
        label="Reset Bindings",
        description=(
            "Reset every keyboard and controller binding back to the engine's "
            "defaults. Display settings are not affected."
        ),
    ),
)


@dataclass(frozen=True)
class _BindingRow:
    action: PlayerAction
    label: str


@dataclass(frozen=True)
class _BindingCategory:
    title: str
    rows: tuple[_BindingRow, ...]


_BINDING_CATEGORIES: tuple[_BindingCategory, ...] = (
    _BindingCategory(
        "Movement",
        (
            _BindingRow(PlayerAction.MOVE_UP, "Move Up"),
            _BindingRow(PlayerAction.MOVE_DOWN, "Move Down"),
            _BindingRow(PlayerAction.MOVE_LEFT, "Move Left"),
            _BindingRow(PlayerAction.MOVE_RIGHT, "Move Right"),
        ),
    ),
    _BindingCategory(
        "Actions",
        (
            _BindingRow(PlayerAction.MINE, "Mine"),
            _BindingRow(PlayerAction.INTERACT, "Interact"),
            _BindingRow(PlayerAction.ROTATE, "Rotate"),
        ),
    ),
    _BindingCategory(
        "Menus",
        (
            _BindingRow(PlayerAction.OPEN_INVENTORY, "Inventory"),
            _BindingRow(PlayerAction.OPEN_ACHIEVEMENTS, "Achievements"),
            _BindingRow(PlayerAction.OPEN_MACHINE, "Inspect Machine"),
            _BindingRow(PlayerAction.OPEN_HELP, "Help"),
            _BindingRow(PlayerAction.TOGGLE_HOTBAR, "Hotbar Page"),
            _BindingRow(PlayerAction.CONFIRM, "Confirm"),
            _BindingRow(PlayerAction.BACK, "Back"),
            _BindingRow(PlayerAction.QUIT, "Quit"),
        ),
    ),
)

_FLAT_BINDING_ROWS: tuple[_BindingRow, ...] = tuple(
    r for cat in _BINDING_CATEGORIES for r in cat.rows
)

_BINDING_SECTIONS: tuple[LabelValueSection, ...] = tuple(
    LabelValueSection(cat.title, tuple(r.label for r in cat.rows))
    for cat in _BINDING_CATEGORIES
)


def _init_rebind_actions() -> list[tuple[str | None, list[tuple[str, str]]]]:
    """Return the binding categories as ``[(title, [(action, label)])]``.

    Provided for :mod:`tests.play.test_rebinding`, which validates that every
    listed action is a real :class:`PlayerAction` with a default keyboard +
    controller binding.
    """
    return [
        (cat.title, [(row.action, row.label) for row in cat.rows])
        for cat in _BINDING_CATEGORIES
    ]


# ---------------------------------------------------------------------------
# Binding-name formatting (unchanged from the previous design)
# ---------------------------------------------------------------------------


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
    if not names:
        return "-"
    parts: list[str] = []
    for n in names:
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
    if not names:
        return "-"
    return ", ".join(_CONTROLLER_DISPLAY.get(n, n) for n in names)


def _format_binding(names: list[str], device: str) -> str:
    if device == "keyboard":
        return _format_key_display(names)
    return _format_controller_display(names)


# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------


_DISPLAY_SECTIONS: tuple[LabelValueSection, ...] = (
    LabelValueSection("Display", ("Fullscreen", "UI scale")),
)


def _format_display_values(fullscreen: bool, ui_scale: int) -> list[str]:
    return [
        "On" if fullscreen else "Off",
        _UI_SCALE_LABELS[ui_scale] if 0 <= ui_scale < len(_UI_SCALE_LABELS) else "?",
    ]


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------


def run_controls_menu(
    screen: pygame.Surface,
    config: PlayerConfig,
) -> tuple[bool, int]:
    """Run the settings menu and return ``(fullscreen, ui_scale)``.

    Binding edits are written into ``config.keyboard`` / ``config.controller``
    in place; the caller is responsible for persistence.
    """
    from factoriax.config import controller_event_to_name, event_to_key_name

    kb_lookup = build_key_lookup(config.keyboard or default_keyboard())
    ctrl_lookup = build_controller_lookup(config.controller or default_controller())

    s = _theme.UI_SCALE
    canvas = ScaledCanvas(1024, s, screen)
    clock = pygame.time.Clock()

    header_font = get_pixel_font(_BASE_TITLE_FONT * s)
    panel_title_font = get_pixel_font(_theme.FONT_BODY)
    body_font = get_pixel_font(_theme.FONT_BODY)
    list_font = get_pixel_font(_theme.FONT_BODY)
    row_font = get_pixel_font(_theme.FONT_BODY)
    heading_font = get_pixel_font(_theme.FONT_HINT)
    hint_font = get_pixel_font(_theme.FONT_HINT)

    header_surf = header_font.render("SETTINGS", False, _theme.TEXT_COLOR)
    header_h = header_surf.get_height()
    hint_h = hint_font.get_height()

    fullscreen = config.fullscreen
    ui_scale = config.ui_scale
    left_idx = 0
    focus_right = False
    right_idx = 0
    listening_action: PlayerAction | None = None
    reset_flash_until = 0

    def _bindings_for(device: str) -> dict[str, list[str]]:
        return config.keyboard if device == "keyboard" else config.controller

    def _binding_values(device: str) -> list[str]:
        bindings = _bindings_for(device)
        out: list[str] = []
        for i, row in enumerate(_FLAT_BINDING_ROWS):
            names = bindings.get(row.action, [])
            if listening_action is not None and i == right_idx:
                out.append("Press a key...")
            else:
                out.append(_format_binding(names, device))
        return out

    def _reset_bindings() -> None:
        config.keyboard = default_keyboard()
        config.controller = default_controller()

    def _cycle_ui_scale(direction: int) -> int:
        return (ui_scale + direction) % len(_UI_SCALE_LABELS)

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
        page = selected_option.action
        if page in {"keyboard", "controller"}:
            n_right_items = len(_FLAT_BINDING_ROWS)
        elif page == "display":
            n_right_items = 2
        else:
            n_right_items = 0
        right_idx = max(0, min(right_idx, max(0, n_right_items - 1)))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return fullscreen, ui_scale
            if event.type == pygame.VIDEORESIZE:
                canvas.handle_resize(event.w, event.h)

            # --- Listening: capture the next input -------------------
            if listening_action is not None and page in {"keyboard", "controller"}:
                if page == "keyboard" and event.type == pygame.KEYDOWN:
                    name = event_to_key_name(event.key, pygame.key.get_mods())
                    config.keyboard[listening_action] = [name]
                    listening_action = None
                    kb_lookup = build_key_lookup(config.keyboard)
                    continue
                if page == "controller" and event.type in (
                    pygame.JOYBUTTONDOWN,
                    pygame.JOYHATMOTION,
                    pygame.JOYAXISMOTION,
                ):
                    ctrl_name = controller_event_to_name(event)
                    if ctrl_name is not None:
                        config.controller[listening_action] = [ctrl_name]
                        listening_action = None
                        ctrl_lookup = build_controller_lookup(config.controller)
                    continue
                # Fall through: still listening, ignore other events.
                continue

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = canvas.to_canvas(*event.pos)
                for i, rect in enumerate(left_row_rects):
                    if rect.collidepoint(mx, my):
                        left_idx = i
                        focus_right = False

            actions = resolve_event(event, kb_lookup, ctrl_lookup)
            if PlayerAction.QUIT in actions:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
                return fullscreen, ui_scale
            if PlayerAction.BACK in actions:
                if focus_right:
                    focus_right = False
                else:
                    return fullscreen, ui_scale
                continue
            if PlayerAction.CONFIRM in actions:
                if focus_right:
                    if page in {"keyboard", "controller"}:
                        listening_action = _FLAT_BINDING_ROWS[right_idx].action
                    elif page == "display":
                        if right_idx == 0:
                            fullscreen = not fullscreen
                        elif right_idx == 1:
                            new_scale = _cycle_ui_scale(1)
                            if new_scale != ui_scale:
                                confirmed = _confirm_scale_change(screen, new_scale)
                                if confirmed:
                                    ui_scale = new_scale
                                applied = ui_scale if ui_scale > 0 else auto_ui_scale()
                                _theme.apply_scale(applied)
                                return fullscreen, ui_scale
                else:
                    if page == "reset":
                        _reset_bindings()
                        kb_lookup = build_key_lookup(config.keyboard)
                        ctrl_lookup = build_controller_lookup(config.controller)
                        reset_flash_until = pygame.time.get_ticks() + 1500
                    else:
                        focus_right = True
                        right_idx = 0
                continue
            if focus_right:
                if PlayerAction.NAV_DOWN in actions and n_right_items > 0:
                    right_idx = (right_idx + 1) % n_right_items
                elif PlayerAction.NAV_UP in actions and n_right_items > 0:
                    right_idx = (right_idx - 1) % n_right_items
                elif page == "display":
                    if PlayerAction.NAV_RIGHT in actions:
                        if right_idx == 0:
                            fullscreen = not fullscreen
                        elif right_idx == 1:
                            new_scale = _cycle_ui_scale(1)
                            if new_scale != ui_scale:
                                confirmed = _confirm_scale_change(screen, new_scale)
                                if confirmed:
                                    ui_scale = new_scale
                                applied = ui_scale if ui_scale > 0 else auto_ui_scale()
                                _theme.apply_scale(applied)
                                return fullscreen, ui_scale
                    elif PlayerAction.NAV_LEFT in actions:
                        if right_idx == 0:
                            fullscreen = not fullscreen
                        elif right_idx == 1:
                            new_scale = _cycle_ui_scale(-1)
                            if new_scale != ui_scale:
                                confirmed = _confirm_scale_change(screen, new_scale)
                                if confirmed:
                                    ui_scale = new_scale
                                applied = ui_scale if ui_scale > 0 else auto_ui_scale()
                                _theme.apply_scale(applied)
                                return fullscreen, ui_scale
            else:
                if PlayerAction.NAV_DOWN in actions:
                    left_idx = (left_idx + 1) % len(_PAGE_OPTIONS)
                elif PlayerAction.NAV_UP in actions:
                    left_idx = (left_idx - 1) % len(_PAGE_OPTIONS)

        # ----- Mouse hover for the left list -----------------------------
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

        # ----- Draw ------------------------------------------------------
        surf = canvas.surface
        surf.fill(_theme.PAGE_BG)
        surf.blit(header_surf, ((sw - header_surf.get_width()) // 2, header_y))

        right_title_map = {
            "keyboard": "Keyboard",
            "controller": "Controller",
            "display": "Display",
            "reset": "Reset Bindings",
        }
        panels.draw_panel(
            surf, list_rect, "Menu", panel_title_font, focused=not focus_right
        )
        panels.draw_panel(
            surf,
            desc_rect,
            right_title_map[page],
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

        if page in {"keyboard", "controller"}:
            panels.draw_label_value_sections(
                surf,
                desc_rect,
                _BINDING_SECTIONS,
                _binding_values(page),
                right_idx,
                focus_right,
                row_font,
                heading_font,
            )
        elif page == "display":
            panels.draw_label_value_sections(
                surf,
                desc_rect,
                _DISPLAY_SECTIONS,
                _format_display_values(fullscreen, ui_scale),
                right_idx,
                focus_right,
                row_font,
                heading_font,
            )
        else:  # reset
            text = selected_option.description
            if pygame.time.get_ticks() < reset_flash_until:
                text = "Bindings reset to defaults.\n\n" + text
            panels.draw_description(surf, desc_rect, text, 0, body_font)

        # Contextual hint bar.
        if listening_action is not None:
            hint = "Press a key to bind...   Backspace cancel"
        elif focus_right and page in {"keyboard", "controller"}:
            hint = "Up/Down field   Enter rebind   Backspace back"
        elif focus_right and page == "display":
            hint = "Up/Down field   Left/Right adjust   Backspace back"
        else:
            hint = "Up/Down select   Enter confirm   Backspace back   Escape quit"
        panels.draw_hint_bar(surf, sw, hint_y, hint, hint_font)

        canvas.present(screen)
        clock.tick(_FPS)
