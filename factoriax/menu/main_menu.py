"""Title screen for FactoriaX.

Two-panel layout mirroring :mod:`factoriax.menu.scenarios_menu`: a list of
menu options on the left, an explanation panel on the right.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from factoriax.config import (
    ControllerLookup,
    KeyLookup,
    PlayerAction,
    resolve_event,
)
from factoriax.ui import panels
from factoriax.ui import theme as _theme
from factoriax.ui.fonts import get_pixel_font
from factoriax.ui.scaling import ScaledCanvas

_BASE_TITLE_FONT: int = 48
_BASE_ICON_SIZE: int = 32
_BASE_HEADER_GAP: int = 24
_FPS: int = 30


@dataclass(frozen=True)
class _MenuOption:
    action: str
    label: str
    description: str


_OPTIONS: tuple[_MenuOption, ...] = (
    _MenuOption(
        action="play",
        label="Play",
        description=(
            "Start a new game with custom environment parameters and a "
            "seed of your choice. The world is generated procedurally "
            "from the engine's general terrain sampler — no pre-built "
            "level, no achievement scoring. Use this for free-form "
            "experimentation."
        ),
    ),
    _MenuOption(
        action="scenarios",
        label="Scenarios",
        description=(
            "Pick a research scenario with pre-configured rules, "
            "achievements, and reward signals. Each scenario fixes the "
            "map, the recipe book, and the per-step budget, so runs are "
            "comparable across agents and seeds."
        ),
    ),
    _MenuOption(
        action="editor",
        label="Editor",
        description=(
            "Open the level editor to author custom maps. Place "
            "machines, paint resource patches, set starting positions, "
            "save and load level files. Useful for authoring fixtures "
            "for new scenarios."
        ),
    ),
    _MenuOption(
        action="settings",
        label="Settings",
        description=(
            "Adjust display settings, UI scale, fullscreen mode, and "
            "key / controller bindings. Changes persist to the config "
            "file on disk."
        ),
    ),
    _MenuOption(
        action="quit",
        label="Quit",
        description="Exit FactoriaX.",
    ),
)


def run_main_menu(
    screen: pygame.Surface,
    kb_lookup: KeyLookup | None = None,
    ctrl_lookup: ControllerLookup | None = None,
) -> str | None:
    """Show the main menu and return the user's choice.

    Returns:
        ``"play"``, ``"scenarios"``, ``"editor"``, ``"settings"``, or
        ``None`` (quit / window closed).
    """
    from factoriax.config import (
        build_controller_lookup,
        build_key_lookup,
        default_controller,
        default_keyboard,
    )

    if kb_lookup is None:
        kb_lookup = build_key_lookup(default_keyboard())
    if ctrl_lookup is None:
        ctrl_lookup = build_controller_lookup(default_controller())

    s = _theme.UI_SCALE
    canvas = ScaledCanvas(1024, s, screen)
    clock = pygame.time.Clock()

    header_font = get_pixel_font(_BASE_TITLE_FONT * s)
    panel_title_font = get_pixel_font(_theme.FONT_BODY)
    body_font = get_pixel_font(_theme.FONT_BODY)
    list_font = get_pixel_font(_theme.FONT_BODY)
    hint_font = get_pixel_font(_theme.FONT_HINT)

    header_surf = header_font.render("FACTORIAX", False, _theme.TEXT_COLOR)
    title_w, title_h = header_surf.get_size()
    strip = panels.DecorationStrip.factoriax_default(icon_size=_BASE_ICON_SIZE * s)
    header_gap = _BASE_HEADER_GAP * s
    header_band_h = max(title_h, strip.height)
    hint_h = hint_font.get_height()

    selected_idx = 0

    def _resolve(action: str) -> str | None:
        return None if action == "quit" else action

    while True:
        sw, sh = canvas.width, canvas.height

        header_y = _theme.PAGE_PAD
        block_w = title_w + header_gap + strip.total_width
        block_x = (sw - block_w) // 2
        title_x = block_x
        title_y = header_y + (header_band_h - title_h) // 2
        strip_x = block_x + title_w + header_gap
        strip_y = header_y + (header_band_h - strip.height) // 2

        panels_top = header_y + header_band_h + _theme.PANEL_TITLE_GAP * 2
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
        row_rects = panels.list_row_rects(list_rect, len(_OPTIONS))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.VIDEORESIZE:
                canvas.handle_resize(event.w, event.h)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return None
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = canvas.to_canvas(*event.pos)
                for i, rect in enumerate(row_rects):
                    if rect.collidepoint(mx, my):
                        return _resolve(_OPTIONS[i].action)
            actions = resolve_event(event, kb_lookup, ctrl_lookup)
            if PlayerAction.CONFIRM in actions:
                return _resolve(_OPTIONS[selected_idx].action)
            if PlayerAction.BACK in actions:
                return None
            if PlayerAction.NAV_DOWN in actions:
                selected_idx = (selected_idx + 1) % len(_OPTIONS)
            elif PlayerAction.NAV_UP in actions:
                selected_idx = (selected_idx - 1) % len(_OPTIONS)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                selected_idx = (selected_idx + 1) % len(_OPTIONS)

        # Mouse hover overrides keyboard focus for the highlight.
        mx, my = canvas.to_canvas(*pygame.mouse.get_pos())
        hovered_idx = next(
            (i for i, rect in enumerate(row_rects) if rect.collidepoint(mx, my)),
            None,
        )
        active_idx = hovered_idx if hovered_idx is not None else selected_idx

        surf = canvas.surface
        surf.fill(_theme.PAGE_BG)

        surf.blit(header_surf, (title_x, title_y))
        strip.draw(surf, strip_x, strip_y, pygame.time.get_ticks())

        panels.draw_panel(surf, list_rect, "Menu", panel_title_font, focused=True)
        panels.draw_panel(
            surf, desc_rect, "Description", panel_title_font, focused=False
        )
        panels.draw_list_rows(
            surf, list_rect, [o.label for o in _OPTIONS], active_idx, list_font
        )
        panels.draw_description(
            surf, desc_rect, _OPTIONS[active_idx].description, 0, body_font
        )

        panels.draw_hint_bar(
            surf,
            sw,
            hint_y,
            "Up/Down select   Enter confirm   Esc quit",
            hint_font,
        )

        canvas.present(screen)
        clock.tick(_FPS)
