"""Scenario picker — two-panel list-and-description view."""

from __future__ import annotations

import inspect
import sys
import textwrap
from dataclasses import dataclass

import pygame

from factoriax import scenarios as scenarios_pkg
from factoriax.config import (
    ControllerLookup,
    KeyLookup,
    PlayerAction,
    resolve_event,
)
from factoriax.scenarios import Scenario
from factoriax.ui import panels
from factoriax.ui import theme as _theme
from factoriax.ui.fonts import get_pixel_font
from factoriax.ui.scaling import ScaledCanvas

_BASE_TITLE_FONT: int = 48
_FPS: int = 30


@dataclass(frozen=True)
class _ScenarioEntry:
    class_name: str
    display_name: str
    instance: Scenario
    docstring: str


def discover_scenarios() -> list[_ScenarioEntry]:
    """Walk :mod:`factoriax.scenarios` for classes implementing ``Scenario``.

    Returns one entry per discovered scenario, sorted by display name.
    Classes that fail to instantiate with no arguments are skipped.
    """
    out: list[_ScenarioEntry] = []
    seen_classes: set[type] = set()
    for name in scenarios_pkg.__all__:
        obj = getattr(scenarios_pkg, name, None)
        if not inspect.isclass(obj) or obj in seen_classes:
            continue
        seen_classes.add(obj)
        try:
            instance = obj()
        except Exception:
            continue
        if not isinstance(instance, Scenario):
            continue
        module = sys.modules.get(obj.__module__)
        doc = textwrap.dedent((module.__doc__ if module else None) or "").strip()
        display = " ".join(word.capitalize() for word in str(instance.name).split("_"))
        out.append(
            _ScenarioEntry(
                class_name=name,
                display_name=display,
                instance=instance,
                docstring=doc,
            )
        )
    out.sort(key=lambda e: e.display_name)
    return out


def run_scenarios_menu(
    screen: pygame.Surface,
    kb_lookup: KeyLookup | None = None,
    ctrl_lookup: ControllerLookup | None = None,
) -> str | None:
    """Show the scenario picker. Returns the chosen scenario's class name, or None."""
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

    entries = discover_scenarios()
    selected_idx = 0
    focus_list = True
    doc_scroll = 0

    header_surf = header_font.render("SCENARIOS", False, _theme.TEXT_COLOR)
    header_h = header_surf.get_height()
    hint_h = hint_font.get_height()

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
        row_rects = panels.list_row_rects(list_rect, len(entries))

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
                        selected_idx = i
                        focus_list = True
                        doc_scroll = 0
                        break
            actions = resolve_event(event, kb_lookup, ctrl_lookup)
            if PlayerAction.BACK in actions:
                return None
            if PlayerAction.CONFIRM in actions and entries:
                return entries[selected_idx].class_name
            if entries:
                if focus_list:
                    if PlayerAction.NAV_DOWN in actions:
                        selected_idx = (selected_idx + 1) % len(entries)
                        doc_scroll = 0
                    elif PlayerAction.NAV_UP in actions:
                        selected_idx = (selected_idx - 1) % len(entries)
                        doc_scroll = 0
                else:
                    if PlayerAction.NAV_DOWN in actions:
                        doc_scroll += 1
                    elif PlayerAction.NAV_UP in actions:
                        doc_scroll = max(0, doc_scroll - 1)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                focus_list = not focus_list
            if event.type == pygame.MOUSEWHEEL and not focus_list:
                doc_scroll = max(0, doc_scroll - event.y)

        surf = canvas.surface
        surf.fill(_theme.PAGE_BG)
        surf.blit(header_surf, ((sw - header_surf.get_width()) // 2, header_y))

        panels.draw_panel(
            surf, list_rect, "Scenarios", panel_title_font, focused=focus_list
        )
        panels.draw_panel(
            surf, desc_rect, "Description", panel_title_font, focused=not focus_list
        )

        if entries:
            panels.draw_list_rows(
                surf,
                list_rect,
                [e.display_name for e in entries],
                selected_idx,
                list_font,
            )
            doc_scroll = panels.draw_description(
                surf,
                desc_rect,
                entries[selected_idx].docstring,
                doc_scroll,
                body_font,
            )
        else:
            empty_surf = body_font.render(
                "No scenarios found.", False, _theme.HINT_COLOR
            )
            surf.blit(
                empty_surf,
                (list_rect.left + _theme.ROW_PAD, list_rect.top + _theme.ROW_PAD),
            )

        panels.draw_hint_bar(
            surf,
            sw,
            hint_y,
            "Up/Down select   Tab focus desc   Enter play   Esc back",
            hint_font,
        )

        canvas.present(screen)
        clock.tick(_FPS)
