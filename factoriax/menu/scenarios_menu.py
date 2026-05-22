"""Scenario picker — two-panel list/description view with a tv-style border."""

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
from factoriax.ui import theme as _theme
from factoriax.ui.fonts import get_pixel_font
from factoriax.ui.scaling import ScaledCanvas

_BG_COLOR: tuple[int, int, int] = (20, 20, 25)
_PANEL_BG: tuple[int, int, int] = (28, 28, 32)
_BORDER_ACTIVE: tuple[int, int, int] = _theme.BORDER[:3]
_BORDER_INACTIVE: tuple[int, int, int] = (
    int(_theme.BORDER[0] * 0.55),
    int(_theme.BORDER[1] * 0.55),
    int(_theme.BORDER[2] * 0.55),
)
_SCROLLBAR_BG: tuple[int, int, int] = _theme.SCROLLBAR_BG[:3]
_SCROLLBAR_THUMB: tuple[int, int, int] = _theme.SCROLLBAR_THUMB[:3]

_BASE_PAGE_PADDING: int = 32
_BASE_PANEL_GUTTER: int = 24
_BASE_PANEL_TITLE_GAP: int = 8
_BASE_TITLE_FONT: int = 48
_BASE_ROW_H: int = 36
_BASE_ROW_PAD: int = 16
_BASE_DOC_PAD: int = 20
_BASE_HINT_GAP: int = 16
_BASE_BORDER_RADIUS: int = 12
_BASE_LINE_GAP: int = 4
_FPS: int = 30


@dataclass(frozen=True)
class _ScenarioEntry:
    class_name: str
    display_name: str
    instance: Scenario
    docstring: str


def discover_scenarios() -> list[_ScenarioEntry]:
    """Walk ``factoriax.scenarios`` for classes implementing the Scenario protocol.

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


def _wrap_paragraph(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = word if not current else f"{current} {word}"
        if font.size(candidate)[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _wrap_docstring(doc: str, font: pygame.font.Font, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in doc.split("\n"):
        lines.extend(_wrap_paragraph(paragraph, font, max_width))
    return lines


def _draw_panel(
    surf: pygame.Surface,
    rect: pygame.Rect,
    title: str,
    title_font: pygame.font.Font,
    focused: bool,
    radius: int,
) -> None:
    pygame.draw.rect(surf, _PANEL_BG, rect, border_radius=radius)
    color = _BORDER_ACTIVE if focused else _BORDER_INACTIVE
    pygame.draw.rect(surf, color, rect, _theme.BORDER_PX, border_radius=radius)

    title_surf = title_font.render(title, False, _theme.TEXT_COLOR)
    tw, th = title_surf.get_size()
    title_x = rect.left + radius + 8
    title_y = rect.top - th // 2
    break_rect = pygame.Rect(
        title_x - 6, rect.top - _theme.BORDER_PX, tw + 12, _theme.BORDER_PX * 2
    )
    pygame.draw.rect(surf, _BG_COLOR, break_rect)
    surf.blit(title_surf, (title_x, title_y))


def _draw_list(
    surf: pygame.Surface,
    rect: pygame.Rect,
    entries: list[_ScenarioEntry],
    selected_idx: int,
    row_h: int,
    row_pad: int,
    font: pygame.font.Font,
    radius: int,
) -> None:
    inner_x = rect.left + row_pad
    inner_w = rect.width - 2 * row_pad
    y = rect.top + row_pad
    for i, entry in enumerate(entries):
        row_rect = pygame.Rect(inner_x, y, inner_w, row_h)
        if i == selected_idx:
            pygame.draw.rect(surf, _BORDER_ACTIVE, row_rect, border_radius=radius // 2)
            text_color: tuple[int, int, int] = _BG_COLOR
            prefix = "> "
        else:
            text_color = _theme.TEXT_COLOR
            prefix = "  "
        text_surf = font.render(prefix + entry.display_name, False, text_color)
        _, th = text_surf.get_size()
        surf.blit(
            text_surf, (row_rect.x + row_pad // 2, row_rect.y + (row_h - th) // 2)
        )
        y += row_h + 4


def _draw_description(
    surf: pygame.Surface,
    rect: pygame.Rect,
    entry: _ScenarioEntry,
    scroll: int,
    body_font: pygame.font.Font,
    doc_pad: int,
    line_gap: int,
) -> tuple[int, int]:
    """Render the description panel and return (total_lines, visible_lines)."""
    inner_x = rect.left + doc_pad
    inner_y = rect.top + doc_pad
    inner_w = rect.width - 2 * doc_pad - _theme.SCROLLBAR_W - 4
    inner_h = rect.height - 2 * doc_pad

    lines = _wrap_docstring(entry.docstring, body_font, inner_w)
    line_h = body_font.get_height() + line_gap
    visible = max(1, inner_h // line_h)
    scroll = max(0, min(scroll, max(0, len(lines) - visible)))

    clip_prev = surf.get_clip()
    surf.set_clip(pygame.Rect(inner_x, inner_y, inner_w, inner_h))
    for row, line in enumerate(lines[scroll : scroll + visible + 1]):
        if not line:
            continue
        line_surf = body_font.render(line, False, _theme.TEXT_COLOR)
        surf.blit(line_surf, (inner_x, inner_y + row * line_h))
    surf.set_clip(clip_prev)

    # Scrollbar
    if len(lines) > visible:
        track_x = rect.right - doc_pad - _theme.SCROLLBAR_W
        track_y = inner_y
        track_h = inner_h
        pygame.draw.rect(
            surf, _SCROLLBAR_BG, (track_x, track_y, _theme.SCROLLBAR_W, track_h)
        )
        thumb_h = max(20, int(track_h * visible / len(lines)))
        thumb_y = track_y + int(
            (track_h - thumb_h) * scroll / max(1, len(lines) - visible)
        )
        pygame.draw.rect(
            surf, _SCROLLBAR_THUMB, (track_x, thumb_y, _theme.SCROLLBAR_W, thumb_h)
        )

    return len(lines), visible


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

    page_pad = _BASE_PAGE_PADDING * s
    gutter = _BASE_PANEL_GUTTER * s
    title_gap = _BASE_PANEL_TITLE_GAP * s
    row_h = _BASE_ROW_H * s
    row_pad = _BASE_ROW_PAD * s
    doc_pad = _BASE_DOC_PAD * s
    hint_gap = _BASE_HINT_GAP * s
    radius = _BASE_BORDER_RADIUS * s
    line_gap = _BASE_LINE_GAP * s

    header_font = get_pixel_font(_BASE_TITLE_FONT * s)
    panel_title_font = get_pixel_font(_theme.FONT_BODY)
    body_font = get_pixel_font(_theme.FONT_BODY)
    list_font = get_pixel_font(_theme.FONT_BODY)
    hint_font = get_pixel_font(_theme.FONT_HINT)

    entries = discover_scenarios()
    selected_idx = 0
    focus_list = True  # False = description has focus for scrolling
    doc_scroll = 0

    header_text = "SCENARIOS"
    header_surf = header_font.render(header_text, False, _theme.TEXT_COLOR)
    header_h = header_surf.get_height()
    hint_h = hint_font.get_height()

    while True:
        sw, sh = canvas.width, canvas.height

        header_y = page_pad
        panels_top = header_y + header_h + title_gap * 2
        hint_y = sh - page_pad - hint_h
        panels_bottom = hint_y - hint_gap
        panels_h = panels_bottom - panels_top

        panels_x = page_pad
        panels_w = sw - 2 * page_pad
        list_w = (panels_w - gutter) // 4
        desc_w = panels_w - gutter - list_w
        list_rect = pygame.Rect(panels_x, panels_top, list_w, panels_h)
        desc_rect = pygame.Rect(
            panels_x + list_w + gutter, panels_top, desc_w, panels_h
        )

        # --- Events ---------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.VIDEORESIZE:
                canvas.handle_resize(event.w, event.h)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = canvas.to_canvas(*event.pos)
                if entries:
                    inner_x = list_rect.left + row_pad
                    inner_w = list_rect.width - 2 * row_pad
                    y = list_rect.top + row_pad
                    for i in range(len(entries)):
                        if inner_x <= mx < inner_x + inner_w and y <= my < y + row_h:
                            selected_idx = i
                            focus_list = True
                            doc_scroll = 0
                            break
                        y += row_h + 4

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

        # --- Draw -----------------------------------------------------------
        surf = canvas.surface
        surf.fill(_BG_COLOR)
        surf.blit(header_surf, ((sw - header_surf.get_width()) // 2, header_y))

        _draw_panel(surf, list_rect, "Scenarios", panel_title_font, focus_list, radius)
        _draw_panel(
            surf, desc_rect, "Description", panel_title_font, not focus_list, radius
        )

        if entries:
            _draw_list(
                surf,
                list_rect,
                entries,
                selected_idx,
                row_h,
                row_pad,
                list_font,
                radius,
            )
            _draw_description(
                surf,
                desc_rect,
                entries[selected_idx],
                doc_scroll,
                body_font,
                doc_pad,
                line_gap,
            )
        else:
            empty_surf = body_font.render(
                "No scenarios found.", False, _theme.HINT_COLOR
            )
            surf.blit(
                empty_surf,
                (
                    list_rect.left + row_pad,
                    list_rect.top + row_pad,
                ),
            )

        hint_text = "Up/Down select   Tab focus desc   Enter play   Esc back"
        hint_surf = hint_font.render(hint_text, False, _theme.HINT_COLOR)
        surf.blit(hint_surf, ((sw - hint_surf.get_width()) // 2, hint_y))

        canvas.present(screen)
        clock.tick(_FPS)
