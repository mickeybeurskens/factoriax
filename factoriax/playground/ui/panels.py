"""Reusable drawing primitives for FactoriaX menus.

All components read sizes and colors from :mod:`factoriax.playground.ui.theme` so the
look stays consistent across menus. Each function is a stateless drawing
operation; menus own the layout and event loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame

from factoriax.playground.ui import theme as _theme


class InputSourceTracker:
    """Tracks whether mouse motion or keyboard nav last moved the selection.

    Menus use this to decide when mouse hover should commit to the selection:
    only while :attr:`mouse_active` is ``True``. The flag flips to ``True``
    when the cursor position changes between :meth:`tick` calls; the caller
    invokes :meth:`mark_keyboard` on every keyboard navigation action so a
    stationary cursor doesn't immediately drag the highlight back to wherever
    the mouse happens to rest.
    """

    def __init__(self) -> None:
        self._prev_mouse: tuple[int, int] = pygame.mouse.get_pos()
        self.mouse_active: bool = False

    def tick(self) -> None:
        """Sample the cursor; sets ``mouse_active`` True if it moved this frame."""
        current = pygame.mouse.get_pos()
        if current != self._prev_mouse:
            self.mouse_active = True
        self._prev_mouse = current

    def mark_keyboard(self) -> None:
        """The keyboard just navigated — mouse hover takes a back seat."""
        self.mouse_active = False


def rgba_to_surface(rgba: np.ndarray) -> pygame.Surface:
    """Convert an RGBA uint8 array of shape ``(H, W, 4)`` to a Surface."""
    h, w = rgba.shape[:2]
    surface = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.surfarray.blit_array(surface, rgba[:, :, :3].transpose(1, 0, 2))
    pygame.surfarray.pixels_alpha(surface)[:] = rgba[:, :, 3].T
    return surface


def _border_color(focused: bool) -> tuple[int, int, int]:
    return _theme.BORDER[:3] if focused else _theme.BORDER_INACTIVE


def draw_panel(
    surf: pygame.Surface,
    rect: pygame.Rect,
    title: str | None = None,
    title_font: pygame.font.Font | None = None,
    *,
    focused: bool = True,
) -> None:
    """Rounded panel with optional title broken into the top border line."""
    radius = _theme.BORDER_RADIUS
    pygame.draw.rect(surf, _theme.MENU_PANEL_BG, rect, border_radius=radius)
    pygame.draw.rect(
        surf, _border_color(focused), rect, _theme.BORDER_PX, border_radius=radius
    )
    if title and title_font is not None:
        title_surf = title_font.render(title, False, _theme.TEXT_COLOR)
        tw, th = title_surf.get_size()
        title_x = rect.left + radius + 8
        title_y = rect.top - th // 2
        break_rect = pygame.Rect(
            title_x - 6, rect.top - _theme.BORDER_PX, tw + 12, _theme.BORDER_PX * 2
        )
        pygame.draw.rect(surf, _theme.PAGE_BG, break_rect)
        surf.blit(title_surf, (title_x, title_y))


def draw_button(
    surf: pygame.Surface,
    rect: pygame.Rect,
    label: str,
    font: pygame.font.Font,
    *,
    focused: bool = False,
    hovered: bool = False,
) -> None:
    """Rounded button matching the panel style."""
    radius = _theme.BORDER_RADIUS
    fill = _theme.BUTTON_HOVER if hovered or focused else _theme.BUTTON_FILL
    pygame.draw.rect(surf, fill, rect, border_radius=radius)
    pygame.draw.rect(
        surf,
        _border_color(focused or hovered),
        rect,
        _theme.BORDER_PX,
        border_radius=radius,
    )
    text_surf = font.render(label, False, _theme.TEXT_COLOR)
    tw, th = text_surf.get_size()
    surf.blit(text_surf, (rect.x + (rect.w - tw) // 2, rect.y + (rect.h - th) // 2))


def draw_hint_bar(
    surf: pygame.Surface,
    sw: int,
    y: int,
    text: str,
    font: pygame.font.Font,
) -> None:
    """Centered hint text in :data:`theme.HINT_COLOR`."""
    hint_surf = font.render(text, False, _theme.HINT_COLOR)
    surf.blit(hint_surf, ((sw - hint_surf.get_width()) // 2, y))


def wrap_text(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
    """Word-wrap ``text`` to fit ``max_width`` using ``font``.

    Newlines become explicit breaks; empty paragraphs become empty lines so
    the source's vertical rhythm is preserved. Words longer than ``max_width``
    are kept on their own line (no character-level splitting).
    """
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        current = ""
        for word in paragraph.split(" "):
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


@dataclass(frozen=True)
class SettingField:
    """One editable numeric setting in a settings menu.

    Attributes:
        key: Identifier the menu uses to look up / write the value (e.g. the
            attribute name on an ``EnvParams`` or a key in a config dict).
        label: Human-readable label shown on the left side of the row.
        is_float: ``True`` for continuous values (probabilities); ``False``
            for integer-typed settings. Controls formatting and rounding.
        step: Amount added or subtracted by one tap of Left/Right.
        min_value: Lower bound, inclusive.
        max_value: Upper bound, inclusive.
    """

    key: str
    label: str
    is_float: bool
    step: float
    min_value: float
    max_value: float

    def clamp(self, value: float) -> float:
        """Clamp ``value`` to ``[min_value, max_value]`` and cast to the type."""
        value = max(self.min_value, min(self.max_value, value))
        if not self.is_float:
            return float(int(value))
        return round(value, 2)

    def format(self, value: float) -> str:
        """Format ``value`` for display in the right-hand column."""
        if self.is_float:
            return f"{value:.2f}"
        return str(int(value))


def list_row_rects(rect: pygame.Rect, n_rows: int) -> list[pygame.Rect]:
    """Return per-row rects laid out top-to-bottom inside ``rect``.

    Single source of truth for the list layout so callers can hit-test
    against the same coordinates :func:`draw_list_rows` draws into.
    """
    rects: list[pygame.Rect] = []
    inner_x = rect.left + _theme.ROW_PAD
    inner_w = rect.width - 2 * _theme.ROW_PAD
    y = rect.top + _theme.ROW_PAD
    for _ in range(n_rows):
        rects.append(pygame.Rect(inner_x, y, inner_w, _theme.ROW_H))
        y += _theme.ROW_H + 4
    return rects


@dataclass(frozen=True)
class SettingSection:
    """A titled group of related :class:`SettingField` rows.

    Used by :func:`draw_setting_sections` to render an editable settings list
    with elegant headings and dividers between groups.
    """

    title: str
    fields: tuple[SettingField, ...]


@dataclass(frozen=True)
class LabelValueSection:
    """A titled group of ``(label, value)`` rows for read-only / non-numeric display.

    Companion to :class:`SettingSection`: same visual layout, but values are
    arbitrary strings the caller pre-formats (key bindings, toggle state,
    enum-style choices). The active row gets the gold-fill highlight but no
    chevron decoration — there are no arrow-adjust semantics to advertise.
    """

    title: str
    labels: tuple[str, ...]


_SECTION_GAP: int = 14
_HEADING_GAP: int = 6
_DIVIDER_H: int = 1


def setting_section_layout(
    rect: pygame.Rect,
    sections: tuple[SettingSection, ...] | list[SettingSection],
    heading_h: int,
) -> tuple[list[pygame.Rect], list[tuple[int, int]], int]:
    """Compute per-field row rects, per-section heading positions, and bottom y.

    Single source of truth for the sectioned-settings layout; the drawer and
    callers that need to position a button below the block (or hit-test the
    rows) share these coordinates.
    """
    inner_x = rect.left + _theme.ROW_PAD
    inner_w = rect.width - 2 * _theme.ROW_PAD
    row_rects: list[pygame.Rect] = []
    heading_positions: list[tuple[int, int]] = []
    y = rect.top + _theme.ROW_PAD

    for sec_i, sec in enumerate(sections):
        if sec_i > 0:
            y += _SECTION_GAP
        heading_positions.append((inner_x, y))
        y += heading_h + _HEADING_GAP
        y += _DIVIDER_H + _HEADING_GAP

        for i in range(len(sec.fields)):
            row_rects.append(pygame.Rect(inner_x, y, inner_w, _theme.ROW_H))
            y += _theme.ROW_H
            if i < len(sec.fields) - 1:
                y += 4
    return row_rects, heading_positions, y


def draw_setting_sections(
    surf: pygame.Surface,
    rect: pygame.Rect,
    sections: tuple[SettingSection, ...] | list[SettingSection],
    values: list[float],
    selected_idx: int,
    focused: bool,
    font: pygame.font.Font,
    heading_font: pygame.font.Font,
) -> tuple[list[pygame.Rect], int]:
    """Render sectioned setting rows with headings and dividers.

    ``values`` is a flat list aligned with the concatenation of every
    section's ``fields``; ``selected_idx`` indexes into that same flat list.

    Returns:
        ``(row_rects, bottom_y)`` — for hit-testing and positioning content
        below the rendered block.
    """
    heading_h = heading_font.get_height()
    row_rects, heading_positions, bottom_y = setting_section_layout(
        rect, sections, heading_h
    )
    inner_w = rect.width - 2 * _theme.ROW_PAD

    for sec, (hx, hy) in zip(sections, heading_positions, strict=True):
        heading_surf = heading_font.render(sec.title.upper(), False, _theme.TEXT_COLOR)
        surf.blit(heading_surf, (hx, hy))
        divider_y = hy + heading_h + _HEADING_GAP
        pygame.draw.rect(
            surf, _theme.BORDER_INACTIVE, (hx, divider_y, inner_w, _DIVIDER_H)
        )

    flat_fields = [f for sec in sections for f in sec.fields]
    for i, (fld, value, row_rect) in enumerate(
        zip(flat_fields, values, row_rects, strict=True)
    ):
        active = focused and i == selected_idx
        if active:
            pygame.draw.rect(
                surf,
                _theme.BORDER[:3],
                row_rect,
                border_radius=_theme.BORDER_RADIUS // 2,
            )
            text_color: tuple[int, int, int] = _theme.PAGE_BG
        else:
            text_color = _theme.TEXT_COLOR
        label_surf = font.render(fld.label, False, text_color)
        surf.blit(
            label_surf,
            (
                row_rect.x + _theme.ROW_PAD // 2,
                row_rect.y + (_theme.ROW_H - label_surf.get_height()) // 2,
            ),
        )
        value_text = fld.format(value)
        decorated = f"<  {value_text}  >" if active else value_text
        value_surf = font.render(decorated, False, text_color)
        surf.blit(
            value_surf,
            (
                row_rect.right - _theme.ROW_PAD // 2 - value_surf.get_width(),
                row_rect.y + (_theme.ROW_H - value_surf.get_height()) // 2,
            ),
        )

    return row_rects, bottom_y


def label_value_section_layout(
    rect: pygame.Rect,
    sections: tuple[LabelValueSection, ...] | list[LabelValueSection],
    heading_h: int,
) -> tuple[list[pygame.Rect], list[tuple[int, int]], int]:
    """Per-row rects, heading positions, and bottom y for label/value sections."""
    inner_x = rect.left + _theme.ROW_PAD
    inner_w = rect.width - 2 * _theme.ROW_PAD
    row_rects: list[pygame.Rect] = []
    heading_positions: list[tuple[int, int]] = []
    y = rect.top + _theme.ROW_PAD

    for sec_i, sec in enumerate(sections):
        if sec_i > 0:
            y += _SECTION_GAP
        heading_positions.append((inner_x, y))
        y += heading_h + _HEADING_GAP
        y += _DIVIDER_H + _HEADING_GAP

        for i in range(len(sec.labels)):
            row_rects.append(pygame.Rect(inner_x, y, inner_w, _theme.ROW_H))
            y += _theme.ROW_H
            if i < len(sec.labels) - 1:
                y += 4
    return row_rects, heading_positions, y


def draw_label_value_sections(
    surf: pygame.Surface,
    rect: pygame.Rect,
    sections: tuple[LabelValueSection, ...] | list[LabelValueSection],
    values: list[str],
    selected_idx: int,
    focused: bool,
    font: pygame.font.Font,
    heading_font: pygame.font.Font,
) -> tuple[list[pygame.Rect], int]:
    """Render sectioned label/value rows for bindings, toggles, choices.

    Same visual treatment as :func:`draw_setting_sections` but values are
    arbitrary strings supplied by the caller; the active row gets the gold
    highlight without chevron decoration. ``values`` is a flat list aligned
    with the concatenation of every section's ``labels``.
    """
    heading_h = heading_font.get_height()
    row_rects, heading_positions, bottom_y = label_value_section_layout(
        rect, sections, heading_h
    )
    inner_w = rect.width - 2 * _theme.ROW_PAD

    for sec, (hx, hy) in zip(sections, heading_positions, strict=True):
        heading_surf = heading_font.render(sec.title.upper(), False, _theme.TEXT_COLOR)
        surf.blit(heading_surf, (hx, hy))
        divider_y = hy + heading_h + _HEADING_GAP
        pygame.draw.rect(
            surf, _theme.BORDER_INACTIVE, (hx, divider_y, inner_w, _DIVIDER_H)
        )

    flat_labels = [lab for sec in sections for lab in sec.labels]
    for i, (label, value, row_rect) in enumerate(
        zip(flat_labels, values, row_rects, strict=True)
    ):
        active = focused and i == selected_idx
        if active:
            pygame.draw.rect(
                surf,
                _theme.BORDER[:3],
                row_rect,
                border_radius=_theme.BORDER_RADIUS // 2,
            )
            text_color: tuple[int, int, int] = _theme.PAGE_BG
        else:
            text_color = _theme.TEXT_COLOR
        label_surf = font.render(label, False, text_color)
        surf.blit(
            label_surf,
            (
                row_rect.x + _theme.ROW_PAD // 2,
                row_rect.y + (_theme.ROW_H - label_surf.get_height()) // 2,
            ),
        )
        value_surf = font.render(value, False, text_color)
        surf.blit(
            value_surf,
            (
                row_rect.right - _theme.ROW_PAD // 2 - value_surf.get_width(),
                row_rect.y + (_theme.ROW_H - value_surf.get_height()) // 2,
            ),
        )

    return row_rects, bottom_y


def draw_setting_rows(
    surf: pygame.Surface,
    rect: pygame.Rect,
    fields: tuple[SettingField, ...] | list[SettingField],
    values: list[float],
    selected_idx: int,
    focused: bool,
    font: pygame.font.Font,
) -> list[pygame.Rect]:
    """Render ``label … value`` rows for an editable settings list.

    When ``focused`` and ``i == selected_idx``, the row gets a gold fill and
    the value is wrapped with ``<  value  >`` chevrons to signal that
    Left/Right adjusts it. Returns the row rects so the caller can hit-test
    mouse clicks against the same coordinates.
    """
    rects = list_row_rects(rect, len(fields))
    for i, (fld, value, row_rect) in enumerate(zip(fields, values, rects, strict=True)):
        active = focused and i == selected_idx
        if active:
            pygame.draw.rect(
                surf,
                _theme.BORDER[:3],
                row_rect,
                border_radius=_theme.BORDER_RADIUS // 2,
            )
            text_color: tuple[int, int, int] = _theme.PAGE_BG
        else:
            text_color = _theme.TEXT_COLOR

        label_surf = font.render(fld.label, False, text_color)
        surf.blit(
            label_surf,
            (
                row_rect.x + _theme.ROW_PAD // 2,
                row_rect.y + (_theme.ROW_H - label_surf.get_height()) // 2,
            ),
        )

        value_text = fld.format(value)
        decorated = f"<  {value_text}  >" if active else value_text
        value_surf = font.render(decorated, False, text_color)
        surf.blit(
            value_surf,
            (
                row_rect.right - _theme.ROW_PAD // 2 - value_surf.get_width(),
                row_rect.y + (_theme.ROW_H - value_surf.get_height()) // 2,
            ),
        )
    return rects


def draw_list_rows(
    surf: pygame.Surface,
    rect: pygame.Rect,
    rows: list[str],
    selected_idx: int,
    font: pygame.font.Font,
) -> None:
    """Render a vertically-stacked selectable list inside ``rect``."""
    for i, (row, row_rect) in enumerate(
        zip(rows, list_row_rects(rect, len(rows)), strict=True)
    ):
        if i == selected_idx:
            pygame.draw.rect(
                surf,
                _theme.BORDER[:3],
                row_rect,
                border_radius=_theme.BORDER_RADIUS // 2,
            )
            text_color: tuple[int, int, int] = _theme.PAGE_BG
            prefix = "> "
        else:
            text_color = _theme.TEXT_COLOR
            prefix = "  "
        text_surf = font.render(prefix + row, False, text_color)
        _, th = text_surf.get_size()
        surf.blit(
            text_surf,
            (
                row_rect.x + _theme.ROW_PAD // 2,
                row_rect.y + (_theme.ROW_H - th) // 2,
            ),
        )


def draw_description(
    surf: pygame.Surface,
    rect: pygame.Rect,
    text: str,
    scroll: int,
    font: pygame.font.Font,
) -> int:
    """Render wrapped scrollable text in ``rect``. Returns the clamped scroll."""
    inner_x = rect.left + _theme.DOC_PAD
    inner_y = rect.top + _theme.DOC_PAD
    inner_w = rect.width - 2 * _theme.DOC_PAD - _theme.SCROLLBAR_W - 4
    inner_h = rect.height - 2 * _theme.DOC_PAD

    lines = wrap_text(text, font, inner_w)
    line_h = font.get_height() + _theme.DOC_LINE_GAP
    visible = max(1, inner_h // line_h)
    scroll = max(0, min(scroll, max(0, len(lines) - visible)))

    clip_prev = surf.get_clip()
    surf.set_clip(pygame.Rect(inner_x, inner_y, inner_w, inner_h))
    for row, line in enumerate(lines[scroll : scroll + visible + 1]):
        if not line:
            continue
        line_surf = font.render(line, False, _theme.TEXT_COLOR)
        surf.blit(line_surf, (inner_x, inner_y + row * line_h))
    surf.set_clip(clip_prev)

    if len(lines) > visible:
        track_x = rect.right - _theme.DOC_PAD - _theme.SCROLLBAR_W
        pygame.draw.rect(
            surf,
            _theme.SCROLLBAR_BG[:3],
            (track_x, inner_y, _theme.SCROLLBAR_W, inner_h),
        )
        thumb_h = max(20, int(inner_h * visible / len(lines)))
        thumb_y = inner_y + int(
            (inner_h - thumb_h) * scroll / max(1, len(lines) - visible)
        )
        pygame.draw.rect(
            surf,
            _theme.SCROLLBAR_THUMB[:3],
            (track_x, thumb_y, _theme.SCROLLBAR_W, thumb_h),
        )

    return scroll


@dataclass(frozen=True)
class DecorationStrip:
    """Horizontal row of game icons with a periodic "ore on belt" animation.

    Stateless: ``draw`` derives the animation phase from ``time_ms`` (typically
    ``pygame.time.get_ticks()``). Build one with the
    :meth:`factoriax_default` factory or by populating the fields manually.

    Attributes:
        surfaces: Pre-rendered icon surfaces drawn left-to-right.
        ore_surface: Small overlay drawn on top of the slot indicated by
            ``anim_targets[(time_ms // anim_step_ms) % anim_cycle_len]`` when
            that index is within ``anim_targets``.
        icon_size: Height of each icon in the row; also used to vertically
            center the ore overlay.
        icon_gap: Pixel gap between consecutive icons.
        anim_targets: Slot indices the ore visits, in order.
        anim_step_ms: Milliseconds per animation step.
        anim_cycle_len: Total steps in one animation cycle; steps past
            ``len(anim_targets)`` hide the ore (a brief idle frame).
    """

    surfaces: tuple[pygame.Surface, ...]
    ore_surface: pygame.Surface
    icon_size: int
    icon_gap: int
    anim_targets: tuple[int, ...]
    anim_step_ms: int = 700
    anim_cycle_len: int = 6

    @classmethod
    def factoriax_default(cls, icon_size: int, icon_gap: int = 10) -> DecorationStrip:
        """Build the standard 8-slot strip: player, miner, 4×belt, arm, pallet.

        Ore travels across the four belt slots (indices 2-5) and onto the arm
        (index 6); the player and miner slots stay clear.
        """
        from factoriax.engine.constants import Direction, ItemType
        from factoriax.playground.ui.icons import (
            create_player_texture,
            render_item_icon,
        )

        surfaces: list[pygame.Surface] = [
            rgba_to_surface(
                create_player_texture(
                    direction=Direction.RIGHT,
                    player_idx=0,
                    is_selected=True,
                    size=icon_size,
                )
            ),
            rgba_to_surface(render_item_icon(ItemType.MINER, icon_size)),
        ]
        belt = rgba_to_surface(render_item_icon(ItemType.CONVEYOR_BELT, icon_size))
        surfaces.extend([belt] * 4)
        surfaces.append(rgba_to_surface(render_item_icon(ItemType.ARM, icon_size)))
        surfaces.append(rgba_to_surface(render_item_icon(ItemType.PALLET, icon_size)))

        ore_size = max(8, icon_size // 2)
        ore_surface = rgba_to_surface(render_item_icon(ItemType.COAL, ore_size))

        return cls(
            surfaces=tuple(surfaces),
            ore_surface=ore_surface,
            icon_size=icon_size,
            icon_gap=icon_gap,
            anim_targets=(2, 3, 4, 5, 6),
        )

    @property
    def total_width(self) -> int:
        if not self.surfaces:
            return 0
        return sum(s.get_width() for s in self.surfaces) + self.icon_gap * (
            len(self.surfaces) - 1
        )

    @property
    def height(self) -> int:
        return self.icon_size

    def draw(self, surf: pygame.Surface, x: int, y: int, time_ms: int) -> None:
        """Blit every icon left-to-right at ``(x, y)`` and overlay the ore."""
        dx = x
        slot_positions: list[int] = []
        for sprite in self.surfaces:
            slot_positions.append(dx)
            surf.blit(sprite, (dx, y))
            dx += sprite.get_width() + self.icon_gap

        anim_step = (time_ms // self.anim_step_ms) % self.anim_cycle_len
        if anim_step < len(self.anim_targets):
            target_idx = self.anim_targets[anim_step]
            if 0 <= target_idx < len(slot_positions):
                ore_size = self.ore_surface.get_width()
                ore_x = slot_positions[target_idx] + (self.icon_size - ore_size) // 2
                ore_y = y + (self.icon_size - ore_size) // 2
                surf.blit(self.ore_surface, (ore_x, ore_y))
