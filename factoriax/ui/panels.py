"""Reusable drawing primitives for FactoriaX menus.

All components read sizes and colors from :mod:`factoriax.ui.theme` so the
look stays consistent across menus. Each function is a stateless drawing
operation; menus own the layout and event loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame

from factoriax.ui import theme as _theme


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
        from factoriax.constants import Direction, ItemType
        from factoriax.ui.icons import create_player_texture, render_item_icon

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
