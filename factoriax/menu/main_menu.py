"""Title screen for FactoriaX.

Renders a full-screen menu with the game title, a decorative row of game
textures, and Play / Editor / Settings / Quit buttons. All drawing happens
on a fixed-resolution canvas that is integer-scaled to the window, so the
layout is pixel-perfect at any window size.
"""

from __future__ import annotations

import numpy as np
import pygame

from factoriax.constants import BlockType, Direction, ItemType
from factoriax.renderer import (
    create_player_texture,
    get_textures,
    render_item_icon,
)
from factoriax.ui import theme as _theme
from factoriax.ui.fonts import get_pixel_font, render_text_rgba
from factoriax.ui.primitives import ClickRegion, hit_test_regions
from factoriax.ui.scaling import ScaledCanvas

# -- Colours -----------------------------------------------------------------

_BG_COLOR: tuple[int, int, int] = (20, 20, 25)
_BTN_FILL: tuple[int, int, int, int] = (35, 35, 40, 255)
_BTN_HOVER: tuple[int, int, int, int] = (55, 55, 60, 255)

# -- Base layout values (multiplied by UI_SCALE at runtime) ------------------

_BASE_BTN_W: int = 400
_BASE_BTN_H: int = 80
_BASE_BTN_GAP: int = 30
_BASE_ICON_SIZE: int = 32
_BASE_ICON_GAP: int = 4
_BASE_TITLE_FONT: int = 192
_BASE_BTN_FONT: int = 44
_FPS: int = 30


# -- Helpers -----------------------------------------------------------------


def _rgba_to_surface(rgba: np.ndarray) -> pygame.Surface:
    """Convert an RGBA numpy array to a pygame Surface with per-pixel alpha.

    Args:
        rgba: uint8 array of shape ``(H, W, 4)``.

    Returns:
        A ``pygame.Surface`` with ``SRCALPHA``.
    """
    h, w = rgba.shape[:2]
    surface = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.surfarray.blit_array(surface, rgba[:, :, :3].transpose(1, 0, 2))
    alpha_arr = rgba[:, :, 3].T
    pygame.surfarray.pixels_alpha(surface)[:] = alpha_arr
    return surface


def _build_decoration_surfaces(icon_size: int) -> list[pygame.Surface]:
    """Build the list of pygame surfaces for the decoration row.

    Args:
        icon_size: Pixel size for each decoration icon.

    Returns:
        Ordered surfaces: player, miner, coal block, 4x belt, arm, chest.
    """
    surfaces: list[pygame.Surface] = []

    player = create_player_texture(
        direction=Direction.RIGHT,
        player_idx=0,
        is_selected=True,
        size=icon_size,
    )
    surfaces.append(_rgba_to_surface(player))

    miner = render_item_icon(ItemType.MINER, icon_size)
    surfaces.append(_rgba_to_surface(miner))

    textures = get_textures(icon_size)
    coal = textures[int(BlockType.COAL)]
    surfaces.append(_rgba_to_surface(coal))

    belt = render_item_icon(ItemType.CONVEYOR_BELT, icon_size)
    belt_surf = _rgba_to_surface(belt)
    for _ in range(4):
        surfaces.append(belt_surf)

    arm = render_item_icon(ItemType.ARM, icon_size)
    surfaces.append(_rgba_to_surface(arm))

    chest = render_item_icon(ItemType.CHEST, icon_size)
    surfaces.append(_rgba_to_surface(chest))

    return surfaces


def _draw_button(
    surface: pygame.Surface,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    font: pygame.font.Font,
    hovered: bool,
    focused: bool,
) -> None:
    """Draw a single menu button with optional hover and focus highlights.

    Args:
        surface: Target surface.
        x: Left edge.
        y: Top edge.
        w: Width.
        h: Height.
        label: Button text.
        font: Font for the label.
        hovered: Whether the mouse is over this button.
        focused: Whether keyboard focus is on this button.
    """
    fill = _BTN_HOVER if hovered or focused else _BTN_FILL
    rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(surface, fill[:3], rect)
    pygame.draw.rect(surface, _theme.BORDER[:3], rect, _theme.BORDER_PX)

    text_surf = font.render(label, False, _theme.TEXT_COLOR)
    tw, th = text_surf.get_size()
    surface.blit(text_surf, (x + (w - tw) // 2, y + (h - th) // 2))


# -- Public API --------------------------------------------------------------


def run_main_menu(screen: pygame.Surface) -> str | None:
    """Show the main menu and return the user's choice.

    Blocks until the player clicks a button or closes the window.
    Renders to a fixed-size canvas that is integer-scaled to the window.

    Args:
        screen: Pygame display surface.

    Returns:
        ``"play"``, ``"editor"``, ``"settings"``, or ``None`` (quit).
    """
    s = _theme.UI_SCALE
    canvas = ScaledCanvas(1024, s, screen)
    clock = pygame.time.Clock()
    sw, sh = canvas.width, canvas.height

    # Scaled layout constants.
    btn_w = _BASE_BTN_W * s
    btn_h = _BASE_BTN_H * s
    btn_gap = _BASE_BTN_GAP * s
    icon_size = _BASE_ICON_SIZE * s
    icon_gap = _BASE_ICON_GAP * s
    title_font_size = _BASE_TITLE_FONT * s
    btn_font_size = _BASE_BTN_FONT * s

    # Fonts.
    title_font = get_pixel_font(title_font_size)
    btn_font = get_pixel_font(btn_font_size)

    # Pre-render title text as a pygame surface for fast blitting.
    title_rgba = render_text_rgba("FACTORIAX", title_font, _theme.TEXT_COLOR)
    title_surf = _rgba_to_surface(title_rgba)
    title_w, title_h = title_surf.get_size()

    # Decoration row surfaces.
    deco_surfs = _build_decoration_surfaces(icon_size)
    deco_total_w = sum(sf.get_width() for sf in deco_surfs) + icon_gap * (
        len(deco_surfs) - 1
    )

    # Vertical positions: title at 25%, buttons at 50%, with overlap guard.
    title_y = int(sh * 0.25) - title_h // 2
    deco_y = title_y + title_h + 8 * s
    deco_bottom = deco_y + icon_size

    num_btns = 4
    btn_block_h = btn_h * num_btns + btn_gap * (num_btns - 1)
    btn_top = max(deco_bottom + 16 * s, int(sh * 0.50) - btn_block_h // 2)
    btn_x = (sw - btn_w) // 2

    # Click regions for hit testing (canvas coordinates).
    labels = ["Play", "Editor", "Settings", "Quit"]
    actions: list[str | None] = ["play", "editor", "settings", None]
    regions = [
        ClickRegion(
            btn_x,
            btn_top + i * (btn_h + btn_gap),
            btn_w,
            btn_h,
            actions[i],
            i,
        )
        for i in range(num_btns)
    ]

    focus_idx = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

            if event.type == pygame.VIDEORESIZE:
                canvas.handle_resize(event.w, event.h)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = canvas.to_canvas(*event.pos)
                hit = hit_test_regions(regions, mx, my)
                if hit is not None:
                    return hit.action

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    return regions[focus_idx].action
                if event.key in (pygame.K_DOWN, pygame.K_TAB):
                    focus_idx = (focus_idx + 1) % len(regions)
                if event.key in (pygame.K_UP,):
                    focus_idx = (focus_idx - 1) % len(regions)

        # -- Draw to canvas --------------------------------------------------
        surf = canvas.surface
        surf.fill(_BG_COLOR)

        # Title (centered horizontally).
        surf.blit(title_surf, ((sw - title_w) // 2, title_y))

        # Decoration row (centered horizontally).
        dx = (sw - deco_total_w) // 2
        for deco in deco_surfs:
            surf.blit(deco, (dx, deco_y))
            dx += deco.get_width() + icon_gap

        # Buttons.
        mx, my = canvas.to_canvas(*pygame.mouse.get_pos())
        for i, region in enumerate(regions):
            hovered = (
                region.x <= mx < region.x + region.w
                and region.y <= my < region.y + region.h
            )
            _draw_button(
                surf,
                region.x,
                region.y,
                region.w,
                region.h,
                labels[i],
                btn_font,
                hovered=hovered,
                focused=(i == focus_idx and not hovered),
            )

        canvas.present(screen)
        clock.tick(_FPS)
