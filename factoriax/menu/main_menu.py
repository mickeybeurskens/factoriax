"""Title screen for FactoriaX.

Renders a full-screen menu with the game title, a decorative row of game
textures, and Play / Editor buttons. The event loop runs at 30 fps and
returns the user's choice when a button is clicked, or ``None`` if the
window is closed.
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
from factoriax.ui.fonts import get_pixel_font, render_text_rgba
from factoriax.ui.primitives import ClickRegion, hit_test_regions
from factoriax.ui.theme import BORDER, BORDER_PX, TEXT_COLOR

# -- Colours -----------------------------------------------------------------

_BG_COLOR: tuple[int, int, int] = (20, 20, 25)
_BTN_FILL: tuple[int, int, int, int] = (35, 35, 40, 255)
_BTN_HOVER: tuple[int, int, int, int] = (55, 55, 60, 255)
_BTN_BORDER: tuple[int, int, int, int] = BORDER

# -- Layout ------------------------------------------------------------------

_BTN_W: int = 400
_BTN_H: int = 80
_BTN_GAP: int = 30
_ICON_SIZE: int = 32
_ICON_GAP: int = 4
_TITLE_FONT_SIZE: int = 192
_BTN_FONT_SIZE: int = 44
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
    # surfarray expects (w, h, 4) layout.
    pygame.surfarray.blit_array(surface, rgba[:, :, :3].transpose(1, 0, 2))
    # Apply alpha channel.
    alpha_arr = rgba[:, :, 3].T
    pygame.surfarray.pixels_alpha(surface)[:] = alpha_arr
    return surface


def _build_decoration_surfaces() -> list[pygame.Surface]:
    """Build the list of pygame surfaces for the decoration row.

    Returns:
        Ordered surfaces: player, miner, coal block, 4x belt, arm, chest.
    """
    surfaces: list[pygame.Surface] = []

    player = create_player_texture(
        direction=Direction.RIGHT,
        player_idx=0,
        is_selected=True,
        size=_ICON_SIZE,
    )
    surfaces.append(_rgba_to_surface(player))

    miner = render_item_icon(ItemType.MINER, _ICON_SIZE)
    surfaces.append(_rgba_to_surface(miner))

    textures = get_textures(_ICON_SIZE)
    coal = textures[int(BlockType.COAL)]
    surfaces.append(_rgba_to_surface(coal))

    belt = render_item_icon(ItemType.CONVEYOR_BELT, _ICON_SIZE)
    belt_surf = _rgba_to_surface(belt)
    for _ in range(4):
        surfaces.append(belt_surf)

    arm = render_item_icon(ItemType.ARM, _ICON_SIZE)
    surfaces.append(_rgba_to_surface(arm))

    chest = render_item_icon(ItemType.CHEST, _ICON_SIZE)
    surfaces.append(_rgba_to_surface(chest))

    return surfaces


def _draw_button(
    screen: pygame.Surface,
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
        screen: Target surface.
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
    pygame.draw.rect(screen, fill[:3], rect)
    pygame.draw.rect(screen, _BTN_BORDER[:3], rect, BORDER_PX)

    text_surf = font.render(label, False, TEXT_COLOR)
    tw, th = text_surf.get_size()
    screen.blit(text_surf, (x + (w - tw) // 2, y + (h - th) // 2))


# -- Public API --------------------------------------------------------------


def run_main_menu(screen: pygame.Surface) -> str | None:
    """Show the main menu and return the user's choice.

    Blocks until the player clicks a button or closes the window. The
    menu renders directly into *screen* at its current resolution.

    Args:
        screen: Pygame display surface.

    Returns:
        ``"play"`` if Play was clicked, ``"editor"`` if Editor was
        clicked, ``None`` if the window was closed.
    """
    clock = pygame.time.Clock()
    sw, sh = screen.get_size()

    # Fonts.
    title_font = get_pixel_font(_TITLE_FONT_SIZE)
    btn_font = get_pixel_font(_BTN_FONT_SIZE)

    # Pre-render title text as a pygame surface for fast blitting.
    title_rgba = render_text_rgba("FACTORIAX", title_font, TEXT_COLOR)
    title_surf = _rgba_to_surface(title_rgba)
    title_w, title_h = title_surf.get_size()

    # Decoration row surfaces.
    deco_surfs = _build_decoration_surfaces()
    deco_total_w = sum(s.get_width() for s in deco_surfs) + _ICON_GAP * (
        len(deco_surfs) - 1
    )

    # Vertical positions (fractions of screen height).
    title_y = int(sh * 0.25) - title_h // 2
    deco_y = title_y + title_h + 8
    num_btns = 4
    btn_block_h = _BTN_H * num_btns + _BTN_GAP * (num_btns - 1)
    btn_top = int(sh * 0.50) - btn_block_h // 2
    btn_x = (sw - _BTN_W) // 2

    # Click regions for hit testing.
    labels = ["Play", "Editor", "Settings", "Quit"]
    actions = ["play", "editor", "settings", None]
    regions = [
        ClickRegion(
            btn_x,
            btn_top + i * (_BTN_H + _BTN_GAP),
            _BTN_W,
            _BTN_H,
            actions[i],
            i,
        )
        for i in range(num_btns)
    ]

    focus_idx = 0

    while True:
        # -- Events ----------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
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

        # -- Draw ------------------------------------------------------------
        screen.fill(_BG_COLOR)

        # Title (centered horizontally).
        screen.blit(title_surf, ((sw - title_w) // 2, title_y))

        # Decoration row (centered horizontally).
        dx = (sw - deco_total_w) // 2
        for surf in deco_surfs:
            screen.blit(surf, (dx, deco_y))
            dx += surf.get_width() + _ICON_GAP

        # Buttons.
        mx, my = pygame.mouse.get_pos()
        for i, region in enumerate(regions):
            hovered = (
                region.x <= mx < region.x + region.w
                and region.y <= my < region.y + region.h
            )
            _draw_button(
                screen,
                region.x,
                region.y,
                region.w,
                region.h,
                labels[i],
                btn_font,
                hovered=hovered,
                focused=(i == focus_idx and not hovered),
            )

        pygame.display.flip()
        clock.tick(_FPS)
