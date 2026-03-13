"""Interactive play script for FactoriaX using pygame."""

import jax
import numpy as np
import pygame
from jax import random

from factoriax.constants import BLOCK_PIXEL_SIZE, Action
from factoriax.envs.factoriax_env import make_factoriax_env
from factoriax.player_ui import render_achievement_menu, render_inventory_menu
from factoriax.renderer import render_pixels


def composite_rgba_over_rgb(
    background: np.ndarray, overlay: np.ndarray
) -> np.ndarray:
    """Composite an RGBA overlay onto an RGB background.

    Args:
        background: RGB image array of shape (H, W, 3)
        overlay: RGBA image array of shape (H, W, 4)

    Returns:
        RGB image array with overlay composited
    """
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    fg = overlay[:, :, :3].astype(np.float32)
    bg = background.astype(np.float32)
    blended = fg * alpha + bg * (1 - alpha)
    return blended.astype(np.uint8)


def calculate_window_size(
    base_width: int, base_height: int, scale_factor: float = 0.8
) -> tuple[int, int]:
    """Calculate window size to fit screen while maintaining aspect ratio.

    Scales the window to use scale_factor (default 80%) of the limiting
    dimension (width or height) while preserving aspect ratio.

    Args:
        base_width: Base render width in pixels
        base_height: Base render height in pixels
        scale_factor: Fraction of screen to use (0.0 to 1.0)

    Returns:
        Tuple of (window_width, window_height) in pixels
    """
    screen_info = pygame.display.Info()
    max_width = screen_info.current_w
    max_height = screen_info.current_h

    aspect_ratio = base_width / base_height

    target_width = int(max_width * scale_factor)
    target_height = int(max_height * scale_factor)

    if target_width / aspect_ratio <= target_height:
        window_width = target_width
        window_height = int(target_width / aspect_ratio)
    else:
        window_height = target_height
        window_width = int(target_height * aspect_ratio)

    return window_width, window_height


def main() -> None:
    """Run the interactive FactoriaX game.

    Controls:
        WASD: Move the selected player
        Space: Mine at current position
        I: Toggle inventory/crafting menu
        Left/Right arrows: Switch between inventory and crafting sections (when menu open)
        Tab: Cycle slot/recipe forward (in focused section)
        [: Cycle slot/recipe backward (in focused section)
        C: Start crafting selected recipe
        E: Place machine from selected inventory slot
        P: Toggle achievement menu
        1-9: Select player (if that many players exist)
        R: Reset the game
        Q/Escape: Quit
    """
    pygame.init()

    env, params = make_factoriax_env()
    base_width = params.map_width * BLOCK_PIXEL_SIZE
    base_height = params.map_height * BLOCK_PIXEL_SIZE

    window_width, window_height = calculate_window_size(base_width, base_height)
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("FactoriaX")
    clock = pygame.time.Clock()

    rng = random.PRNGKey(42)
    rng, reset_key = random.split(rng)
    obs, state = env.reset_env(reset_key, params)

    step_fn = jax.jit(env.step_env)

    key_to_action = {
        pygame.K_a: Action.LEFT,
        pygame.K_d: Action.RIGHT,
        pygame.K_w: Action.UP,
        pygame.K_s: Action.DOWN,
        pygame.K_SPACE: Action.MINE,
        pygame.K_c: Action.CRAFT,
        pygame.K_e: Action.PLACE,
    }

    key_to_player = {
        pygame.K_1: 0,
        pygame.K_2: 1,
        pygame.K_3: 2,
        pygame.K_4: 3,
        pygame.K_5: 4,
        pygame.K_6: 5,
        pygame.K_7: 6,
        pygame.K_8: 7,
        pygame.K_9: 8,
    }

    inventory_open = False
    achievement_open = False
    menu_focus = "inventory"

    running = True
    while running:
        action = Action.NOOP

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_i:
                    inventory_open = not inventory_open
                    if inventory_open:
                        achievement_open = False
                elif event.key == pygame.K_p:
                    achievement_open = not achievement_open
                    if achievement_open:
                        inventory_open = False
                elif event.key == pygame.K_r:
                    rng, reset_key = random.split(rng)
                    obs, state = env.reset_env(reset_key, params)
                elif inventory_open and event.key == pygame.K_LEFT:
                    menu_focus = "inventory"
                elif inventory_open and event.key == pygame.K_RIGHT:
                    menu_focus = "crafting"
                elif event.key == pygame.K_TAB:
                    if inventory_open and menu_focus == "crafting":
                        action = Action.NEXT_RECIPE
                    else:
                        action = Action.NEXT_SLOT
                elif event.key == pygame.K_LEFTBRACKET:
                    if inventory_open and menu_focus == "crafting":
                        action = Action.PREV_RECIPE
                    else:
                        action = Action.PREV_SLOT
                elif event.key in key_to_player:
                    player_idx = key_to_player[event.key]
                    if player_idx < params.num_players:
                        state = state.replace(selected_player=player_idx)
                elif event.key in key_to_action:
                    action = key_to_action[event.key]

        if action != Action.NOOP:
            rng, step_key = random.split(rng)
            obs, state, reward, done, info = step_fn(step_key, state, action, params)

            if done:
                rng, reset_key = random.split(rng)
                obs, state = env.reset_env(reset_key, params)

        pixels = render_pixels(state)

        if inventory_open:
            menu_overlay = render_inventory_menu(
                state, base_width, base_height, menu_focus
            )
            pixels = composite_rgba_over_rgb(pixels, menu_overlay)

        if achievement_open:
            ach_overlay = render_achievement_menu(state, base_width, base_height)
            pixels = composite_rgba_over_rgb(pixels, ach_overlay)

        base_surface = pygame.surfarray.make_surface(np.transpose(pixels, (1, 0, 2)))
        scaled_surface = pygame.transform.scale(
            base_surface, (window_width, window_height)
        )
        screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
