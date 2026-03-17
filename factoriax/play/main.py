"""Interactive play script for FactoriaX using pygame."""

import jax
import numpy as np
import pygame
from jax import random

from factoriax.constants import BLOCK_PIXEL_SIZE, Action
from factoriax.envs.factoriax_env import make_factoriax_env
from factoriax.play.ui import (
    ClickRegion,
    render_achievement_menu,
    render_inventory_menu,
    render_pause_menu,
)
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


def hit_test_regions(
    regions: list[ClickRegion], x: int, y: int
) -> ClickRegion | None:
    """Find the first click region containing the given point.

    Args:
        regions: List of click regions to test.
        x: X coordinate in base resolution.
        y: Y coordinate in base resolution.

    Returns:
        The first matching ClickRegion, or None if no hit.
    """
    for region in regions:
        if (
            region.x <= x < region.x + region.w
            and region.y <= y < region.y + region.h
        ):
            return region
    return None


def calculate_window_size(
    base_width: int, base_height: int, scale_factor: float = 0.8
) -> tuple[int, int]:
    """Calculate window size using integer scaling for crisp pixel art.

    Uses the largest integer scale factor that fits within scale_factor
    (default 80%) of the screen. Integer scaling ensures every pixel is
    rendered at exactly the same size, preventing blurry text and artifacts.

    Args:
        base_width: Base render width in pixels
        base_height: Base render height in pixels
        scale_factor: Fraction of screen to use (0.0 to 1.0)

    Returns:
        Tuple of (window_width, window_height) in pixels
    """
    screen_info = pygame.display.Info()
    max_width = int(screen_info.current_w * scale_factor)
    max_height = int(screen_info.current_h * scale_factor)

    # Find the largest integer scale that fits the screen
    max_scale_w = max_width // base_width
    max_scale_h = max_height // base_height
    scale = max(1, min(max_scale_w, max_scale_h))

    return base_width * scale, base_height * scale


def main() -> None:
    """Run the interactive FactoriaX game.

    Controls:
        WASD: Move the selected player
        Space: Mine at current position
        I: Toggle inventory/crafting menu
        Tab: Switch between inventory and crafting sections (when menu open)
        Left/Right arrows: Navigate inventory slots (in inventory section)
        Up/Down arrows: Navigate recipes (in crafting section)
        C: Start crafting selected recipe
        E: Place (in world/inventory) or Craft (in crafting section)
        P: Toggle achievement menu
        1-5: Quick-select inventory slot 1-5
        Shift+1-5: Quick-select inventory slot 6-10
        Ctrl+1-9: Select player (if that many players exist)
        R: Reset the game
        Escape: Close menus / Open pause menu
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
    }

    key_to_slot = {
        pygame.K_1: 0,
        pygame.K_2: 1,
        pygame.K_3: 2,
        pygame.K_4: 3,
        pygame.K_5: 4,
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
    pause_open = False
    pause_selection = 0
    menu_focus = "inventory"

    scale = window_width // base_width
    click_regions: list[ClickRegion] = []

    running = True
    while running:
        action = Action.NOOP

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                base_x = event.pos[0] // scale
                base_y = event.pos[1] // scale
                hit = hit_test_regions(click_regions, base_x, base_y)
                if hit is not None:
                    if hit.action == "select_slot":
                        menu_focus = "inventory"
                        selected_player = int(state.selected_player)
                        new_slots = state.selected_slots.at[selected_player].set(
                            hit.param
                        )
                        state = state.replace(selected_slots=new_slots)
                    elif hit.action == "select_recipe":
                        menu_focus = "crafting"
                        selected_player = int(state.selected_player)
                        new_recipes = state.selected_recipes.at[selected_player].set(
                            hit.param
                        )
                        state = state.replace(selected_recipes=new_recipes)
                    elif hit.action == "focus_inventory":
                        menu_focus = "inventory"
                    elif hit.action == "focus_crafting":
                        menu_focus = "crafting"
                    elif hit.action == "pause_option":
                        if hit.param == 0:
                            pause_open = False
                        else:
                            running = False
            elif event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                shift_held = mods & pygame.KMOD_SHIFT
                ctrl_held = mods & pygame.KMOD_CTRL

                if event.key == pygame.K_ESCAPE:
                    if pause_open:
                        pause_open = False
                    elif inventory_open:
                        inventory_open = False
                    elif achievement_open:
                        achievement_open = False
                    else:
                        pause_open = True
                        pause_selection = 0
                elif pause_open:
                    if event.key == pygame.K_UP:
                        pause_selection = max(0, pause_selection - 1)
                    elif event.key == pygame.K_DOWN:
                        pause_selection = min(1, pause_selection + 1)
                    elif event.key in (pygame.K_RETURN, pygame.K_e):
                        if pause_selection == 0:
                            pause_open = False
                        else:
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
                elif inventory_open and event.key == pygame.K_TAB:
                    if menu_focus == "inventory":
                        menu_focus = "crafting"
                    else:
                        menu_focus = "inventory"
                elif inventory_open and menu_focus == "inventory":
                    if event.key == pygame.K_LEFT:
                        action = Action.PREV_SLOT
                    elif event.key == pygame.K_RIGHT:
                        action = Action.NEXT_SLOT
                elif inventory_open and menu_focus == "crafting":
                    if event.key == pygame.K_UP:
                        action = Action.PREV_RECIPE
                    elif event.key == pygame.K_DOWN:
                        action = Action.NEXT_RECIPE
                    elif event.key == pygame.K_e:
                        action = Action.CRAFT
                elif event.key == pygame.K_e:
                    action = Action.PLACE
                elif ctrl_held and event.key in key_to_player:
                    player_idx = key_to_player[event.key]
                    if player_idx < params.num_players:
                        state = state.replace(selected_player=player_idx)
                elif event.key in key_to_slot:
                    slot_idx = key_to_slot[event.key]
                    if shift_held:
                        slot_idx += 5
                    selected_player = int(state.selected_player)
                    new_slots = state.selected_slots.at[selected_player].set(slot_idx)
                    state = state.replace(selected_slots=new_slots)
                elif event.key in key_to_action:
                    action = key_to_action[event.key]

        if action != Action.NOOP:
            rng, step_key = random.split(rng)
            obs, state, reward, done, info = step_fn(step_key, state, action, params)

            if done:
                rng, reset_key = random.split(rng)
                obs, state = env.reset_env(reset_key, params)

        pixels = render_pixels(state)
        click_regions = []

        if inventory_open:
            menu_overlay, inv_regions = render_inventory_menu(
                state, base_width, base_height, menu_focus
            )
            pixels = composite_rgba_over_rgb(pixels, menu_overlay)
            click_regions.extend(inv_regions)

        if achievement_open:
            ach_overlay = render_achievement_menu(state, base_width, base_height)
            pixels = composite_rgba_over_rgb(pixels, ach_overlay)

        if pause_open:
            pause_overlay, pause_regions = render_pause_menu(
                base_width, base_height, pause_selection
            )
            pixels = composite_rgba_over_rgb(pixels, pause_overlay)
            click_regions.extend(pause_regions)

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
