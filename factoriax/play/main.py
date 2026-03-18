"""Interactive play script for FactoriaX using pygame."""

import jax
import jax.numpy as jnp
import numpy as np
import pygame
from jax import random

from factoriax.constants import (
    BLOCK_PIXEL_SIZE,
    MACHINE_NUM_SLOTS,
    NUM_INVENTORY_SLOTS,
    Action,
    MachineType,
)
from factoriax.envs.factoriax_env import make_factoriax_env
from factoriax.play.transfer import deposit_to_machine, withdraw_from_machine
from factoriax.play.ui import (
    SCROLL_STEP,
    ClickRegion,
    render_achievement_menu,
    render_inventory_menu,
    render_machine_menu,
    render_pause_menu,
    render_welcome_screen,
)
from factoriax.renderer import render_pixels


def composite_rgba_over_rgb(background: np.ndarray, overlay: np.ndarray) -> np.ndarray:
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


def hit_test_regions(regions: list[ClickRegion], x: int, y: int) -> ClickRegion | None:
    """Find the first click region containing the given point.

    Args:
        regions: List of click regions to test.
        x: X coordinate in base resolution.
        y: Y coordinate in base resolution.

    Returns:
        The first matching ClickRegion, or None if no hit.
    """
    for region in regions:
        if region.x <= x < region.x + region.w and region.y <= y < region.y + region.h:
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


def _tile_in_front(state: object, player_idx: int) -> tuple[int, int]:
    """Return the (x, y) tile immediately in front of a player.

    Uses the player's current facing direction to compute a one-tile offset.
    The result may be out of map bounds — callers must validate before use.

    Args:
        state: Current environment state.
        player_idx: Index of the player.

    Returns:
        ``(tx, ty)`` tile coordinates in front of the player.
    """
    pos = np.array(state.player_positions[player_idx])  # type: ignore[attr-defined]
    direction = int(state.player_directions[player_idx])  # type: ignore[attr-defined]
    offsets: dict[int, tuple[int, int]] = {
        int(Action.LEFT): (-1, 0),
        int(Action.RIGHT): (1, 0),
        int(Action.UP): (0, -1),
        int(Action.DOWN): (0, 1),
    }
    dx, dy = offsets.get(direction, (0, 0))
    return int(pos[0]) + dx, int(pos[1]) + dy


def main() -> None:
    """Run the interactive FactoriaX game.

    Controls:
        WASD: Move the selected player
        Space: Mine at current position
        F: Inspect machine in front of the selected player
        I: Toggle inventory/crafting menu
        Tab: Switch between inventory and crafting sections (when menu open)
        Left/Right arrows: Navigate inventory slots / machine slots
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

    # Trigger JIT compilation before the display loop so the first real
    # keypress is not delayed by tracing.
    rng, warmup_key = random.split(rng)
    step_fn(warmup_key, state, jnp.int32(Action.NOOP), params)[0].block_until_ready()

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

    welcome_open = True
    inventory_open = False
    achievement_open = False
    achievement_scroll = 0
    pause_open = False
    pause_selection = 0
    menu_focus = "inventory"
    machine_open = False
    machine_tx = 0
    machine_ty = 0
    machine_panel_active = True  # True = machine slots focused, False = player strip

    scale = window_width // base_width
    click_regions: list[ClickRegion] = []

    running = True
    while running:
        action = Action.NOOP

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and not welcome_open
            ):
                base_x = event.pos[0] // scale
                base_y = event.pos[1] // scale
                hit = hit_test_regions(click_regions, base_x, base_y)
                if hit is not None:
                    if hit.action == "select_slot":
                        selected_player = int(state.selected_player)
                        new_slots = state.selected_slots.at[selected_player].set(
                            hit.param
                        )
                        state = state.replace(selected_slots=new_slots)
                        if machine_open:
                            machine_panel_active = False
                        else:
                            menu_focus = "inventory"
                    elif hit.action == "select_recipe":
                        menu_focus = "crafting"
                        selected_player = int(state.selected_player)
                        new_recipes = state.selected_recipes.at[selected_player].set(
                            hit.param
                        )
                        state = state.replace(selected_recipes=new_recipes)
                    elif hit.action == "select_machine_slot":
                        machine_type = int(state.machine_types[machine_ty, machine_tx])
                        num_slots = int(MACHINE_NUM_SLOTS[machine_type])
                        if 0 <= hit.param < num_slots:
                            new_sel = state.machine_selected_slot.at[
                                machine_ty, machine_tx
                            ].set(hit.param)
                            state = state.replace(machine_selected_slot=new_sel)
                        machine_panel_active = True
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

                if welcome_open:
                    if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        welcome_open = False
                    continue

                if event.key == pygame.K_ESCAPE:
                    if pause_open:
                        pause_open = False
                    elif inventory_open:
                        inventory_open = False
                    elif achievement_open:
                        achievement_open = False
                    elif machine_open:
                        machine_open = False
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
                elif event.key == pygame.K_f:
                    if machine_open:
                        machine_open = False
                    else:
                        selected_player = int(state.selected_player)
                        tx, ty = _tile_in_front(state, selected_player)
                        map_h, map_w = state.map.shape
                        if (
                            0 <= tx < map_w
                            and 0 <= ty < map_h
                            and int(state.machine_types[ty, tx])
                            != int(MachineType.NONE)
                        ):
                            machine_tx, machine_ty = tx, ty
                            machine_open = True
                            machine_panel_active = True
                            inventory_open = False
                            achievement_open = False
                            pause_open = False
                elif machine_open:
                    if event.key == pygame.K_TAB:
                        machine_panel_active = not machine_panel_active
                    elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                        delta = -1 if event.key == pygame.K_LEFT else 1
                        selected_player = int(state.selected_player)
                        if machine_panel_active:
                            machine_type = int(
                                state.machine_types[machine_ty, machine_tx]
                            )
                            num_slots = int(MACHINE_NUM_SLOTS[machine_type])
                            if num_slots > 0:
                                current = int(
                                    state.machine_selected_slot[machine_ty, machine_tx]
                                )
                                new_slot = (current + delta) % num_slots
                                new_sel = state.machine_selected_slot.at[
                                    machine_ty, machine_tx
                                ].set(new_slot)
                                state = state.replace(machine_selected_slot=new_sel)
                        else:
                            current = int(state.selected_slots[selected_player])
                            new_slot = (current + delta) % NUM_INVENTORY_SLOTS
                            new_slots = state.selected_slots.at[selected_player].set(
                                new_slot
                            )
                            state = state.replace(selected_slots=new_slots)
                    elif event.key == pygame.K_e:
                        selected_player = int(state.selected_player)
                        machine_slot = int(
                            state.machine_selected_slot[machine_ty, machine_tx]
                        )
                        player_slot = int(state.selected_slots[selected_player])
                        if machine_panel_active:
                            state = withdraw_from_machine(
                                state, selected_player,
                                machine_tx, machine_ty, machine_slot,
                            )
                        else:
                            state = deposit_to_machine(
                                state, selected_player,
                                machine_tx, machine_ty, machine_slot, player_slot,
                            )
                elif event.key == pygame.K_i:
                    inventory_open = not inventory_open
                    if inventory_open:
                        achievement_open = False
                        machine_open = False
                elif event.key == pygame.K_p:
                    achievement_open = not achievement_open
                    if achievement_open:
                        inventory_open = False
                        achievement_scroll = 0
                elif achievement_open:
                    if event.key == pygame.K_UP:
                        achievement_scroll = max(0, achievement_scroll - SCROLL_STEP)
                    elif event.key == pygame.K_DOWN:
                        achievement_scroll += SCROLL_STEP
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

        if machine_open:
            machine_overlay, machine_regions = render_machine_menu(
                state, base_width, base_height, machine_tx, machine_ty,
                machine_panel_active,
            )
            pixels = composite_rgba_over_rgb(pixels, machine_overlay)
            click_regions.extend(machine_regions)

        if inventory_open:
            menu_overlay, inv_regions = render_inventory_menu(
                state, base_width, base_height, menu_focus
            )
            pixels = composite_rgba_over_rgb(pixels, menu_overlay)
            click_regions.extend(inv_regions)

        if achievement_open:
            ach_overlay = render_achievement_menu(
                state, base_width, base_height, achievement_scroll
            )
            pixels = composite_rgba_over_rgb(pixels, ach_overlay)

        if pause_open:
            pause_overlay, pause_regions = render_pause_menu(
                base_width, base_height, pause_selection
            )
            pixels = composite_rgba_over_rgb(pixels, pause_overlay)
            click_regions.extend(pause_regions)

        if welcome_open:
            welcome_overlay = render_welcome_screen(base_width, base_height)
            pixels = composite_rgba_over_rgb(pixels, welcome_overlay)

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
