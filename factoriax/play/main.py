"""Interactive play script for FactoriaX using pygame."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pygame
from jax import random

from factoriax.achievements import NUM_ACHIEVEMENTS
from factoriax.constants import (
    MACHINE_NUM_SLOTS,
    NUM_ASSEMBLER_RECIPES,
    NUM_INVENTORY_SLOTS,
    PLACEABLE_ITEMS,
    Action,
    MachineType,
)
from factoriax.envs.factoriax_env import make_factoriax_env
from factoriax.levels import Level
from factoriax.play.transfer import (
    deposit_to_machine,
    rotate_machine,
    swap_inventory_slots,
    withdraw_from_machine,
)
from factoriax.play.ui import (
    _HOTBAR_H,
    ClickRegion,
    render_achievement_menu,
    render_help_overlay,
    render_hotbar,
    render_inventory_menu,
    render_machine_menu,
    render_pause_menu,
    render_welcome_screen,
)
from factoriax.renderer import render_pixels
from factoriax.state import EnvParams


def composite_rgba_over_rgb(
    background: np.ndarray, overlay: np.ndarray
) -> None:
    """Composite an RGBA overlay onto an RGB background in-place.

    Only blends pixels within the bounding box of non-transparent
    overlay content, skipping the float arithmetic for the large
    fully-transparent regions that surround a centered menu panel.

    Args:
        background: RGB image array of shape (H, W, 3), modified
            in place.
        overlay: RGBA image array of shape (H, W, 4).
    """
    alpha_chan = overlay[:, :, 3]
    row_has_alpha = np.any(alpha_chan > 0, axis=1)
    if not np.any(row_has_alpha):
        return
    col_has_alpha = np.any(alpha_chan > 0, axis=0)

    r0 = int(np.argmax(row_has_alpha))
    r1 = len(row_has_alpha) - int(np.argmax(row_has_alpha[::-1]))
    c0 = int(np.argmax(col_has_alpha))
    c1 = len(col_has_alpha) - int(np.argmax(col_has_alpha[::-1]))

    a = overlay[r0:r1, c0:c1, 3:4].astype(np.float32) / 255.0
    fg = overlay[r0:r1, c0:c1, :3].astype(np.float32)
    bg = background[r0:r1, c0:c1].astype(np.float32)
    background[r0:r1, c0:c1] = (fg * a + bg * (1 - a)).astype(
        np.uint8
    )


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


_monitor_size: tuple[int, int] | None = None


def _get_monitor_size() -> tuple[int, int]:
    """Return the monitor resolution, cached on first call.

    ``pygame.display.Info()`` reports the monitor size before any
    display mode is set, but returns the *window* size afterwards.
    This function captures the true monitor dimensions once and
    reuses them for all subsequent calls.

    Returns:
        ``(width, height)`` of the primary monitor in pixels.
    """
    global _monitor_size  # noqa: PLW0603
    if _monitor_size is None:
        info = pygame.display.Info()
        _monitor_size = (info.current_w, info.current_h)
    return _monitor_size


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
    monitor_w, monitor_h = _get_monitor_size()
    max_width = int(monitor_w * scale_factor)
    max_height = int(monitor_h * scale_factor)

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


def play_level(
    level: Level,
    num_players: int = 1,
    screen: pygame.Surface | None = None,
) -> None:
    """Play a level with the full game UI.

    Provides the complete play experience including inventory, crafting,
    machine inspection, achievements, and pause menus.  When called from
    the editor the existing *screen* surface is reused and the function
    returns on quit instead of terminating pygame.

    Args:
        level: Level to play.
        num_players: Number of players to spawn.
        screen: Existing pygame display surface.  If ``None`` a new
            window is created and destroyed on exit.
    """
    owns_pygame = screen is None
    if owns_pygame:
        pygame.init()

    env, _ = make_factoriax_env()
    params = EnvParams(
        map_width=level.map_width,
        map_height=level.map_height,
        num_players=num_players,
    )

    prev_size = screen.get_size() if screen is not None else None
    game_win_w, game_win_h = calculate_window_size(_UI_SIZE, _UI_SIZE)
    if screen is None:
        screen = pygame.display.set_mode((game_win_w, game_win_h))
    else:
        screen = pygame.display.set_mode(
            (game_win_w, game_win_h),
            pygame.RESIZABLE,
        )

    pygame.display.set_caption(f"FactoriaX - {level.name}")

    obs, state = env.reset_from_level(level, params)
    rng = random.PRNGKey(42)

    step_fn = jax.jit(env.step_env)
    rng, warmup_key = random.split(rng)
    step_fn(
        warmup_key,
        state,
        jnp.int32(Action.NOOP),
        params,
    )[0].block_until_ready()

    _play_loop(env, state, params, level, screen, rng)

    if owns_pygame:
        pygame.quit()
    elif prev_size is not None:
        pygame.display.set_mode(prev_size, pygame.RESIZABLE)


_UI_SIZE = 1024

_PLACEABLE_ITEM_SET: frozenset[int] = frozenset(int(x) for x in PLACEABLE_ITEMS)


def _tile_pixel_size(map_w: int, map_h: int) -> int:
    """Choose a tile pixel size so the map fits within the UI canvas.

    Picks the largest size that keeps the full map visible above the
    hotbar, with a minimum of 8 pixels per tile so blocks remain
    distinguishable.

    Args:
        map_w: Map width in tiles.
        map_h: Map height in tiles.

    Returns:
        Tile side length in pixels.
    """
    world_h = _UI_SIZE - _HOTBAR_H
    return max(8, min(_UI_SIZE // map_w, world_h // map_h))


def _play_loop(
    env: object,
    state: object,
    params: EnvParams,
    level: Level | None,
    screen: pygame.Surface,
    rng: jax.Array,
) -> None:
    """Run the full interactive game loop with all menus and controls.

    The game world is rendered at a tile size chosen so the full map
    fits within the fixed ``_UI_SIZE x _UI_SIZE`` canvas.  Menus
    always render at the same canvas resolution, so their proportions
    are independent of map dimensions.

    Args:
        env: FactoriaX environment instance.
        state: Initial environment state.
        params: Environment parameters.
        level: Source level for reset, or ``None`` for procedural reset.
        screen: Pygame display surface.
        rng: JAX random key.
    """
    window_width, window_height = screen.get_size()
    step_fn = jax.jit(env.step_env)  # type: ignore[union-attr]
    clock = pygame.time.Clock()

    ui_w = _UI_SIZE
    ui_h = _UI_SIZE
    world_area_h = ui_h - _HOTBAR_H
    tile_px = _tile_pixel_size(params.map_width, params.map_height)
    world_pw = params.map_width * tile_px
    world_ph = params.map_height * tile_px
    world_ox = (ui_w - world_pw) // 2
    world_oy = (world_area_h - world_ph) // 2

    nav_keys = {
        pygame.K_w: "up",
        pygame.K_s: "down",
        pygame.K_a: "left",
        pygame.K_d: "right",
    }

    key_to_action = {
        pygame.K_SPACE: Action.MINE,
    }

    key_to_slot = {
        pygame.K_1: 0,
        pygame.K_2: 1,
        pygame.K_3: 2,
        pygame.K_4: 3,
        pygame.K_5: 4,
        pygame.K_6: 5,
        pygame.K_7: 6,
        pygame.K_8: 7,
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
    achievement_scroll = 0
    achievement_selection = 0
    pause_open = False
    pause_selection = 0
    help_open = False
    menu_focus = "inventory"
    machine_open = False
    machine_tx = 0
    machine_ty = 0
    machine_panel_active = True
    hotbar_page = 0
    held_slot: int | None = None
    welcome_open = True

    win_scale = max(1, min(window_width // ui_w, window_height // ui_h))
    win_ox = (window_width - ui_w * win_scale) // 2
    win_oy = (window_height - ui_h * win_scale) // 2
    click_regions: list[ClickRegion] = []

    frame_tick = 0
    running = True
    while running:
        action = Action.NOOP

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif welcome_open:
                if event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_SPACE,
                    pygame.K_RETURN,
                    pygame.K_ESCAPE,
                ):
                    welcome_open = False
                continue
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                base_x = (event.pos[0] - win_ox) // win_scale
                base_y = (event.pos[1] - win_oy) // win_scale
                hit = hit_test_regions(click_regions, base_x, base_y)
                if hit is not None:
                    if hit.action == "select_slot":
                        selected_player = int(state.selected_player)  # type: ignore[union-attr]
                        new_slots = state.selected_slots.at[selected_player].set(  # type: ignore[union-attr]
                            hit.param
                        )
                        state = state.replace(selected_slots=new_slots)  # type: ignore[union-attr]
                        if machine_open:
                            machine_panel_active = False
                        else:
                            menu_focus = "inventory"
                    elif hit.action == "select_recipe":
                        menu_focus = "crafting"
                        selected_player = int(state.selected_player)  # type: ignore[union-attr]
                        new_recipes = state.selected_recipes.at[selected_player].set(  # type: ignore[union-attr]
                            hit.param
                        )
                        state = state.replace(selected_recipes=new_recipes)  # type: ignore[union-attr]
                    elif hit.action == "select_machine_slot":
                        machine_type = int(state.machine_types[machine_ty, machine_tx])  # type: ignore[union-attr]
                        num_slots = int(MACHINE_NUM_SLOTS[machine_type])
                        if 0 <= hit.param < num_slots:
                            new_sel = state.machine_selected_slot.at[  # type: ignore[union-attr]
                                machine_ty, machine_tx
                            ].set(hit.param)
                            state = state.replace(machine_selected_slot=new_sel)  # type: ignore[union-attr]
                        machine_panel_active = True
                    elif hit.action == "toggle_held":
                        if held_slot is None:
                            held_slot = hit.param
                            selected_player = int(state.selected_player)  # type: ignore[union-attr]
                            new_slots = state.selected_slots.at[selected_player].set(  # type: ignore[union-attr]
                                hit.param
                            )
                            state = state.replace(selected_slots=new_slots)  # type: ignore[union-attr]
                        elif held_slot == hit.param:
                            held_slot = None
                        else:
                            selected_player = int(state.selected_player)  # type: ignore[union-attr]
                            state = swap_inventory_slots(
                                state,
                                selected_player,
                                held_slot,
                                hit.param,
                            )
                            held_slot = None
                    elif hit.action == "hotbar_page":
                        hotbar_page = 1 - hotbar_page
                    elif hit.action == "focus_inventory":
                        menu_focus = "inventory"
                    elif hit.action == "focus_crafting":
                        menu_focus = "crafting"
                    elif hit.action == "pause_option":
                        if hit.param == 0:
                            pause_open = False
                        else:
                            running = False
                elif not (
                    inventory_open
                    or achievement_open
                    or machine_open
                    or pause_open
                    or welcome_open
                    or help_open
                ):
                    selected_player = int(state.selected_player)  # type: ignore[union-attr]
                    slot_idx = int(state.selected_slots[selected_player])  # type: ignore[union-attr]
                    item_type = int(state.inventory_items[selected_player, slot_idx])  # type: ignore[union-attr]
                    if item_type in _PLACEABLE_ITEM_SET:
                        action = Action.PLACE
            elif event.type == pygame.KEYDOWN:
                if help_open:
                    help_open = False
                    continue

                mods = pygame.key.get_mods()
                shift_held = mods & pygame.KMOD_SHIFT
                ctrl_held = mods & pygame.KMOD_CTRL

                if event.key == pygame.K_ESCAPE:
                    if pause_open:
                        pause_open = False
                    elif inventory_open:
                        inventory_open = False
                        held_slot = None
                    elif achievement_open:
                        achievement_open = False
                    elif machine_open:
                        machine_open = False
                    else:
                        pause_open = True
                        pause_selection = 0
                elif pause_open:
                    if event.key == pygame.K_w:
                        pause_selection = max(0, pause_selection - 1)
                    elif event.key == pygame.K_s:
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
                        selected_player = int(state.selected_player)  # type: ignore[union-attr]
                        tx, ty = _tile_in_front(state, selected_player)
                        map_h, map_w = state.map.shape  # type: ignore[union-attr]
                        if (
                            0 <= tx < map_w
                            and 0 <= ty < map_h
                            and int(state.machine_types[ty, tx])  # type: ignore[union-attr]
                            != int(MachineType.NONE)
                        ):
                            machine_tx, machine_ty = tx, ty
                            machine_open = True
                            machine_panel_active = True
                            inventory_open = False
                            achievement_open = False
                            pause_open = False
                elif machine_open:
                    if event.key in (pygame.K_w, pygame.K_s):
                        machine_panel_active = not machine_panel_active
                    elif event.key in (pygame.K_a, pygame.K_d):
                        delta = -1 if event.key == pygame.K_a else 1
                        selected_player = int(state.selected_player)  # type: ignore[union-attr]
                        if machine_panel_active:
                            machine_type = int(
                                state.machine_types[machine_ty, machine_tx]  # type: ignore[union-attr]
                            )
                            num_slots = int(MACHINE_NUM_SLOTS[machine_type])
                            if num_slots > 0:
                                current = int(
                                    state.machine_selected_slot[machine_ty, machine_tx]  # type: ignore[union-attr]
                                )
                                new_slot = (current + delta) % num_slots
                                new_sel = state.machine_selected_slot.at[  # type: ignore[union-attr]
                                    machine_ty, machine_tx
                                ].set(new_slot)
                                state = state.replace(machine_selected_slot=new_sel)  # type: ignore[union-attr]
                        else:
                            current = int(state.selected_slots[selected_player])  # type: ignore[union-attr]
                            new_slot = (current + delta) % NUM_INVENTORY_SLOTS
                            new_slots = state.selected_slots.at[selected_player].set(  # type: ignore[union-attr]
                                new_slot
                            )
                            state = state.replace(selected_slots=new_slots)  # type: ignore[union-attr]
                    elif event.key == pygame.K_e:
                        selected_player = int(state.selected_player)  # type: ignore[union-attr]
                        machine_slot = int(
                            state.machine_selected_slot[machine_ty, machine_tx]  # type: ignore[union-attr]
                        )
                        player_slot = int(state.selected_slots[selected_player])  # type: ignore[union-attr]
                        if machine_panel_active:
                            state = withdraw_from_machine(
                                state,
                                selected_player,
                                machine_tx,
                                machine_ty,
                                machine_slot,
                            )
                        else:
                            state = deposit_to_machine(
                                state,
                                selected_player,
                                machine_tx,
                                machine_ty,
                                machine_slot,
                                player_slot,
                            )
                elif event.key == pygame.K_q:
                    machine_type = int(
                        state.machine_types[machine_ty, machine_tx]  # type: ignore[union-attr]
                    )
                    is_idle = int(state.machine_power[machine_ty, machine_tx]) == 0  # type: ignore[union-attr]
                    mc = state.machine_inventory_counts  # type: ignore[union-attr]
                    has_inputs = (
                        int(mc[machine_ty, machine_tx, 0]) > 0
                        or int(mc[machine_ty, machine_tx, 1]) > 0
                        or int(mc[machine_ty, machine_tx, 2]) > 0
                    )
                    if (
                        machine_type == int(MachineType.ASSEMBLER)
                        and is_idle
                        and not has_inputs
                    ):
                        cur_recipe = int(
                            state.machine_selected_recipe[machine_ty, machine_tx]  # type: ignore[union-attr]
                        )
                        new_recipe = (cur_recipe + 1) % NUM_ASSEMBLER_RECIPES
                        new_sel = state.machine_selected_recipe.at[  # type: ignore[union-attr]
                            machine_ty, machine_tx
                        ].set(new_recipe)
                        state = state.replace(machine_selected_recipe=new_sel)  # type: ignore[union-attr]
                    elif machine_type != int(MachineType.ASSEMBLER):
                        hotbar_page = 1 - hotbar_page
                elif event.key == pygame.K_i:
                    inventory_open = not inventory_open
                    if inventory_open:
                        achievement_open = False
                        machine_open = False
                    else:
                        held_slot = None
                elif event.key == pygame.K_p:
                    achievement_open = not achievement_open
                    if achievement_open:
                        inventory_open = False
                        achievement_scroll = 0
                        achievement_selection = 0
                elif achievement_open:
                    row_h = 36
                    if event.key == pygame.K_w:
                        achievement_selection = max(
                            0, achievement_selection - 1
                        )
                    elif event.key == pygame.K_s:
                        achievement_selection = min(
                            NUM_ACHIEVEMENTS - 1,
                            achievement_selection + 1,
                        )
                    sel_top = achievement_selection * row_h
                    sel_bot = sel_top + row_h
                    if sel_top < achievement_scroll:
                        achievement_scroll = sel_top
                    elif sel_bot > achievement_scroll + 8 * row_h:
                        achievement_scroll = sel_bot - 8 * row_h
                elif event.key == pygame.K_QUESTION or (
                    event.key == pygame.K_SLASH and shift_held
                ):
                    help_open = True
                elif event.key == pygame.K_r:
                    if level is not None:
                        obs, state = env.reset_from_level(level, params)  # type: ignore[union-attr]
                    else:
                        rng, reset_key = random.split(rng)
                        obs, state = env.reset_env(reset_key, params)  # type: ignore[union-attr]
                elif inventory_open and menu_focus == "inventory":
                    selected_player = int(state.selected_player)  # type: ignore[union-attr]
                    current = int(state.selected_slots[selected_player])  # type: ignore[union-attr]
                    col = current % 5
                    if event.key == pygame.K_a:
                        if col == 0:
                            pass  # leftmost column, nowhere to go
                        else:
                            action = Action.PREV_SLOT
                    elif event.key == pygame.K_d:
                        if col == 4:
                            menu_focus = "crafting"
                        else:
                            action = Action.NEXT_SLOT
                    elif event.key == pygame.K_w:
                        if current >= 5:
                            new_sel = state.selected_slots.at[  # type: ignore[union-attr]
                                selected_player
                            ].set(current - 5)
                            state = state.replace(selected_slots=new_sel)  # type: ignore[union-attr]
                    elif event.key == pygame.K_s:
                        if current < 5:
                            new_sel = state.selected_slots.at[  # type: ignore[union-attr]
                                selected_player
                            ].set(current + 5)
                            state = state.replace(selected_slots=new_sel)  # type: ignore[union-attr]
                elif inventory_open and menu_focus == "crafting":
                    if event.key == pygame.K_w:
                        action = Action.PREV_RECIPE
                    elif event.key == pygame.K_s:
                        action = Action.NEXT_RECIPE
                    elif event.key == pygame.K_a:
                        menu_focus = "inventory"
                    elif event.key == pygame.K_e:
                        action = Action.CRAFT
                elif event.key == pygame.K_e:
                    selected_player = int(state.selected_player)  # type: ignore[union-attr]
                    tx, ty = _tile_in_front(state, selected_player)
                    map_h, map_w = state.map.shape  # type: ignore[union-attr]
                    has_machine = (
                        0 <= tx < map_w
                        and 0 <= ty < map_h
                        and int(state.machine_types[ty, tx])  # type: ignore[union-attr]
                        != int(MachineType.NONE)
                    )
                    action = Action.PICKUP if has_machine else Action.PLACE
                elif event.key == pygame.K_t:
                    selected_player = int(state.selected_player)  # type: ignore[union-attr]
                    tx, ty = _tile_in_front(state, selected_player)
                    map_h, map_w = state.map.shape  # type: ignore[union-attr]
                    if (
                        0 <= tx < map_w
                        and 0 <= ty < map_h
                    ):
                        state = rotate_machine(state, tx, ty)
                elif ctrl_held and event.key in key_to_player:
                    player_idx = key_to_player[event.key]
                    if player_idx < params.num_players:
                        state = state.replace(selected_player=player_idx)  # type: ignore[union-attr]
                elif event.key in key_to_slot:
                    slot_idx = key_to_slot[event.key]
                    if shift_held:
                        slot_idx += 8
                    if slot_idx < NUM_INVENTORY_SLOTS:
                        selected_player = int(state.selected_player)  # type: ignore[union-attr]
                        new_slots = state.selected_slots.at[selected_player].set(  # type: ignore[union-attr]
                            slot_idx,
                        )
                        state = state.replace(selected_slots=new_slots)  # type: ignore[union-attr]
                elif event.key in key_to_action:
                    action = key_to_action[event.key]
                elif event.key in nav_keys:
                    nav_dir = nav_keys[event.key]
                    nav_to_action = {
                        "up": Action.UP,
                        "down": Action.DOWN,
                        "left": Action.LEFT,
                        "right": Action.RIGHT,
                    }
                    action = nav_to_action[nav_dir]

        if action != Action.NOOP:
            rng, step_key = random.split(rng)
            obs, state, reward, done, info = step_fn(
                step_key,
                state,
                action,
                params,
            )
            if done:
                if level is not None:
                    obs, state = env.reset_from_level(level, params)  # type: ignore[union-attr]
                else:
                    rng, reset_key = random.split(rng)
                    obs, state = env.reset_env(reset_key, params)  # type: ignore[union-attr]

        pixels = render_pixels(
            state, block_pixel_size=tile_px, frame_tick=frame_tick
        )
        click_regions = []

        # Build the UI frame.  The world is rendered at a tile size
        # chosen to fit the fixed canvas.  Menus overlay at the same
        # canvas resolution so their proportions never change.
        ui_frame = np.zeros((ui_h, ui_w, 3), dtype=np.uint8)
        ph, pw = pixels.shape[:2]
        ui_frame[world_oy : world_oy + ph, world_ox : world_ox + pw] = pixels

        # Persistent hotbar at the bottom of every frame.
        hotbar_overlay, hotbar_regions = render_hotbar(
            state, ui_w, ui_h, hotbar_page
        )
        composite_rgba_over_rgb(ui_frame, hotbar_overlay)
        click_regions.extend(hotbar_regions)

        if machine_open:
            machine_overlay, machine_regions = render_machine_menu(
                state,
                ui_w,
                ui_h,
                machine_tx,
                machine_ty,
                machine_panel_active,
            )
            composite_rgba_over_rgb(ui_frame, machine_overlay)
            click_regions.extend(machine_regions)

        if inventory_open:
            menu_overlay, inv_regions = render_inventory_menu(
                state,
                ui_w,
                ui_h,
                menu_focus,
                held_slot,
            )
            composite_rgba_over_rgb(ui_frame, menu_overlay)
            click_regions.extend(inv_regions)

        if achievement_open:
            ach_overlay = render_achievement_menu(
                state,
                ui_w,
                ui_h,
                achievement_scroll,
                achievement_selection,
            )
            composite_rgba_over_rgb(ui_frame, ach_overlay)

        if pause_open:
            pause_overlay, pause_regions = render_pause_menu(
                ui_w,
                ui_h,
                pause_selection,
            )
            composite_rgba_over_rgb(ui_frame, pause_overlay)
            click_regions.extend(pause_regions)

        if help_open:
            composite_rgba_over_rgb(
                ui_frame, render_help_overlay(ui_w, ui_h)
            )

        if welcome_open:
            composite_rgba_over_rgb(
                ui_frame, render_welcome_screen(ui_w, ui_h)
            )

        final_surface = pygame.surfarray.make_surface(
            np.transpose(ui_frame, (1, 0, 2)),
        )
        scaled_surface = pygame.transform.scale(
            final_surface,
            (ui_w * win_scale, ui_h * win_scale),
        )
        screen.fill((0, 0, 0))
        screen.blit(scaled_surface, (win_ox, win_oy))
        pygame.display.flip()
        frame_tick += 1
        clock.tick(30)


def main() -> None:
    """Run the interactive FactoriaX game.

    Controls:
        WASD: Move player (world), navigate menus (context-dependent)
        Space: Mine ore at current tile
        E: Place/pick up (world), transfer (machine), craft (crafting)
        F: Inspect machine in front of player
        I: Toggle inventory/crafting menu
        A/D: Select inventory slot, edge-wrap to crafting panel
        W/S: Navigate rows (inventory), recipes (crafting), panels (machine)
        P: Toggle achievement menu
        1-8: Quick-select inventory slot 1-8
        Shift+1-2: Quick-select inventory slot 9-10
        Ctrl+1-9: Select player (if that many players exist)
        R: Reset the game
        Escape: Close menus / Open pause menu
    """
    pygame.init()

    env, params = make_factoriax_env()

    window_width, window_height = calculate_window_size(
        _UI_SIZE,
        _UI_SIZE,
    )
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("FactoriaX")

    rng = random.PRNGKey(42)
    rng, reset_key = random.split(rng)
    obs, state = env.reset_env(reset_key, params)

    step_fn = jax.jit(env.step_env)
    rng, warmup_key = random.split(rng)
    step_fn(
        warmup_key,
        state,
        jnp.int32(Action.NOOP),
        params,
    )[0].block_until_ready()

    _play_loop(env, state, params, None, screen, rng)

    pygame.quit()


if __name__ == "__main__":
    main()
