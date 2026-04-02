"""Interactive play script for FactoriaX using pygame."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC

import jax
import jax.numpy as jnp
import numpy as np
import pygame
from jax import random

from factoriax.achievements import ACHIEVEMENT_INFO, NUM_ACHIEVEMENTS
from factoriax.constants import (
    MACHINE_NUM_SLOTS,
    NUM_INVENTORY_SLOTS,
    NUM_TECHNOLOGIES,
    PLACEABLE_ITEMS,
    Action,
    Direction,
    MachineType,
)
from factoriax.envs.factoriax_env import FactoriaXEnv, make_factoriax_env
from factoriax.levels import Level
from factoriax.play.play_state import PlayState
from factoriax.play.transfer import swap_inventory_slots
from factoriax.play.ui import (
    _HOTBAR_H,
    render_achievement_menu,
    render_help_overlay,
    render_hotbar,
    render_info_panel,
    render_inventory_menu,
    render_machine_menu,
    render_pause_menu,
    render_research_menu,
    render_victory_screen,
    render_welcome_screen,
)
from factoriax.recipes import NUM_ASSEMBLER_RECIPES, NUM_RECIPES
from factoriax.renderer import render_pixels
from factoriax.state import EnvParams, EnvState
from factoriax.ui.compositing import composite_rgba_over_rgb
from factoriax.ui.primitives import ClickRegion, hit_test_regions
from factoriax.ui.window import calculate_window_size

_ROCKET_ACHIEVEMENT_IDX: int = next(
    i for i, a in enumerate(ACHIEVEMENT_INFO) if a.id == "rocket_complete"
)


def _run_with_loading_screen(
    screen: pygame.Surface,
    message: str,
    fn: Callable[[], object],
) -> object:
    """Run *fn* on a background thread while showing a loading message.

    Keeps the pygame event loop alive so the OS does not flag the
    window as unresponsive during long JAX compilations.

    Args:
        screen: Pygame display surface.
        message: Text to show while waiting.
        fn: Blocking callable to run in the background.

    Returns:
        Whatever *fn* returned.
    """
    import threading

    result: list[object] = []

    def _worker() -> None:
        result.append(fn())

    thread = threading.Thread(target=_worker)
    thread.start()

    font = pygame.font.SysFont(None, 28)
    clock = pygame.time.Clock()
    dots = 0

    while thread.is_alive():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                thread.join()
                raise SystemExit
        dots = (dots + 1) % 4
        label = message + "." * dots
        screen.fill((20, 20, 25))
        text = font.render(label, True, (180, 180, 180))
        cx = (screen.get_width() - text.get_width()) // 2
        cy = (screen.get_height() - text.get_height()) // 2
        screen.blit(text, (cx, cy))
        pygame.display.flip()
        clock.tick(8)

    thread.join()
    return result[0] if result else None


def _tile_in_front(state: EnvState, player_idx: int) -> tuple[int, int]:
    """Return the (x, y) tile immediately in front of a player.

    Uses the player's current facing direction to compute a one-tile offset.
    The result may be out of map bounds — callers must validate before use.

    Args:
        state: Current environment state.
        player_idx: Index of the player.

    Returns:
        ``(tx, ty)`` tile coordinates in front of the player.
    """
    pos = np.array(state.player_positions[player_idx])
    direction = int(state.player_directions[player_idx])
    offsets: dict[int, tuple[int, int]] = {
        int(Direction.LEFT): (-1, 0),
        int(Direction.RIGHT): (1, 0),
        int(Direction.UP): (0, -1),
        int(Direction.DOWN): (0, 1),
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

    reset_result: tuple[jax.Array, EnvState] = _run_with_loading_screen(  # type: ignore[assignment]
        screen, "Building world",
        lambda: env.reset_from_level(level, params),
    )
    _, state = reset_result
    rng = random.PRNGKey(42)

    step_fn = jax.jit(env.step_env)
    rng, warmup_key = random.split(rng)
    _warmup_key = warmup_key

    def _warmup() -> None:
        step_fn(
            _warmup_key, state, jnp.int32(Action.NOOP), params,
        )[0].block_until_ready()

    _run_with_loading_screen(
        screen, "Compiling JAX (first run only)", _warmup,
    )

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


def _handle_welcome_event(
    event: pygame.event.Event,
    ps: PlayState,
    state: EnvState,
    win_ox: int,
    win_oy: int,
    win_scale: int,
    ui_w: int,
    ui_h: int,
) -> tuple[PlayState, EnvState]:
    """Process events while the welcome screen is showing.

    Args:
        event: Pygame event.
        ps: Current play state.
        state: Current environment state.
        win_ox: Window X offset for coordinate transform.
        win_oy: Window Y offset for coordinate transform.
        win_scale: Window scale factor.
        ui_w: UI canvas width.
        ui_h: UI canvas height.

    Returns:
        Updated play state and environment state.
    """
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_r:
            ps.record_enabled = not ps.record_enabled
        elif event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_ESCAPE):
            ps.welcome_open = False
            if ps.record_enabled:
                ps.recorded_states.append(state)
    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        mx = (event.pos[0] - win_ox) // win_scale
        my = (event.pos[1] - win_oy) // win_scale
        _, welcome_regions = render_welcome_screen(ui_w, ui_h, ps.record_enabled)
        hit = hit_test_regions(welcome_regions, mx, my)
        if hit is not None and hit.action == "toggle_record":
            ps.record_enabled = not ps.record_enabled
    return ps, state


def _handle_click(
    event: pygame.event.Event,
    ps: PlayState,
    state: EnvState,
    env: FactoriaXEnv,
    params: EnvParams,
    rng: jax.Array,
    level: Level | None,
    win_ox: int,
    win_oy: int,
    win_scale: int,
) -> tuple[PlayState, EnvState, jax.Array, int, bool]:
    """Process a mouse click against the current click regions.

    Args:
        event: Pygame MOUSEBUTTONDOWN event.
        ps: Current play state.
        state: Current environment state.
        env: FactoriaX environment instance.
        params: Environment parameters.
        rng: JAX random key.
        level: Source level for reset.
        win_ox: Window X offset for coordinate transform.
        win_oy: Window Y offset for coordinate transform.
        win_scale: Window scale factor.

    Returns:
        Tuple of (play_state, env_state, rng, action, running).
    """
    action = int(Action.NOOP)
    running = True

    base_x = (event.pos[0] - win_ox) // win_scale
    base_y = (event.pos[1] - win_oy) // win_scale
    hit = hit_test_regions(ps.click_regions, base_x, base_y)

    if hit is not None:
        if hit.action == "select_slot":
            selected_player = int(state.selected_player)
            new_slots = state.selected_slots.at[selected_player].set(hit.param)
            state = state.replace(selected_slots=new_slots)
            if ps.machine_open:
                ps.machine_panel_active = False
            else:
                ps.menu_focus = "inventory"
        elif hit.action == "select_recipe":
            ps.menu_focus = "crafting"
            ps.selected_recipe = hit.param
        elif hit.action == "select_machine_slot":
            machine_type = int(state.machine_types[ps.machine_ty, ps.machine_tx])
            num_slots = int(MACHINE_NUM_SLOTS[machine_type])
            if 0 <= hit.param < num_slots:
                new_sel = state.machine_selected_slot.at[
                    ps.machine_ty, ps.machine_tx
                ].set(hit.param)
                state = state.replace(machine_selected_slot=new_sel)
            ps.machine_panel_active = True
        elif hit.action == "toggle_held":
            if ps.held_slot is None:
                ps.held_slot = hit.param
                selected_player = int(state.selected_player)
                new_slots = state.selected_slots.at[selected_player].set(hit.param)
                state = state.replace(selected_slots=new_slots)
            elif ps.held_slot == hit.param:
                ps.held_slot = None
            else:
                selected_player = int(state.selected_player)
                state = swap_inventory_slots(
                    state, selected_player, ps.held_slot, hit.param,
                )
                ps.held_slot = None
        elif hit.action == "hotbar_page":
            ps.hotbar_page = 1 - ps.hotbar_page
        elif hit.action == "focus_inventory":
            ps.menu_focus = "inventory"
        elif hit.action == "focus_crafting":
            ps.menu_focus = "crafting"
        elif hit.action == "pause_option":
            if hit.param == 0:
                ps.pause_open = False
            elif hit.param == 1:
                ps.pause_open = False
                if level is not None:
                    _, state = env.reset_from_level(level, params)
                else:
                    rng, reset_key = random.split(rng)
                    _, state = env.reset_env(reset_key, params)
            else:
                running = False
    elif not (
        ps.inventory_open or ps.achievement_open or ps.research_open
        or ps.machine_open or ps.pause_open or ps.welcome_open or ps.help_open
    ):
        selected_player = int(state.selected_player)
        slot_idx = int(state.selected_slots[selected_player])
        item_type = int(state.inventory_items[selected_player, slot_idx])
        if item_type in _PLACEABLE_ITEM_SET:
            action = int(Action.PLACE)

    return ps, state, rng, action, running


def _handle_keydown(
    event: pygame.event.Event,
    ps: PlayState,
    state: EnvState,
    env: FactoriaXEnv,
    params: EnvParams,
    rng: jax.Array,
    level: Level | None,
) -> tuple[PlayState, EnvState, jax.Array, int, bool]:
    """Dispatch a KEYDOWN event to the appropriate context handler.

    Args:
        event: Pygame KEYDOWN event.
        ps: Current play state.
        state: Current environment state.
        env: FactoriaX environment instance.
        params: Environment parameters.
        rng: JAX random key.
        level: Source level for reset.

    Returns:
        Tuple of (play_state, env_state, rng, action, running).
    """
    action = int(Action.NOOP)
    running = True

    if ps.help_open:
        ps.help_open = False
        return ps, state, rng, action, running

    mods = pygame.key.get_mods()
    shift_held = mods & pygame.KMOD_SHIFT
    ctrl_held = mods & pygame.KMOD_CTRL

    if event.key == pygame.K_ESCAPE:
        if ps.pause_open:
            ps.pause_open = False
        elif ps.inventory_open:
            ps.inventory_open = False
            ps.held_slot = None
        elif ps.achievement_open:
            ps.achievement_open = False
        elif ps.research_open:
            ps.research_open = False
        elif ps.machine_open:
            ps.machine_open = False
        else:
            ps.pause_open = True
            ps.pause_selection = 0
    elif ps.pause_open:
        ps, state, rng, running = _handle_pause_keys(
            event, ps, state, env, params, rng, level
        )
    elif event.key == pygame.K_f:
        ps, state = _handle_machine_toggle(event, ps, state)
    elif ps.machine_open:
        ps, state, action = _handle_machine_keys(event, ps, state)
    elif event.key == pygame.K_q:
        ps, state = _handle_assembler_recipe_or_hotbar(ps, state)
    elif event.key == pygame.K_i:
        ps.inventory_open = not ps.inventory_open
        if ps.inventory_open:
            ps.achievement_open = False
            ps.machine_open = False
        else:
            ps.held_slot = None
    elif event.key == pygame.K_p:
        ps.achievement_open = not ps.achievement_open
        if ps.achievement_open:
            ps.inventory_open = False
            ps.achievement_scroll = 0
            ps.achievement_selection = 0
    elif ps.achievement_open:
        ps = _handle_achievement_keys(event, ps)
    elif event.key == pygame.K_t and not ps.inventory_open:
        ps.research_open = not ps.research_open
        if ps.research_open:
            ps.inventory_open = False
            ps.achievement_open = False
            ps.research_selection = 0
    elif ps.research_open:
        ps, action = _handle_research_keys(event, ps)
    elif event.key == pygame.K_QUESTION or (
        event.key == pygame.K_SLASH and shift_held
    ):
        ps.help_open = True
    elif ps.inventory_open and ps.menu_focus == "inventory":
        ps, state, action = _handle_inventory_nav(event, ps, state)
    elif ps.inventory_open and ps.menu_focus == "crafting":
        ps, action = _handle_crafting_nav(event, ps)
    elif event.key == pygame.K_e:
        action = _handle_world_interact(state)
    elif event.key == pygame.K_r:
        action = int(Action.ROTATE)
    elif ctrl_held and event.key in _KEY_TO_PLAYER:
        player_idx = _KEY_TO_PLAYER[event.key]
        if player_idx < params.num_players:
            state = state.replace(selected_player=player_idx)
    elif event.key in _KEY_TO_SLOT:
        slot_idx = _KEY_TO_SLOT[event.key]
        if shift_held:
            slot_idx += 8
        if slot_idx < NUM_INVENTORY_SLOTS:
            selected_player = int(state.selected_player)
            new_slots = state.selected_slots.at[selected_player].set(slot_idx)
            state = state.replace(selected_slots=new_slots)
    elif event.key in _KEY_TO_ACTION:
        action = _KEY_TO_ACTION[event.key]
    elif event.key in _DIR_KEYS:
        want_dir, move_act, face_act = _DIR_KEYS[event.key]
        sel = int(state.selected_player)
        facing = int(state.player_directions[sel])
        action = move_act if facing == want_dir else face_act
    elif event.key in _NAV_KEYS:
        action = _NAV_KEYS[event.key]

    return ps, state, rng, action, running


def _handle_pause_keys(
    event: pygame.event.Event,
    ps: PlayState,
    state: EnvState,
    env: FactoriaXEnv,
    params: EnvParams,
    rng: jax.Array,
    level: Level | None,
) -> tuple[PlayState, EnvState, jax.Array, bool]:
    """Handle keyboard input while the pause menu is open.

    Args:
        event: Pygame KEYDOWN event.
        ps: Current play state.
        state: Current environment state.
        env: FactoriaX environment instance.
        params: Environment parameters.
        rng: JAX random key.
        level: Source level for reset.

    Returns:
        Tuple of (play_state, env_state, rng, running).
    """
    running = True
    if event.key == pygame.K_w:
        ps.pause_selection = max(0, ps.pause_selection - 1)
    elif event.key == pygame.K_s:
        ps.pause_selection = min(2, ps.pause_selection + 1)
    elif event.key in (pygame.K_RETURN, pygame.K_e):
        if ps.pause_selection == 0:
            ps.pause_open = False
        elif ps.pause_selection == 1:
            ps.pause_open = False
            if level is not None:
                _, state = env.reset_from_level(level, params)
            else:
                rng, reset_key = random.split(rng)
                _, state = env.reset_env(reset_key, params)
        else:
            running = False
    return ps, state, rng, running


def _handle_machine_toggle(
    event: pygame.event.Event,
    ps: PlayState,
    state: EnvState,
) -> tuple[PlayState, EnvState]:
    """Toggle the machine inspection menu with the F key.

    Args:
        event: Pygame KEYDOWN event.
        ps: Current play state.
        state: Current environment state.

    Returns:
        Updated play state and environment state.
    """
    if ps.machine_open:
        ps.machine_open = False
    else:
        selected_player = int(state.selected_player)
        tx, ty = _tile_in_front(state, selected_player)
        map_h, map_w = state.map.shape
        if (
            0 <= tx < map_w
            and 0 <= ty < map_h
            and int(state.machine_types[ty, tx]) != int(MachineType.NONE)
        ):
            ps.machine_tx, ps.machine_ty = tx, ty
            ps.machine_open = True
            ps.machine_panel_active = True
            ps.inventory_open = False
            ps.achievement_open = False
            ps.pause_open = False
    return ps, state


def _handle_machine_keys(
    event: pygame.event.Event,
    ps: PlayState,
    state: EnvState,
) -> tuple[PlayState, EnvState, int]:
    """Handle keyboard input while the machine menu is open.

    Args:
        event: Pygame KEYDOWN event.
        ps: Current play state.
        state: Current environment state.

    Returns:
        Tuple of (play_state, env_state, action).
    """
    action = int(Action.NOOP)
    if event.key in (pygame.K_w, pygame.K_s):
        ps.machine_panel_active = not ps.machine_panel_active
    elif event.key in (pygame.K_a, pygame.K_d):
        if ps.machine_panel_active:
            action = int(
                Action.PREV_MACHINE_SLOT if event.key == pygame.K_a
                else Action.NEXT_MACHINE_SLOT
            )
        else:
            delta = -1 if event.key == pygame.K_a else 1
            selected_player = int(state.selected_player)
            current = int(state.selected_slots[selected_player])
            new_slot = (current + delta) % NUM_INVENTORY_SLOTS
            new_slots = state.selected_slots.at[selected_player].set(new_slot)
            state = state.replace(selected_slots=new_slots)
    elif event.key == pygame.K_e:
        action = int(Action.WITHDRAW if ps.machine_panel_active else Action.DEPOSIT)
    return ps, state, action


def _handle_assembler_recipe_or_hotbar(
    ps: PlayState,
    state: EnvState,
) -> tuple[PlayState, EnvState]:
    """Handle Q key: cycle assembler recipe or toggle hotbar page.

    Args:
        ps: Current play state.
        state: Current environment state.

    Returns:
        Updated play state and environment state.
    """
    machine_type = int(state.machine_types[ps.machine_ty, ps.machine_tx])
    is_idle = int(state.machine_power[ps.machine_ty, ps.machine_tx]) == 0
    mc = state.machine_inventory_counts
    has_inputs = (
        int(mc[ps.machine_ty, ps.machine_tx, 0]) > 0
        or int(mc[ps.machine_ty, ps.machine_tx, 1]) > 0
        or int(mc[ps.machine_ty, ps.machine_tx, 2]) > 0
    )
    if machine_type == int(MachineType.ASSEMBLER) and is_idle and not has_inputs:
        cur_recipe = int(state.machine_selected_recipe[ps.machine_ty, ps.machine_tx])
        new_recipe = (cur_recipe + 1) % NUM_ASSEMBLER_RECIPES
        new_sel = state.machine_selected_recipe.at[
            ps.machine_ty, ps.machine_tx
        ].set(new_recipe)
        state = state.replace(machine_selected_recipe=new_sel)
    elif machine_type != int(MachineType.ASSEMBLER):
        ps.hotbar_page = 1 - ps.hotbar_page
    return ps, state


def _handle_achievement_keys(
    event: pygame.event.Event,
    ps: PlayState,
) -> PlayState:
    """Handle keyboard navigation in the achievement menu.

    Args:
        event: Pygame KEYDOWN event.
        ps: Current play state.

    Returns:
        Updated play state.
    """
    row_h = 36
    if event.key == pygame.K_w:
        ps.achievement_selection = max(0, ps.achievement_selection - 1)
    elif event.key == pygame.K_s:
        ps.achievement_selection = min(
            NUM_ACHIEVEMENTS - 1, ps.achievement_selection + 1
        )
    sel_top = ps.achievement_selection * row_h
    sel_bot = sel_top + row_h
    if sel_top < ps.achievement_scroll:
        ps.achievement_scroll = sel_top
    elif sel_bot > ps.achievement_scroll + 8 * row_h:
        ps.achievement_scroll = sel_bot - 8 * row_h
    return ps


def _handle_research_keys(
    event: pygame.event.Event,
    ps: PlayState,
) -> tuple[PlayState, int]:
    """Handle keyboard navigation in the research menu.

    W/S navigate between technologies, E spends a science pack
    (triggers RESEARCH action).

    Args:
        event: Pygame KEYDOWN event.
        ps: Current play state.

    Returns:
        Tuple of (updated play state, action to execute).
    """
    action = int(Action.NOOP)
    if event.key == pygame.K_w:
        ps.research_selection = max(0, ps.research_selection - 1)
    elif event.key == pygame.K_s:
        ps.research_selection = min(
            NUM_TECHNOLOGIES - 1, ps.research_selection + 1
        )
    elif event.key == pygame.K_e:
        action = int(Action.RESEARCH)
    return ps, action


def _handle_inventory_nav(
    event: pygame.event.Event,
    ps: PlayState,
    state: EnvState,
) -> tuple[PlayState, EnvState, int]:
    """Handle keyboard navigation in the inventory panel.

    Args:
        event: Pygame KEYDOWN event.
        ps: Current play state.
        state: Current environment state.

    Returns:
        Tuple of (play_state, env_state, action).
    """
    action = int(Action.NOOP)
    selected_player = int(state.selected_player)
    current = int(state.selected_slots[selected_player])
    col = current % 5
    if event.key == pygame.K_a:
        if col > 0:
            action = int(Action.PREV_SLOT)
    elif event.key == pygame.K_d:
        if col == 4:
            ps.menu_focus = "crafting"
        else:
            action = int(Action.NEXT_SLOT)
    elif event.key == pygame.K_w:
        if current >= 5:
            new_sel = state.selected_slots.at[selected_player].set(current - 5)
            state = state.replace(selected_slots=new_sel)
    elif event.key == pygame.K_s:
        if current < 5:
            new_sel = state.selected_slots.at[selected_player].set(current + 5)
            state = state.replace(selected_slots=new_sel)
    return ps, state, action


def _handle_crafting_nav(
    event: pygame.event.Event,
    ps: PlayState,
) -> tuple[PlayState, int]:
    """Handle keyboard navigation in the crafting panel.

    Args:
        event: Pygame KEYDOWN event.
        ps: Current play state.

    Returns:
        Tuple of (play_state, action).
    """
    action = int(Action.NOOP)
    if event.key == pygame.K_w:
        ps.selected_recipe = (ps.selected_recipe - 1) % NUM_RECIPES
    elif event.key == pygame.K_s:
        ps.selected_recipe = (ps.selected_recipe + 1) % NUM_RECIPES
    elif event.key == pygame.K_a:
        ps.menu_focus = "inventory"
    elif event.key == pygame.K_e:
        action = int(Action.CRAFT_MINER) + ps.selected_recipe
    return ps, action


def _handle_world_interact(state: EnvState) -> int:
    """Determine action for E key in the world (pickup or place).

    Args:
        state: Current environment state.

    Returns:
        Action integer (PICKUP or PLACE).
    """
    selected_player = int(state.selected_player)
    tx, ty = _tile_in_front(state, selected_player)
    map_h, map_w = state.map.shape
    has_machine = (
        0 <= tx < map_w
        and 0 <= ty < map_h
        and int(state.machine_types[ty, tx]) != int(MachineType.NONE)
    )
    return int(Action.PICKUP if has_machine else Action.PLACE)


def _render_frame(
    state: EnvState,
    ps: PlayState,
    ui_w: int,
    ui_h: int,
    tile_px: int,
    world_ox: int,
    world_oy: int,
) -> tuple[np.ndarray, list[ClickRegion]]:
    """Build one UI frame with all active overlays.

    Args:
        state: Current environment state.
        ps: Current play state.
        ui_w: UI canvas width.
        ui_h: UI canvas height.
        tile_px: Tile pixel size.
        world_ox: World X offset within canvas.
        world_oy: World Y offset within canvas.

    Returns:
        Tuple of (rgb frame array, click regions for this frame).
    """
    pixels = render_pixels(state, block_pixel_size=tile_px, frame_tick=ps.frame_tick)
    click_regions: list[ClickRegion] = []

    ui_frame = np.zeros((ui_h, ui_w, 3), dtype=np.uint8)
    ph, pw = pixels.shape[:2]
    ui_frame[world_oy : world_oy + ph, world_ox : world_ox + pw] = pixels

    # Tile highlight: draw a bright border around the hovered tile.
    htx, hty = ps.hover_tile_x, ps.hover_tile_y
    if htx >= 0 and hty >= 0:
        hx = world_ox + htx * tile_px
        hy = world_oy + hty * tile_px
        hx2 = hx + tile_px
        hy2 = hy + tile_px
        highlight = np.array([255, 255, 255], dtype=np.uint8)
        # Blend 50% with existing pixels for semi-transparency.
        for row in (hy, hy2 - 1):
            if 0 <= row < ui_h:
                c0, c1 = max(0, hx), min(ui_w, hx2)
                ui_frame[row, c0:c1] = (
                    ui_frame[row, c0:c1] // 2 + highlight // 2
                )
        for col in (hx, hx2 - 1):
            if 0 <= col < ui_w:
                r0, r1 = max(0, hy), min(ui_h, hy2)
                ui_frame[r0:r1, col] = (
                    ui_frame[r0:r1, col] // 2 + highlight // 2
                )

    hotbar_overlay, hotbar_regions = render_hotbar(state, ui_w, ui_h, ps.hotbar_page)
    composite_rgba_over_rgb(ui_frame, hotbar_overlay)
    click_regions.extend(hotbar_regions)

    # Info panel (right 1/3 of bottom bar).
    info_overlay = render_info_panel(state, ui_w, ui_h, htx, hty)
    composite_rgba_over_rgb(ui_frame, info_overlay)

    if ps.machine_open:
        machine_overlay, machine_regions = render_machine_menu(
            state, ui_w, ui_h, ps.machine_tx, ps.machine_ty, ps.machine_panel_active,
        )
        composite_rgba_over_rgb(ui_frame, machine_overlay)
        click_regions.extend(machine_regions)

    if ps.inventory_open:
        menu_overlay, inv_regions = render_inventory_menu(
            state, ui_w, ui_h, ps.menu_focus, ps.held_slot,
            ps.selected_recipe,
        )
        composite_rgba_over_rgb(ui_frame, menu_overlay)
        click_regions.extend(inv_regions)

    if ps.achievement_open:
        ach_overlay = render_achievement_menu(
            state, ui_w, ui_h, ps.achievement_scroll, ps.achievement_selection,
        )
        composite_rgba_over_rgb(ui_frame, ach_overlay)

    if ps.research_open:
        research_overlay = render_research_menu(
            state, ui_w, ui_h, ps.research_selection,
        )
        composite_rgba_over_rgb(ui_frame, research_overlay)

    if ps.pause_open:
        pause_overlay, pause_regions = render_pause_menu(
            ui_w, ui_h, ps.pause_selection,
        )
        composite_rgba_over_rgb(ui_frame, pause_overlay)
        click_regions.extend(pause_regions)

    if ps.help_open:
        composite_rgba_over_rgb(ui_frame, render_help_overlay(ui_w, ui_h))

    if ps.victory_open:
        composite_rgba_over_rgb(ui_frame, render_victory_screen(ui_w, ui_h))

    if ps.welcome_open:
        welcome_overlay, _ = render_welcome_screen(ui_w, ui_h, ps.record_enabled)
        composite_rgba_over_rgb(ui_frame, welcome_overlay)

    return ui_frame, click_regions


# Module-level key mappings (constant, no need to rebuild per frame).
# Maps direction keys to (Direction, move_Action, face_Action).
# If the player already faces that direction, emit the move action.
# Otherwise emit the face action to turn first.
_DIR_KEYS: dict[int, tuple[int, int, int]] = {
    pygame.K_w: (Direction.UP, int(Action.UP), int(Action.FACE_UP)),
    pygame.K_s: (Direction.DOWN, int(Action.DOWN), int(Action.FACE_DOWN)),
    pygame.K_a: (Direction.LEFT, int(Action.LEFT), int(Action.FACE_LEFT)),
    pygame.K_d: (Direction.RIGHT, int(Action.RIGHT), int(Action.FACE_RIGHT)),
    pygame.K_UP: (Direction.UP, int(Action.UP), int(Action.FACE_UP)),
    pygame.K_DOWN: (Direction.DOWN, int(Action.DOWN), int(Action.FACE_DOWN)),
    pygame.K_LEFT: (Direction.LEFT, int(Action.LEFT), int(Action.FACE_LEFT)),
    pygame.K_RIGHT: (Direction.RIGHT, int(Action.RIGHT), int(Action.FACE_RIGHT)),
}

_NAV_KEYS: dict[int, int] = {
    pygame.K_q: int(Action.TURN_LEFT),
    pygame.K_e: int(Action.TURN_RIGHT),
}

_KEY_TO_ACTION = {
    pygame.K_SPACE: Action.MINE,
    pygame.K_g: Action.REPAIR,
}

_KEY_TO_SLOT = {
    pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3,
    pygame.K_5: 4, pygame.K_6: 5, pygame.K_7: 6, pygame.K_8: 7,
}

_KEY_TO_PLAYER = {
    pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3,
    pygame.K_5: 4, pygame.K_6: 5, pygame.K_7: 6, pygame.K_8: 7,
    pygame.K_9: 8,
}


_MOUSE_DIR_TO_FACE: dict[int, int] = {
    int(Direction.UP): int(Action.FACE_UP),
    int(Direction.DOWN): int(Action.FACE_DOWN),
    int(Direction.LEFT): int(Action.FACE_LEFT),
    int(Direction.RIGHT): int(Action.FACE_RIGHT),
}


def _mouse_facing_direction(
    mx: int,
    my: int,
    player_screen_x: int,
    player_screen_y: int,
) -> int:
    """Determine which cardinal direction the mouse points relative to player.

    Picks the dominant axis (largest absolute delta) and returns the
    corresponding Direction value. Returns 0 if the mouse is exactly
    on the player center.

    Args:
        mx: Mouse x in UI coordinates.
        my: Mouse y in UI coordinates.
        player_screen_x: Player center x in UI coordinates.
        player_screen_y: Player center y in UI coordinates.

    Returns:
        Direction enum value (1-4), or 0 if undetermined.
    """
    dx = mx - player_screen_x
    dy = my - player_screen_y
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) >= abs(dy):
        return Direction.RIGHT if dx > 0 else Direction.LEFT
    return Direction.DOWN if dy > 0 else Direction.UP


def _play_loop(
    env: FactoriaXEnv,
    state: EnvState,
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
    step_fn = jax.jit(env.step_env)
    clock = pygame.time.Clock()

    ui_w = _UI_SIZE
    ui_h = _UI_SIZE
    world_area_h = ui_h - _HOTBAR_H
    tile_px = _tile_pixel_size(params.map_width, params.map_height)
    world_pw = params.map_width * tile_px
    world_ph = params.map_height * tile_px
    world_ox = (ui_w - world_pw) // 2
    world_oy = (world_area_h - world_ph) // 2
    win_scale = max(1, min(window_width // ui_w, window_height // ui_h))
    win_ox = (window_width - ui_w * win_scale) // 2
    win_oy = (window_height - ui_h * win_scale) // 2

    ps = PlayState()
    running = True

    while running:
        action = int(Action.NOOP)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif ps.welcome_open:
                ps, state = _handle_welcome_event(
                    event, ps, state, win_ox, win_oy, win_scale, ui_w, ui_h,
                )
                continue
            elif ps.victory_open:
                if event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_SPACE, pygame.K_RETURN, pygame.K_ESCAPE,
                ):
                    ps.victory_open = False
                continue
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                ps, state, rng, action, running = _handle_click(
                    event, ps, state, env, params, rng, level,
                    win_ox, win_oy, win_scale,
                )
            elif event.type == pygame.KEYDOWN:
                ps, state, rng, action, running = _handle_keydown(
                    event, ps, state, env, params, rng, level,
                )

        # Highlight the tile the player is facing.
        ftx, fty = _tile_in_front(state, int(state.selected_player))
        map_h = int(state.map.shape[0])
        map_w = int(state.map.shape[1])
        if 0 <= ftx < map_w and 0 <= fty < map_h:
            ps.hover_tile_x = ftx
            ps.hover_tile_y = fty
        else:
            ps.hover_tile_x = -1
            ps.hover_tile_y = -1

        if action != int(Action.NOOP):
            rng, step_key = random.split(rng)
            obs, state, reward, done, info = step_fn(
                step_key, state, action, params,
            )
            if ps.record_enabled:
                ps.recorded_actions.append(int(action))
                ps.recorded_rewards.append(float(reward))
                ps.recorded_states.append(state)
            if done:
                if level is not None:
                    _, state = env.reset_from_level(level, params)
                else:
                    rng, reset_key = random.split(rng)
                    _, state = env.reset_env(reset_key, params)

            if not ps.victory_shown:
                rocket_unlocked = bool(
                    state.achievements_unlocked[_ROCKET_ACHIEVEMENT_IDX]
                )
                if rocket_unlocked:
                    ps.victory_open = True
                    ps.victory_shown = True

        ui_frame, ps.click_regions = _render_frame(
            state, ps, ui_w, ui_h, tile_px, world_ox, world_oy,
        )

        final_surface = pygame.surfarray.make_surface(
            np.transpose(ui_frame, (1, 0, 2)),
        )
        scaled_surface = pygame.transform.scale(
            final_surface, (ui_w * win_scale, ui_h * win_scale),
        )
        screen.fill((0, 0, 0))
        screen.blit(scaled_surface, (win_ox, win_oy))
        pygame.display.flip()
        ps.frame_tick += 1
        clock.tick(30)

    if ps.record_enabled and ps.recorded_states:
        _save_recorded_trajectory(
            ps.recorded_states, ps.recorded_actions, ps.recorded_rewards,
        )


def _save_recorded_trajectory(
    states: list[EnvState],
    actions: list[int],
    rewards: list[float],
) -> None:
    """Save a recorded play session as a timestamped .npz trajectory.

    Args:
        states: List of EnvState snapshots.
        actions: List of action integers.
        rewards: List of reward floats.
    """
    from datetime import datetime

    from factoriax.analysis.trajectory import states_to_trajectory

    # Pad actions/rewards to match states length (states has initial + per-step).
    act = np.array(actions + [0] * (len(states) - len(actions)), dtype=np.int32)
    rew = np.array(rewards + [0.0] * (len(states) - len(rewards)), dtype=np.float32)
    from dataclasses import replace

    traj = states_to_trajectory(states, actions=act, rewards=rew)
    traj = replace(traj, observation_scheme={"type": 3})  # PLAYER
    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    path = f"trajectory_{ts}.npz"
    traj.save(path)
    print(f"Saved trajectory: {path} ({len(states)} steps)")


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
        Escape: Close menus / Open pause menu (with Reset option)
    """
    pygame.init()

    window_width, window_height = calculate_window_size(
        _UI_SIZE,
        _UI_SIZE,
    )
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("FactoriaX")

    env_result = _run_with_loading_screen(
        screen, "Initialising environment",
        make_factoriax_env,
    )
    env: FactoriaXEnv = env_result[0]  # type: ignore[index]
    params: EnvParams = env_result[1]  # type: ignore[index]

    rng = random.PRNGKey(42)
    rng, reset_key = random.split(rng)
    _reset_key = reset_key
    reset_result = _run_with_loading_screen(
        screen, "Generating world",
        lambda: env.reset_env(_reset_key, params),
    )
    state: EnvState = reset_result[1]  # type: ignore[index]

    step_fn = jax.jit(env.step_env)
    rng, warmup_key = random.split(rng)
    _warmup_key2 = warmup_key

    def _warmup_main() -> None:
        step_fn(
            _warmup_key2, state, jnp.int32(Action.NOOP), params,
        )[0].block_until_ready()

    _run_with_loading_screen(
        screen, "Compiling JAX (first run only)", _warmup_main,
    )

    _play_loop(env, state, params, None, screen, rng)

    pygame.quit()


if __name__ == "__main__":
    main()
