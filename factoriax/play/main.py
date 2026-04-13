"""Interactive play script for FactoriaX using pygame.

Uses :class:`~factoriax.play.game_ui.GameUI` for all menu rendering
and input dispatch. This module handles the pygame window, environment
stepping, trajectory recording, and play-specific screens (welcome,
victory).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC

import jax
import numpy as np
import pygame
from jax import random

from factoriax.achievements import ACHIEVEMENT_INFO
from factoriax.config import (
    KeyLookup,
    build_key_lookup,
    default_keyboard,
)
from factoriax.constants import Action, Direction
from factoriax.envs.achievement_wrapper import AchievementState, AchievementWrapper
from factoriax.envs.factoriax_env import FactoriaXEnv, make_factoriax_env
from factoriax.levels import Level
from factoriax.play.game_ui import GameUI
from factoriax.play.play_state import PlayState
from factoriax.play.ui import _hotbar_h, render_welcome_screen
from factoriax.state import EnvParams, EnvState
from factoriax.ui import theme as _play_theme
from factoriax.ui.compositing import composite_rgba_over_rgb
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

    inner_env, _ = make_factoriax_env()
    env = AchievementWrapper(inner_env)
    params = EnvParams(
        map_width=level.map_width,
        map_height=level.map_height,
        num_players=num_players,
    )

    prev_size = screen.get_size() if screen is not None else None
    scaled = _BASE_UI_SIZE * _play_theme.UI_SCALE
    game_win_w, game_win_h = calculate_window_size(scaled, scaled)
    if screen is None:
        screen = pygame.display.set_mode((game_win_w, game_win_h))
    else:
        screen = pygame.display.set_mode(
            (game_win_w, game_win_h),
            pygame.RESIZABLE,
        )

    pygame.display.set_caption(f"FactoriaX - {level.name}")

    reset_result: tuple[jax.Array, AchievementState] = _run_with_loading_screen(  # type: ignore[assignment]
        screen,
        "Building world",
        lambda: env.reset_from_level(level, params),
    )
    _, state = reset_result
    rng = random.PRNGKey(42)

    _play_loop(env, state, params, level, screen, rng)

    if owns_pygame:
        pygame.quit()
    elif prev_size is not None:
        pygame.display.set_mode(prev_size, pygame.RESIZABLE)


_BASE_UI_SIZE = 1024

_MOUSE_DIR_TO_FACE: dict[int, int] = {
    int(Direction.UP): int(Action.FACE_UP),
    int(Direction.DOWN): int(Action.FACE_DOWN),
    int(Direction.LEFT): int(Action.FACE_LEFT),
    int(Direction.RIGHT): int(Action.FACE_RIGHT),
}


def _tile_pixel_size(map_w: int, map_h: int) -> int:
    """Choose a tile pixel size so the map fits within the UI canvas.

    Args:
        map_w: Map width in tiles.
        map_h: Map height in tiles.

    Returns:
        Tile side length in pixels.
    """
    world_h = _BASE_UI_SIZE * _play_theme.UI_SCALE - _hotbar_h()
    return max(8, min(_BASE_UI_SIZE * _play_theme.UI_SCALE // map_w, world_h // map_h))


def _mouse_facing_direction(
    mx: int,
    my: int,
    player_screen_x: int,
    player_screen_y: int,
) -> int:
    """Determine which cardinal direction the mouse points relative to player.

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


def _handle_welcome_event(
    event: pygame.event.Event,
    ps: PlayState,
    state: AchievementState,
    win_ox: int,
    win_oy: int,
    win_scale: int,
    ui_w: int,
    ui_h: int,
) -> tuple[PlayState, AchievementState]:
    """Process events while the welcome screen is showing.

    Args:
        event: Pygame event.
        ps: Current play state.
        state: Current wrapped state.
        win_ox: Window X offset for coordinate transform.
        win_oy: Window Y offset for coordinate transform.
        win_scale: Window scale factor.
        ui_w: UI canvas width.
        ui_h: UI canvas height.

    Returns:
        Updated play state and wrapped state.
    """
    if event.type == pygame.KEYDOWN:
        if event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_ESCAPE):
            ps.welcome_open = False
    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        ps.welcome_open = False
    return ps, state


def _play_loop(
    env: AchievementWrapper,
    state: AchievementState,
    params: EnvParams,
    level: Level | None,
    screen: pygame.Surface,
    rng: jax.Array,
    kb_lookup: KeyLookup | None = None,
) -> None:
    """Run the full interactive game loop with all menus and controls.

    Args:
        env: Achievement-wrapped environment instance.
        state: Initial wrapped state.
        params: Environment parameters.
        level: Source level for reset, or ``None`` for procedural reset.
        screen: Pygame display surface.
        rng: JAX random key.
        kb_lookup: Key lookup table from :func:`build_key_lookup`. Built
            from default bindings when ``None``.
    """
    if kb_lookup is None:
        kb_lookup = build_key_lookup(default_keyboard())
    window_width, window_height = screen.get_size()
    step_fn = jax.jit(env.step_env)
    rng, warmup_key = random.split(rng)
    _wk = warmup_key
    _st = state

    def _warmup() -> tuple[jax.Array, AchievementState]:
        _, s, _, _, _ = step_fn(_wk, _st, int(Action.NOOP), params)
        return _wk, s

    warmup_result = _run_with_loading_screen(screen, "Compiling JAX", _warmup)
    _, state = warmup_result  # type: ignore[misc]

    clock = pygame.time.Clock()

    ui_w = _BASE_UI_SIZE * _play_theme.UI_SCALE
    ui_h = _BASE_UI_SIZE * _play_theme.UI_SCALE
    world_area_h = ui_h - _hotbar_h()
    tile_px = _tile_pixel_size(params.map_width, params.map_height)
    world_pw = params.map_width * tile_px
    world_ph = params.map_height * tile_px
    world_ox = (ui_w - world_pw) // 2
    world_oy = (world_area_h - world_ph) // 2
    win_scale = max(1, min(window_width // ui_w, window_height // ui_h))
    win_ox = (window_width - ui_w * win_scale) // 2
    win_oy = (window_height - ui_h * win_scale) // 2

    ui = GameUI(params, kb_lookup, welcome_open=True)
    ps = ui.play_state
    running = True

    while running:
        action = int(Action.NOOP)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif ps.welcome_open:
                ps, state = _handle_welcome_event(
                    event, ps, state,
                    win_ox, win_oy, win_scale, ui_w, ui_h,
                )
                continue
            elif ps.victory_open:
                if event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_SPACE,
                    pygame.K_RETURN,
                    pygame.K_ESCAPE,
                ):
                    ps.victory_open = False
                continue
            elif event.type in (
                pygame.MOUSEBUTTONDOWN, pygame.KEYDOWN,
            ):
                result = ui.handle_event(event, state.env_state)
                if result.state is not None:
                    state = state.replace(env_state=result.state)
                if result.action is not None:
                    action = result.action
                if result.quit:
                    running = False
                if result.reset:
                    if level is not None:
                        _, state = env.reset_from_level(level, params)
                    else:
                        rng, reset_key = random.split(rng)
                        _, state = env.reset_env(reset_key, params)

        # Highlight the tile the player is facing.
        ui.update_hover(state.env_state)

        if action != int(Action.NOOP):
            rng, step_key = random.split(rng)
            obs, state, reward, done, info = step_fn(
                step_key, state, action, params,
            )
            if ps.record_enabled:
                ps.recorded_actions.append(int(action))
                ps.recorded_rewards.append(float(reward))
                ps.recorded_states.append(state.env_state)
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

        ui_frame, ps.click_regions = ui.render_frame(
            state.env_state, ui_w, ui_h, tile_px, world_ox, world_oy,
            achievements=state.achievements_unlocked,
        )

        # Welcome screen overlay (managed outside GameUI).
        if ps.welcome_open:
            welcome_overlay, _ = render_welcome_screen(
                ui_w, ui_h, ps.record_enabled,
            )
            composite_rgba_over_rgb(ui_frame, welcome_overlay)

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
            ps.recorded_states,
            ps.recorded_actions,
            ps.recorded_rewards,
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
        _BASE_UI_SIZE * _play_theme.UI_SCALE,
        _BASE_UI_SIZE * _play_theme.UI_SCALE,
    )
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("FactoriaX")

    env_result = _run_with_loading_screen(
        screen,
        "Initialising environment",
        make_factoriax_env,
    )
    inner_env: FactoriaXEnv = env_result[0]  # type: ignore[index]
    params: EnvParams = env_result[1]  # type: ignore[index]
    env = AchievementWrapper(inner_env)

    rng = random.PRNGKey(42)
    rng, reset_key = random.split(rng)

    reset_result: tuple[jax.Array, AchievementState] = _run_with_loading_screen(  # type: ignore[assignment]
        screen,
        "Generating world",
        lambda: env.reset_env(reset_key, params),
    )
    _, state = reset_result

    _play_loop(env, state, params, None, screen, rng)
    pygame.quit()
