"""Main loop for the RL trajectory inspector."""

from __future__ import annotations

import numpy as np
import pygame

from factoriax.analysis.trajectory import Trajectory
from factoriax.inspector.layout import (
    ACTION_STRIP_HEIGHT,
    CHART_HEIGHT,
    INFO_PANEL_WIDTH,
    MENU_BAR_HEIGHT,
    MIN_CANVAS_H,
    MIN_CANVAS_W,
    TIMELINE_HEIGHT,
    compute_base_dimensions,
    rebuild_caches,
    render_frame,
)
from factoriax.inspector.panels import render_timeline
from factoriax.inspector.state import InspectorState
from factoriax.ui.primitives import hit_test_regions
from factoriax.ui.window import calculate_window_size


def main(path: str) -> None:
    """Launch the trajectory inspector.

    Args:
        path: Path to a ``.npz`` trajectory file.
    """
    pygame.init()

    traj = Trajectory.load(path)
    state = InspectorState()

    canvas_w = max(MIN_CANVAS_W, 400)
    canvas_h = max(MIN_CANVAS_H, 300)
    base_w, base_h = compute_base_dimensions(canvas_w, canvas_h)

    window_w, window_h = calculate_window_size(base_w, base_h)
    screen = pygame.display.set_mode((window_w, window_h), pygame.RESIZABLE)
    pygame.display.set_caption(
        f"FactoriaX Inspector - {path} "
        f"({traj.num_episodes} eps, {traj.episode_length} steps)"
    )

    scale = max(1, min(window_w // base_w, window_h // base_h))
    rebuild_caches(traj, state, base_w)

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type == pygame.VIDEORESIZE:
                window_w, window_h = event.w, event.h
                canvas_w = max(MIN_CANVAS_W, window_w - INFO_PANEL_WIDTH)
                canvas_h = max(
                    MIN_CANVAS_H,
                    window_h
                    - MENU_BAR_HEIGHT
                    - TIMELINE_HEIGHT
                    - CHART_HEIGHT
                    - ACTION_STRIP_HEIGHT,
                )
                base_w, base_h = compute_base_dimensions(canvas_w, canvas_h)
                scale = max(1, min(window_w // base_w, window_h // base_h))
                rebuild_caches(traj, state, base_w)

            if event.type == pygame.KEYDOWN:
                _handle_key(event, traj, state)

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx = event.pos[0] // scale
                my = event.pos[1] // scale
                _handle_click(mx, my, event.button, traj, state, base_w, canvas_h)

            if event.type == pygame.MOUSEBUTTONUP:
                state.timeline_dragging = False

            if event.type == pygame.MOUSEMOTION and state.timeline_dragging:
                mx = event.pos[0] // scale
                _handle_scrub(mx, traj, state, base_w)

        # Playback advance.
        if state.playing and traj is not None:
            total_steps = traj.episode_length
            state.current_step = min(
                state.current_step + state.playback_speed, total_steps - 1
            )
            if state.current_step >= total_steps - 1:
                state.playing = False

        # Render.
        frame = render_frame(traj, state, base_w, base_h, canvas_w, canvas_h)

        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        scaled = pygame.transform.scale(surface, (base_w * scale, base_h * scale))
        screen.fill((0, 0, 0))
        screen.blit(scaled, (0, 0))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def _handle_key(
    event: pygame.event.Event,
    traj: Trajectory,
    state: InspectorState,
) -> None:
    """Process keyboard input.

    Args:
        event: pygame KEYDOWN event.
        traj: Loaded trajectory.
        state: Inspector state (mutated in place).
    """
    key = event.key
    total_steps = traj.episode_length

    if key == pygame.K_ESCAPE:
        pygame.event.post(pygame.event.Event(pygame.QUIT))
    elif key == pygame.K_SPACE:
        state.playing = not state.playing
    elif key == pygame.K_RIGHT:
        state.current_step = min(state.current_step + 1, total_steps - 1)
        state.playing = False
    elif key == pygame.K_LEFT:
        state.current_step = max(state.current_step - 1, 0)
        state.playing = False
    elif key == pygame.K_HOME:
        state.current_step = 0
        state.playing = False
    elif key == pygame.K_END:
        state.current_step = total_steps - 1
        state.playing = False
    elif key == pygame.K_TAB:
        if traj.is_multi_player:
            state.selected_player = (state.selected_player + 1) % traj.num_players
    elif key == pygame.K_PERIOD:
        if state.selected_episode < traj.num_episodes - 1:
            state.selected_episode += 1
            state.current_step = 0
            state.playing = False
    elif key == pygame.K_COMMA:
        if state.selected_episode > 0:
            state.selected_episode -= 1
            state.current_step = 0
            state.playing = False


def _handle_click(
    mx: int,
    my: int,
    button: int,
    traj: Trajectory,
    state: InspectorState,
    base_w: int,
    canvas_h: int,
) -> None:
    """Handle mouse click events.

    Args:
        mx: Mouse X in base resolution.
        my: Mouse Y in base resolution.
        button: Mouse button (1=left).
        traj: Loaded trajectory.
        state: Inspector state.
        base_w: Frame width.
        canvas_h: Canvas height.
    """
    if button != 1:
        return

    tl_y = MENU_BAR_HEIGHT + canvas_h
    if tl_y <= my < tl_y + TIMELINE_HEIGHT:
        _, regions = render_timeline(
            state.current_step,
            traj.episode_length,
            base_w,
            TIMELINE_HEIGHT,
        )
        # Offset regions to frame coordinates.
        adjusted = [
            type(r)(r.x, r.y + tl_y, r.w, r.h, r.action, r.param) for r in regions
        ]
        hit = hit_test_regions(adjusted, mx, my)
        if hit is not None:
            if hit.action == "scrub":
                state.timeline_dragging = True
                _handle_scrub(mx, traj, state, base_w)
            elif hit.action == "first":
                state.current_step = 0
                state.playing = False
            elif hit.action == "prev":
                state.current_step = max(0, state.current_step - 1)
                state.playing = False
            elif hit.action == "play_pause":
                state.playing = not state.playing
            elif hit.action == "last":
                state.current_step = traj.episode_length - 1
                state.playing = False


def _handle_scrub(
    mx: int,
    traj: Trajectory,
    state: InspectorState,
    base_w: int,
) -> None:
    """Update current step based on mouse X during timeline scrub.

    Args:
        mx: Mouse X in base resolution.
        traj: Loaded trajectory.
        state: Inspector state.
        base_w: Frame width.
    """
    track_x = 60
    track_w = base_w - 120
    total_steps = traj.episode_length
    if track_w <= 0 or total_steps <= 1:
        return
    frac = (mx - track_x) / track_w
    frac = max(0.0, min(1.0, frac))
    state.current_step = int(frac * (total_steps - 1))
    state.playing = False
