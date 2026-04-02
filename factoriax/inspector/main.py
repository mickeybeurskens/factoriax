"""Main loop for the RL trajectory inspector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygame

from factoriax.analysis.trajectory import Trajectory
from factoriax.inspector.dialogs import FileBrowserDialog
from factoriax.inspector.layout import (
    MENU_BAR_HEIGHT,
    TIMELINE_HEIGHT,
    compute_base_dimensions,
    rebuild_caches,
    render_frame,
)
from factoriax.inspector.panels import render_timeline
from factoriax.inspector.state import InspectorState
from factoriax.ui.compositing import composite_rgba_over_rgb
from factoriax.ui.primitives import hit_test_regions
from factoriax.ui.window import calculate_window_size


def _render_frames_from_states(
    traj: Trajectory, episode: int
) -> list[np.ndarray] | None:
    """Render frames from trajectory state data if available.

    Args:
        traj: Loaded trajectory.
        episode: Episode index.

    Returns:
        List of RGB frames, or None if state data is missing.
    """
    if traj.block_map is None:
        return None
    from factoriax.analysis.trajectory import trajectory_to_states
    from factoriax.renderer import render_pixels

    env_states = trajectory_to_states(traj, episode=episode)
    return [render_pixels(s, block_pixel_size=24) for s in env_states]


def _render_jax_hud_frames(
    traj: Trajectory, episode: int, state: InspectorState
) -> list[np.ndarray] | None:
    """Render JAX HUD frames from trajectory state data.

    Uses the JaxRenderer stored in inspector state to render the
    map + 4-quadrant HUD for each timestep. Returns numpy arrays
    for display via pygame.

    Args:
        traj: Loaded trajectory.
        episode: Episode index.
        state: Inspector state with jax_renderer.

    Returns:
        List of RGB frames, or None if state data or renderer missing.
    """
    if traj.block_map is None or state.jax_renderer is None:
        return None
    from factoriax.analysis.trajectory import trajectory_to_states

    env_states = trajectory_to_states(traj, episode=episode)
    renderer = state.jax_renderer
    return [np.array(renderer.jit_render_hud(s)) for s in env_states]


def _build_frames(
    traj: Trajectory,
    state: InspectorState,
    level_path: str | None,
) -> list[np.ndarray] | None:
    """Build rendered frames from the best available source.

    Args:
        traj: Loaded trajectory.
        state: Inspector state (for episode index).
        level_path: Optional level file for action replay.

    Returns:
        List of RGB frames, or None.
    """
    if level_path is not None:
        from factoriax.inspector.replay import load_replay_frames

        return load_replay_frames(level_path, traj, episode=state.selected_episode)
    return _render_frames_from_states(traj, state.selected_episode)


def main(path: str, level_path: str | None = None) -> None:
    """Launch the trajectory inspector.

    Args:
        path: Path to a ``.npz`` trajectory file.
        level_path: Optional path to a level JSON file. When provided,
            the recorded actions are replayed through the level to
            produce pixel-perfect game world frames.
    """
    pygame.init()

    traj = Trajectory.load(path)

    from factoriax.jax_renderer import JaxRenderer

    state = InspectorState(jax_renderer=JaxRenderer(tile_px=8))

    frames = _build_frames(traj, state, level_path)
    state.jax_hud_frames = _render_jax_hud_frames(traj, state.selected_episode, state)
    if frames is not None:
        print(f"Rendered {len(frames)} frames.")
    elif traj.block_map is None and level_path is None:
        print(
            "No state data in trajectory (block_map missing). "
            "Game world will not render.\n"
            "  Record with states_to_trajectory(), or provide --level."
        )

    # Render at a compact base resolution, then integer-scale to fill
    # the screen. This keeps pixel-font text crisp and readable.
    canvas_w = 420
    canvas_h = 260
    base_w, base_h = compute_base_dimensions(canvas_w, canvas_h)
    window_w, window_h = calculate_window_size(base_w, base_h)
    scale = max(1, min(window_w // base_w, window_h // base_h))

    screen = pygame.display.set_mode((window_w, window_h), pygame.RESIZABLE)
    _update_caption(path, traj, state)

    rebuild_caches(traj, state, base_w)

    clock = pygame.time.Clock()
    running = True
    dialog: FileBrowserDialog | None = None
    last_frame: np.ndarray | None = None

    def _load_trajectory(new_path: str) -> None:
        nonlocal traj, frames, path
        path = new_path
        traj = Trajectory.load(new_path)
        state.current_step = 0
        state.selected_episode = 0
        state.selected_player = 0
        state.playing = False
        frames = _build_frames(traj, state, level_path)
        state.jax_hud_frames = _render_jax_hud_frames(
            traj, state.selected_episode, state
        )
        rebuild_caches(traj, state, base_w)
        _update_caption(path, traj, state)

    def _load_level(new_level_path: str) -> None:
        nonlocal frames, level_path
        level_path = new_level_path
        frames = _build_frames(traj, state, level_path)
        state.jax_hud_frames = _render_jax_hud_frames(
            traj, state.selected_episode, state
        )

    def _on_episode_change() -> None:
        nonlocal frames
        state.current_step = 0
        state.playing = False
        frames = _build_frames(traj, state, level_path)
        state.jax_hud_frames = _render_jax_hud_frames(
            traj, state.selected_episode, state
        )
        rebuild_caches(traj, state, base_w)
        _update_caption(path, traj, state)

    def _on_player_change() -> None:
        rebuild_caches(traj, state, base_w)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type == pygame.VIDEORESIZE:
                window_w, window_h = event.w, event.h
                scale = max(1, min(window_w // base_w, window_h // base_h))

            # Help overlay: any key dismisses it.
            if state.show_help:
                if event.type == pygame.KEYDOWN:
                    state.show_help = False
                continue

            # Dialog modal: consume all events while open.
            if dialog is not None:
                result = dialog.handle_event(event)
                if result == "ok":
                    chosen = dialog.get_path()
                    if chosen is not None:
                        if dialog.pattern.endswith(".npz"):
                            _load_trajectory(chosen)
                        else:
                            _load_level(chosen)
                    dialog = None
                elif result == "cancel":
                    dialog = None
                continue

            if event.type == pygame.KEYDOWN:
                key = event.key
                if key == pygame.K_l:
                    dialog = FileBrowserDialog(
                        title="Load Trajectory", pattern="**/*.npz"
                    )
                elif key == pygame.K_k:
                    dialog = FileBrowserDialog(
                        title="Load Level (for replay)",
                        pattern="**/*.json",
                    )
                elif key == pygame.K_TAB:
                    if traj.is_multi_player:
                        state.selected_player = (
                            state.selected_player + 1
                        ) % traj.num_players
                        _on_player_change()
                elif key == pygame.K_PERIOD:
                    if state.selected_episode < traj.num_episodes - 1:
                        state.selected_episode += 1
                        _on_episode_change()
                elif key == pygame.K_COMMA:
                    if state.selected_episode > 0:
                        state.selected_episode -= 1
                        _on_episode_change()
                elif key == pygame.K_LEFTBRACKET:
                    state.playback_speed = max(1, state.playback_speed - 1)
                elif key == pygame.K_RIGHTBRACKET:
                    state.playback_speed = min(20, state.playback_speed + 1)
                elif key == pygame.K_s and last_frame is not None:
                    _export_png(last_frame, path, state)
                elif key == pygame.K_v:
                    _export_mp4(frames, path, state)
                elif key == pygame.K_o:
                    state.show_obs_overlay = not state.show_obs_overlay
                elif key == pygame.K_SLASH or key == pygame.K_QUESTION:
                    state.show_help = True
                else:
                    _handle_key(event, traj, state)

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx = event.pos[0] // scale
                my = event.pos[1] // scale
                # Scroll wheel on game world area: no-op for now.
                if event.button in (4, 5):
                    continue
                _handle_click(mx, my, event.button, traj, state, base_w, canvas_h)

            if event.type == pygame.MOUSEBUTTONUP:
                state.timeline_dragging = False

            if event.type == pygame.MOUSEMOTION and state.timeline_dragging:
                _handle_scrub(event.pos[0] // scale, traj, state, base_w)

        # Playback advance.
        if state.playing and traj is not None:
            total_steps = traj.episode_length
            state.current_step = min(
                state.current_step + state.playback_speed,
                total_steps - 1,
            )
            if state.current_step >= total_steps - 1:
                state.playing = False

        # Render.
        frame: np.ndarray = render_frame(
            traj,
            state,
            base_w,
            base_h,
            canvas_w,
            canvas_h,
            frames=frames,
        )

        last_frame = frame

        # Overlay dialog or help if open.
        if dialog is not None:
            overlay = dialog.render(base_w, base_h)
            composite_rgba_over_rgb(frame, overlay)
        if state.show_help:
            from factoriax.inspector.panels import render_help_overlay

            help_overlay = render_help_overlay(base_w, base_h)
            composite_rgba_over_rgb(frame, help_overlay)

        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        scaled_surf = pygame.transform.scale(surface, (base_w * scale, base_h * scale))
        screen.fill((0, 0, 0))
        screen.blit(scaled_surf, (0, 0))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def _update_caption(path: str, traj: Trajectory, state: InspectorState) -> None:
    """Update the window title with current trajectory info."""
    ep = f"ep {state.selected_episode + 1}/{traj.num_episodes}"
    speed = f"x{state.playback_speed}" if state.playback_speed > 1 else ""
    parts = [f"FactoriaX Inspector - {path}", ep]
    if traj.is_multi_player:
        parts.append(f"P{state.selected_player}")
    if speed:
        parts.append(speed)
    pygame.display.set_caption("  ".join(parts))


def _export_png(frame: np.ndarray, path: str, state: InspectorState) -> None:
    """Save the current frame as a PNG."""
    out = Path(path).stem + f"_step{state.current_step}.png"
    try:
        import imageio.v3 as iio

        iio.imwrite(out, frame)
        print(f"Saved screenshot: {out}")
    except ImportError:
        print("imageio required for PNG export.")


def _export_mp4(
    frames: list[np.ndarray] | None,
    path: str,
    state: InspectorState,
) -> None:
    """Export the game world frames as an MP4 video."""
    if frames is None or not frames:
        print("No frames to export.")
        return
    out = Path(path).stem + f"_ep{state.selected_episode}.mp4"
    try:
        import warnings

        import imageio.v3 as iio

        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="os.fork()",
            )
            iio.imwrite(
                str(out_path),
                np.stack([f.astype(np.uint8) for f in frames]),
                plugin="FFMPEG",
                fps=10,
                codec="libx264",
                pixelformat="yuv420p",
            )
        print(f"Saved video: {out}")
    except ImportError:
        print("imageio[ffmpeg] required for MP4 export.")


def _handle_key(
    event: pygame.event.Event,
    traj: Trajectory,
    state: InspectorState,
) -> None:
    """Process keyboard input for playback controls.

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
