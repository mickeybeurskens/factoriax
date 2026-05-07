"""Debugger class: interactive step-through viewer for RL evaluation.

Spawns a pygame window that lets the user step through a live JAX
environment one action at a time, or replay a pre-recorded trajectory.

**Live mode** (created via ``Debugger(env, params, state, ...)``):
  In AI mode, pressing ``N`` queries the policy. In human mode,
  pressing ``N`` waits for a game action via the full play UI.
  ``[`` and ``]`` navigate the view cursor through history.

**Replay mode** (created via ``Debugger.from_trajectory(path)``):
  Loads a ``.npz`` trajectory file and replays it with play/pause,
  speed control, episode/player switching, and action visualizations.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pygame
from jax import random

from factoriax.agentdebugger.layout import (
    MIN_QUADRANT_H,
    MIN_QUADRANT_W,
    action_strip_step_from_click,
    compute_debugger_dimensions,
    rebuild_replay_caches,
    render_debugger_frame,
    render_replay_frame,
)
from factoriax.agentdebugger.state import DebuggerState

if TYPE_CHECKING:
    from factoriax.agentdebugger.dialogs import FileBrowserDialog
from factoriax.config import build_key_lookup, default_keyboard
from factoriax.constants import Action
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.observations import global_array
from factoriax.play.game_ui import GameUI
from factoriax.state import EnvParams, EnvState
from factoriax.ui.compositing import composite_rgba_over_rgb
from factoriax.ui.fonts import get_pixel_font, render_text_rgba
from factoriax.ui.window import calculate_window_size

logger = logging.getLogger(__name__)


class Debugger:
    """Interactive step-through debugger for FactoriaX environments.

    Spawns a pygame window that blocks until the user closes it.
    Each press of ``N`` advances the environment by one step, either
    by querying the AI policy or by waiting for human input through
    the full game UI.

    For trajectory replay, use the :meth:`from_trajectory` classmethod
    instead of this constructor.

    Args:
        env: FactoriaXEnv instance.
        params: Environment parameters.
        state: Initial environment state.
        policy: AI policy callable ``(obs) -> action``. Optional.
        obs_fn: Observation extraction function
            ``(state, params, player_idx) -> obs``. Defaults to
            :func:`~factoriax.observations.global_array`.
        reward_fn: Reward function
            ``(prev_state, new_state, params) -> scalar``. Optional.
        constraint_fn: Constraint cost function
            ``(prev_state, new_state, params) -> cost_vector``. Optional.
        constraint_names: Labels for each constraint dimension.
        player_idx: Which player the human controls (default 0).
        seed: Random seed for the PRNG (default 42).

    Note:
        The :meth:`from_trajectory` classmethod bypasses this
        constructor via ``object.__new__()``. If you add logic here,
        mirror it in ``from_trajectory`` where applicable.
    """

    def __init__(
        self,
        env: FactoriaXEnv,
        params: EnvParams,
        state: EnvState,
        *,
        policy: Callable[[jax.Array], jax.Array] | None = None,
        obs_fn: (Callable[[EnvState, EnvParams, int], jax.Array] | None) = None,
        reward_fn: (Callable[[EnvState, EnvState, EnvParams], jax.Array] | None) = None,
        constraint_fn: (
            Callable[[EnvState, EnvState, EnvParams], jax.Array] | None
        ) = None,
        constraint_names: list[str] | None = None,
        player_idx: int = 0,
        seed: int = 42,
    ) -> None:
        self._env = env
        self._params = params
        self._step_fn = jax.jit(env.step_env)
        self._policy = policy
        self._obs_fn = obs_fn or global_array
        self._reward_fn = reward_fn
        self._constraint_fn = constraint_fn
        self._constraint_names = constraint_names or []
        self._player_idx = player_idx
        self._rng = random.PRNGKey(seed)

        # Append-only history.
        self._states: list[EnvState] = [state]
        self._actions: list[int] = []
        self._rewards: list[float] = []
        self._costs: list[np.ndarray] = []

        self._dbg = DebuggerState()
        self._awaiting_human_input = False
        # Replay-only field; the from_trajectory constructor rebinds it.
        self._level_path: str | None = None

    # ------------------------------------------------------------------
    # Trajectory replay constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_trajectory(
        cls,
        path: str,
        *,
        level_path: str | None = None,
        episode: int = 0,
        player_idx: int = 0,
    ) -> Debugger:
        """Create a replay-mode debugger from a trajectory file.

        Loads a ``.npz`` trajectory and pre-renders game frames if
        state data or a level file is available. The debugger opens
        with play/pause controls, episode/player switching, and action
        visualizations in Q3/Q4.

        Args:
            path: Path to a ``.npz`` trajectory file.
            level_path: Optional level JSON file for action replay
                rendering (used when trajectory lacks state data).
            episode: Initial episode index to display.
            player_idx: Initial player index.

        Returns:
            A :class:`Debugger` in replay mode.
        """
        from factoriax.analysis.trajectory import Trajectory

        traj = Trajectory.load(path)
        states = _load_trajectory_data(traj, episode, level_path)

        # Extract actions and rewards for the selected episode.
        ep_data = traj.episode(episode)
        if ep_data.is_multi_player:
            actions = [
                int(ep_data.actions[0, t, player_idx])
                for t in range(ep_data.episode_length)
            ]
        else:
            actions = [
                int(ep_data.actions[0, t]) for t in range(ep_data.episode_length)
            ]

        rewards: list[float] = []
        if traj.rewards is not None:
            rewards = [
                float(traj.rewards[episode, t]) for t in range(traj.episode_length)
            ]

        # Bypass __init__ -- replay mode needs none of the live fields.
        self = object.__new__(cls)
        self._env = None  # type: ignore[assignment]
        self._params = None  # type: ignore[assignment]
        self._step_fn = None  # type: ignore[assignment]
        self._policy = None
        self._obs_fn = None  # type: ignore[assignment]
        self._reward_fn = None
        self._constraint_fn = None
        self._constraint_names = []
        self._player_idx = player_idx
        self._rng = None  # type: ignore[assignment]

        self._states = states
        self._actions = actions
        self._rewards = rewards
        self._costs = []
        self._awaiting_human_input = False

        self._dbg = DebuggerState(
            replay_mode=True,
            trajectory=traj,
            trajectory_path=path,
            selected_episode=episode,
            selected_player=player_idx,
        )

        # Store for episode reloading.
        self._level_path = level_path

        return self

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Launch the debugger window. Blocks until the user closes it."""
        pygame.init()
        pygame.font.init()

        quadrant_w = max(MIN_QUADRANT_W, 320)
        quadrant_h = max(MIN_QUADRANT_H, 240)
        base_w, base_h = compute_debugger_dimensions(quadrant_w, quadrant_h)
        window_w, window_h = calculate_window_size(base_w, base_h)
        win_scale = max(
            1,
            min(window_w // base_w, window_h // base_h),
        )

        screen = pygame.display.set_mode((window_w, window_h))

        if self._dbg.replay_mode:
            self._update_replay_caption()
            rebuild_replay_caches(self._dbg, quadrant_w, quadrant_h)
        else:
            pygame.display.set_caption("FactoriaX Debugger")
            self._warmup(screen)

        # GameUI for human input mode (live mode only).
        ui: GameUI | None = None
        if not self._dbg.replay_mode:
            kb_lookup = build_key_lookup(default_keyboard())
            ui = GameUI(self._params, kb_lookup, welcome_open=False)
            ui.set_window_transform(0, 0, win_scale)

        win_ox = (window_w - base_w * win_scale) // 2
        win_oy = (window_h - base_h * win_scale) // 2

        clock = pygame.time.Clock()
        running = True
        dialog: FileBrowserDialog | None = None

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    continue
                if event.type == pygame.VIDEORESIZE:
                    window_w, window_h = event.w, event.h
                    win_scale = max(
                        1,
                        min(window_w // base_w, window_h // base_h),
                    )
                    win_ox = (window_w - base_w * win_scale) // 2
                    win_oy = (window_h - base_h * win_scale) // 2
                    if ui is not None:
                        ui.set_window_transform(win_ox, win_oy, win_scale)
                    continue
                if (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and self._dbg.replay_mode
                    and self._dbg.trajectory is not None
                ):
                    step = action_strip_step_from_click(
                        event.pos[0],
                        event.pos[1],
                        win_ox,
                        win_oy,
                        win_scale,
                        quadrant_w,
                        quadrant_h,
                        self._dbg.trajectory.episode_length,
                    )
                    if step is not None:
                        self._dbg.current_step = step
                        self._dbg.playing = False
                    continue
                if event.type != pygame.KEYDOWN:
                    continue

                # File browser dialog consumes all keys while open.
                if dialog is not None:
                    result = dialog.handle_event(event)
                    if result == "ok":
                        chosen = dialog.get_path()
                        if chosen is not None:
                            self._load_new_trajectory(
                                chosen,
                                quadrant_w,
                                quadrant_h,
                            )
                        dialog = None
                    elif result == "cancel":
                        dialog = None
                    continue

                if self._dbg.replay_mode:
                    running, dialog = self._handle_replay_keydown(
                        event,
                        quadrant_w,
                        quadrant_h,
                    )
                else:
                    running = self._handle_keydown(event, ui)

            # Playback auto-advance (replay mode). ``playback_direction``
            # is ±1 — same playback_speed applies in both directions.
            if self._dbg.replay_mode and self._dbg.playing:
                traj = self._dbg.trajectory
                total = traj.episode_length if traj is not None else 1
                step_delta = self._dbg.playback_speed * self._dbg.playback_direction
                new_step = self._dbg.current_step + step_delta
                self._dbg.current_step = max(0, min(new_step, total - 1))
                # Stop when the bound in the current direction is hit.
                if (
                    self._dbg.playback_direction > 0
                    and self._dbg.current_step >= total - 1
                ) or (self._dbg.playback_direction < 0 and self._dbg.current_step <= 0):
                    self._dbg.playing = False

            # Render.
            if self._dbg.replay_mode:
                frame = render_replay_frame(
                    self._dbg,
                    self._states,
                    base_w,
                    base_h,
                    quadrant_w,
                    quadrant_h,
                )
            else:
                frame = render_debugger_frame(
                    self._dbg,
                    self._states,
                    self._rewards,
                    self._actions,
                    self._costs,
                    self._constraint_names,
                    base_w,
                    base_h,
                    quadrant_w,
                    quadrant_h,
                    has_reward=self._reward_fn is not None,
                    has_cost=self._constraint_fn is not None,
                    player_idx=self._player_idx,
                )

            # Help overlay.
            if self._dbg.show_help:
                if self._dbg.replay_mode:
                    overlay = _render_replay_help_overlay(base_w, base_h)
                else:
                    overlay = _render_help_overlay(base_w, base_h)
                composite_rgba_over_rgb(frame, overlay)

            # Awaiting input indicator (live mode only).
            if self._awaiting_human_input:
                indicator = _render_awaiting_overlay(base_w, base_h)
                composite_rgba_over_rgb(frame, indicator)

            # Dialog overlay.
            if dialog is not None:
                dlg_overlay = dialog.render(base_w, base_h)
                composite_rgba_over_rgb(frame, dlg_overlay)

            surface = pygame.surfarray.make_surface(
                np.transpose(frame, (1, 0, 2)),
            )
            scaled = pygame.transform.scale(
                surface,
                (base_w * win_scale, base_h * win_scale),
            )
            screen.fill((0, 0, 0))
            screen.blit(scaled, (win_ox, win_oy))
            pygame.display.flip()
            self._dbg.frame_tick += 1
            clock.tick(30)

        pygame.quit()

    # ------------------------------------------------------------------
    # JIT warmup (live mode only)
    # ------------------------------------------------------------------

    def _warmup(self, screen: pygame.Surface) -> None:
        """JIT-compile the step function with a loading screen."""
        import threading

        state = self._states[0]
        params = self._params
        step_fn = self._step_fn
        rng = self._rng

        done_flag = [False]

        def _compile() -> None:
            step_fn(
                rng,
                state,
                jnp.int32(Action.NOOP),
                params,
            )[0].block_until_ready()
            done_flag[0] = True

        thread = threading.Thread(target=_compile)
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
            label = "Compiling JAX" + "." * dots
            screen.fill((20, 20, 25))
            text = font.render(label, True, (180, 180, 180))
            cx = (screen.get_width() - text.get_width()) // 2
            cy = (screen.get_height() - text.get_height()) // 2
            screen.blit(text, (cx, cy))
            pygame.display.flip()
            clock.tick(8)

        thread.join()

    # ------------------------------------------------------------------
    # Live-mode key handling
    # ------------------------------------------------------------------

    def _handle_keydown(
        self,
        event: pygame.event.Event,
        ui: GameUI | None,
    ) -> bool:
        """Process a keyboard event (live mode). Returns False to quit."""
        key = event.key

        # If awaiting human input, pass to GameUI.
        if self._awaiting_human_input and ui is not None:
            state = self._states[-1]
            result = ui.handle_event(event, state)
            if result.state is not None:
                self._states[-1] = result.state
            if result.action is not None:
                self._execute_step(result.action)
                self._awaiting_human_input = False
            return True

        if key == pygame.K_ESCAPE:
            if self._dbg.show_help:
                self._dbg.show_help = False
                return True
            if ui is not None and ui.has_menu_open():
                ui.handle_event(
                    event,
                    self._states[self._dbg.current_step],
                )
                return True
            return False

        if self._dbg.show_help:
            self._dbg.show_help = False
            return True

        if key == pygame.K_LEFTBRACKET:
            if self._dbg.current_step > 0:
                self._dbg.current_step -= 1
            return True

        if key == pygame.K_RIGHTBRACKET:
            if self._dbg.current_step < len(self._states) - 1:
                self._dbg.current_step += 1
            return True

        if key == pygame.K_TAB:
            self._dbg.mode = "ai" if self._dbg.mode == "human" else "human"
            self._awaiting_human_input = False
            return True

        if key == pygame.K_n:
            self._step_forward(ui)
            return True

        if key == pygame.K_PERIOD:
            num_players = self._params.num_players
            self._player_idx = (self._player_idx + 1) % num_players
            return True

        if key in (pygame.K_SLASH, pygame.K_QUESTION):
            self._dbg.show_help = True
            return True

        return True

    # ------------------------------------------------------------------
    # Replay-mode key handling
    # ------------------------------------------------------------------

    def _handle_replay_keydown(
        self,
        event: pygame.event.Event,
        quadrant_w: int,
        quadrant_h: int,
    ) -> tuple[bool, FileBrowserDialog | None]:
        """Process a keyboard event (replay mode).

        Args:
            event: pygame KEYDOWN event.
            quadrant_w: Quadrant width (for cache rebuilding).
            quadrant_h: Quadrant height (for cache rebuilding).

        Returns:
            ``(running, dialog)`` where running is False to quit and
            dialog is a :class:`FileBrowserDialog` or None.
        """
        key = event.key
        traj = self._dbg.trajectory
        total = traj.episode_length if traj is not None else 1

        if key == pygame.K_ESCAPE:
            if self._dbg.show_help:
                self._dbg.show_help = False
                return True, None
            return False, None

        if self._dbg.show_help:
            self._dbg.show_help = False
            return True, None

        if key == pygame.K_SPACE:
            self._dbg.playing = not self._dbg.playing

        elif key in (pygame.K_LEFT, pygame.K_LEFTBRACKET):
            self._dbg.current_step = max(0, self._dbg.current_step - 1)
            self._dbg.playing = False

        elif key in (pygame.K_RIGHT, pygame.K_RIGHTBRACKET, pygame.K_n):
            self._dbg.current_step = min(
                total - 1,
                self._dbg.current_step + 1,
            )
            self._dbg.playing = False

        elif key in (pygame.K_HOME, pygame.K_0):
            self._dbg.current_step = 0
            self._dbg.playing = False

        elif key == pygame.K_END:
            self._dbg.current_step = total - 1
            self._dbg.playing = False

        elif key == pygame.K_MINUS:
            self._dbg.playback_speed = max(1, self._dbg.playback_speed - 1)

        elif key in (pygame.K_EQUALS, pygame.K_PLUS):
            self._dbg.playback_speed = min(
                20,
                self._dbg.playback_speed + 1,
            )

        elif key == pygame.K_COMMA:
            if traj is not None and self._dbg.selected_episode > 0:
                self._dbg.selected_episode -= 1
                self._on_episode_change(quadrant_w, quadrant_h)

        elif key == pygame.K_PERIOD:
            if traj is not None and self._dbg.selected_episode < traj.num_episodes - 1:
                self._dbg.selected_episode += 1
                self._on_episode_change(quadrant_w, quadrant_h)

        elif key == pygame.K_TAB:
            if traj is not None and traj.is_multi_player:
                self._dbg.selected_player = (
                    self._dbg.selected_player + 1
                ) % traj.num_players
                self._on_player_change(quadrant_w, quadrant_h)

        elif key == pygame.K_l:
            from factoriax.agentdebugger.dialogs import FileBrowserDialog

            dialog = FileBrowserDialog(
                title="Load Trajectory",
                pattern="**/*.npz",
            )
            return True, dialog

        elif key == pygame.K_b:
            # Flip playback direction; same speed applies in reverse.
            self._dbg.playback_direction = -self._dbg.playback_direction

        elif key == pygame.K_o:
            self._dbg.show_obs_overlay = not self._dbg.show_obs_overlay

        elif key == pygame.K_s:
            self._export_png()

        elif key == pygame.K_v:
            self._export_mp4()

        elif key in (pygame.K_SLASH, pygame.K_QUESTION):
            self._dbg.show_help = True

        return True, None

    # ------------------------------------------------------------------
    # Live-mode stepping
    # ------------------------------------------------------------------

    def _step_forward(self, ui: GameUI | None) -> None:
        """Advance one step in the appropriate mode."""
        if self._dbg.done:
            return

        # If the cursor isn't at the end, just advance it.
        if self._dbg.current_step < len(self._states) - 1:
            self._dbg.current_step += 1
            return

        if self._dbg.mode == "ai":
            if self._policy is None:
                logger.warning("No policy provided; cannot step in AI mode")
                return
            state = self._states[-1]
            state_p = state.replace(
                selected_player=self._player_idx,
            )
            obs = self._obs_fn(state_p, self._params, self._player_idx)
            action = int(self._policy(obs))
            self._execute_step(action)
        else:
            self._awaiting_human_input = True

    def _execute_step(self, human_action: int) -> None:
        """Step all players and record the result.

        The controlled player uses *human_action*. Other players use
        the policy (if available) or NOOP.

        Args:
            human_action: Action for the controlled player.
        """
        prev_state = self._states[-1]
        state = prev_state
        num_players = self._params.num_players
        done = jnp.array(False)

        for p in range(num_players):
            if p == self._player_idx:
                action = human_action
            elif self._policy is not None:
                state_p = state.replace(selected_player=p)
                obs = self._obs_fn(state_p, self._params, p)
                action = int(self._policy(obs))
            else:
                action = int(Action.NOOP)

            self._rng, step_key = random.split(self._rng)
            _, state, _, done, _ = self._step_fn(
                step_key,
                state.replace(selected_player=p),
                action,
                self._params,
            )

        self._states.append(state)
        self._actions.append(human_action)

        if self._reward_fn is not None:
            reward = float(
                self._reward_fn(prev_state, state, self._params),
            )
            self._rewards.append(reward)

        if self._constraint_fn is not None:
            cost = np.asarray(
                self._constraint_fn(prev_state, state, self._params),
            )
            self._costs.append(cost)

        self._dbg.current_step = len(self._states) - 1
        self._dbg.reward_chart_cache = None
        self._dbg.cost_chart_cache = None

        if bool(done):
            self._dbg.done = True

    # ------------------------------------------------------------------
    # Replay-mode helpers
    # ------------------------------------------------------------------

    def _on_episode_change(
        self,
        quadrant_w: int,
        quadrant_h: int,
    ) -> None:
        """Reload data and caches after switching episodes."""
        traj = self._dbg.trajectory
        if traj is None:
            return
        ep = self._dbg.selected_episode
        level_path = getattr(self, "_level_path", None)

        self._states = _load_trajectory_data(traj, ep, level_path)
        self._dbg.current_step = 0
        self._dbg.playing = False

        # Re-extract actions/rewards.
        ep_data = traj.episode(ep)
        if ep_data.is_multi_player:
            self._actions = [
                int(ep_data.actions[0, t, self._dbg.selected_player])
                for t in range(ep_data.episode_length)
            ]
        else:
            self._actions = [
                int(ep_data.actions[0, t]) for t in range(ep_data.episode_length)
            ]
        if traj.rewards is not None:
            self._rewards = [
                float(traj.rewards[ep, t]) for t in range(traj.episode_length)
            ]
        else:
            self._rewards = []

        rebuild_replay_caches(self._dbg, quadrant_w, quadrant_h)
        self._update_replay_caption()

    def _on_player_change(
        self,
        quadrant_w: int,
        quadrant_h: int,
    ) -> None:
        """Rebuild per-player caches after switching players."""
        rebuild_replay_caches(self._dbg, quadrant_w, quadrant_h)

    def _load_new_trajectory(
        self,
        new_path: str,
        quadrant_w: int,
        quadrant_h: int,
    ) -> None:
        """Load a new trajectory from the file browser."""
        from factoriax.analysis.trajectory import Trajectory

        traj = Trajectory.load(new_path)
        self._dbg.trajectory = traj
        self._dbg.trajectory_path = new_path
        self._dbg.selected_episode = 0
        self._dbg.selected_player = 0
        self._dbg.current_step = 0
        self._dbg.playing = False

        level_path = getattr(self, "_level_path", None)
        self._states = _load_trajectory_data(traj, 0, level_path)

        ep_data = traj.episode(0)
        if ep_data.is_multi_player:
            self._actions = [
                int(ep_data.actions[0, t, 0]) for t in range(ep_data.episode_length)
            ]
        else:
            self._actions = [
                int(ep_data.actions[0, t]) for t in range(ep_data.episode_length)
            ]
        if traj.rewards is not None:
            self._rewards = [
                float(traj.rewards[0, t]) for t in range(traj.episode_length)
            ]
        else:
            self._rewards = []

        rebuild_replay_caches(self._dbg, quadrant_w, quadrant_h)
        self._update_replay_caption()

    def _update_replay_caption(self) -> None:
        """Update window title with trajectory info."""
        traj = self._dbg.trajectory
        path = self._dbg.trajectory_path or "trajectory"
        parts = [f"FactoriaX Debugger - {path}"]
        if traj is not None:
            parts.append(
                f"ep {self._dbg.selected_episode + 1}/{traj.num_episodes}",
            )
            if traj.is_multi_player:
                parts.append(f"P{self._dbg.selected_player}")
        if self._dbg.playback_speed > 1:
            parts.append(f"x{self._dbg.playback_speed}")
        pygame.display.set_caption("  ".join(parts))

    def _export_png(self) -> None:
        """Save the current frame as PNG (rendered on demand)."""
        if not self._states or self._dbg.current_step >= len(self._states):
            print("No frame to export.")
            return
        from factoriax.jax_renderer import JaxRenderer

        state = self._states[self._dbg.current_step]
        renderer = JaxRenderer(tile_px=24)
        frame = np.asarray(renderer.jit_render_map(state))
        path = self._dbg.trajectory_path or "trajectory"
        out = Path(path).stem + f"_step{self._dbg.current_step}.png"
        try:
            import imageio.v3 as iio

            iio.imwrite(out, frame)
            print(f"Saved screenshot: {out}")
        except ImportError:
            print("imageio required for PNG export.")

    def _export_mp4(self) -> None:
        """Export the episode as an MP4, rendering each frame on demand.

        Renders once per export rather than relying on a cached frame
        list; export is a one-shot user action, so the few-second
        render cost is preferable to keeping a multi-GB cache in RAM.
        """
        if not self._states:
            print("No frames to export.")
            return
        from factoriax.jax_renderer import JaxRenderer

        path = self._dbg.trajectory_path or "trajectory"
        out = Path(path).stem + f"_ep{self._dbg.selected_episode}.mp4"
        try:
            import warnings

            import imageio.v3 as iio

            out_path = Path(out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Rendering {len(self._states)} frames for export...")
            renderer = JaxRenderer(tile_px=24)
            rendered = np.stack(
                [
                    np.asarray(renderer.jit_render_map(s), dtype=np.uint8)
                    for s in self._states
                ]
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    message="os.fork()",
                )
                iio.imwrite(
                    str(out_path),
                    rendered,
                    plugin="FFMPEG",
                    fps=10,
                    codec="libx264",
                    pixelformat="yuv420p",
                )
            print(f"Saved video: {out}")
        except ImportError:
            print("imageio[ffmpeg] required for MP4 export.")


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _load_trajectory_data(
    traj: object,
    episode: int,
    level_path: str | None,
) -> list[EnvState]:
    """Load per-step environment states from a trajectory.

    Returns states only — frames are rendered on demand inside
    :func:`render_replay_frame`. Pre-rendering the whole frame list
    upfront at ~550 KB per frame caused multi-GB RAM spikes on long
    trajectories; the state list is ~45x smaller and the JIT'd JAX
    renderer is fast enough (~0.3 ms at 24 px / 32x32) to redo per
    frame.

    Args:
        traj: :class:`~factoriax.analysis.trajectory.Trajectory`.
        episode: Episode index.
        level_path: Optional level JSON for action replay (when the
            trajectory only has actions, not state snapshots).

    Returns:
        List of :class:`EnvState` (empty when neither a level nor a
        block_map is available — callers then fall back to the
        grid-world view in layout.py).
    """
    from factoriax.analysis.trajectory import Trajectory

    if not isinstance(traj, Trajectory):
        return []

    if level_path is not None:
        from factoriax.agentdebugger.replay import load_replay_states

        return load_replay_states(level_path, traj, episode)

    if traj.block_map is not None:
        from factoriax.analysis.trajectory import trajectory_to_states

        return trajectory_to_states(traj, episode=episode)

    return []


# ------------------------------------------------------------------
# Overlay rendering
# ------------------------------------------------------------------


def _render_help_overlay(
    width: int,
    height: int,
) -> np.ndarray:
    """Render a translucent help overlay for live mode.

    Args:
        width: Overlay width.
        height: Overlay height.

    Returns:
        RGBA uint8 array.
    """
    lines = [
        "DEBUGGER CONTROLS",
        "",
        "N         Step forward (AI or Human)",
        "[  /  ]   Navigate history backward / forward",
        "Tab       Toggle Human / AI mode",
        ".         Cycle selected player",
        "Esc       Quit (or close menu)",
        "?         Toggle this help",
        "",
        "HUMAN MODE (after pressing N):",
        "  WASD / Arrows   Move",
        "  Space           Mine",
        "  E               Place / Pick up",
        "  R               Rotate",
        "  I               Inventory",
        "  1-8             Select slot",
    ]
    return _render_overlay_with_lines(width, height, lines)


def _render_replay_help_overlay(
    width: int,
    height: int,
) -> np.ndarray:
    """Render a translucent help overlay for replay mode.

    Args:
        width: Overlay width.
        height: Overlay height.

    Returns:
        RGBA uint8 array.
    """
    lines = [
        "REPLAY CONTROLS",
        "",
        "Space       Play / Pause",
        "B           Reverse playback direction",
        "Left/Right  Step backward / forward",
        "[  /  ]     Step backward / forward",
        "N           Step forward",
        "Home / 0    Go to start",
        "End         Go to end",
        "-  /  +     Decrease / increase speed",
        ",  /  .     Previous / next episode",
        "Tab         Cycle player",
        "L           Load trajectory file",
        "Click       Seek on action strip",
        "O           Toggle fog-of-war overlay",
        "S           Save screenshot (PNG)",
        "V           Export video (MP4)",
        "?           Toggle this help",
        "Esc         Quit",
    ]
    return _render_overlay_with_lines(width, height, lines)


def _render_overlay_with_lines(
    width: int,
    height: int,
    lines: list[str],
) -> np.ndarray:
    """Render a translucent help overlay with text lines.

    Args:
        width: Overlay width.
        height: Overlay height.
        lines: Text lines to display.

    Returns:
        RGBA uint8 array.
    """
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    overlay[:, :, 3] = 180

    font = get_pixel_font(11)
    y = 20
    for line in lines:
        if not line:
            y += 10
            continue
        text_rgba = render_text_rgba(line, font, (200, 200, 190))
        th, tw = text_rgba.shape[:2]
        x = 20
        if x + tw <= width and y + th <= height:
            mask = text_rgba[:, :, 3:4].astype(np.float32) / 255.0
            overlay[y : y + th, x : x + tw, :3] = (
                text_rgba[:, :, :3].astype(np.float32) * mask
                + overlay[y : y + th, x : x + tw, :3].astype(np.float32) * (1.0 - mask)
            ).astype(np.uint8)
            overlay[y : y + th, x : x + tw, 3] = np.maximum(
                overlay[y : y + th, x : x + tw, 3],
                text_rgba[:, :, 3],
            )
        y += th + 2

    return overlay


def _render_awaiting_overlay(
    width: int,
    height: int,
) -> np.ndarray:
    """Render a small 'AWAITING INPUT' indicator overlay.

    Args:
        width: Overlay width.
        height: Overlay height.

    Returns:
        RGBA uint8 array.
    """
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    font = get_pixel_font(12)
    text_rgba = render_text_rgba(
        "PRESS AN ACTION KEY",
        font,
        (255, 220, 80),
    )
    th, tw = text_rgba.shape[:2]
    x = (width - tw) // 2
    y = 8
    if x >= 0 and y + th <= height and x + tw <= width:
        pad = 4
        bx0 = max(0, x - pad)
        by0 = max(0, y - pad)
        bx1 = min(width, x + tw + pad)
        by1 = min(height, y + th + pad)
        overlay[by0:by1, bx0:bx1, :3] = (30, 30, 35)
        overlay[by0:by1, bx0:bx1, 3] = 220

        mask = text_rgba[:, :, 3:4].astype(np.float32) / 255.0
        overlay[y : y + th, x : x + tw, :3] = (
            text_rgba[:, :, :3].astype(np.float32) * mask
            + overlay[y : y + th, x : x + tw, :3].astype(np.float32) * (1.0 - mask)
        ).astype(np.uint8)
        overlay[y : y + th, x : x + tw, 3] = np.maximum(
            overlay[y : y + th, x : x + tw, 3],
            text_rgba[:, :, 3],
        )

    return overlay
