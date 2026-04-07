"""Debugger class: interactive step-through viewer for RL evaluation.

Spawns a pygame window that lets the user step through a live JAX
environment one action at a time. In AI mode, pressing ``N`` queries
the policy. In human mode, pressing ``N`` waits for a game action
via the full play UI (inventory, crafting, movement, etc.). The
``[`` and ``]`` keys navigate the view cursor through history without
taking new steps.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pygame
from jax import random

from factoriax.agentdebugger.layout import (
    MIN_QUADRANT_H,
    MIN_QUADRANT_W,
    compute_debugger_dimensions,
    render_debugger_frame,
)
from factoriax.agentdebugger.state import DebuggerState
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
    """

    def __init__(
        self,
        env: FactoriaXEnv,
        params: EnvParams,
        state: EnvState,
        *,
        policy: Callable[[jax.Array], jax.Array] | None = None,
        obs_fn: (
            Callable[[EnvState, EnvParams, int], jax.Array] | None
        ) = None,
        reward_fn: (
            Callable[[EnvState, EnvState, EnvParams], jax.Array] | None
        ) = None,
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

    def run(self) -> None:
        """Launch the debugger window. Blocks until the user closes it."""
        pygame.init()
        pygame.font.init()

        quadrant_w = max(MIN_QUADRANT_W, 320)
        quadrant_h = max(MIN_QUADRANT_H, 240)
        base_w, base_h = compute_debugger_dimensions(quadrant_w, quadrant_h)
        window_w, window_h = calculate_window_size(base_w, base_h)
        win_scale = max(
            1, min(window_w // base_w, window_h // base_h),
        )

        screen = pygame.display.set_mode((window_w, window_h))
        pygame.display.set_caption("FactoriaX Debugger")

        # JIT warmup.
        self._warmup(screen)

        # GameUI for human input mode.
        kb_lookup = build_key_lookup(default_keyboard())
        ui = GameUI(self._params, kb_lookup, welcome_open=False)
        ui.set_window_transform(0, 0, win_scale)

        win_ox = (window_w - base_w * win_scale) // 2
        win_oy = (window_h - base_h * win_scale) // 2

        clock = pygame.time.Clock()
        running = True

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
                    ui.set_window_transform(win_ox, win_oy, win_scale)
                    continue
                if event.type != pygame.KEYDOWN:
                    continue

                running = self._handle_keydown(event, ui)

            # Render.
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
                overlay = _render_help_overlay(base_w, base_h)
                composite_rgba_over_rgb(frame, overlay)

            # Awaiting input indicator.
            if self._awaiting_human_input:
                indicator = _render_awaiting_overlay(base_w, base_h)
                composite_rgba_over_rgb(frame, indicator)

            surface = pygame.surfarray.make_surface(
                np.transpose(frame, (1, 0, 2)),
            )
            scaled = pygame.transform.scale(
                surface, (base_w * win_scale, base_h * win_scale),
            )
            screen.fill((0, 0, 0))
            screen.blit(scaled, (win_ox, win_oy))
            pygame.display.flip()
            self._dbg.frame_tick += 1
            clock.tick(30)

        pygame.quit()

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
                rng, state, jnp.int32(Action.NOOP), params,
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

    def _handle_keydown(
        self,
        event: pygame.event.Event,
        ui: GameUI,
    ) -> bool:
        """Process a keyboard event. Returns False to quit."""
        key = event.key

        # If awaiting human input, pass to GameUI.
        if self._awaiting_human_input:
            state = self._states[-1]
            result = ui.handle_event(event, state)
            if result.state is not None:
                # Slot swaps etc. — update the latest state.
                self._states[-1] = result.state
            if result.action is not None:
                self._execute_step(result.action)
                self._awaiting_human_input = False
            return True

        if key == pygame.K_ESCAPE:
            if self._dbg.show_help:
                self._dbg.show_help = False
                return True
            if ui.has_menu_open():
                ui.handle_event(event, self._states[self._dbg.current_step])
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
            self._dbg.mode = (
                "ai" if self._dbg.mode == "human" else "human"
            )
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

    def _step_forward(self, ui: GameUI) -> None:
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


def _render_help_overlay(
    width: int, height: int,
) -> np.ndarray:
    """Render a translucent help overlay with keybinding reference.

    Args:
        width: Overlay width.
        height: Overlay height.

    Returns:
        RGBA uint8 array.
    """
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    overlay[:, :, 3] = 180  # translucent black

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
            # Alpha-composite text onto overlay.
            mask = text_rgba[:, :, 3:4].astype(np.float32) / 255.0
            overlay[y : y + th, x : x + tw, :3] = (
                text_rgba[:, :, :3].astype(np.float32) * mask
                + overlay[y : y + th, x : x + tw, :3].astype(np.float32)
                * (1.0 - mask)
            ).astype(np.uint8)
            overlay[y : y + th, x : x + tw, 3] = np.maximum(
                overlay[y : y + th, x : x + tw, 3],
                text_rgba[:, :, 3],
            )
        y += th + 2

    return overlay


def _render_awaiting_overlay(
    width: int, height: int,
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
        "PRESS AN ACTION KEY", font, (255, 220, 80),
    )
    th, tw = text_rgba.shape[:2]
    # Center horizontally near the top.
    x = (width - tw) // 2
    y = 8
    if x >= 0 and y + th <= height and x + tw <= width:
        # Draw background box.
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
            + overlay[y : y + th, x : x + tw, :3].astype(np.float32)
            * (1.0 - mask)
        ).astype(np.uint8)
        overlay[y : y + th, x : x + tw, 3] = np.maximum(
            overlay[y : y + th, x : x + tw, 3],
            text_rgba[:, :, 3],
        )

    return overlay
