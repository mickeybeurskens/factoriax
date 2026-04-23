"""Mutable debugger state for the step-through viewer."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from factoriax.analysis.trajectory import Trajectory


@dataclasses.dataclass
class DebuggerState:
    """All mutable UI state for the debugger.

    Live-mode fields are used when stepping through a live JAX
    environment. Replay-mode fields (prefixed with a comment block)
    are only meaningful when ``replay_mode`` is ``True``, i.e. when
    the debugger was created via
    :meth:`~factoriax.agentdebugger.Debugger.from_trajectory`.

    Attributes:
        current_step: View cursor index into the history (0 = initial
            state). This is display-only: stepping forward always
            appends at the end and auto-advances the cursor.
        mode: Stepping mode, either ``"human"`` or ``"ai"``.
        reward_chart_cache: Pre-rendered reward chart RGB image, or
            ``None`` when invalidated.
        reward_chart_plot_bounds: ``(plot_x0, plot_x1)`` column range of
            the matplotlib plot area inside ``reward_chart_cache``, used
            by :func:`draw_cursor` to place the cursor on the plot line
            rather than in the axis gutter. ``None`` when no cache.
        cost_chart_cache: Pre-rendered cost chart RGB image, or
            ``None`` when invalidated.
        cost_chart_plot_bounds: ``(plot_x0, plot_x1)`` column range for
            the cost chart's matplotlib plot area. See
            ``reward_chart_plot_bounds``.
        show_help: Whether the help overlay is visible.
        frame_tick: Frame counter for animation timing.
        done: Whether the environment returned ``done=True``.
        replay_mode: Whether this debugger is replaying a trajectory.
        playing: Whether automatic playback is active (replay only).
        playback_speed: Steps advanced per frame during playback.
        selected_episode: Episode index within the trajectory.
        selected_player: Player index for per-player charts.
        action_strip_cache: Pre-rendered action strip RGB image.
        action_legend_cache: Pre-rendered action legend RGB image.
        sankey_cache: Pre-rendered Sankey heatmap RGB image.
        show_obs_overlay: Whether fog-of-war overlay is active.
        trajectory_path: File path of the loaded trajectory.
        rendered_frames: Pre-rendered game world frames (replay only).
        trajectory: Reference to the loaded Trajectory (replay only).
    """

    # -- Shared fields --
    current_step: int = 0
    mode: str = "ai"
    reward_chart_cache: np.ndarray | None = None
    reward_chart_plot_bounds: tuple[int, int] | None = None
    cost_chart_cache: np.ndarray | None = None
    cost_chart_plot_bounds: tuple[int, int] | None = None
    show_help: bool = False
    frame_tick: int = 0
    done: bool = False

    # -- Replay-mode fields (defaults preserve live-mode behavior) --
    replay_mode: bool = False
    playing: bool = False
    playback_speed: int = 1
    # Direction of auto-advance during ``playing``. +1 = forward,
    # -1 = backward. Toggled with the ``B`` key.
    playback_direction: int = 1
    selected_episode: int = 0
    selected_player: int = 0
    action_strip_cache: np.ndarray | None = None
    action_legend_cache: np.ndarray | None = None
    sankey_cache: np.ndarray | None = None
    show_obs_overlay: bool = True
    trajectory_path: str | None = None
    rendered_frames: list[np.ndarray] | None = None
    trajectory: Trajectory | None = None
