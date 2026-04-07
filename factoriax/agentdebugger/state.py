"""Mutable debugger state for the step-through viewer."""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class DebuggerState:
    """All mutable UI state for the debugger.

    Attributes:
        current_step: View cursor index into the history (0 = initial
            state). This is display-only: stepping forward always
            appends at the end and auto-advances the cursor.
        mode: Stepping mode, either ``"human"`` or ``"ai"``.
        reward_chart_cache: Pre-rendered reward chart RGB image, or
            ``None`` when invalidated.
        cost_chart_cache: Pre-rendered cost chart RGB image, or
            ``None`` when invalidated.
        show_help: Whether the help overlay is visible.
        frame_tick: Frame counter for animation timing.
        done: Whether the environment returned ``done=True``.
    """

    current_step: int = 0
    mode: str = "ai"
    reward_chart_cache: np.ndarray | None = None
    cost_chart_cache: np.ndarray | None = None
    show_help: bool = False
    frame_tick: int = 0
    done: bool = False
