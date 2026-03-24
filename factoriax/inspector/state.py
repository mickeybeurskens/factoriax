"""Mutable inspector state for the trajectory viewer."""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class InspectorState:
    """All mutable state for the inspector UI.

    Attributes:
        current_step: Index of the currently displayed timestep.
        playing: Whether automatic playback is active.
        playback_speed: Steps advanced per frame tick during playback.
        selected_player: Player index for the info panel and action strip.
        selected_episode: Episode index within the loaded trajectory.
        reward_chart_cache: Pre-rendered reward chart RGB image.
        action_strip_cache: Pre-rendered action color strip RGB image.
        timeline_dragging: Whether the user is scrubbing the timeline.
    """

    current_step: int = 0
    playing: bool = False
    playback_speed: int = 1
    selected_player: int = 0
    selected_episode: int = 0
    reward_chart_cache: np.ndarray | None = None
    action_strip_cache: np.ndarray | None = None
    action_legend_cache: np.ndarray | None = None
    sankey_cache: np.ndarray | None = None
    timeline_dragging: bool = False
    show_help: bool = False
