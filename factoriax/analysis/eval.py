"""Eval-rollout container and summary plots.

A single-episode rollout (states, actions, achievement masks) is the
shared input format for offline inspection visualizations: per-item
inventory trajectories, per-action histograms, and achievement
unlock timing. Both the PPO eval pipeline and the scripted-agent
runners produce :class:`EvalRollout` instances and call
:func:`generate_eval_plots`.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import numpy as np

from factoriax.engine.constants import NUM_ACTIONS, NUM_ITEM_TYPES, Action, ItemType

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EvalRollout:
    """Collected artifacts from a single deterministic eval episode.

    Attributes:
        frames: Rendered RGB frames, length ``T + 1`` (includes the
            pre-step state so the video starts from the initial
            state). ``None`` when the producer streamed frames to disk
            instead of buffering them in memory.
        actions: Actions taken, shape ``(T,)`` int32.
        env_states: Inner ``EnvState`` at each step, length ``T + 1``.
            Consumed by ``factoriax.analysis.states_to_trajectory``.
        ach_per_step: Per-step achievement masks, shape
            ``(T + 1, MAX_ACHIEVEMENTS)`` bool. ``ach_per_step[t]`` is
            the latched mask after step ``t-1`` (row 0 = all False).
    """

    frames: list[np.ndarray] | None
    actions: np.ndarray
    env_states: list[Any]
    ach_per_step: np.ndarray

    @property
    def final_ach_mask(self) -> np.ndarray:
        """Latched achievement mask at the last recorded step."""
        mask: np.ndarray = self.ach_per_step[-1]
        return mask


def plot_item_counts(
    traj: Any,
    out_path: Any,
    *,
    title: str = "Item counts over time",
) -> Any:
    """Line plot of player item counts over time.

    Draws one line per :class:`ItemType` that exceeds zero at some
    point in the episode. The dense ``player_inventory`` field on the
    trajectory (shape ``(1, T+1, P, N)``) is already per-item, so no
    slot-to-item aggregation is needed.

    Args:
        traj: Trajectory pytree from
            :func:`factoriax.analysis.trajectory.states_to_trajectory`.
        out_path: PNG destination.
        title: Figure title.

    Returns:
        ``out_path`` as written.

    Raises:
        ValueError: If the trajectory has no ``player_inventory``.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if traj.player_inventory is None:
        raise ValueError("trajectory is missing player_inventory")
    inv = np.asarray(traj.player_inventory)[0, :, 0, :]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Item count in inventory")
    ax.set_title(title)

    drawn = 0
    for i in range(1, NUM_ITEM_TYPES):
        series = inv[:, i]
        if series.max() == 0:
            continue
        ax.plot(series, label=ItemType(i).name, linewidth=1.2)
        drawn += 1

    if drawn:
        ncol = 2 if drawn > 8 else 1
        ax.legend(fontsize="small", ncol=ncol, loc="upper left")
    ax.set_xlim(0, inv.shape[0] - 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_action_counts(
    actions: np.ndarray,
    out_path: Any,
    *,
    title: str = "Action counts",
) -> Any:
    """Bar chart of per-action counts over a single eval episode.

    Single-episode action distribution over time is noisy, so this
    helper plots a simple counts view. A moving-average distribution
    is a better fit for multi-episode analyses.

    Args:
        actions: 1-D int32 array of action indices.
        out_path: PNG destination.
        title: Figure title.

    Returns:
        ``out_path`` as written.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    counts = np.bincount(actions, minlength=NUM_ACTIONS)
    order = np.argsort(counts)[::-1]
    kept = [i for i in order if counts[i] > 0]
    names = [Action(int(i)).name for i in kept]
    vals = [int(counts[i]) for i in kept]

    height = max(4.0, 0.25 * len(kept))
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(range(len(kept)), vals[::-1], color="steelblue", edgecolor="black")
    ax.set_yticks(range(len(kept)))
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel(f"Count over {len(actions)} episode steps")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_eval_plots(
    rollout: EvalRollout,
    out_dir: Any,
    *,
    achievement_labels: list[str],
    num_achievements: int,
    title_prefix: str = "Final rollout",
) -> dict[str, Any]:
    """Render inventory, action-count, and achievement-timing plots.

    Args:
        rollout: Recorded episode artifacts.
        out_dir: Destination directory; created if missing.
        achievement_labels: Per-achievement display names; length
            equals ``num_achievements``.
        num_achievements: How many leading slots of
            ``rollout.ach_per_step`` are real achievements (the
            trailing slots are padding from ``MAX_ACHIEVEMENTS``).
        title_prefix: Prepended to each figure title.

    Returns:
        Mapping ``{"items": path, "actions": path, "achievements": path}``.
        Keys are omitted when their plot fails to generate.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")

    from factoriax.analysis.milestones import (  # noqa: PLC0415
        plot_achievement_timing,
    )
    from factoriax.analysis.trajectory import (  # noqa: PLC0415
        states_to_trajectory,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    actions_padded = np.concatenate(
        [rollout.actions, np.zeros((1,), dtype=np.int32)],
    )
    base_traj = states_to_trajectory(rollout.env_states, actions=actions_padded)
    traj = dataclasses.replace(
        base_traj,
        achievements=rollout.ach_per_step[np.newaxis, :, :num_achievements],
    )

    paths: dict[str, Any] = {}
    try:
        paths["items"] = plot_item_counts(
            traj,
            out_dir / "final_items.png",
            title=f"{title_prefix} — item counts over time",
        )
    except Exception:  # noqa: BLE001
        logger.exception("Item count plot failed.")

    try:
        paths["actions"] = plot_action_counts(
            rollout.actions,
            out_dir / "final_actions.png",
            title=f"{title_prefix} — action counts",
        )
    except Exception:  # noqa: BLE001
        logger.exception("Action count plot failed.")

    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, _ = plot_achievement_timing(
            traj,
            achievement_labels=achievement_labels,
            title=f"{title_prefix} — achievement unlock timing",
        )
        ach_path = out_dir / "final_achievements.png"
        fig.savefig(ach_path, dpi=150)
        plt.close(fig)
        paths["achievements"] = ach_path
    except Exception:  # noqa: BLE001
        logger.exception("Achievement timing plot failed.")

    return paths
