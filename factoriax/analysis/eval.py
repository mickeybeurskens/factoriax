"""Hold one eval episode and draw its summary plots.

:class:`EvalRollout` is the shared format for a single episode: the
states, the actions, and the achievement mask at each step. The PPO
eval pipeline and the scripted-agent runners both build one and pass
it to :func:`generate_eval_plots`.

That call writes three figures. One shows the item counts over time,
one the action totals, and one the timestep at which each achievement
unlocked.
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
    """The recorded output of one deterministic eval episode.

    Attributes
    ----------
    frames :
        RGB frames for a video, or ``None`` when the run rendered
        none.
    actions :
        The action taken at each step, as a 1-D int array of length
        ``T``.
    env_states :
        The state at each step. This list holds ``T + 1`` entries,
        one more than ``actions``, because it records the state
        before the first action and after the last.
    ach_per_step :
        The achievement mask at each step, of shape
        ``(T + 1, MAX_ACHIEVEMENTS)``. The mask is latched, so a bit
        stays set once it is set. Trailing slots past the achievement
        count of the scenario are padding.
    """

    frames: list[np.ndarray] | None
    actions: np.ndarray
    env_states: list[Any]
    ach_per_step: np.ndarray

    @property
    def final_ach_mask(self) -> np.ndarray:
        """The achievement mask at the last recorded step.

        Returns
        -------
        numpy.ndarray
            One row of ``ach_per_step``, of length
            ``MAX_ACHIEVEMENTS``. Because the mask is latched, this row
            names every achievement the episode reached.
        """
        mask: np.ndarray = self.ach_per_step[-1]
        return mask


def plot_item_counts(
    traj: Any,
    out_path: Any,
    *,
    title: str = "Item counts over time",
) -> Any:
    """Line plot of player item counts over time.

    The plot holds one line for each item type that rises above zero
    at some point. An item that stays at zero is left out, which keeps
    the legend short.

    The function reads episode 0 and player 0 only, from
    ``player_inventory`` of shape ``(1, T + 1, P, N)``. That field
    already holds a count for each item, so nothing has to map slots
    to items.

    Parameters
    ----------
    traj :
        A trajectory from
        :func:`factoriax.analysis.trajectory.states_to_trajectory`.
    out_path :
        Where to write the PNG. Parent directories are not created.
    title :
        The figure title.

    Returns
    -------
    Any
        ``out_path`` unchanged, after the write.

    Raises
    ------
    ValueError
        When the trajectory carries no ``player_inventory``.

    Notes
    -----
    The function writes a file and closes its own figure.
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

    Bars are sorted by count, and an action with a count of zero is
    left out. The figure grows taller with the number of actions kept,
    so the labels stay readable.

    One episode gives a noisy distribution over time, so this plot
    shows plain totals. For several episodes, a moving average of the
    distribution says more.

    Parameters
    ----------
    actions :
        Action ids in a 1-D int array. Every entry counts, so a
        trajectory padded with zeros reports those pad steps as real
        ``NOOP`` actions.
    out_path :
        Where to write the PNG. Parent directories are not created.
    title :
        The figure title.

    Returns
    -------
    Any
        ``out_path`` unchanged, after the write.

    Notes
    -----
    The function writes a file and closes its own figure.
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

    Each plot is drawn inside its own ``try``. A plot that raises is
    logged with its traceback and left out of the result, so one
    failure does not cost the other two.

    Parameters
    ----------
    rollout :
        The recorded episode.
    out_dir :
        Where to write the three PNG files. Missing parent directories
        are created.
    achievement_labels :
        The display name of each achievement. The list must hold
        ``num_achievements`` entries.
    num_achievements :
        How many leading slots of ``rollout.ach_per_step`` name a real
        achievement. The slots after them are padding up to
        ``MAX_ACHIEVEMENTS``.
    title_prefix :
        Text put in front of each figure title.

    Returns
    -------
    dict
        ``{"items": path, "actions": path, "achievements": path}``. A
        key is absent when its plot raised, so a caller must not
        assume all three.

    Notes
    -----
    The function selects the Agg matplotlib backend for the whole
    process. It also pads ``rollout.actions`` with one zero, to match
    the state list, which is one longer. Action 0 is ``NOOP``, so the
    action-count plot plots one extra no-op.
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
            title=f"{title_prefix}: item counts over time",
        )
    except Exception:  # noqa: BLE001
        logger.exception("Item count plot failed.")

    try:
        paths["actions"] = plot_action_counts(
            rollout.actions,
            out_dir / "final_actions.png",
            title=f"{title_prefix}: action counts",
        )
    except Exception:  # noqa: BLE001
        logger.exception("Action count plot failed.")

    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, _ = plot_achievement_timing(
            traj,
            achievement_labels=achievement_labels,
            title=f"{title_prefix}: achievement unlock timing",
        )
        ach_path = out_dir / "final_achievements.png"
        fig.savefig(ach_path, dpi=150)
        plt.close(fig)
        paths["achievements"] = ach_path
    except Exception:  # noqa: BLE001
        logger.exception("Achievement timing plot failed.")

    return paths
