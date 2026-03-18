"""Trajectory container for storing and validating episode rollout data.

A Trajectory holds batched sequences of actions and (optionally) state-derived
features from one or more episodes.  Every analysis function in this package
accepts a Trajectory as its primary input.

Typical usage
-------------
>>> import numpy as np
>>> from factoriax.analysis.trajectory import Trajectory
>>> actions = np.random.randint(0, 12, size=(16, 200, 2))  # (B, T, P)
>>> traj = Trajectory(actions=actions)

With full state information:
>>> traj = Trajectory(
...     actions=actions,
...     positions=positions,           # (B, T, P, 2)
...     inventory_items=inv_items,     # (B, T, P, num_slots)
...     inventory_counts=inv_counts,   # (B, T, P, num_slots)
...     achievements=achievements,     # (B, T, num_achievements)
...     rewards=rewards,               # (B, T) or (B, T, P)
...     timesteps=timesteps,           # (B, T)
... )
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    """Immutable container for batched episode data.

    All array fields follow the convention (B, T, ...) where B is the batch
    (episode) dimension and T is the time dimension.  For multi-player
    environments the player dimension P follows T.

    Parameters
    ----------
    actions : np.ndarray
        **Required.**  Integer action IDs.
        Shape ``(B, T, P)`` for multi-player or ``(B, T)`` for single-player.
        Values should be in ``[0, num_actions)``.
    positions : np.ndarray, optional
        Player (x, y) positions.  Shape ``(B, T, P, 2)``.
    inventory_items : np.ndarray, optional
        Item type IDs per inventory slot.  Shape ``(B, T, P, num_slots)``.
    inventory_counts : np.ndarray, optional
        Stack counts per inventory slot.  Shape ``(B, T, P, num_slots)``.
    machine_types : np.ndarray, optional
        Machine type grid snapshots.  Shape ``(B, T, H, W)``.
    achievements : np.ndarray, optional
        Boolean achievement flags.  Shape ``(B, T, num_achievements)``.
    rewards : np.ndarray, optional
        Per-step rewards.  Shape ``(B, T)`` or ``(B, T, P)``.
    timesteps : np.ndarray, optional
        Timestep indices.  Shape ``(B, T)``.  Inferred as ``arange(T)`` if
        not provided.
    metadata : dict, optional
        Arbitrary metadata (hyperparameters, run ID, etc.).
    """

    actions: np.ndarray

    # Optional state-derived fields
    positions: np.ndarray | None = None
    inventory_items: np.ndarray | None = None
    inventory_counts: np.ndarray | None = None
    machine_types: np.ndarray | None = None
    achievements: np.ndarray | None = None
    rewards: np.ndarray | None = None
    timesteps: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Coerce to numpy
        object.__setattr__(self, "actions", np.asarray(self.actions))

        actions = self.actions
        if actions.ndim == 1:
            # Single episode, single player -> (1, T)
            object.__setattr__(self, "actions", actions[np.newaxis, :])
        elif actions.ndim == 2:
            # Could be (B, T) single-player or (T, P) single-episode multi-player.
            # We assume (B, T) — users should add a batch dim if needed.
            pass
        elif actions.ndim == 3:
            pass  # (B, T, P)
        else:
            raise ValueError(
                f"actions must be 2D (B, T) or 3D (B, T, P), got shape {actions.shape}"
            )

    # ---- Shape properties ----

    @property
    def num_episodes(self) -> int:
        return self.actions.shape[0]

    @property
    def episode_length(self) -> int:
        return self.actions.shape[1]

    @property
    def num_players(self) -> int:
        if self.actions.ndim == 3:
            return self.actions.shape[2]
        return 1

    @property
    def is_multi_player(self) -> bool:
        return self.num_players > 1

    # ---- Slicing helpers ----

    def episode(self, idx: int) -> Trajectory:
        """Return a Trajectory containing a single episode."""
        if idx < 0:
            idx = self.num_episodes + idx
        return self.episodes(slice(idx, idx + 1))

    def episodes(self, s: slice | np.ndarray) -> Trajectory:
        """Slice along the batch dimension."""
        kwargs: dict = {"actions": self.actions[s]}
        for name in (
            "positions",
            "inventory_items",
            "inventory_counts",
            "machine_types",
            "achievements",
            "rewards",
            "timesteps",
        ):
            val = getattr(self, name)
            if val is not None:
                kwargs[name] = val[s]
        kwargs["metadata"] = self.metadata
        return Trajectory(**kwargs)

    def player(self, idx: int) -> Trajectory:
        """Return a single-player view (actions become (B, T))."""
        if not self.is_multi_player:
            return self
        kwargs: dict = {"actions": self.actions[:, :, idx]}
        if self.positions is not None:
            kwargs["positions"] = self.positions[:, :, idx, :]
        if self.inventory_items is not None:
            kwargs["inventory_items"] = self.inventory_items[:, :, idx, :]
        if self.inventory_counts is not None:
            kwargs["inventory_counts"] = self.inventory_counts[:, :, idx, :]
        # achievements, rewards, timesteps may or may not have player dim
        for name in ("achievements", "rewards", "timesteps"):
            val = getattr(self, name)
            if val is not None:
                kwargs[name] = val
        kwargs["metadata"] = self.metadata
        return Trajectory(**kwargs)

    def time_slice(self, start: int, end: int) -> Trajectory:
        """Slice along the time dimension."""
        kwargs: dict = {"actions": self.actions[:, start:end]}
        for name in (
            "positions",
            "inventory_items",
            "inventory_counts",
            "machine_types",
            "achievements",
            "rewards",
            "timesteps",
        ):
            val = getattr(self, name)
            if val is not None:
                kwargs[name] = val[:, start:end]
        kwargs["metadata"] = self.metadata
        return Trajectory(**kwargs)

    # ---- I/O helpers ----

    def save(self, path: str) -> None:
        """Save trajectory to a compressed .npz file."""
        arrays: dict = {"actions": self.actions}
        for name in (
            "positions",
            "inventory_items",
            "inventory_counts",
            "machine_types",
            "achievements",
            "rewards",
            "timesteps",
        ):
            val = getattr(self, name)
            if val is not None:
                arrays[name] = val
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str, **metadata) -> Trajectory:
        """Load trajectory from a .npz file."""
        data = np.load(path)
        return cls(
            actions=data["actions"],
            positions=data.get("positions"),
            inventory_items=data.get("inventory_items"),
            inventory_counts=data.get("inventory_counts"),
            machine_types=data.get("machine_types"),
            achievements=data.get("achievements"),
            rewards=data.get("rewards"),
            timesteps=data.get("timesteps"),
            metadata=metadata,
        )

    def __repr__(self) -> str:
        parts = [
            f"Trajectory(episodes={self.num_episodes}",
            f"steps={self.episode_length}",
            f"players={self.num_players}",
        ]
        extras = []
        for name in (
            "positions",
            "inventory_items",
            "inventory_counts",
            "machine_types",
            "achievements",
            "rewards",
        ):
            if getattr(self, name) is not None:
                extras.append(name)
        if extras:
            parts.append(f"fields=[{', '.join(extras)}]")
        return ", ".join(parts) + ")"
