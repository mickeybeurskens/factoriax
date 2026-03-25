"""Rollout recording utilities for building Trajectory objects.

:class:`RolloutRecorder` — **zero-change integration.**  Call
:meth:`record` after each ``collect_fn`` call, passing the trajectory
struct and env states that are *already returned* by your collection
function.  Accumulates across iterations and produces a
:class:`~factoriax.analysis.trajectory.Trajectory` via :meth:`finish`.


Typical usage
-------------------------------------------------------
>>> from factoriax.analysis.recorder import RolloutRecorder
>>>
>>> recorder = RolloutRecorder(max_episodes=32)
>>>
>>> for it in range(total_iters):
...     trajectories, env_states, obs, last_values, _ = collect_fn(...)
...     # One line added to your training loop:
...     recorder.record(trajectories, env_states)
...     # ... rest of your PPO update ...
>>>
>>> traj = recorder.finish()
>>> # traj is now a Trajectory ready for analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .trajectory import Trajectory


@dataclass
class RolloutRecorder:
    """Accumulates rollout data across training iterations.

    This recorder is designed to slot into an existing training loop
    with **zero changes** to your ``collect_fn`` or JIT'd code.  It
    works by extracting numpy arrays from the trajectory struct and
    env states that your collection function already returns.

    The recorder segments the continuous stream of ``(T, N)`` rollout
    chunks into complete episodes using the ``done`` flags.

    Parameters
    ----------
    max_episodes : int, optional
        Stop recording after this many complete episodes.  If *None*,
        record indefinitely until :meth:`finish` is called.
    record_states : bool
        If *True*, also extract ``player_positions``, ``inventory_items``,
        and ``inventory_counts`` from ``env_states`` at each step.
        This increases memory usage but enables state-evolution analyses.
    state_fields : list[str], optional
        Which ``EnvState`` fields to record when ``record_states=True``.
        Defaults to ``["player_positions", "inventory_items",
        "inventory_counts"]``.

    Examples
    --------
    Minimal integration (actions + rewards only):

    >>> recorder = RolloutRecorder(max_episodes=64)
    >>> for it in range(total_iters):
    ...     trajectories, env_states, obs, last_values, _ = collect_fn(...)
    ...     recorder.record(trajectories)
    ...     if recorder.is_full:
    ...         break
    >>> traj = recorder.finish()

    With state recording:

    >>> recorder = RolloutRecorder(max_episodes=32, record_states=True)
    >>> for it in range(total_iters):
    ...     trajectories, env_states, obs, last_values, _ = collect_fn(...)
    ...     recorder.record(trajectories, env_states)
    >>> traj = recorder.finish()
    """

    max_episodes: int | None = None
    record_states: bool = False
    state_fields: list[str] = field(
        default_factory=lambda: [
            "player_positions",
            "inventory_items",
            "inventory_counts",
        ]
    )

    # Internal buffers — populated by record()
    _action_chunks: list[np.ndarray] = field(default_factory=list, repr=False)
    _reward_chunks: list[np.ndarray] = field(default_factory=list, repr=False)
    _done_chunks: list[np.ndarray] = field(default_factory=list, repr=False)
    _state_chunks: dict[str, list[np.ndarray]] = field(default_factory=dict, repr=False)
    _num_complete_episodes: int = field(default=0, repr=False)

    def record(
        self,
        trajectories: Any,
        env_states: Any = None,
    ) -> None:
        """Record one chunk of rollout data.

        Parameters
        ----------
        trajectories
            The trajectory struct returned by ``collect_fn``.  Must have
            ``.action``, ``.reward``, and ``.done`` attributes, each
            shaped ``(T, N)`` or ``(T, N, ...)``.
        env_states : optional
            The ``EnvState`` pytree returned by ``collect_fn``.  Only
            needed if ``record_states=True``.  The recorder extracts
            fields listed in ``self.state_fields``.

            **Important:** ``env_states`` as returned by ``collect_fn``
            is typically the *final* state after the rollout, not the
            per-step states.  If your ``collect_fn`` returns per-step
            states inside the trajectory struct, pass those instead.
            See the note on ``record_states`` below.
        """
        if self.is_full:
            return

        # Extract arrays — handles both JAX and numpy transparently
        actions = np.asarray(trajectories.action)  # (T, N) or (T, N, P)
        rewards = np.asarray(trajectories.reward)  # (T, N)
        dones = np.asarray(trajectories.done)  # (T, N)

        self._action_chunks.append(actions)
        self._reward_chunks.append(rewards)
        self._done_chunks.append(dones)

        # Count newly completed episodes
        new_dones = int(dones.sum())
        self._num_complete_episodes += new_dones

        # Optional state recording
        if self.record_states and env_states is not None:
            for fname in self.state_fields:
                val = getattr(env_states, fname, None)
                if val is None:
                    continue
                arr = np.asarray(val)
                if fname not in self._state_chunks:
                    self._state_chunks[fname] = []
                self._state_chunks[fname].append(arr)

    @property
    def is_full(self) -> bool:
        """Whether we've recorded enough complete episodes."""
        if self.max_episodes is None:
            return False
        return self._num_complete_episodes >= self.max_episodes

    @property
    def num_recorded_steps(self) -> int:
        """Total timesteps recorded so far."""
        if not self._action_chunks:
            return 0
        return sum(chunk.shape[0] for chunk in self._action_chunks)

    def finish(self, pad_incomplete: bool = True) -> Trajectory:
        """Segment recorded chunks into complete episodes and build a Trajectory.

        This method concatenates all recorded chunks along the time axis,
        then splits them into individual episodes using the ``done`` flags.

        Parameters
        ----------
        pad_incomplete : bool
            If *True*, include the last (possibly incomplete) episode in
            each environment, zero-padded to match the longest episode.
            If *False*, only include fully completed episodes.

        Returns
        -------
        Trajectory
            With actions shaped ``(B, T_max)`` or ``(B, T_max, P)`` for
            multi-player, where B is the number of episodes and T_max
            is the length of the longest episode.
        """
        if not self._action_chunks:
            raise ValueError("No data recorded. Call record() first.")

        # Concatenate chunks along time axis: (T_total, N, ...)
        all_actions = np.concatenate(self._action_chunks, axis=0)
        all_rewards = np.concatenate(self._reward_chunks, axis=0)
        all_dones = np.concatenate(self._done_chunks, axis=0)

        T_total, N = all_dones.shape[:2]

        # Segment into episodes per environment
        episodes_actions: list[np.ndarray] = []
        episodes_rewards: list[np.ndarray] = []

        for env_idx in range(N):
            env_actions = all_actions[:, env_idx]  # (T_total, ...)
            env_rewards = all_rewards[:, env_idx]  # (T_total,)
            env_dones = all_dones[:, env_idx]  # (T_total,)

            # Find episode boundaries
            done_indices = np.where(env_dones)[0]
            starts = np.concatenate([[0], done_indices[:-1] + 1])
            ends = done_indices + 1

            for s, e in zip(starts, ends):
                episodes_actions.append(env_actions[s:e])
                episodes_rewards.append(env_rewards[s:e])

            # Handle trailing incomplete episode
            if pad_incomplete and (
                len(done_indices) == 0 or done_indices[-1] < T_total - 1
            ):
                last_start = 0 if len(done_indices) == 0 else done_indices[-1] + 1
                episodes_actions.append(env_actions[last_start:])
                episodes_rewards.append(env_rewards[last_start:])

        if not episodes_actions:
            raise ValueError(
                "No complete episodes found. Try recording more steps "
                "or set pad_incomplete=True."
            )

        # Respect max_episodes limit
        if self.max_episodes is not None:
            episodes_actions = episodes_actions[: self.max_episodes]
            episodes_rewards = episodes_rewards[: self.max_episodes]

        # Pad to uniform length
        max_len = max(ep.shape[0] for ep in episodes_actions)
        B = len(episodes_actions)

        action_shape = episodes_actions[0].shape[1:]  # () or (P,)
        padded_actions = np.zeros(
            (B, max_len, *action_shape), dtype=episodes_actions[0].dtype
        )
        padded_rewards = np.zeros((B, max_len), dtype=np.float32)

        for i, (a, r) in enumerate(zip(episodes_actions, episodes_rewards)):
            L = a.shape[0]
            padded_actions[i, :L] = a
            padded_rewards[i, :L] = r

        return Trajectory(
            actions=padded_actions,
            rewards=padded_rewards,
        )

    def reset(self) -> None:
        """Clear all recorded data."""
        self._action_chunks.clear()
        self._reward_chunks.clear()
        self._done_chunks.clear()
        self._state_chunks.clear()
        self._num_complete_episodes = 0
