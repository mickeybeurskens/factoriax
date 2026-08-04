"""Collect rollout chunks from a training loop into a trajectory.

:class:`RolloutRecorder` takes the arrays that a collection function
already returns and turns them into a
:class:`~factoriax.analysis.trajectory.Trajectory`. A training loop
therefore needs one new call, and no change to ``collect_fn`` or to
any code under ``jit``.

Call :meth:`~RolloutRecorder.record` after each collection call, then
:meth:`~RolloutRecorder.finish` once at the end.

The sketch below names ``collect_fn`` and ``total_iters`` from the
training loop it plugs into, so it does not run on its own:

.. code-block:: python

    from factoriax.analysis.recorder import RolloutRecorder

    recorder = RolloutRecorder(max_episodes=32)
    for _ in range(total_iters):
        trajectories, env_states, obs, last_values, _ = collect_fn(...)
        recorder.record(trajectories, env_states)
    traj = recorder.finish()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .trajectory import _STATE_TO_TRAJ, Trajectory


@dataclass
class RolloutRecorder:
    """Accumulates rollout data across training iterations.

    The recorder reads the arrays that a collection function already
    returns. A training loop therefore needs one new call, and no
    change to ``collect_fn`` or to any code under ``jit``.

    Each call to :meth:`record` stores one ``(T, N)`` chunk.
    :meth:`finish` joins the chunks and cuts them into episodes at the
    ``done`` flags.

    Attributes
    ----------
    max_episodes :
        Stop storing chunks once this many episodes have finished, and
        keep at most this many episodes in the result. ``None`` means
        no limit, and :meth:`finish` then decides the count.
    record_states :
        True to also store the fields named in ``state_fields`` from
        the env states. This costs memory in proportion to the size of
        those fields, and it is what makes a state-over-time analysis
        possible.
    state_fields :
        Which ``EnvState`` fields to store when ``record_states`` is
        True.

    Notes
    -----
    The episode count in :attr:`is_full` counts ``done`` flags across
    every parallel environment, and not per environment.
    """

    max_episodes: int | None = None
    record_states: bool = False
    state_fields: list[str] = field(
        default_factory=lambda: [
            "player_positions",
            "player_inventory",
        ]
    )

    # Internal buffers, populated by record()
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
        """Store one chunk of rollout data.

        The call returns at once when the recorder is already full, so
        a training loop can keep calling it without a guard. The chunk
        is then dropped.

        Parameters
        ----------
        trajectories :
            The struct that ``collect_fn`` returns. It must carry
            ``action``, ``reward``, and ``done``. Actions have shape
            ``(T, N)`` or ``(T, N, P)``, and the other two ``(T, N)``,
            where ``T`` is the chunk length and ``N`` the number of
            parallel environments. JAX arrays and NumPy arrays both
            work.
        env_states :
            The env states from the same call, or ``None``. They are
            read only when ``record_states`` is True. A field named in
            ``state_fields`` but absent from the states is skipped in
            silence.
        """
        if self.is_full:
            return

        # Extract arrays. Handles both JAX and numpy.
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
        """Whether the episode count has reached ``max_episodes``.

        Returns
        -------
        bool
            Always False when ``max_episodes`` is ``None``.
        """
        if self.max_episodes is None:
            return False
        return self._num_complete_episodes >= self.max_episodes

    @property
    def num_recorded_steps(self) -> int:
        """Total timesteps stored so far, summed over the chunks.

        Returns
        -------
        int
            The length of the time axis, not multiplied by the number
            of parallel environments. 0 before the first
            :meth:`record`.
        """
        if not self._action_chunks:
            return 0
        return sum(chunk.shape[0] for chunk in self._action_chunks)

    def finish(self, pad_incomplete: bool = True) -> Trajectory:
        """Cut the recorded chunks into episodes and build a trajectory.

        The chunks are joined along the time axis and then split at the
        ``done`` flags, one environment at a time. State fields are cut
        at the same boundaries.

        Episodes differ in length, so every episode is padded with
        zeros to the length of the longest one.

        Parameters
        ----------
        pad_incomplete :
            True to keep the last episode of each environment even when
            no ``done`` flag ended it. False to keep finished episodes
            only.

        Returns
        -------
        :class:`~factoriax.analysis.trajectory.Trajectory`
            Actions of shape ``(B, T_max)`` or ``(B, T_max, P)``, plus
            rewards, plus any recorded state fields under their
            trajectory names. ``B`` is the episode count after the
            ``max_episodes`` limit.

        Raises
        ------
        ValueError
            When no chunk was recorded, or when the split found no
            episode at all. The second case happens with
            ``pad_incomplete=False`` and no ``done`` flag anywhere.

        Notes
        -----
        The padding is zeros, and action 0 is ``NOOP``. Padded steps
        therefore read as real no-op actions to every analysis
        function, and the trajectory carries no mask that marks them.
        A per-episode length is the only way to tell them apart, and
        this method does not return one.

        The method does not clear the buffers. A second call rebuilds
        the same trajectory. Call :meth:`reset` to start again.
        """
        if not self._action_chunks:
            raise ValueError("No data recorded. Call record() first.")

        # Concatenate chunks along time axis: (T_total, N, ...)
        all_actions = np.concatenate(self._action_chunks, axis=0)
        all_rewards = np.concatenate(self._reward_chunks, axis=0)
        all_dones = np.concatenate(self._done_chunks, axis=0)

        # Concatenate state chunks: {field_name: (T_total, N, ...)}
        all_states: dict[str, np.ndarray] = {}
        for fname, chunks in self._state_chunks.items():
            if chunks:
                all_states[fname] = np.concatenate(chunks, axis=0)

        T_total, N = all_dones.shape[:2]

        # Segment into episodes per environment
        episodes_actions: list[np.ndarray] = []
        episodes_rewards: list[np.ndarray] = []
        episodes_states: dict[str, list[np.ndarray]] = {
            fname: [] for fname in all_states
        }

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
                for fname, arr in all_states.items():
                    episodes_states[fname].append(arr[s:e, env_idx])

            # Handle trailing incomplete episode
            if pad_incomplete and (
                len(done_indices) == 0 or done_indices[-1] < T_total - 1
            ):
                last_start = 0 if len(done_indices) == 0 else done_indices[-1] + 1
                episodes_actions.append(env_actions[last_start:])
                episodes_rewards.append(env_rewards[last_start:])
                for fname, arr in all_states.items():
                    episodes_states[fname].append(arr[last_start:, env_idx])

        if not episodes_actions:
            raise ValueError(
                "No complete episodes found. Try recording more steps "
                "or set pad_incomplete=True."
            )

        # Respect max_episodes limit
        if self.max_episodes is not None:
            episodes_actions = episodes_actions[: self.max_episodes]
            episodes_rewards = episodes_rewards[: self.max_episodes]
            for fname in episodes_states:
                episodes_states[fname] = episodes_states[fname][: self.max_episodes]

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

        # Pad state fields to (B, T_max, ...)
        padded_states: dict[str, np.ndarray] = {}
        for fname, ep_list in episodes_states.items():
            if not ep_list:
                continue
            extra_shape = ep_list[0].shape[1:]
            padded = np.zeros((B, max_len, *extra_shape), dtype=ep_list[0].dtype)
            for i, arr in enumerate(ep_list):
                L = arr.shape[0]
                padded[i, :L] = arr
            # Map EnvState field names to Trajectory field names
            traj_name = _STATE_TO_TRAJ.get(fname, fname)
            padded_states[traj_name] = padded

        return Trajectory(
            actions=padded_actions,
            rewards=padded_rewards,
            **padded_states,  # type: ignore[arg-type]
        )

    def reset(self) -> None:
        """Drop every stored chunk and reset the episode count.

        The configuration fields keep their values, so the recorder is
        ready for another run right away.
        """
        self._action_chunks.clear()
        self._reward_chunks.clear()
        self._done_chunks.clear()
        self._state_chunks.clear()
        self._num_complete_episodes = 0
