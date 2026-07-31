"""Hold the recorded episodes that every analysis function reads.

:class:`Trajectory` is the one input format of this package. It holds
a batch of episodes, and each field is an array whose first two axes
are the episode and the timestep.

Axis order
----------
Every array starts with ``(B, T, ...)``, where ``B`` counts episodes
and ``T`` counts timesteps. A field that belongs to one player adds a
player axis after ``T``, giving ``(B, T, P, ...)``. A map field adds
the two map axes instead, giving ``(B, T, H, W, ...)``.

Only ``actions`` is required. Every other array field defaults to
``None``, which means "not recorded" and never "empty".

Scheme descriptors
------------------
Four dictionary fields record how a trajectory was produced: the
observation scheme, the reward scheme, the cost scheme, and the engine
parameters. :meth:`Trajectory.save` writes them to the archive as
JSON, under keys with a leading underscore, which keeps them apart
from the array fields.

Examples
--------
Build a trajectory from actions alone:

>>> import numpy as np
>>> from factoriax.analysis.trajectory import Trajectory
>>> actions = np.random.randint(0, 12, size=(16, 200, 2))
>>> traj = Trajectory(actions=actions)
>>> traj.num_episodes, traj.episode_length, traj.num_players
(16, 200, 2)

Record how it was produced:

>>> traj = Trajectory(
...     actions=actions,
...     observation_scheme={"type": "local", "radius": 7},
... )
>>> traj.observation_scheme["radius"]
7

Nothing runs these examples. ``pyproject.toml`` sets no
``--doctest-modules``, so they are text and can go stale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import orjson

if TYPE_CHECKING:
    from factoriax.engine.state import EnvParams, EnvState

#: Every optional array field, in declaration order. Save, load, the
#: slice methods, and ``__repr__`` all walk this tuple, so a new field
#: has to be added here as well as to the class.
_OPTIONAL_ARRAY_FIELDS: tuple[str, ...] = (
    # Player fields, shape (B, T, P, ...).
    "positions",
    "player_directions",
    "player_inventory",
    # Map fields, shape (B, T, H, W, ...).
    "block_map",
    "block_resources",
    "machine_types",
    "tile_entity",
    # Entity fields, shape (B, T, MAX_M, ...).
    "ent_y",
    "ent_x",
    "ent_type",
    "ent_direction",
    "ent_power",
    "ent_buf_type",
    "ent_buf_count",
    "ent_asm_in_type",
    "ent_asm_in_count",
    "ent_asm_out_type",
    "ent_asm_out_count",
    "ent_health",
    # Global fields, shape (B, T, ...).
    "selected_player",
    "achievements",
    "achievements_unlocked",
    "items_mined",
    "science_consumed_step",
    "rewards",
    "timesteps",
)

#: Fields that carry a player axis after ``(B, T)``.
#: :meth:`Trajectory.player` reads this set to decide which fields to
#: slice on that axis. Every name here must also appear in
#: :data:`_OPTIONAL_ARRAY_FIELDS`, or it is never reached.
_PLAYER_FIELDS: frozenset[str] = frozenset(
    {
        "positions",
        "player_directions",
        "player_inventory",
    }
)


@dataclass(frozen=True)
class Trajectory:
    """Immutable container for batched episode data.

    Every array field starts with ``(B, T, ...)``. ``B`` counts
    episodes and ``T`` counts timesteps. A player field adds a player
    axis after ``T``. A map field adds the map axes instead.

    The class is frozen, so a slice method returns a new trajectory
    and never changes the one it was called on. The arrays inside are
    not copied, so two trajectories can share memory and a caller that
    writes into one array affects both.

    Attributes
    ----------
    actions :
        The action taken at each step. Shape ``(B, T)`` for one
        player, or ``(B, T, P)`` for several. A 1-D array is accepted
        and gets a batch axis of 1.
    positions :
        Player positions as ``(B, T, P, 2)``, ordered ``(x, y)`` in
        tiles. Note the order: x is first, unlike the map fields,
        which are indexed ``[y, x]`` in row-major order.
    player_directions :
        The facing of each player as ``(B, T, P)``.
    player_inventory :
        Item counts as ``(B, T, P, NUM_ITEM_TYPES)``, indexed by
        ``ItemType`` value. This is a count for each item, not a list
        of slots.
    block_map :
        The tile type of each map cell as ``(B, T, H, W)``.
    block_resources :
        The resource left in each map cell as ``(B, T, H, W)``.
    machine_types :
        The machine on each map cell as ``(B, T, H, W)``.
    tile_entity :
        The entity index on each map cell as ``(B, T, H, W)``, or -1
        for an empty cell.
    ent_y, ent_x :
        Entity positions as ``(B, T, MAX_M)``. A free slot holds a
        negative y.
    ent_type, ent_direction, ent_power, ent_health :
        Per-entity fields as ``(B, T, MAX_M)``.
    ent_buf_type, ent_buf_count :
        The buffer of each entity as ``(B, T, MAX_M)``.
    ent_asm_in_type, ent_asm_in_count, ent_asm_out_type,
    ent_asm_out_count :
        Assembler input and output slots as ``(B, T, MAX_M)``.
    selected_player :
        The index of the acting player as ``(B, T)``.
    achievements :
        The achievement mask at each step as ``(B, T, A)``, where
        ``A`` is the achievement count of the scenario.
    achievements_unlocked :
        The latched mask carried by the engine state, padded to
        ``MAX_ACHIEVEMENTS``. This is wider than ``achievements`` and
        is not the same field.
    items_mined :
        Items mined so far as ``(B, T, ...)``.
    science_consumed_step :
        Science consumed at each step as ``(B, T, ...)``.
    rewards :
        The reward at each step as ``(B, T)``.
    timesteps :
        The engine timestep of each step as ``(B, T)``. This can
        differ from the position along the time axis when a recording
        skips steps.
    observation_scheme, reward_scheme, cost_scheme, env_params_scheme :
        Plain dictionaries that record how the trajectory was
        produced. ``None`` means "not recorded".

    Raises
    ------
    ValueError
        When ``actions`` has more than three axes, or fewer than one.

    Notes
    -----
    Nothing checks that the fields agree with each other. A trajectory
    whose ``rewards`` are shorter than its ``actions`` is built without
    complaint and fails later, inside whichever function reads both.
    """

    actions: np.ndarray

    # Player fields, shape (B, T, P, ...).
    positions: np.ndarray | None = None
    player_directions: np.ndarray | None = None
    player_inventory: np.ndarray | None = None

    # Map fields, shape (B, T, H, W, ...).
    block_map: np.ndarray | None = None
    block_resources: np.ndarray | None = None
    machine_types: np.ndarray | None = None
    tile_entity: np.ndarray | None = None

    # Entity fields, shape (B, T, MAX_M, ...).
    ent_y: np.ndarray | None = None
    ent_x: np.ndarray | None = None
    ent_type: np.ndarray | None = None
    ent_direction: np.ndarray | None = None
    ent_power: np.ndarray | None = None
    ent_buf_type: np.ndarray | None = None
    ent_buf_count: np.ndarray | None = None
    ent_asm_in_type: np.ndarray | None = None
    ent_asm_in_count: np.ndarray | None = None
    ent_asm_out_type: np.ndarray | None = None
    ent_asm_out_count: np.ndarray | None = None
    ent_health: np.ndarray | None = None

    # Global fields, shape (B, T, ...).
    selected_player: np.ndarray | None = None
    achievements: np.ndarray | None = None
    achievements_unlocked: np.ndarray | None = None
    items_mined: np.ndarray | None = None
    science_consumed_step: np.ndarray | None = None
    rewards: np.ndarray | None = None
    timesteps: np.ndarray | None = None

    # Scheme descriptors. They capture how the trajectory was produced.
    observation_scheme: dict[str, object] | None = None
    reward_scheme: dict[str, object] | None = None
    cost_scheme: dict[str, object] | None = None
    # Snapshot of ``env_params_to_dict(params)`` at recording time so
    # replay tooling can rebuild :class:`EnvParams` with the live
    # ``player_mining_yield``, ``miner_mining_rate``, and the others.
    env_params_scheme: dict[str, object] | None = None

    def __post_init__(self) -> None:
        """Convert ``actions`` to a NumPy array and add a batch axis.

        A 1-D array describes one episode and gets a batch axis of 1. A
        2-D or 3-D array is left as it is.

        Raises
        ------
        ValueError
            When ``actions`` has four or more axes, or zero.
        """
        object.__setattr__(self, "actions", np.asarray(self.actions))

        actions = self.actions
        if actions.ndim == 1:
            object.__setattr__(self, "actions", actions[np.newaxis, :])
        elif actions.ndim == 2:
            pass  # (B, T)
        elif actions.ndim == 3:
            pass  # (B, T, P)
        else:
            raise ValueError(
                f"actions must be 2D (B, T) or 3D (B, T, P), got shape {actions.shape}"
            )

    # ---- Shape properties ----

    @property
    def num_episodes(self) -> int:
        """Number of episodes in the batch."""
        return int(self.actions.shape[0])

    @property
    def episode_length(self) -> int:
        """Number of timesteps per episode."""
        return int(self.actions.shape[1])

    @property
    def num_players(self) -> int:
        """Number of players (1 if single-player)."""
        if self.actions.ndim == 3:
            return int(self.actions.shape[2])
        return 1

    @property
    def is_multi_player(self) -> bool:
        """Whether the trajectory has more than one player."""
        return self.num_players > 1

    # ---- Slicing helpers ----

    def episode(self, idx: int) -> Trajectory:
        """Return one episode as a trajectory of batch size 1.

        Parameters
        ----------
        idx :
            The episode to keep. A negative index counts from the end,
            as in a Python list.

        Returns
        -------
        Trajectory
            A new trajectory with ``B`` of 1. The time axis is kept, so
            the result is not squeezed to a single episode shape.
        """
        if idx < 0:
            idx = self.num_episodes + idx
        return self.episodes(slice(idx, idx + 1))

    def episodes(self, s: slice | np.ndarray) -> Trajectory:
        """Return the selected episodes as a new trajectory.

        Parameters
        ----------
        s :
            Anything NumPy accepts on the first axis: a slice, an
            array of indexes, or a boolean mask.

        Returns
        -------
        Trajectory
            A new trajectory holding those episodes. Every recorded
            field is sliced the same way, and the scheme dictionaries
            are carried over unchanged.
        """
        kwargs: dict[str, Any] = {"actions": self.actions[s]}
        for name in _OPTIONAL_ARRAY_FIELDS:
            val = getattr(self, name)
            if val is not None:
                kwargs[name] = val[s]
        kwargs.update(self._scheme_kwargs())
        return Trajectory(**kwargs)

    def player(self, idx: int) -> Trajectory:
        """Return the view of one player as a new trajectory.

        Parameters
        ----------
        idx :
            The player to keep.

        Returns
        -------
        Trajectory
            A new trajectory with the player axis dropped from the
            player fields. A trajectory that is not multi-player is
            returned unchanged, including one of shape ``(B, T, 1)``,
            which therefore keeps its player axis. Use
            :func:`factoriax.analysis.utils.resolve_player_actions`
            when the result must always have two axes.
        """
        if not self.is_multi_player:
            return self
        kwargs: dict[str, Any] = {"actions": self.actions[:, :, idx]}
        for name in _OPTIONAL_ARRAY_FIELDS:
            val = getattr(self, name)
            if val is None:
                continue
            if name in _PLAYER_FIELDS:
                # Slice the player dimension (axis 2).
                if name == "positions":
                    kwargs[name] = val[:, :, idx, :]
                elif val.ndim > 2 and val.shape[2] > 1:
                    kwargs[name] = val[:, :, idx]
                else:
                    kwargs[name] = val
            else:
                kwargs[name] = val
        kwargs.update(self._scheme_kwargs())
        return Trajectory(**kwargs)

    def time_slice(self, start: int, end: int) -> Trajectory:
        """Return the timesteps from ``start`` to ``end`` as a trajectory.

        Parameters
        ----------
        start :
            The first timestep to keep.
        end :
            The timestep after the last one to keep, as in a Python
            slice.

        Returns
        -------
        Trajectory
            A new trajectory holding that window of every recorded
            field. The ``timesteps`` field is sliced too, so it keeps
            the original engine timesteps and does not restart at 0.
        """
        kwargs: dict[str, Any] = {"actions": self.actions[:, start:end]}
        for name in _OPTIONAL_ARRAY_FIELDS:
            val = getattr(self, name)
            if val is not None:
                kwargs[name] = val[:, start:end]
        kwargs.update(self._scheme_kwargs())
        return Trajectory(**kwargs)

    # ---- Internal helpers ----

    # Scheme field names, used by save/load/slice to avoid repetition.
    _SCHEME_FIELDS: tuple[str, ...] = (
        "observation_scheme",
        "reward_scheme",
        "cost_scheme",
        "env_params_scheme",
    )

    def _scheme_kwargs(self) -> dict[str, Any]:
        """Return the scheme fields that are set, as constructor keywords.

        Returns
        -------
        dict
            One entry for each scheme field that is not ``None``. The
            slice methods pass this on, so a slice keeps the record of
            how the trajectory was produced.
        """
        return {
            name: getattr(self, name)
            for name in self._SCHEME_FIELDS
            if getattr(self, name) is not None
        }

    # ---- I/O helpers ----

    def save(self, path: str) -> None:
        """Write the trajectory to a compressed ``.npz`` archive.

        Each array field is stored under its own name. Each scheme
        dictionary is stored as JSON bytes under its name with a
        leading underscore, such as ``_observation_scheme``.

        A field that is ``None`` is left out of the archive. An
        unrecorded field is therefore absent, and not present as a
        null, so :meth:`load` can tell the two apart.

        Parameters
        ----------
        path :
            Where to write the archive. NumPy appends ``.npz`` when
            the name does not end with it. Parent directories are not
            created.
        """
        arrays: dict[str, Any] = {"actions": self.actions}
        for name in _OPTIONAL_ARRAY_FIELDS:
            val = getattr(self, name)
            if val is not None:
                arrays[name] = val
        for name in self._SCHEME_FIELDS:
            val = getattr(self, name)
            if val is not None:
                arrays[f"_{name}"] = np.void(orjson.dumps(val))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str) -> Trajectory:
        """Read a trajectory back from a ``.npz`` archive.

        A field absent from the archive stays ``None``. Each JSON entry
        with a leading underscore is read back into its scheme
        dictionary, so the values return as plain JSON types.

        An old archive that stored ``_obs_type`` and ``_obs_radius`` as
        scalars is folded into ``observation_scheme``. That fold runs
        only when the archive holds no ``observation_scheme`` of its
        own.

        Parameters
        ----------
        path :
            The archive to read.

        Returns
        -------
        Trajectory
            The trajectory the archive holds.

        Notes
        -----
        The load runs with ``allow_pickle=True``, which is what lets
        the JSON entries come back. Read only archives you trust, and
        treat one from outside the project as untrusted input.
        """
        data = np.load(path, allow_pickle=True)
        kwargs: dict[str, Any] = {"actions": data["actions"]}
        for name in _OPTIONAL_ARRAY_FIELDS:
            if name in data:
                kwargs[name] = data[name]
        # Restore scheme dicts from JSON byte strings.
        for name in cls._SCHEME_FIELDS:
            npz_key = f"_{name}"
            if npz_key in data:
                kwargs[name] = orjson.loads(bytes(data[npz_key]))
        # Legacy migration: old files stored _obs_type / _obs_radius
        # as scalar int32 arrays.  Fold them into observation_scheme.
        if "_obs_type" in data and "observation_scheme" not in kwargs:
            legacy: dict[str, int] = {"type": int(data["_obs_type"])}
            if "_obs_radius" in data:
                legacy["radius"] = int(data["_obs_radius"])
            kwargs["observation_scheme"] = legacy
        return cls(**kwargs)

    def __repr__(self) -> str:
        """Return a one-line summary for the console.

        Returns
        -------
        str
            The episode, step, and player counts, then the names of
            the recorded array fields and scheme fields. The arrays
            themselves are not printed.
        """
        parts = [
            f"Trajectory(episodes={self.num_episodes}",
            f"steps={self.episode_length}",
            f"players={self.num_players}",
        ]
        extras = [
            name for name in _OPTIONAL_ARRAY_FIELDS if getattr(self, name) is not None
        ]
        if extras:
            parts.append(f"fields=[{', '.join(extras)}]")
        schemes = [
            name for name in self._SCHEME_FIELDS if getattr(self, name) is not None
        ]
        if schemes:
            parts.append(f"schemes=[{', '.join(schemes)}]")
        return ", ".join(parts) + ")"


# ---------------------------------------------------------------------------
# EnvState conversion (Trajectory knows about EnvState, not vice versa)
# ---------------------------------------------------------------------------

# Mapping from Trajectory field name -> EnvState attribute name.
# Only fields where the names differ need entries.
_TRAJ_TO_STATE: dict[str, str] = {
    "positions": "player_positions",
    "block_map": "map",
}

# Trajectory fields with no EnvState counterpart, skipped during
# reconstruction because EnvState rejects them as unknown keywords.
# ``achievements`` is per-step and as wide as the scenario's set, where
# EnvState carries the latched ``achievements_unlocked`` at MAX_ACHIEVEMENTS.
_TRAJ_ONLY_FIELDS: frozenset[str] = frozenset({"achievements"})

# Inverse mapping.
_STATE_TO_TRAJ: dict[str, str] = {v: k for k, v in _TRAJ_TO_STATE.items()}


def states_to_trajectory(
    states: list[EnvState],
    actions: np.ndarray | None = None,
    rewards: np.ndarray | None = None,
    params: EnvParams | None = None,
) -> Trajectory:
    """Stack a list of engine states into a one-episode trajectory.

    Every array field of the states is stacked along a new time axis,
    and a batch axis of 1 is added in front. The result therefore has
    ``B`` of 1 and ``T`` equal to the number of states.

    Parameters
    ----------
    states :
        The states, one for each timestep, in order.
    actions :
        The action at each step, of shape ``(T,)`` or ``(T, P)``, or
        ``None`` for all zeros. Nothing compares its length against
        the state count, so a mismatch gives a trajectory whose action
        axis and time axis disagree.
    rewards :
        The reward at each step, of shape ``(T,)``, or ``None`` to
        record none.
    params :
        The engine parameters that produced the states, or ``None``.
        When given, their dictionary form is stored in
        ``env_params_scheme`` so replay code can rebuild them.

    Returns
    -------
    Trajectory
        A trajectory with ``B`` of 1. Its ``timesteps`` field counts
        from 0 to ``T - 1``, whatever the states themselves report.

    Raises
    ------
    ValueError
        When ``states`` is empty.
    """
    if not states:
        raise ValueError("states list is empty.")

    T = len(states)

    # Build actions if not provided.
    if actions is not None:
        act = np.asarray(actions)
        if act.ndim == 1:
            act = act[np.newaxis, :]  # (1, T)
        elif act.ndim == 2:
            act = act[np.newaxis, :, :]  # (1, T, P)
    else:
        act = np.zeros((1, T), dtype=np.int32)

    kwargs: dict[str, Any] = {"actions": act}

    if rewards is not None:
        r = np.asarray(rewards, dtype=np.float32)
        kwargs["rewards"] = r[np.newaxis, :]  # (1, T)

    # Stack each state field along time, then add batch dim.
    for traj_name in _OPTIONAL_ARRAY_FIELDS:
        if traj_name in ("rewards", "timesteps"):
            continue  # Handled separately.
        state_name = _TRAJ_TO_STATE.get(traj_name, traj_name)
        first_val = getattr(states[0], state_name, None)
        if first_val is None:
            continue
        arr = np.stack(
            [np.asarray(getattr(s, state_name)) for s in states],
            axis=0,
        )  # (T, ...)
        kwargs[traj_name] = arr[np.newaxis, :]  # (1, T, ...)

    kwargs["timesteps"] = np.arange(T, dtype=np.int32)[np.newaxis, :]

    if params is not None:
        from factoriax.playground.config import env_params_to_dict

        kwargs["env_params_scheme"] = env_params_to_dict(params)

    return Trajectory(**kwargs)


def trajectory_to_states(
    traj: Trajectory,
    episode: int = 0,
) -> list[EnvState]:
    """Rebuild the engine states of one episode from a trajectory.

    This is the inverse of :func:`states_to_trajectory`. Only the
    fields the trajectory recorded are set. A field that ``EnvState``
    requires but the trajectory lacks raises inside ``EnvState``, and
    not here.

    The ``achievements`` field is skipped. It is per-step and only as
    wide as the achievement count of the scenario, while ``EnvState``
    carries the latched ``achievements_unlocked`` at
    ``MAX_ACHIEVEMENTS``.

    Parameters
    ----------
    traj :
        The recording to read.
    episode :
        Which episode to rebuild.

    Returns
    -------
    list
        One ``EnvState`` for each timestep. The ``timestep`` of each
        state comes from the ``timesteps`` field, or from the position
        along the time axis when that field was not recorded.

    Notes
    -----
    A trajectory from :meth:`RolloutRecorder.finish` is padded with
    zeros to a common length. The states rebuilt from those pad steps
    look like real states at the origin with empty inventories.
    """
    from factoriax.engine.state import EnvState

    T = traj.episode_length
    states = []

    for t in range(T):
        state_kwargs: dict[str, Any] = {}
        for traj_name in _OPTIONAL_ARRAY_FIELDS:
            if traj_name in ("rewards", "timesteps"):
                continue
            if traj_name in _TRAJ_ONLY_FIELDS:
                continue
            val = getattr(traj, traj_name)
            if val is None:
                continue
            state_name = _TRAJ_TO_STATE.get(traj_name, traj_name)
            step_val = val[episode, t]
            # Scalars (selected_player, timestep) need unwrapping.
            if step_val.ndim == 0:
                state_kwargs[state_name] = int(step_val)
            else:
                import jax.numpy as jnp

                state_kwargs[state_name] = jnp.array(step_val)

        # Set timestep from the trajectory or from index.
        if traj.timesteps is not None:
            state_kwargs["timestep"] = int(traj.timesteps[episode, t])
        else:
            state_kwargs["timestep"] = t

        states.append(EnvState(**state_kwargs))

    return states
