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

With scheme descriptors that capture how the trajectory was produced:
>>> traj = Trajectory(
...     actions=actions,
...     observation_scheme={"type": "local", "radius": 7},
...     reward_scheme={"type": "shaped", "weights": {"mine": 1.0}},
...     cost_scheme={"type": "action_penalty", "scale": 0.01},
... )

State round-trip (requires factoriax.state):
>>> from factoriax.analysis.trajectory import states_to_trajectory, trajectory_to_states
>>> traj = states_to_trajectory(states, actions)
>>> reconstructed = trajectory_to_states(traj, episode=0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import orjson

if TYPE_CHECKING:
    from factoriax.state import EnvState

# All optional array fields in the order they are declared.
# Used by save/load/slice/repr to avoid hardcoding the list in 6 places.
_OPTIONAL_ARRAY_FIELDS: tuple[str, ...] = (
    # Player fields — shape (B, T, P, ...).
    "positions",
    "player_directions",
    "inventory_items",
    "inventory_counts",
    "selected_slots",
    "crafting_recipe",
    "craft_progress",
    # Map fields — shape (B, T, H, W, ...).
    "block_map",
    "block_resources",
    "machine_types",
    "machine_power",
    "machine_inventory_items",
    "machine_inventory_counts",
    "machine_selected_recipe",
    "machine_selected_slot",
    "machine_direction",
    # Global fields — shape (B, T, ...).
    "selected_player",
    "achievements",
    "items_mined",
    "research_progress",
    "research_unlocked",
    "machine_health",
    "biter_positions",
    "biter_health",
    "scent_field",
    "rewards",
    "timesteps",
)

# Fields that have a per-player dimension after (B, T).
_PLAYER_FIELDS: frozenset[str] = frozenset(
    {
        "positions",
        "player_directions",
        "inventory_items",
        "inventory_counts",
        "selected_slots",
        "crafting_recipe",
        "craft_progress",
    }
)


@dataclass(frozen=True)
class Trajectory:
    """Immutable container for batched episode data.

    All array fields follow the convention ``(B, T, ...)`` where *B* is the
    batch (episode) dimension and *T* is the time dimension.  For multi-player
    environments the player dimension *P* follows *T* in player-specific
    fields.

    Parameters
    ----------
    actions : np.ndarray
        **Required.**  Integer action IDs.
        Shape ``(B, T, P)`` for multi-player or ``(B, T)`` for single-player.
    positions : np.ndarray, optional
        Player ``(x, y)`` positions.  Shape ``(B, T, P, 2)``.
    player_directions : np.ndarray, optional
        Player facing directions.  Shape ``(B, T, P)``.
    inventory_items : np.ndarray, optional
        Item type IDs per inventory slot.  Shape ``(B, T, P, num_slots)``.
    inventory_counts : np.ndarray, optional
        Stack counts per inventory slot.  Shape ``(B, T, P, num_slots)``.
    selected_slots : np.ndarray, optional
        Selected inventory slot per player.  Shape ``(B, T, P)``.
    crafting_recipe : np.ndarray, optional
        Recipe in progress per player.  Shape ``(B, T, P)``.
    craft_progress : np.ndarray, optional
        Crafting ticks remaining per player.  Shape ``(B, T, P)``.
    block_map : np.ndarray, optional
        Block type grid snapshots.  Shape ``(B, T, H, W)``.
    block_resources : np.ndarray, optional
        Per-tile resource counts.  Shape ``(B, T, H, W)``.
    machine_types : np.ndarray, optional
        Machine type grid snapshots.  Shape ``(B, T, H, W)``.
    machine_power : np.ndarray, optional
        Machine power levels.  Shape ``(B, T, H, W)``.
    machine_inventory_items : np.ndarray, optional
        Machine slot items.  Shape ``(B, T, H, W, S)``.
    machine_inventory_counts : np.ndarray, optional
        Machine slot counts.  Shape ``(B, T, H, W, S)``.
    machine_selected_recipe : np.ndarray, optional
        Assembler recipe per tile.  Shape ``(B, T, H, W)``.
    machine_selected_slot : np.ndarray, optional
        UI-focused slot per tile.  Shape ``(B, T, H, W)``.
    machine_direction : np.ndarray, optional
        Machine facing direction per tile.  Shape ``(B, T, H, W)``.
    selected_player : np.ndarray, optional
        Active player index per step.  Shape ``(B, T)``.
    achievements : np.ndarray, optional
        Boolean achievement flags.  Shape ``(B, T, num_achievements)``.
    items_mined : np.ndarray, optional
        Lifetime mined count per item type.  Shape ``(B, T, num_item_types)``.
    rewards : np.ndarray, optional
        Per-step rewards.  Shape ``(B, T)`` or ``(B, T, P)``.
    timesteps : np.ndarray, optional
        Timestep indices.  Shape ``(B, T)``.
    observation_scheme : dict, optional
        Describes how observations were produced (e.g. type, radius,
        channels).  Persisted through save/load as JSON.
    reward_scheme : dict, optional
        Describes the reward function used to generate the ``rewards``
        array (e.g. type, shaping weights, sparse vs dense).
    cost_scheme : dict, optional
        Describes any cost or penalty function applied during training
        (e.g. action penalty scale, entropy bonus coefficient).
    """

    actions: np.ndarray

    # Player fields — (B, T, P, ...).
    positions: np.ndarray | None = None
    player_directions: np.ndarray | None = None
    inventory_items: np.ndarray | None = None
    inventory_counts: np.ndarray | None = None
    selected_slots: np.ndarray | None = None
    crafting_recipe: np.ndarray | None = None
    craft_progress: np.ndarray | None = None

    # Map fields — (B, T, H, W, ...).
    block_map: np.ndarray | None = None
    block_resources: np.ndarray | None = None
    machine_types: np.ndarray | None = None
    machine_power: np.ndarray | None = None
    machine_inventory_items: np.ndarray | None = None
    machine_inventory_counts: np.ndarray | None = None
    machine_selected_recipe: np.ndarray | None = None
    machine_selected_slot: np.ndarray | None = None
    machine_direction: np.ndarray | None = None

    # Global fields — (B, T, ...).
    selected_player: np.ndarray | None = None
    achievements: np.ndarray | None = None
    items_mined: np.ndarray | None = None
    research_progress: np.ndarray | None = None
    research_unlocked: np.ndarray | None = None
    machine_health: np.ndarray | None = None
    biter_positions: np.ndarray | None = None
    biter_health: np.ndarray | None = None
    scent_field: np.ndarray | None = None
    rewards: np.ndarray | None = None
    timesteps: np.ndarray | None = None

    # Scheme descriptors — capture how the trajectory was produced.
    observation_scheme: dict[str, object] | None = None
    reward_scheme: dict[str, object] | None = None
    cost_scheme: dict[str, object] | None = None

    def __post_init__(self) -> None:
        """Coerce actions to numpy and ensure batch dimension."""
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
        """Return a Trajectory containing a single episode."""
        if idx < 0:
            idx = self.num_episodes + idx
        return self.episodes(slice(idx, idx + 1))

    def episodes(self, s: slice | np.ndarray) -> Trajectory:
        """Slice along the batch dimension."""
        kwargs: dict[str, Any] = {"actions": self.actions[s]}
        for name in _OPTIONAL_ARRAY_FIELDS:
            val = getattr(self, name)
            if val is not None:
                kwargs[name] = val[s]
        kwargs.update(self._scheme_kwargs())
        return Trajectory(**kwargs)

    def player(self, idx: int) -> Trajectory:
        """Return a single-player view (actions become (B, T))."""
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
        """Slice along the time dimension."""
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
    )

    def _scheme_kwargs(self) -> dict[str, Any]:
        """Return a dict of non-None scheme fields for forwarding."""
        return {
            name: getattr(self, name)
            for name in self._SCHEME_FIELDS
            if getattr(self, name) is not None
        }

    # ---- I/O helpers ----

    def save(self, path: str) -> None:
        """Save trajectory to a compressed ``.npz`` file.

        Scheme dicts are serialized as JSON byte strings stored under
        keys with an underscore prefix (e.g. ``_observation_scheme``).
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
        """Load trajectory from a ``.npz`` file.

        Underscore-prefixed JSON entries are deserialized back into
        their corresponding scheme dicts.  Legacy files with scalar
        ``_obs_type`` / ``_obs_radius`` keys are migrated into
        ``observation_scheme`` automatically.
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
        """Human-readable summary."""
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
    "achievements": "achievements_unlocked",
}

# Inverse mapping.
_STATE_TO_TRAJ: dict[str, str] = {v: k for k, v in _TRAJ_TO_STATE.items()}


def states_to_trajectory(
    states: list[EnvState],
    actions: np.ndarray | None = None,
    rewards: np.ndarray | None = None,
) -> Trajectory:
    """Convert a sequence of EnvState snapshots into a Trajectory.

    Builds a single-episode trajectory (batch dim = 1) by stacking
    every array field from the state list along a new time axis.

    Args:
        states: List of ``EnvState`` objects, one per timestep.
        actions: Optional action array of shape ``(T,)`` or ``(T, P)``.
            If ``None``, a zeros array is used.
        rewards: Optional reward array of shape ``(T,)``.

    Returns:
        A ``Trajectory`` with batch dimension 1 and all state fields
        populated.
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

    return Trajectory(**kwargs)


def trajectory_to_states(
    traj: Trajectory,
    episode: int = 0,
) -> list[EnvState]:
    """Reconstruct EnvState objects from a Trajectory.

    Only fields that are present in the trajectory are set. Missing
    fields will cause the reconstruction to fail if they are required
    by ``EnvState``.

    Args:
        traj: Trajectory with state fields populated.
        episode: Episode index to reconstruct.

    Returns:
        List of ``EnvState`` objects, one per timestep.

    Raises:
        ImportError: If ``factoriax.state`` is not available.
        ValueError: If required fields are missing from the trajectory.
    """
    from factoriax.state import EnvState

    T = traj.episode_length
    states = []

    for t in range(T):
        state_kwargs: dict[str, Any] = {}
        for traj_name in _OPTIONAL_ARRAY_FIELDS:
            if traj_name in ("rewards", "timesteps"):
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
