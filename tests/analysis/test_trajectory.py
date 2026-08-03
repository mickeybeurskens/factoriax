"""Tests for :mod:`factoriax.analysis.trajectory`.

The first group covers construction, the shape properties, slicing, and
the ``.npz`` round trip. The second group covers the ``env_params_scheme``
field.

A replay needs the engine parameters that made the recording. A value
such as ``player_mining_yield`` changes how many items one mine action
gives. Under different parameters, a replayed rollout therefore
diverges from the recording.

:class:`Trajectory` carries these parameters in ``env_params_scheme``,
a plain dictionary from :func:`env_params_to_dict`. ``save`` writes the
dictionary to the ``.npz`` archive as JSON, under the key
``_env_params_scheme``. The leading underscore separates it from the
array fields, which keep their own names.

The field is optional and defaults to ``None``. ``save`` omits a
``None`` scheme from the archive and writes no null. An old reader
therefore finds no such key, and not a key with nothing in it.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.analysis.trajectory import Trajectory, states_to_trajectory
from factoriax.engine.constants import BlockType
from factoriax.engine.state import EnvParams
from factoriax.playground.config import env_params_to_dict
from tests.helpers.trajectories import (
    make_multi_player_traj,
    make_single_player_traj,
)


class TestEnvParamsSchemeField:
    """The field itself, and how it survives the archive."""

    def test_default_is_none(self) -> None:
        """Report ``None`` for a trajectory built without a scheme.

        A recording made before the field existed loads into this
        state. ``None`` must therefore mean "not recorded" and not
        "empty".
        """
        traj = Trajectory(actions=np.zeros((1, 4), dtype=np.int32))
        assert traj.env_params_scheme is None

    def test_assigned_value_round_trips_via_save(self, tmp_path: Path) -> None:
        """Load a saved scheme back equal to the one that was saved.

        The dictionary goes to JSON, so the values that return are
        plain JSON types. The first assertion covers the whole
        dictionary. The second covers one field, and catches a round
        trip that keeps the keys but turns an integer into a string.
        """
        scheme = env_params_to_dict(EnvParams(player_mining_yield=3))
        traj = Trajectory(
            actions=np.zeros((1, 4), dtype=np.int32),
            env_params_scheme=scheme,
        )
        path = tmp_path / "with_scheme.npz"
        traj.save(str(path))
        loaded = Trajectory.load(str(path))
        assert loaded.env_params_scheme == scheme
        assert loaded.env_params_scheme["player_mining_yield"] == 3

    def test_none_is_absent_in_load(self, tmp_path: Path) -> None:
        """Write no key at all for a ``None`` scheme.

        An explicit null makes an unrecorded scheme look the same as a
        scheme recorded as empty.
        """
        traj = Trajectory(actions=np.zeros((1, 4), dtype=np.int32))
        path = tmp_path / "no_scheme.npz"
        traj.save(str(path))
        data = np.load(str(path), allow_pickle=True)
        assert "_env_params_scheme" not in data.files
        loaded = Trajectory.load(str(path))
        assert loaded.env_params_scheme is None


class TestStatesToTrajectoryWithParams:
    """``states_to_trajectory`` packing the ``params`` argument.

    The states are synthetic and not the output of a real rollout.
    ``states_to_trajectory`` stacks the state fields along a time axis.
    Separately, it converts ``params`` to a dictionary. No part of the
    states changes that conversion. Direct construction is therefore
    sufficient, and it replaces a rollout of about ten seconds with
    less than one second of work.
    """

    @staticmethod
    def _fake_states(state_factory, count: int):
        """Return ``count`` synthetic states and matching actions.

        Parameters
        ----------
        state_factory :
            The ``state_factory`` fixture from the root ``conftest``.
        count :
            Number of states to build.

        Returns
        -------
        tuple
            ``(states, actions)``. The action array gets one entry for
            each state. ``states_to_trajectory`` does not compare the
            two lengths. A mismatch therefore gives a trajectory whose
            action axis and time axis disagree.
        """
        world_map = jnp.full((8, 8), BlockType.DIRT, dtype=jnp.int32)
        states = [state_factory(world_map=world_map) for _ in range(count)]
        actions = np.zeros(count, dtype=np.int32)
        return states, actions

    def test_params_kwarg_populates_scheme(self, state_factory) -> None:
        """When the caller gives ``params``, record them on the trajectory."""
        params = EnvParams(player_mining_yield=3)
        states, actions = self._fake_states(state_factory, count=4)
        traj = states_to_trajectory(states, actions=actions, params=params)
        assert traj.env_params_scheme is not None
        assert traj.env_params_scheme["player_mining_yield"] == 3

    def test_omitting_params_leaves_scheme_none(self, state_factory) -> None:
        """When the caller omits ``params``, record no parameters.

        The caller asks for the conversion. A caller without an
        ``EnvParams`` gets a trajectory that says so. It does not get a
        trajectory with engine defaults that no rollout used.
        """
        states, actions = self._fake_states(state_factory, count=4)
        traj = states_to_trajectory(states, actions=actions)
        assert traj.env_params_scheme is None


class TestTrajectory:
    """Tests for Trajectory construction, shape properties, slicing, and I/O."""

    def test_1d_actions_coerced(self) -> None:
        """1D action array is promoted to (1, T)."""
        traj = Trajectory(actions=np.zeros(50, dtype=np.int32))
        assert traj.num_episodes == 1
        assert traj.episode_length == 50

    def test_2d_single_player(self) -> None:
        """2D actions give single-player trajectory."""
        traj = make_single_player_traj(8, 50)
        assert traj.is_multi_player is False
        assert traj.num_players == 1

    def test_3d_multi_player(self) -> None:
        """3D actions give multi-player trajectory with correct player count."""
        traj = make_multi_player_traj(8, 50, num_p=3)
        assert traj.is_multi_player is True
        assert traj.num_players == 3

    def test_4d_raises_value_error(self) -> None:
        """4D action array raises ValueError."""
        with pytest.raises(ValueError):
            Trajectory(actions=np.zeros((2, 3, 4, 5), dtype=np.int32))

    def test_episode_returns_one_episode(self) -> None:
        """episode(i) returns a single-episode Trajectory with correct data."""
        traj = make_single_player_traj(8, 50)
        ep = traj.episode(2)
        assert ep.num_episodes == 1
        np.testing.assert_array_equal(ep.actions[0], traj.actions[2])

    def test_episode_negative_index(self) -> None:
        """episode(-1) returns the last episode."""
        traj = make_single_player_traj(8, 50)
        ep = traj.episode(-1)
        assert ep.num_episodes == 1
        np.testing.assert_array_equal(ep.actions[0], traj.actions[-1])

    def test_episodes_slice(self) -> None:
        """episodes(slice(1, 3)) returns two episodes."""
        traj = make_single_player_traj(8, 50)
        sliced = traj.episodes(slice(1, 3))
        assert sliced.actions.shape == (2, 50)

    def test_episodes_array_index(self) -> None:
        """episodes(np.array([0, 2, 4])) returns three specific episodes."""
        traj = make_single_player_traj(8, 50)
        indexed = traj.episodes(np.array([0, 2, 4]))
        assert indexed.actions.shape == (3, 50)

    def test_player_on_single_player(self) -> None:
        """player(0) on a single-player trajectory returns a single-player view."""
        traj = make_single_player_traj(8, 50)
        result = traj.player(0)
        assert result.is_multi_player is False

    def test_player_extracts_correct_player(self) -> None:
        """player(1) actions match traj.actions[:, :, 1]."""
        traj = make_multi_player_traj(8, 50, num_p=2)
        p1 = traj.player(1)
        np.testing.assert_array_equal(p1.actions, traj.actions[:, :, 1])

    def test_player_preserves_optional_fields(self) -> None:
        """player(0) squeezes positions to (num_eps, num_steps, 2)."""
        traj = make_multi_player_traj(8, 50, num_p=2)
        p0 = traj.player(0)
        assert p0.positions is not None
        assert p0.positions.shape == (8, 50, 2)

    def test_time_slice_shape(self) -> None:
        """time_slice(10, 40) returns episode_length == 30."""
        traj = make_single_player_traj(8, 50)
        sliced = traj.time_slice(10, 40)
        assert sliced.episode_length == 30

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """All array shapes and dtypes survive save/load."""
        traj = make_multi_player_traj(4, 30, num_p=2)
        save_path = str(tmp_path / "traj.npz")
        traj.save(save_path)
        loaded = Trajectory.load(save_path)
        np.testing.assert_array_equal(loaded.actions, traj.actions)
        assert loaded.positions is not None
        assert traj.positions is not None
        assert loaded.positions.shape == traj.positions.shape
        assert loaded.actions.dtype == traj.actions.dtype
        assert loaded.achievements is not None
        assert traj.achievements is not None
        assert loaded.achievements.shape == traj.achievements.shape


# ---------------------------------------------------------------------------
# TestActions
# ---------------------------------------------------------------------------
