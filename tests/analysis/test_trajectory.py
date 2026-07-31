"""Tests for the ``env_params_scheme`` field on :class:`Trajectory`.

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

from factoriax.analysis.trajectory import Trajectory, states_to_trajectory
from factoriax.engine.constants import BlockType
from factoriax.engine.state import EnvParams
from factoriax.playground.config import env_params_to_dict


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
