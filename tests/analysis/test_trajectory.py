"""Tests for ``Trajectory.env_params_scheme`` round-trip.

The trajectory format carries the ``EnvParams`` snapshot under the
``_env_params_scheme`` underscore-prefixed JSON metadata key so that
replay tooling can reconstruct the engine parameters that produced the
recording (including the new ``player_mining_yield``).
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
    """Direct field assertions on Trajectory."""

    def test_default_is_none(self) -> None:
        """Without env_params_scheme, the field is None."""
        traj = Trajectory(actions=np.zeros((1, 4), dtype=np.int32))
        assert traj.env_params_scheme is None

    def test_assigned_value_round_trips_via_save(self, tmp_path: Path) -> None:
        """A non-default scheme survives save → load byte-for-byte."""
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
        """Saving with scheme=None does not write the key."""
        traj = Trajectory(actions=np.zeros((1, 4), dtype=np.int32))
        path = tmp_path / "no_scheme.npz"
        traj.save(str(path))
        data = np.load(str(path), allow_pickle=True)
        assert "_env_params_scheme" not in data.files
        loaded = Trajectory.load(str(path))
        assert loaded.env_params_scheme is None


class TestStatesToTrajectoryWithParams:
    """``states_to_trajectory(states, params=...)`` populates the scheme.

    Uses ``state_factory`` (function-scoped, no env stepping) instead
    of a real ``canonical_env_8x8_1p`` rollout. The behavior under test
    is purely how ``states_to_trajectory`` packs the ``params`` kwarg
    into ``env_params_scheme``; the contents of the state list don't
    influence that pack step, so synthetic ``EnvState`` objects are
    sufficient. Replaces the ~10s rollout setup+call with sub-second
    state construction.
    """

    @staticmethod
    def _fake_states(state_factory, count: int):
        """Return ``count`` synthetic states + matching int32 actions.

        Parameters
        ----------
        state_factory
            The root conftest ``state_factory`` fixture.
        count
            Number of states to build.

        Returns
        -------
        tuple
            ``(states, actions)`` where ``actions`` is padded to match
            ``len(states)`` exactly, which is what ``states_to_trajectory``
            expects when it computes per-step deltas across the rollout.
        """
        world_map = jnp.full((8, 8), BlockType.DIRT, dtype=jnp.int32)
        states = [state_factory(world_map=world_map) for _ in range(count)]
        actions = np.zeros(count, dtype=np.int32)
        return states, actions

    def test_params_kwarg_populates_scheme(self, state_factory) -> None:
        """Passing params records env_params_to_dict on the trajectory."""
        params = EnvParams(player_mining_yield=3)
        states, actions = self._fake_states(state_factory, count=4)
        traj = states_to_trajectory(states, actions=actions, params=params)
        assert traj.env_params_scheme is not None
        assert traj.env_params_scheme["player_mining_yield"] == 3
        assert traj.env_params_scheme["player_mining_yield"] == 3

    def test_omitting_params_leaves_scheme_none(self, state_factory) -> None:
        """Default call (no params kwarg) keeps env_params_scheme=None."""
        states, actions = self._fake_states(state_factory, count=4)
        traj = states_to_trajectory(states, actions=actions)
        assert traj.env_params_scheme is None
