"""Tests for ``Trajectory.env_params_scheme`` round-trip.

The trajectory format carries the ``EnvParams`` snapshot under the
``_env_params_scheme`` underscore-prefixed JSON metadata key so that
replay tooling can reconstruct the engine parameters that produced the
recording (including the new ``player_mining_yield``).
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np

from factoriax.analysis.trajectory import Trajectory, states_to_trajectory
from factoriax.config import env_params_to_dict
from factoriax.state import EnvParams


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

    Consumes ``canonical_env_8x8_1p`` (root conftest, session-scoped)
    so the JIT compile of ``env.step_env`` is paid once for the entire
    session instead of twice per test class. ``env.reset_env`` is
    cheap and is called locally with the test-specific ``params`` so
    ``player_mining_yield`` variation flows through correctly.
    """

    def _run_short_episode(self, canonical_env, params: EnvParams):
        """Run a deterministic 3-step NOOP rollout for the given params.

        Args:
            canonical_env: The ``canonical_env_8x8_1p`` fixture tuple.
            params: Override params for this run; the env's reset
                uses these (cheap, non-JIT) and the shared
                ``jit_step_fn`` steps through them as traced data.

        Returns:
            Tuple ``(states, actions)`` where ``states`` is the list of
            states across the rollout and ``actions`` is the matching
            int32 action sequence.
        """
        env, _, jit_step_fn, _ = canonical_env
        rng = jax.random.PRNGKey(0)
        _, state = env.reset_env(rng, params)
        states = [state]
        actions = []
        for _ in range(3):
            rng, k = jax.random.split(rng)
            # Pass action as a Python int — the canonical form used by
            # ``test_jit_retrace.py`` and ``test_science_lab.py``. Passing
            # ``jnp.int32(0)`` here would key the JIT cache differently
            # and force a retrace when other migrated tests share the
            # ``canonical_env_8x8_1p`` fixture.
            a = 0
            _, state, _, _, _ = jit_step_fn(k, state, a, params)
            states.append(state)
            actions.append(a)
        return states, np.asarray(actions, dtype=np.int32)

    def test_params_kwarg_populates_scheme(self, canonical_env_8x8_1p) -> None:
        """Passing params records env_params_to_dict on the trajectory."""
        params = EnvParams(
            map_width=8, map_height=8, num_players=1, player_mining_yield=3
        )
        states, actions = self._run_short_episode(canonical_env_8x8_1p, params)
        # Pad actions to match states length (one final state after last action).
        actions = np.pad(actions, (0, len(states) - len(actions)))
        traj = states_to_trajectory(states, actions=actions, params=params)
        assert traj.env_params_scheme is not None
        assert traj.env_params_scheme["player_mining_yield"] == 3
        assert traj.env_params_scheme["map_width"] == 8

    def test_omitting_params_leaves_scheme_none(self, canonical_env_8x8_1p) -> None:
        """Default call (no params kwarg) keeps env_params_scheme=None."""
        params = EnvParams(map_width=8, map_height=8, num_players=1)
        states, actions = self._run_short_episode(canonical_env_8x8_1p, params)
        actions = np.pad(actions, (0, len(states) - len(actions)))
        traj = states_to_trajectory(states, actions=actions)
        assert traj.env_params_scheme is None
