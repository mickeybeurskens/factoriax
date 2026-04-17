"""Tests for the Debugger stepping logic.

These tests exercise _execute_step and _step_forward without
launching a pygame window.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.agentdebugger.main import Debugger
from factoriax.constants import Action
from factoriax.envs.factoriax_env import make_factoriax_env
from factoriax.observations import global_array
from factoriax.state import EnvParams

# Each test steps a real env through the debugger, triggering the env
# JIT compile. Gated behind ``@slow`` for the pre-commit inner loop.
pytestmark = pytest.mark.slow


class TestExecuteStep:
    """Tests for Debugger._execute_step."""

    def test_appends_state(self, debugger: Debugger) -> None:
        """Stepping appends a new state to the history."""
        assert len(debugger._states) == 1
        debugger._execute_step(int(Action.NOOP))
        assert len(debugger._states) == 2

    def test_appends_action(self, debugger: Debugger) -> None:
        """Stepping records the action."""
        debugger._execute_step(int(Action.NOOP))
        assert debugger._actions == [int(Action.NOOP)]

    def test_appends_reward(self, debugger: Debugger) -> None:
        """Stepping records a reward when reward_fn is set."""
        debugger._execute_step(int(Action.NOOP))
        assert len(debugger._rewards) == 1
        assert debugger._rewards[0] == 1.0

    def test_appends_cost(self, debugger: Debugger) -> None:
        """Stepping records costs when constraint_fn is set."""
        debugger._execute_step(int(Action.NOOP))
        assert len(debugger._costs) == 1
        np.testing.assert_array_almost_equal(
            debugger._costs[0],
            [0.5, 0.1],
        )

    def test_advances_cursor(self, debugger: Debugger) -> None:
        """Cursor auto-advances to the latest step."""
        assert debugger._dbg.current_step == 0
        debugger._execute_step(int(Action.NOOP))
        assert debugger._dbg.current_step == 1

    def test_invalidates_chart_caches(
        self,
        debugger: Debugger,
    ) -> None:
        """Chart caches are cleared after stepping."""
        debugger._dbg.reward_chart_cache = np.zeros((10, 10, 3))
        debugger._dbg.cost_chart_cache = np.zeros((10, 10, 3))
        debugger._execute_step(int(Action.NOOP))
        assert debugger._dbg.reward_chart_cache is None
        assert debugger._dbg.cost_chart_cache is None

    def test_multiple_steps_accumulate(
        self,
        debugger: Debugger,
    ) -> None:
        """Multiple steps build up the history correctly."""
        for i in range(5):
            debugger._execute_step(int(Action.NOOP))
        assert len(debugger._states) == 6
        assert len(debugger._actions) == 5
        assert len(debugger._rewards) == 5
        assert len(debugger._costs) == 5
        assert debugger._dbg.current_step == 5

    def test_state_changes_after_step(
        self,
        debugger: Debugger,
    ) -> None:
        """The new state differs from the initial state (timestep advances)."""
        initial = debugger._states[0]
        debugger._execute_step(int(Action.NOOP))
        stepped = debugger._states[1]
        assert int(stepped.timestep) == int(initial.timestep) + 1


class TestStepForward:
    """Tests for Debugger._step_forward."""

    def test_ai_mode_queries_policy(
        self,
        debugger: Debugger,
    ) -> None:
        """In AI mode, step_forward calls the policy and steps."""
        debugger._dbg.mode = "ai"
        # Need a GameUI stub; use None since _step_forward in AI mode
        # never touches the UI.
        from unittest.mock import MagicMock

        mock_ui = MagicMock()
        debugger._step_forward(mock_ui)
        assert len(debugger._states) == 2

    def test_human_mode_sets_awaiting(
        self,
        debugger: Debugger,
    ) -> None:
        """In human mode, step_forward sets the awaiting flag."""
        debugger._dbg.mode = "human"
        from unittest.mock import MagicMock

        mock_ui = MagicMock()
        debugger._step_forward(mock_ui)
        assert debugger._awaiting_human_input is True
        assert len(debugger._states) == 1  # No step taken yet.

    def test_done_blocks_stepping(
        self,
        debugger: Debugger,
    ) -> None:
        """No step is taken after the environment is done."""
        debugger._dbg.done = True
        from unittest.mock import MagicMock

        mock_ui = MagicMock()
        debugger._step_forward(mock_ui)
        assert len(debugger._states) == 1

    def test_cursor_mid_history_advances_without_step(
        self,
        debugger: Debugger,
    ) -> None:
        """When cursor is behind history head, advance cursor only."""
        debugger._execute_step(int(Action.NOOP))
        debugger._execute_step(int(Action.NOOP))
        debugger._dbg.current_step = 0  # Rewind cursor.
        from unittest.mock import MagicMock

        mock_ui = MagicMock()
        debugger._step_forward(mock_ui)
        assert debugger._dbg.current_step == 1
        assert len(debugger._states) == 3  # No new step.


class TestNoRewardOrConstraint:
    """Tests for Debugger with no reward/constraint functions."""

    @pytest.fixture
    def bare_debugger(self, env_and_state: tuple) -> Debugger:
        """Debugger with no reward or constraint functions."""
        env, params, state = env_and_state
        return Debugger(
            env,
            params,
            state,
            policy=lambda obs: jnp.int32(Action.NOOP),
            seed=0,
        )

    def test_no_rewards_recorded(
        self,
        bare_debugger: Debugger,
    ) -> None:
        """No rewards list entries when reward_fn is None."""
        bare_debugger._execute_step(int(Action.NOOP))
        assert bare_debugger._rewards == []

    def test_no_costs_recorded(
        self,
        bare_debugger: Debugger,
    ) -> None:
        """No costs list entries when constraint_fn is None."""
        bare_debugger._execute_step(int(Action.NOOP))
        assert bare_debugger._costs == []

    def test_stepping_still_works(
        self,
        bare_debugger: Debugger,
    ) -> None:
        """Environment stepping works without reward/constraint."""
        bare_debugger._execute_step(int(Action.MINE))
        assert len(bare_debugger._states) == 2
        assert bare_debugger._actions == [int(Action.MINE)]


class TestMultiPlayer:
    """Tests for multi-player stepping behavior."""

    @pytest.fixture
    def mp_debugger(self) -> Debugger:
        """Two-player debugger where human controls player 0."""
        env, _ = make_factoriax_env()
        params = EnvParams(
            map_width=8,
            map_height=8,
            num_players=2,
        )
        _, state = env.reset_env(jax.random.PRNGKey(0), params)

        actions_seen: list[int] = []

        def tracking_policy(obs: jax.Array) -> jax.Array:
            actions_seen.append(1)
            return jnp.int32(Action.MINE)

        dbg = Debugger(
            env,
            params,
            state,
            policy=tracking_policy,
            obs_fn=global_array,
            player_idx=0,
            seed=0,
        )
        dbg._policy_calls = actions_seen  # type: ignore[attr-defined]
        return dbg

    def test_both_players_stepped(
        self,
        mp_debugger: Debugger,
    ) -> None:
        """A single execute_step advances the timestep for all players."""
        initial_ts = int(mp_debugger._states[0].timestep)
        mp_debugger._execute_step(int(Action.NOOP))
        # Two players stepped, so timestep advances by 2.
        final_ts = int(mp_debugger._states[-1].timestep)
        assert final_ts == initial_ts + 2

    def test_policy_called_for_other_player(
        self,
        mp_debugger: Debugger,
    ) -> None:
        """The policy is queried for the non-controlled player."""
        mp_debugger._execute_step(int(Action.NOOP))
        # Player 0 uses human_action, player 1 uses policy.
        assert len(mp_debugger._policy_calls) == 1  # type: ignore[attr-defined]

    def test_recorded_action_is_human_action(
        self,
        mp_debugger: Debugger,
    ) -> None:
        """The recorded action is the human player's action."""
        mp_debugger._execute_step(int(Action.UP))
        assert mp_debugger._actions == [int(Action.UP)]
