"""Tests for trajectory replay mode in the agent debugger."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pygame
import pytest

from factoriax.agentdebugger.charts import (
    render_action_legend,
    render_action_sankey,
    render_action_strip,
    render_reward_chart,
)
from factoriax.agentdebugger.layout import (
    compute_debugger_dimensions,
    rebuild_replay_caches,
    render_replay_frame,
)
from factoriax.agentdebugger.main import Debugger
from factoriax.analysis.trajectory import Trajectory, states_to_trajectory
from factoriax.envs.factoriax_env import make_factoriax_env
from factoriax.state import EnvParams

# Replay tests run a real episode through the debugger before
# exercising the replay UI — slow, gated behind ``@slow``.
pytestmark = pytest.mark.slow

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def minimal_trajectory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a minimal trajectory .npz with only actions + rewards."""
    t = 20
    traj = Trajectory(
        actions=np.random.randint(0, 5, size=(2, t)).astype(np.int32),
        rewards=np.random.randn(2, t).astype(np.float32),
    )
    path = tmp_path_factory.mktemp("traj") / "minimal.npz"
    traj.save(str(path))
    return path


@pytest.fixture(scope="module")
def stateful_trajectory(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Create a trajectory with full state data (block_map etc.)."""
    env, _ = make_factoriax_env()
    params = EnvParams(map_width=8, map_height=8, num_players=1)
    rng = jax.random.PRNGKey(42)
    _, state = env.reset_env(rng, params)
    step_fn = jax.jit(env.step_env)

    states = [state]
    actions_list = []
    for _ in range(10):
        action = np.random.randint(0, 5)
        actions_list.append(action)
        rng, key = jax.random.split(rng)
        _, state, _, done, _ = step_fn(key, state, action, params)
        states.append(state)
        if bool(done):
            break

    act_arr = np.array(actions_list, dtype=np.int32)
    # states_to_trajectory uses len(states) as T, pad actions to match.
    if len(act_arr) < len(states):
        act_arr = np.pad(act_arr, (0, len(states) - len(act_arr)))
    traj = states_to_trajectory(states, actions=act_arr)

    path = tmp_path_factory.mktemp("traj") / "stateful.npz"
    traj.save(str(path))
    return path


# ------------------------------------------------------------------
# from_trajectory tests
# ------------------------------------------------------------------


class TestFromTrajectory:
    """Tests for Debugger.from_trajectory()."""

    def test_loads_and_sets_replay_mode(
        self,
        minimal_trajectory: Path,
    ) -> None:
        """Replay mode flag is set on construction."""
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        assert dbg._dbg.replay_mode is True

    def test_trajectory_stored(self, minimal_trajectory: Path) -> None:
        """The trajectory reference is stored in state."""
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        assert dbg._dbg.trajectory is not None
        assert dbg._dbg.trajectory.num_episodes == 2

    def test_actions_extracted(self, minimal_trajectory: Path) -> None:
        """Actions are extracted from the trajectory."""
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        assert len(dbg._actions) == 20

    def test_rewards_extracted(self, minimal_trajectory: Path) -> None:
        """Rewards are extracted from the trajectory."""
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        assert len(dbg._rewards) == 20

    def test_no_frames_without_block_map(
        self,
        minimal_trajectory: Path,
    ) -> None:
        """No rendered frames when trajectory has no state data."""
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        assert dbg._dbg.rendered_frames is None

    def test_frames_with_state_data(
        self,
        stateful_trajectory: Path,
    ) -> None:
        """Rendered frames are produced from state data."""
        dbg = Debugger.from_trajectory(str(stateful_trajectory))
        assert dbg._dbg.rendered_frames is not None
        assert len(dbg._dbg.rendered_frames) > 0

    def test_episode_selection(self, minimal_trajectory: Path) -> None:
        """Can select a specific episode."""
        dbg = Debugger.from_trajectory(
            str(minimal_trajectory),
            episode=1,
        )
        assert dbg._dbg.selected_episode == 1

    def test_player_selection(self, minimal_trajectory: Path) -> None:
        """Can select a specific player index."""
        dbg = Debugger.from_trajectory(
            str(minimal_trajectory),
            player_idx=0,
        )
        assert dbg._player_idx == 0


# ------------------------------------------------------------------
# Replay keybinding tests
# ------------------------------------------------------------------


class TestReplayKeybindings:
    """Test keyboard handling in replay mode."""

    def _make_event(self, key: int) -> pygame.event.Event:
        """Create a fake KEYDOWN event."""
        return pygame.event.Event(pygame.KEYDOWN, key=key, unicode="")

    def test_step_forward(self, minimal_trajectory: Path) -> None:
        """Right bracket advances the step."""
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        assert dbg._dbg.current_step == 0
        dbg._handle_replay_keydown(
            self._make_event(pygame.K_RIGHTBRACKET),
            320,
            240,
        )
        assert dbg._dbg.current_step == 1

    def test_step_backward(self, minimal_trajectory: Path) -> None:
        """Left bracket decreases the step (clamped to 0)."""
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        dbg._dbg.current_step = 5
        dbg._handle_replay_keydown(
            self._make_event(pygame.K_LEFTBRACKET),
            320,
            240,
        )
        assert dbg._dbg.current_step == 4

    def test_step_backward_clamped(
        self,
        minimal_trajectory: Path,
    ) -> None:
        """Step doesn't go below 0."""
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        dbg._handle_replay_keydown(
            self._make_event(pygame.K_LEFTBRACKET),
            320,
            240,
        )
        assert dbg._dbg.current_step == 0

    def test_play_pause_toggle(self, minimal_trajectory: Path) -> None:
        """Space toggles playing."""
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        assert dbg._dbg.playing is False
        dbg._handle_replay_keydown(
            self._make_event(pygame.K_SPACE),
            320,
            240,
        )
        assert dbg._dbg.playing is True
        dbg._handle_replay_keydown(
            self._make_event(pygame.K_SPACE),
            320,
            240,
        )
        assert dbg._dbg.playing is False

    def test_home_jumps_to_start(
        self,
        minimal_trajectory: Path,
    ) -> None:
        """Home key jumps to step 0."""
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        dbg._dbg.current_step = 10
        dbg._handle_replay_keydown(
            self._make_event(pygame.K_HOME),
            320,
            240,
        )
        assert dbg._dbg.current_step == 0

    def test_end_jumps_to_last(self, minimal_trajectory: Path) -> None:
        """End key jumps to the last step."""
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        dbg._handle_replay_keydown(
            self._make_event(pygame.K_END),
            320,
            240,
        )
        assert dbg._dbg.current_step == 19

    def test_n_steps_forward(self, minimal_trajectory: Path) -> None:
        """N key also steps forward."""
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        dbg._handle_replay_keydown(
            self._make_event(pygame.K_n),
            320,
            240,
        )
        assert dbg._dbg.current_step == 1

    def test_episode_navigation(
        self,
        minimal_trajectory: Path,
    ) -> None:
        """Period/comma navigate episodes."""
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        assert dbg._dbg.selected_episode == 0
        dbg._handle_replay_keydown(
            self._make_event(pygame.K_PERIOD),
            320,
            240,
        )
        assert dbg._dbg.selected_episode == 1
        dbg._handle_replay_keydown(
            self._make_event(pygame.K_COMMA),
            320,
            240,
        )
        assert dbg._dbg.selected_episode == 0

    def test_arrow_keys_step(self, minimal_trajectory: Path) -> None:
        """Arrow keys step backward / forward."""
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        dbg._handle_replay_keydown(
            self._make_event(pygame.K_RIGHT),
            320,
            240,
        )
        assert dbg._dbg.current_step == 1
        dbg._handle_replay_keydown(
            self._make_event(pygame.K_LEFT),
            320,
            240,
        )
        assert dbg._dbg.current_step == 0

    def test_speed_control(self, minimal_trajectory: Path) -> None:
        """+/- keys control playback speed."""
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        assert dbg._dbg.playback_speed == 1
        dbg._handle_replay_keydown(
            self._make_event(pygame.K_EQUALS),
            320,
            240,
        )
        assert dbg._dbg.playback_speed == 2
        dbg._handle_replay_keydown(
            self._make_event(pygame.K_MINUS),
            320,
            240,
        )
        assert dbg._dbg.playback_speed == 1

    def test_zero_goes_to_start(self, minimal_trajectory: Path) -> None:
        """0 key jumps to step 0."""
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        dbg._dbg.current_step = 10
        dbg._handle_replay_keydown(
            self._make_event(pygame.K_0),
            320,
            240,
        )
        assert dbg._dbg.current_step == 0

    def test_help_toggle(self, minimal_trajectory: Path) -> None:
        """Question mark toggles help overlay."""
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        dbg._handle_replay_keydown(
            self._make_event(pygame.K_QUESTION),
            320,
            240,
        )
        assert dbg._dbg.show_help is True

    def test_escape_quits(self, minimal_trajectory: Path) -> None:
        """Escape returns running=False."""
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        running, _ = dbg._handle_replay_keydown(
            self._make_event(pygame.K_ESCAPE),
            320,
            240,
        )
        assert running is False


# ------------------------------------------------------------------
# Replay layout tests
# ------------------------------------------------------------------


class TestReplayLayout:
    """Tests for render_replay_frame."""

    def test_output_shape(self, minimal_trajectory: Path) -> None:
        """Frame has the expected shape."""
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        qw, qh = 320, 240
        base_w, base_h = compute_debugger_dimensions(qw, qh)
        rebuild_replay_caches(dbg._dbg, qw, qh)
        frame = render_replay_frame(dbg._dbg, base_w, base_h, qw, qh)
        assert frame.shape == (base_h, base_w, 3)
        assert frame.dtype == np.uint8

    def test_not_all_black(self, minimal_trajectory: Path) -> None:
        """Frame contains non-black content."""
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        qw, qh = 320, 240
        base_w, base_h = compute_debugger_dimensions(qw, qh)
        rebuild_replay_caches(dbg._dbg, qw, qh)
        frame = render_replay_frame(dbg._dbg, base_w, base_h, qw, qh)
        assert frame.sum() > 0


# ------------------------------------------------------------------
# Cache rebuilding tests
# ------------------------------------------------------------------


class TestRebuildReplayCaches:
    """Tests for rebuild_replay_caches."""

    def test_caches_populated(self, minimal_trajectory: Path) -> None:
        """All chart caches are non-None after rebuild."""
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        rebuild_replay_caches(dbg._dbg, 320, 240)
        assert dbg._dbg.reward_chart_cache is not None
        assert dbg._dbg.action_strip_cache is not None
        assert dbg._dbg.action_legend_cache is not None
        assert dbg._dbg.sankey_cache is not None


# ------------------------------------------------------------------
# Chart function tests (ported from inspector)
# ------------------------------------------------------------------


class TestActionCharts:
    """Tests for action strip, legend, and sankey rendering."""

    @pytest.fixture
    def traj(self) -> Trajectory:
        """Simple trajectory with known actions."""
        return Trajectory(
            actions=np.array([[0, 1, 2, 1, 0, 3, 2, 1]], dtype=np.int32),
            rewards=np.random.randn(1, 8).astype(np.float32),
        )

    def test_action_strip_shape(self, traj: Trajectory) -> None:
        """Action strip has correct output shape."""
        img = render_action_strip(traj, 0, 0, 200, 24)
        assert img.shape == (24, 200, 3)
        assert img.dtype == np.uint8

    def test_action_strip_not_black(self, traj: Trajectory) -> None:
        """Action strip contains colored pixels."""
        img = render_action_strip(traj, 0, 0, 200, 24)
        assert img.sum() > 0

    def test_action_legend_shape(self, traj: Trajectory) -> None:
        """Action legend has correct output shape."""
        img = render_action_legend(traj, 0, 0, 200, 20)
        assert img.shape == (20, 200, 3)

    def test_action_sankey_shape(self, traj: Trajectory) -> None:
        """Action sankey has correct output shape."""
        img = render_action_sankey(traj, 0, 0, 200, 120)
        assert img.shape == (120, 200, 3)

    def test_reward_chart_shape(self, traj: Trajectory) -> None:
        """Reward chart has correct output shape."""
        img = render_reward_chart(traj, 0, 200, 100)
        assert img.shape == (100, 200, 3)


# ------------------------------------------------------------------
# Live mode regression
# ------------------------------------------------------------------


class TestLiveModeUnchanged:
    """Verify the standard constructor still works."""

    def test_live_mode_flag(self, debugger: Debugger) -> None:
        """Standard constructor produces live mode."""
        assert debugger._dbg.replay_mode is False

    def test_live_mode_has_env(self, debugger: Debugger) -> None:
        """Live mode has environment references."""
        assert debugger._env is not None
        assert debugger._params is not None
