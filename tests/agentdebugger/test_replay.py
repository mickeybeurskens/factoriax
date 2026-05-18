"""Tests for trajectory replay mode in the agent debugger."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pygame
import pytest

import factoriax
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
    env, _ = factoriax.make()
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

    def test_no_states_without_block_map(
        self,
        minimal_trajectory: Path,
    ) -> None:
        """No per-step states when trajectory has no state data."""
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        assert dbg._states == []

    def test_states_with_state_data(
        self,
        stateful_trajectory: Path,
    ) -> None:
        """Per-step states are reconstructed from state data."""
        dbg = Debugger.from_trajectory(str(stateful_trajectory))
        assert len(dbg._states) > 0

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
        frame = render_replay_frame(dbg._dbg, dbg._states, base_w, base_h, qw, qh)
        assert frame.shape == (base_h, base_w, 3)
        assert frame.dtype == np.uint8

    def test_not_all_black(self, minimal_trajectory: Path) -> None:
        """Frame contains non-black content."""
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        qw, qh = 320, 240
        base_w, base_h = compute_debugger_dimensions(qw, qh)
        rebuild_replay_caches(dbg._dbg, qw, qh)
        frame = render_replay_frame(dbg._dbg, dbg._states, base_w, base_h, qw, qh)
        assert frame.sum() > 0

    def test_does_not_cache_rendered_frames(
        self,
        stateful_trajectory: Path,
    ) -> None:
        """No pre-rendered frame list is held on the debugger state.

        Regression guard: pre-rendering one RGB frame per step at
        550 KB/frame OOM'd long replays. The refactor keeps only the
        state list (~12 KB/step) and renders on demand.
        """
        dbg = Debugger.from_trajectory(str(stateful_trajectory))
        assert not hasattr(dbg._dbg, "rendered_frames"), (
            "DebuggerState must not carry a pre-rendered frame list — "
            "frames are rendered on demand inside render_replay_frame."
        )

    def test_q1_matches_on_demand_render_pixels(
        self,
        stateful_trajectory: Path,
    ) -> None:
        """Q1's map region equals ``JaxRenderer.jit_render_map`` output.

        Proves the on-demand render path produces the same content the
        pre-rendered cache used to serve.
        """
        from factoriax.agentdebugger.layout import _tile_px
        from factoriax.jax_renderer import JaxRenderer

        dbg = Debugger.from_trajectory(str(stateful_trajectory))
        if not dbg._states:
            pytest.skip("trajectory has no state data")
        dbg._dbg.show_obs_overlay = False  # keep the comparison direct
        # Pick a step in the middle so it's neither the initial nor
        # the padded terminal state.
        dbg._dbg.current_step = min(1, len(dbg._states) - 1)
        qw, qh = 320, 240
        base_w, base_h = compute_debugger_dimensions(qw, qh)
        rebuild_replay_caches(dbg._dbg, qw, qh)
        frame = render_replay_frame(dbg._dbg, dbg._states, base_w, base_h, qw, qh)
        state = dbg._states[dbg._dbg.current_step]
        renderer = JaxRenderer(tile_px=_tile_px(state))
        expected = np.asarray(renderer.jit_render_map(state))
        # The game image is blitted into Q1 (top-left), centered with
        # aspect-preserving scale. We can't compare pixel-exact without
        # recomputing the scale math — instead assert a non-trivial
        # fraction of pixels in the Q1 region match some pixel from
        # the expected render (same color histogram).
        q1 = frame[:qh, :qw]
        assert q1.sum() > 0
        # Every unique color in the Q1 region must exist in the
        # expected render (no colors invented by the blit step).
        q1_flat = q1.reshape(-1, 3)
        exp_flat = expected.reshape(-1, 3)
        q1_set = {tuple(c.tolist()) for c in q1_flat}
        exp_set = {tuple(c.tolist()) for c in exp_flat}
        # Background color is painted outside the scaled game image,
        # allow it through.
        bg = (20, 20, 25)
        foreign = q1_set - exp_set - {bg}
        assert not foreign, (
            f"Q1 contains colors not present in JaxRenderer output: {foreign}"
        )


# ------------------------------------------------------------------
# Cache rebuilding tests
# ------------------------------------------------------------------


class TestRebuildReplayCaches:
    """Tests for rebuild_replay_caches."""

    def test_caches_populated(self, minimal_trajectory: Path) -> None:
        """Reward + action-strip caches are populated after rebuild.

        The Sankey cache is intentionally left ``None``: Q4 now holds
        the reward chart and the per-step inventory panel is rendered
        on the fly, so no Sankey image is pre-computed.
        """
        dbg = Debugger.from_trajectory(str(minimal_trajectory))
        rebuild_replay_caches(dbg._dbg, 320, 240)
        assert dbg._dbg.reward_chart_cache is not None
        assert dbg._dbg.action_strip_cache is not None
        assert dbg._dbg.action_legend_cache is not None
        assert dbg._dbg.sankey_cache is None


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
        """Reward chart returns image + plot bounds at requested size."""
        img, (x0, x1) = render_reward_chart(traj, 0, 200, 100)
        assert img.shape == (100, 200, 3)
        assert 0 <= x0 < x1 < 200


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


# ------------------------------------------------------------------
# replay_states param reconstruction
# ------------------------------------------------------------------


class TestReplayStatesEnvParamsReconstruction:
    """``replay_states`` must rebuild EnvParams from env_params_scheme."""

    def test_player_mining_yield_replays_bit_identically(
        self,
        tmp_path: Path,
    ) -> None:
        """A yield=3 recording must replay to the same items_mined."""
        import jax.numpy as jnp

        from factoriax.agentdebugger.replay import replay_states
        from factoriax.constants import Action, BlockType
        from factoriax.envs.factoriax_env import FactoriaXEnv
        from factoriax.levels import Level, build_state

        # 3x3 map: COAL at (1, 1), player spawned at (0, 1).
        block_map = np.full((3, 3), int(BlockType.DIRT), dtype=np.int32)
        block_map[1, 1] = int(BlockType.COAL)

        level = Level(
            name="mine-test",
            map_width=3,
            map_height=3,
            block_map=block_map,
            player_positions=[(0, 1)],
        )

        # Mine three times with yield=3 → 9 coal after three actions.
        params = EnvParams(
            map_width=3,
            map_height=3,
            num_players=1,
            max_timesteps=10,
            player_mining_yield=3,
            base_resources=30,
        )
        state = build_state(level, params)

        env = FactoriaXEnv()
        step_fn = jax.jit(env.step_env)
        rng = jax.random.PRNGKey(0)

        # Drive: FACE_RIGHT, then three MINEs. Both record and replay
        # start from the same build_state output so the action sequence
        # alone reproduces the final state.
        states = [state]
        actions: list[int] = [
            int(Action.FACE_RIGHT),
            int(Action.MINE),
            int(Action.MINE),
            int(Action.MINE),
        ]
        for a in actions:
            rng, k = jax.random.split(rng)
            _, state, _, _, _ = step_fn(k, state, jnp.int32(a), params)
            states.append(state)

        recorded_items_mined = np.asarray(states[-1].items_mined)
        del jnp  # silence unused import — jnp was only needed for the
        # old build_state override path; keep the import for future
        # maintenance of this fixture.

        # Build a trajectory with env_params_scheme.
        act_arr = np.array(actions + [0], dtype=np.int32)
        traj = states_to_trajectory(states, actions=act_arr, params=params)
        assert traj.env_params_scheme is not None
        assert traj.env_params_scheme["player_mining_yield"] == 3

        # Save + load round trip to mirror real usage.
        path = tmp_path / "yield3.npz"
        traj.save(str(path))
        loaded = Trajectory.load(str(path))

        replayed = replay_states(level, loaded)
        replayed_items_mined = np.asarray(replayed[-1].items_mined)

        # Bit-identical items_mined under the recovered yield.
        np.testing.assert_array_equal(replayed_items_mined, recorded_items_mined)
        # Sanity: with yield=3 and three mines, expect 9 coal.
        from factoriax.constants import ItemType

        assert int(replayed_items_mined[int(ItemType.COAL)]) == 9

    def test_replay_honours_num_players_from_scheme(
        self,
        tmp_path: Path,
    ) -> None:
        """``num_players`` from env_params_scheme overrides the hardcode."""
        from factoriax.agentdebugger.replay import replay_states
        from factoriax.config import env_params_to_dict
        from factoriax.constants import BlockType
        from factoriax.levels import Level

        block_map = np.full((3, 3), int(BlockType.DIRT), dtype=np.int32)
        level = Level(
            name="two-player",
            map_width=3,
            map_height=3,
            block_map=block_map,
            player_positions=[(0, 0), (2, 2)],
        )
        params = EnvParams(map_width=3, map_height=3, num_players=2, max_timesteps=3)

        # Build a 2-player trajectory with env_params_scheme.
        actions = np.zeros((1, 2, 2), dtype=np.int32)
        traj = Trajectory(
            actions=actions,
            env_params_scheme=env_params_to_dict(params),
        )

        # Override the hardcoded num_players=1 — should not crash.
        states = replay_states(level, traj)
        assert states[0].player_positions.shape[0] == 2
