"""Round-trip test: EnvState -> Trajectory -> EnvState.

Verifies that every EnvState field survives conversion to a Trajectory
and back. Also tests save/load round-trip through .npz files.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import numpy.testing as npt

from factoriax.analysis.trajectory import (
    Trajectory,
    states_to_trajectory,
    trajectory_to_states,
)
from factoriax.benchmarks.basic_skills.levels import BASIC_SKILLS_LEVELS
from factoriax.constants import Action, BlockType, MachineType
from factoriax.envs import FactoriaXEnv
from factoriax.levels import LevelBuilder, build_state
from factoriax.state import EnvParams, EnvState


def _run_episode(level_idx: int = 0, steps: int = 10) -> list[EnvState]:
    """Run a short random episode and return per-step states."""
    bl = BASIC_SKILLS_LEVELS[level_idx]
    params = bl.env_params
    state = build_state(bl.level, params)
    env = FactoriaXEnv()
    jit_step = jax.jit(env.step_env)
    rng = jax.random.PRNGKey(42)

    states = [state]
    for _ in range(steps):
        action = jax.random.randint(rng, (), 0, len(Action))
        rng, subkey = jax.random.split(rng)
        _, state, _, done, _ = jit_step(subkey, state, action, params)
        states.append(state)
        if bool(done):
            break
    return states


def _run_episode_with_machines(steps: int = 5) -> list[EnvState]:
    """Run an episode on a level with pre-placed machines."""
    level = (
        LevelBuilder(7, 7)
        .fill_rect(1, 1, 3, 3, BlockType.IRON, resources=100)
        .place_machine(5, 3, MachineType.CHEST)
        .build("test_machines")
    )
    params = EnvParams(map_width=7, map_height=7, num_players=1, max_timesteps=steps)
    state = build_state(level, params)
    env = FactoriaXEnv()
    jit_step = jax.jit(env.step_env)
    rng = jax.random.PRNGKey(99)

    states = [state]
    for _ in range(steps):
        action = jax.random.randint(rng, (), 0, len(Action))
        rng, subkey = jax.random.split(rng)
        _, state, _, done, _ = jit_step(subkey, state, action, params)
        states.append(state)
        if bool(done):
            break
    return states


class TestStatesToTrajectory:
    """Converting EnvState list to Trajectory preserves all fields."""

    def test_basic_conversion(self) -> None:
        """All state fields should appear in the trajectory."""
        states = _run_episode(steps=5)
        traj = states_to_trajectory(states)

        assert traj.num_episodes == 1
        assert traj.episode_length == len(states)

        # Every state field should be present.
        assert traj.block_map is not None
        assert traj.positions is not None
        assert traj.player_directions is not None
        assert traj.inventory_items is not None
        assert traj.inventory_counts is not None
        assert traj.machine_types is not None
        assert traj.machine_power is not None
        assert traj.machine_direction is not None
        assert traj.achievements is not None
        assert traj.items_mined is not None
        assert traj.selected_player is not None

    def test_shapes(self) -> None:
        """Trajectory arrays should have correct (1, T, ...) shapes."""
        states = _run_episode(steps=3)
        traj = states_to_trajectory(states)
        t_len = len(states)

        assert traj.block_map.shape[:2] == (1, t_len)
        assert traj.positions.shape[:2] == (1, t_len)
        assert traj.machine_types.shape[:2] == (1, t_len)

    def test_with_actions_and_rewards(self) -> None:
        """Actions and rewards should be stored when provided."""
        states = _run_episode(steps=5)
        actions = np.random.randint(0, 15, size=(len(states),))
        rewards = np.random.uniform(size=(len(states),)).astype(np.float32)
        traj = states_to_trajectory(states, actions=actions, rewards=rewards)

        assert traj.actions.shape == (1, len(states))
        assert traj.rewards.shape == (1, len(states))


class TestTrajectoryToStates:
    """Reconstructing EnvState from Trajectory preserves field values."""

    def test_round_trip_field_values(self) -> None:
        """Every field should match after state -> traj -> state."""
        original_states = _run_episode(steps=5)
        traj = states_to_trajectory(original_states)
        reconstructed = trajectory_to_states(traj, episode=0)

        assert len(reconstructed) == len(original_states)

        for t in range(len(original_states)):
            orig = original_states[t]
            recon = reconstructed[t]

            npt.assert_array_equal(
                np.asarray(recon.map),
                np.asarray(orig.map),
                err_msg=f"map mismatch at step {t}",
            )
            npt.assert_array_equal(
                np.asarray(recon.player_positions),
                np.asarray(orig.player_positions),
                err_msg=f"player_positions mismatch at step {t}",
            )
            npt.assert_array_equal(
                np.asarray(recon.player_directions),
                np.asarray(orig.player_directions),
                err_msg=f"player_directions mismatch at step {t}",
            )
            assert int(recon.timestep) == int(orig.timestep), (
                f"timestep mismatch at step {t}"
            )
            npt.assert_array_equal(
                np.asarray(recon.inventory_items),
                np.asarray(orig.inventory_items),
                err_msg=f"inventory_items mismatch at step {t}",
            )
            npt.assert_array_equal(
                np.asarray(recon.inventory_counts),
                np.asarray(orig.inventory_counts),
                err_msg=f"inventory_counts mismatch at step {t}",
            )
            npt.assert_array_equal(
                np.asarray(recon.machine_types),
                np.asarray(orig.machine_types),
                err_msg=f"machine_types mismatch at step {t}",
            )
            npt.assert_array_equal(
                np.asarray(recon.machine_direction),
                np.asarray(orig.machine_direction),
                err_msg=f"machine_direction mismatch at step {t}",
            )
            npt.assert_array_equal(
                np.asarray(recon.block_resources),
                np.asarray(orig.block_resources),
                err_msg=f"block_resources mismatch at step {t}",
            )
            npt.assert_array_equal(
                np.asarray(recon.achievements_unlocked),
                np.asarray(orig.achievements_unlocked),
                err_msg=f"achievements mismatch at step {t}",
            )
            npt.assert_array_equal(
                np.asarray(recon.items_mined),
                np.asarray(orig.items_mined),
                err_msg=f"items_mined mismatch at step {t}",
            )

    def test_round_trip_with_machines(self) -> None:
        """Round-trip works on levels with pre-placed machines."""
        original_states = _run_episode_with_machines(steps=3)
        traj = states_to_trajectory(original_states)
        reconstructed = trajectory_to_states(traj, episode=0)

        for t in range(len(original_states)):
            orig = original_states[t]
            recon = reconstructed[t]
            npt.assert_array_equal(
                np.asarray(recon.machine_types),
                np.asarray(orig.machine_types),
            )
            npt.assert_array_equal(
                np.asarray(recon.machine_inventory_items),
                np.asarray(orig.machine_inventory_items),
            )
            npt.assert_array_equal(
                np.asarray(recon.machine_inventory_counts),
                np.asarray(orig.machine_inventory_counts),
            )


class TestSaveLoadRoundTrip:
    """Save to .npz and load back preserves all fields."""

    def test_save_load_preserves_fields(self, tmp_path: Path) -> None:
        """All fields survive a save/load cycle."""
        states = _run_episode(steps=5)
        traj = states_to_trajectory(states)

        path = str(tmp_path / "test_traj.npz")
        traj.save(path)
        loaded = Trajectory.load(path)

        assert loaded.num_episodes == traj.num_episodes
        assert loaded.episode_length == traj.episode_length
        npt.assert_array_equal(loaded.actions, traj.actions)
        npt.assert_array_equal(loaded.block_map, traj.block_map)
        npt.assert_array_equal(loaded.positions, traj.positions)
        npt.assert_array_equal(loaded.machine_types, traj.machine_types)
        npt.assert_array_equal(loaded.achievements, traj.achievements)

    def test_full_round_trip_through_file(self, tmp_path: Path) -> None:
        """State -> traj -> .npz -> traj -> state preserves values."""
        original_states = _run_episode(steps=3)
        traj = states_to_trajectory(original_states)

        path = str(tmp_path / "full_trip.npz")
        traj.save(path)
        loaded = Trajectory.load(path)
        reconstructed = trajectory_to_states(loaded, episode=0)

        for t in range(len(original_states)):
            orig = original_states[t]
            recon = reconstructed[t]
            npt.assert_array_equal(
                np.asarray(recon.map),
                np.asarray(orig.map),
            )
            npt.assert_array_equal(
                np.asarray(recon.player_positions),
                np.asarray(orig.player_positions),
            )
            assert int(recon.timestep) == int(orig.timestep)


class TestBackwardCompatibility:
    """Old trajectories with fewer fields still load correctly."""

    def test_load_actions_only(self, tmp_path: Path) -> None:
        """A trajectory with only actions should load fine."""
        actions = np.random.randint(0, 15, size=(1, 20))
        path = str(tmp_path / "minimal.npz")
        np.savez_compressed(path, actions=actions)
        traj = Trajectory.load(path)
        assert traj.block_map is None
        assert traj.positions is None
        npt.assert_array_equal(traj.actions, actions)
