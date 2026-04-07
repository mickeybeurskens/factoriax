"""Tests for agentdebugger chart rendering."""

import numpy as np

from factoriax.agentdebugger.charts import (
    build_partial_trajectory,
    render_cost_chart,
)


class TestBuildPartialTrajectory:
    """Tests for build_partial_trajectory."""

    def test_basic_shape(self) -> None:
        """Trajectory has batch=1 and correct time dimension."""
        traj = build_partial_trajectory(
            rewards=[1.0, 2.0, 3.0],
            actions=[0, 1, 2],
        )
        assert traj.actions.shape == (1, 3)
        assert traj.rewards is not None
        assert traj.rewards.shape == (1, 3)

    def test_dtypes(self) -> None:
        """Actions are int32, rewards are float32."""
        traj = build_partial_trajectory(
            rewards=[0.5], actions=[7],
        )
        assert traj.actions.dtype == np.int32
        assert traj.rewards is not None
        assert traj.rewards.dtype == np.float32

    def test_empty_lists(self) -> None:
        """Empty inputs produce a trajectory with time=1."""
        traj = build_partial_trajectory(rewards=[], actions=[])
        assert traj.actions.shape == (1, 1)
        assert traj.rewards is not None
        assert traj.rewards.shape == (1, 1)

    def test_mismatched_lengths_padded(self) -> None:
        """Shorter array is zero-padded to match the longer one."""
        traj = build_partial_trajectory(
            rewards=[1.0, 2.0, 3.0],
            actions=[0],
        )
        assert traj.actions.shape == (1, 3)
        assert traj.rewards is not None
        assert traj.rewards.shape == (1, 3)

    def test_values_preserved(self) -> None:
        """Input values appear in the trajectory arrays."""
        traj = build_partial_trajectory(
            rewards=[10.0, 20.0],
            actions=[3, 7],
        )
        np.testing.assert_array_equal(traj.actions[0], [3, 7])
        assert traj.rewards is not None
        np.testing.assert_array_almost_equal(
            traj.rewards[0], [10.0, 20.0],
        )


class TestRenderCostChart:
    """Tests for render_cost_chart."""

    def test_output_shape_and_dtype(self) -> None:
        """Returns RGB uint8 array of requested dimensions."""
        costs = [np.array([0.5, 0.1]), np.array([0.3, 0.2])]
        img = render_cost_chart(costs, ["a", "b"], 200, 100)
        assert img.shape == (100, 200, 3)
        assert img.dtype == np.uint8

    def test_empty_costs(self) -> None:
        """Empty cost list produces a valid placeholder image."""
        img = render_cost_chart([], [], 200, 100)
        assert img.shape == (100, 200, 3)
        assert img.dtype == np.uint8

    def test_not_all_black(self) -> None:
        """Chart with data should have non-zero pixels."""
        costs = [np.array([1.0]) for _ in range(10)]
        img = render_cost_chart(costs, ["x"], 200, 100)
        assert img.sum() > 0

    def test_plot_bounds_embedded(self) -> None:
        """Plot area fractions are encoded in the top-left pixels."""
        costs = [np.array([0.5]) for _ in range(5)]
        img = render_cost_chart(costs, ["x"], 300, 150)
        frac_x0 = int(img[0, 0, 0])
        frac_x1 = int(img[0, 1, 0])
        assert frac_x0 > 0
        assert frac_x1 > frac_x0
