"""Build a :class:`Trajectory` for an analysis test.

Four test files under ``tests/analysis/`` share these builders. A builder
returns a trajectory with random but seeded contents, so a test that asserts a
shape or a plot contract needs one line of setup.

The builders carry no assertion of their own. A test that needs a specific
value builds its own trajectory.
"""

from __future__ import annotations

import numpy as np

from factoriax.analysis.trajectory import Trajectory


def make_single_player_traj(
    num_eps: int, num_steps: int, num_actions: int = 6, seed: int = 0
) -> Trajectory:
    """Build a (num_eps, num_steps) single-player trajectory with random actions.

    Parameters
    ----------
    num_eps
        Number of episodes.
    num_steps
        Episode length.
    num_actions
        Action space size.
    seed
        RNG seed for reproducibility.

    Returns
    -------
    Trajectory
        Single-player trajectory with integer actions in
        ``[0, num_actions)``.
    """
    rng = np.random.default_rng(seed)
    actions = rng.integers(0, num_actions, size=(num_eps, num_steps)).astype(np.int32)
    return Trajectory(actions=actions)


def make_multi_player_traj(
    num_eps: int,
    num_steps: int,
    num_p: int = 2,
    num_actions: int = 6,
    seed: int = 0,
) -> Trajectory:
    """Build a multi-player trajectory with all optional fields populated.

    Parameters
    ----------
    num_eps
        Number of episodes.
    num_steps
        Episode length.
    num_p
        Number of players.
    num_actions
        Action space size.
    seed
        RNG seed for reproducibility.

    Returns
    -------
    Trajectory
        Multi-player trajectory with actions, positions, inventory,
        achievements, rewards, and timesteps.
    """
    rng = np.random.default_rng(seed)
    actions = rng.integers(0, num_actions, size=(num_eps, num_steps, num_p)).astype(
        np.int32
    )
    positions = rng.integers(0, 32, size=(num_eps, num_steps, num_p, 2)).astype(
        np.int32
    )
    player_inventory = rng.integers(0, 20, size=(num_eps, num_steps, num_p, 3)).astype(
        np.int32
    )
    achievements = np.zeros((num_eps, num_steps, 2), dtype=bool)
    achievements[:, num_steps // 2 :, 0] = True
    achievements[:, num_steps // 4 :, 1] = True
    rewards = rng.standard_normal(size=(num_eps, num_steps)).astype(np.float32)
    timesteps = (
        np.broadcast_to(np.arange(num_steps), (num_eps, num_steps))
        .copy()
        .astype(np.int32)
    )
    return Trajectory(
        actions=actions,
        positions=positions,
        player_inventory=player_inventory,
        achievements=achievements,
        rewards=rewards,
        timesteps=timesteps,
    )


def make_minimal_traj(num_eps: int, num_steps: int) -> Trajectory:
    """Build a trajectory with only zero actions and no optional fields.

    Parameters
    ----------
    num_eps
        Number of episodes.
    num_steps
        Episode length.

    Returns
    -------
    Trajectory
        Minimal trajectory for error-condition tests.
    """
    return Trajectory(actions=np.zeros((num_eps, num_steps), dtype=np.int32))

