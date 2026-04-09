"""Replay recorded actions through a level to produce rendered frames.

Given a level and an action sequence from a trajectory, steps through
the environment and captures an RGB frame at each timestep using the
game's own ``render_pixels`` function.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from factoriax.analysis.trajectory import Trajectory
from factoriax.levels import Level, build_state, load_level
from factoriax.renderer import render_pixels
from factoriax.state import EnvParams


def replay_frames(
    level: Level,
    traj: Trajectory,
    episode: int = 0,
    player: int = 0,
    tile_size: int = 24,
) -> list[np.ndarray]:
    """Replay a trajectory's actions on a level and capture rendered frames.

    Steps the environment using the recorded actions and calls
    ``render_pixels`` at each step to produce RGB images of the game
    world.

    Args:
        level: Level to replay on.
        traj: Trajectory containing the action sequence.
        episode: Episode index within the trajectory.
        player: Player index whose actions to replay (for multi-player
            trajectories with sequential player cycling, only player 0's
            actions drive the environment).
        tile_size: Pixel size per tile for rendering.

    Returns:
        List of RGB uint8 arrays, one per timestep plus a final frame.
    """
    from factoriax.envs import FactoriaXEnv

    params = EnvParams(
        map_width=level.map_width,
        map_height=level.map_height,
        num_players=1,
        max_timesteps=traj.episode_length,
    )
    state = build_state(level, params)

    env = FactoriaXEnv()
    jit_step = jax.jit(env.step_env)
    rng = jax.random.PRNGKey(0)

    ep = traj.episode(episode)
    if ep.is_multi_player:
        actions = ep.actions[0, :, player]
    else:
        actions = ep.actions[0]

    frames: list[np.ndarray] = []
    for t in range(len(actions)):
        frames.append(render_pixels(state, block_pixel_size=tile_size))
        action = jnp.int32(actions[t])
        rng, subkey = jax.random.split(rng)
        _, state, _, done, _ = jit_step(subkey, state, action, params)
        if bool(done):
            break

    frames.append(render_pixels(state, block_pixel_size=tile_size))
    return frames


def load_replay_frames(
    level_path: str,
    traj: Trajectory,
    episode: int = 0,
    tile_size: int = 24,
) -> list[np.ndarray]:
    """Load a level from disk and replay to get frames.

    Convenience wrapper around :func:`replay_frames` that handles
    level loading.

    Args:
        level_path: Path to a level JSON file.
        traj: Trajectory containing the action sequence.
        episode: Episode index.
        tile_size: Pixel size per tile.

    Returns:
        List of RGB uint8 arrays.
    """
    level = load_level(Path(level_path))
    return replay_frames(level, traj, episode, tile_size=tile_size)
