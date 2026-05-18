"""Replay recorded actions through a level to reconstruct env states.

Given a level and an action sequence, steps the environment and
captures an :class:`EnvState` at each timestep. The debugger renders
those states on demand in :func:`render_replay_frame`, so this module
no longer owns the render step — the state list is ~45x cheaper to
hold in memory than pre-rendered frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from factoriax.analysis.trajectory import Trajectory
from factoriax.levels import Level, build_state, load_level
from factoriax.state import EnvParams, EnvState


def replay_states(
    level: Level,
    traj: Trajectory,
    episode: int = 0,
    player: int = 0,
) -> list[EnvState]:
    """Replay a trajectory's actions on a level and collect env states.

    Steps the environment using the recorded actions and snapshots the
    :class:`EnvState` at each timestep. Callers render frames on demand
    — holding a full rendered-frame list for long trajectories is the
    OOM source this module used to be.

    Args:
        level: Level to replay on.
        traj: Trajectory containing the action sequence.
        episode: Episode index within the trajectory.
        player: Player index whose actions to replay (for multi-player
            trajectories with sequential player cycling, only player 0's
            actions drive the environment).

    Returns:
        List of :class:`EnvState`, one per timestep plus a final state
        captured after the last action.
    """
    from factoriax.envs import FactoriaXEnv

    # Pull engine knobs from the trajectory's captured scheme when
    # present. ``num_players`` is no longer hardcoded — multi-player
    # recordings replay against the correct player count, and engine
    # knobs like ``player_mining_yield`` flow through so ``items_mined``
    # reproduces bit-identically.
    scheme: dict[str, Any] = traj.env_params_scheme or {}

    def _scheme_int(name: str, default: int) -> int:
        val = scheme.get(name)
        return int(val) if val is not None else default

    params = EnvParams(
        map_width=_scheme_int("map_width", level.map_width),
        map_height=_scheme_int("map_height", level.map_height),
        num_players=_scheme_int("num_players", 1),
        max_timesteps=_scheme_int("max_timesteps", traj.episode_length),
        miner_mining_rate=_scheme_int(
            "miner_mining_rate", EnvParams().miner_mining_rate
        ),
        player_mining_yield=_scheme_int(
            "player_mining_yield", EnvParams().player_mining_yield
        ),
        base_resources=_scheme_int("base_resources", EnvParams().base_resources),
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

    states: list[EnvState] = []
    for t in range(len(actions)):
        states.append(state)
        action = jnp.int32(actions[t])
        rng, subkey = jax.random.split(rng)
        _, state, _, done, _ = jit_step(subkey, state, action, params)
        if bool(done):
            break

    states.append(state)
    return states


def load_replay_states(
    level_path: str,
    traj: Trajectory,
    episode: int = 0,
) -> list[EnvState]:
    """Load a level from disk and replay to reconstruct env states.

    Convenience wrapper around :func:`replay_states`.

    Args:
        level_path: Path to a level JSON file.
        traj: Trajectory containing the action sequence.
        episode: Episode index.

    Returns:
        List of :class:`EnvState` snapshots.
    """
    level: Any = load_level(Path(level_path))
    return replay_states(level, traj, episode)
