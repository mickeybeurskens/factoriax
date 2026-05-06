"""Fault-analysis dump for the advanced-factory agent.

Runs the agent with a wrapped planner that records ``repr(goal)`` on
every FAIL / VERIFY event, then prints:

- a chronological timeline keyed by goal repr (so we know *which*
  placements were starving for inventory and which crafts FAILed);
- per-goal-class FAIL_GIVEUP counts;
- the most recent verify-failure diagnostic block;
- the player's final inventory by item.

Run via ``uv run python scripts/fault_advanced_factory.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from baselines.rocket.scripted.agent_advanced_factory import (
    build_advanced_factory_goals,
    make_advanced_factory_rocket_agent,
)
from baselines.rocket.scripted.layout import (
    diff_layout,
    expected_layout_from_goals,
)
from baselines.rocket.scripted.planner import Planner
from baselines.rocket.scripted.world_model import decode_observation
from factoriax.benchmarks.rocket import (
    ROCKET_BLOCKED_ACTIONS,
    ROCKET_RECIPE_BOOK,
    ROCKET_RECIPE_TABLE,
    build_rocket_level,
    rocket_conditions,
)
from factoriax.constants import MAX_ACHIEVEMENTS, ItemType, MachineType
from factoriax.envs import FactoriaXEnv
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.levels import build_state
from factoriax.observations import global_array
from factoriax.state import EnvParams

_MAP_SIZE = 32
_FAIL_KINDS = {
    "FAIL_RETRY",
    "FAIL_GIVEUP",
    "VERIFY_HALT",
    "VERIFY_RETRY",
    "VERIFY_GIVEUP",
    "VERIFY_IGNORE",
}


def _instrument(planner: Planner) -> list[tuple[int, str, str, str]]:
    """Wrap planner.step so every event captures ``repr(self._current.goal)``."""
    rich_events: list[tuple[int, str, str, str]] = []
    real_log = planner._log  # type: ignore[attr-defined]

    def rich_log(kind: str, name: str) -> None:
        goal_repr = ""
        if planner._current is not None:  # type: ignore[attr-defined]
            goal_repr = repr(planner._current.goal)  # type: ignore[attr-defined]
        rich_events.append(
            (planner._tick, kind, name, goal_repr)  # type: ignore[attr-defined]
        )
        real_log(kind, name)

    planner._log = rich_log  # type: ignore[attr-defined]
    return rich_events


def main() -> None:
    """Run the agent and dump fault-analysis information."""
    max_steps = 12000
    env_params = EnvParams(
        map_width=_MAP_SIZE,
        map_height=_MAP_SIZE,
        num_players=1,
        max_timesteps=max_steps,
        recipe_table=ROCKET_RECIPE_TABLE,
    )
    level = build_rocket_level()
    state = build_state(level, env_params)
    env = ActionMaskWrapper(
        FactoriaXEnv(achievement_fn=rocket_conditions),
        ROCKET_BLOCKED_ACTIONS,
    )
    jit_step = jax.jit(env.step_env)
    jit_obs = jax.jit(lambda s: global_array(s, env_params, 0))

    agent = make_advanced_factory_rocket_agent(env_params, book=ROCKET_RECIPE_BOOK)
    agent.planner._debug = True
    rich_events = _instrument(agent.planner)

    key = jax.random.PRNGKey(0)
    last_state = state
    final_tick = 0
    for t in range(max_steps):
        obs = np.asarray(jit_obs(last_state))
        action = agent.act(obs)
        key, subkey = jax.random.split(key)
        _, last_state, _, done, _ = jit_step(
            subkey, last_state, jnp.int32(action), env_params
        )
        final_tick = t + 1
        if agent.is_done or bool(done):
            break
        if agent.planner.is_done:
            break
    env_state = last_state

    print(f"=== Run terminated after {final_tick} ticks ===")
    print(f"Total events recorded: {len(rich_events)}\n")

    print("=== Phase boundaries (first START tick per goal class) ===")
    seen_phase: dict[str, int] = {}
    for tick, kind, name, _repr in rich_events:
        if kind != "START":
            continue
        if name not in seen_phase:
            seen_phase[name] = tick
    for name, tick in sorted(seen_phase.items(), key=lambda kv: kv[1]):
        print(f"  tick={tick:>5}  first START  {name}")
    print(f"  tick={final_tick:>5}  end\n")

    print("=== FAIL/VERIFY timeline (one line per event, with repr) ===")
    fail_events = [(t, k, n, r) for t, k, n, r in rich_events if k in _FAIL_KINDS]
    if not fail_events:
        print("  (none)")
    for tick, kind, name, goal_repr in fail_events:
        # Compress repr for readability
        short = goal_repr[:120]
        print(f"  tick={tick:>5}  {kind:<14}  {short}")
    print()

    print("=== FAIL_GIVEUP / VERIFY_HALT distinct goal reprs ===")
    halt_repr_counts: dict[str, int] = {}
    for tick, kind, name, goal_repr in rich_events:
        if kind in {"FAIL_GIVEUP", "VERIFY_HALT", "VERIFY_GIVEUP"}:
            halt_repr_counts[goal_repr] = halt_repr_counts.get(goal_repr, 0) + 1
    for goal_repr, c in sorted(halt_repr_counts.items(), key=lambda kv: -kv[1]):
        print(f"  x{c:<3}  {goal_repr}")
    print()

    diag = agent.planner.verify_diagnostic
    print("=== Most recent verify-failure diagnostic ===")
    if diag is None:
        print("  (none)")
    else:
        print(diag.format())
    print()

    print("=== Player final inventory (non-zero) ===")
    inv = np.asarray(env_state.player_inventory[0])
    for item_idx, count in enumerate(inv):
        if int(count) > 0:
            print(f"  {ItemType(item_idx).name:<22} {int(count)}")
    print()

    final_view = decode_observation(
        np.asarray(jit_obs(env_state)),
        env_params.map_height,
        env_params.map_width,
        env_params.max_timesteps,
    )
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    expected_from_goals = expected_layout_from_goals(goals)
    pre_placed = {
        (15, 16): (int(MachineType.FURNACE), 4),
        (17, 16): (int(MachineType.ASSEMBLER), 4),
    }
    expected_layout = {**expected_from_goals, **pre_placed}
    layout_diff = diff_layout(final_view, expected_layout)
    print("=== Layout diff summary ===")
    by_kind: dict[str, int] = {}
    for m in layout_diff:
        by_kind[m.kind] = by_kind.get(m.kind, 0) + 1
    print(
        f"  total={len(layout_diff)} "
        f"({', '.join(f'{k}={v}' for k, v in sorted(by_kind.items()))})"
    )
    print()

    mask = np.asarray(last_state.achievements_unlocked)
    print(f"=== Achievements unlocked: {int(mask.sum())} / {MAX_ACHIEVEMENTS} ===")


if __name__ == "__main__":
    main()
