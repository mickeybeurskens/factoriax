"""End-of-run diagnostic for the M4 advanced-factory agent.

Runs the agent and dumps:
- the iron-cell area + coal trunk (rows 9-11, cols 0-10) tile-by-tile
- the goal log (FAIL_RETRY / FAIL_GIVEUP / VERIFY_* events)
- the layout diff (count by kind)
- the player's final inventory by item

Run via ``uv run python scripts/diagnose_advanced_factory.py``.
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
from baselines.rocket.scripted.world_model import decode_observation
from factoriax.benchmarks.rocket import (
    ROCKET_BLOCKED_ACTIONS,
    ROCKET_RECIPE_BOOK,
    ROCKET_RECIPE_TABLE,
    build_rocket_level,
    rocket_conditions,
)
from factoriax.constants import Direction, ItemType, MachineType
from factoriax.envs import FactoriaXEnv
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.levels import build_state
from factoriax.observations import global_array
from factoriax.state import EnvParams

_MAP_SIZE = 32


def main() -> None:
    """Run the agent, dump diagnostics to stdout."""
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

    env_state = last_state

    print(
        f"\n=== Phase 1 cell area dump (rows 9-21, cols 0-10) "
        f"after {final_tick} ticks ==="
    )
    machine_types = np.asarray(env_state.machine_types)
    ent_direction = np.asarray(env_state.ent_direction)
    tile_entity = np.asarray(env_state.tile_entity)
    ent_buf_type = np.asarray(env_state.ent_buf_type)
    ent_buf_count = np.asarray(env_state.ent_buf_count)
    ent_asm_in_type = np.asarray(env_state.ent_asm_in_type)
    ent_asm_in_count = np.asarray(env_state.ent_asm_in_count)
    ent_asm_out_type = np.asarray(env_state.ent_asm_out_type)
    ent_asm_out_count = np.asarray(env_state.ent_asm_out_count)
    for y in range(9, 22):
        for x in range(0, 19):
            mt = int(machine_types[y, x])
            if mt == 0:
                continue
            mt_name = MachineType(mt).name
            eidx = int(tile_entity[y, x])
            d = int(ent_direction[eidx]) if eidx >= 0 else 0
            d_name = Direction(d).name if d != 0 else "NONE"
            buf_type = int(ent_buf_type[eidx]) if eidx >= 0 else 0
            buf_count = int(ent_buf_count[eidx]) if eidx >= 0 else 0
            buf_name = ItemType(buf_type).name if buf_type != 0 else "EMPTY"
            extra = f"buf=({buf_name}, {buf_count})"
            if (
                mt in (int(MachineType.FURNACE), int(MachineType.ASSEMBLER))
                and eidx >= 0
            ):
                in0_t = int(ent_asm_in_type[eidx, 0])
                in0_c = int(ent_asm_in_count[eidx, 0])
                in1_t = int(ent_asm_in_type[eidx, 1])
                in1_c = int(ent_asm_in_count[eidx, 1])
                out_t = int(ent_asm_out_type[eidx])
                out_c = int(ent_asm_out_count[eidx])
                power = int(np.asarray(env_state.ent_power)[eidx])
                in0_name = ItemType(in0_t).name if in0_t != 0 else "EMPTY"
                in1_name = ItemType(in1_t).name if in1_t != 0 else "EMPTY"
                out_name = ItemType(out_t).name if out_t != 0 else "EMPTY"
                extra = (
                    f"asm_in=[({in0_name}, {in0_c}), ({in1_name}, {in1_c})] "
                    f"asm_out=({out_name}, {out_c}) power={power}"
                )
            print(f"  ({x:>2}, {y:>2}) {mt_name:<14} {d_name:<5}  {extra}")

    print("\n=== FAIL / VERIFY events ===")
    fail_kinds = {
        "FAIL_RETRY",
        "FAIL_GIVEUP",
        "VERIFY_HALT",
        "VERIFY_RETRY",
        "VERIFY_GIVEUP",
        "VERIFY_IGNORE",
    }
    counts: dict[tuple[str, str], int] = {}
    for _tick, kind, name in agent.planner._events:
        if kind in fail_kinds:
            counts[(kind, name)] = counts.get((kind, name), 0) + 1
    if not counts:
        print("  (none)")
    for (kind, name), n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {kind:<13} {name:<30} x{n}")

    final_view = decode_observation(
        np.asarray(jit_obs(env_state)),
        env_params.map_height,
        env_params.map_width,
        env_params.max_timesteps,
    )
    goals = build_advanced_factory_goals(book=ROCKET_RECIPE_BOOK)
    expected_from_goals = expected_layout_from_goals(goals)
    pre_placed = {
        (15, 16): (int(MachineType.FURNACE), int(Direction.DOWN)),
        (17, 16): (int(MachineType.ASSEMBLER), int(Direction.DOWN)),
    }
    expected_layout = {**expected_from_goals, **pre_placed}
    layout_diff = diff_layout(final_view, expected_layout)
    print("\n=== Layout diff ===")
    by_kind: dict[str, int] = {}
    for m in layout_diff:
        by_kind[m.kind] = by_kind.get(m.kind, 0) + 1
    print(
        f"  total={len(layout_diff)} "
        f"({', '.join(f'{k}={v}' for k, v in sorted(by_kind.items()))})"
    )
    for m in layout_diff[:20]:
        print(f"  {m}")

    print("\n=== Player final inventory (non-zero) ===")
    inv = np.asarray(env_state.player_inventory[0])
    for item_idx, count in enumerate(inv):
        if int(count) > 0:
            print(f"  {ItemType(item_idx).name:<22} {int(count)}")

    mask = np.asarray(last_state.achievements_unlocked)
    print(f"\n=== Achievements unlocked: {int(mask.sum())} ===")


if __name__ == "__main__":
    main()
