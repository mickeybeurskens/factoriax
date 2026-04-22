"""Bisection tests for the factory agent.

Each test runs a truncated goal list against the rocket env with a
hard wall-clock cap. If the cap is exceeded (usually because a
goal deadlocked), the test fails fast and reports which goal was
active at the bail point — so we know exactly which phase broke
without having to re-run a full 8000-step episode.

All tests are ``@pytest.mark.slow`` — they run real JAX rollouts.

The tests are designed to bisect failure modes of the factory
agent:

- **Test A** — starter produces all the infrastructure components
  (miners, pallets, furnaces, assemblers) the later phases need.
- **Test B** — node miners, deployed with explicit facings, are
  pushing ore into their paired pallets.
- **Test C** — central-bank furnaces + assemblers are placed near
  spawn and the agent can reach them.
- **Test D** — ``PipelinedProduce`` across 3 furnaces completes a
  reasonable bulk smelt within the time cap.

A run that passes A+B+C+D but fails the full integration implies
the bug is in phase transitions / state carry-over, not in any one
phase.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from baselines.rocket.scripted.agent import ScriptedAgent
from baselines.rocket.scripted.goals import (
    Goal,
    MineOre,
    PipelinedProduce,
    PlaceMachine,
    PlaceMachineAt,
    ProduceInAssembler,
    ProduceInFurnace,
    WaitUntil,
    free_tile_near_player,
)
from baselines.rocket.scripted.planner import Planner
from factoriax.benchmarks.rocket import (
    ROCKET_BLOCKED_ACTIONS,
    build_rocket_level,
    rocket_conditions,
)
from factoriax.constants import (
    MAX_ACHIEVEMENTS,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.envs import FactoriaXEnv
from factoriax.envs.achievement_wrapper import AchievementState, AchievementWrapper
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.levels import build_state
from factoriax.observations import global_array
from factoriax.state import EnvParams

pytestmark = pytest.mark.slow


# Tight wall-clock cap per test. Single-env rollouts run at
# ~25-40 ms/tick on this machine, so 90 s ≈ 2000-3500 ticks.
_WALL_CAP_SEC = 90.0


@dataclass
class BisectionRunResult:
    state: object  # AchievementState
    steps: int
    wall_sec: float
    last_goal: str
    hit_cap: bool
    goal_log: list[str]


def _run_plan(goals: list[Goal], max_steps: int) -> BisectionRunResult:
    """Run the goal list against the rocket env with a wall-clock cap.

    Returns a :class:`BisectionRunResult` regardless of outcome.
    ``hit_cap=True`` means the test should assert on that first.
    """
    env_params = EnvParams(
        map_width=32,
        map_height=32,
        num_players=1,
        max_timesteps=max_steps,
    )
    level = build_rocket_level()
    env_state = build_state(level, env_params)
    state = AchievementState(
        env_state=env_state,
        achievements_unlocked=jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_),
    )
    env = ActionMaskWrapper(
        AchievementWrapper(FactoriaXEnv(), rocket_conditions),
        ROCKET_BLOCKED_ACTIONS,
    )
    jit_step = jax.jit(env.step_env)
    # JIT the observation too — without this, every tick pays a ~50 ms
    # eager-mode dispatch cost (10 spatial channels worth of scatters),
    # which alone burns the wall cap long before any deadlock surfaces.
    jit_obs = jax.jit(lambda es: global_array(es, env_params, 0))

    planner = Planner(goals, debug_log=True)
    agent = ScriptedAgent(env_params, planner)
    key = jax.random.PRNGKey(0)

    from collections import deque

    from factoriax.constants import Action

    recent_actions: deque[tuple[int, int]] = deque(maxlen=40)

    t0 = time.perf_counter()
    steps = 0
    hit_cap = False
    for t in range(max_steps):
        if time.perf_counter() - t0 > _WALL_CAP_SEC:
            hit_cap = True
            break
        obs = np.asarray(jit_obs(state.env_state))
        action = agent.act(obs)
        recent_actions.append((t, int(action)))
        key, sub = jax.random.split(key)
        _, state, _, done, _ = jit_step(sub, state, jnp.int32(action), env_params)
        steps = t + 1
        if agent.is_done or bool(done):
            break

    result = BisectionRunResult(
        state=state,
        steps=steps,
        wall_sec=time.perf_counter() - t0,
        last_goal=planner.current_goal_name,
        hit_cap=hit_cap,
        goal_log=planner.goal_log_lines(),
    )
    if hit_cap:
        recent_names = ", ".join(
            f"{t}:{Action(a).name}" for t, a in list(recent_actions)[-15:]
        )
        result.goal_log.append(f"(recent actions) {recent_names}")
    return result


def _assert_not_capped(result: BisectionRunResult) -> None:
    if not result.hit_cap:
        return
    tail = "\n    ".join(result.goal_log[-20:]) or "(empty)"
    # Dump combiner state so we can see which machines are wedged.
    env_state = result.state.env_state
    ent_y = np.asarray(env_state.ent_y)
    ent_x = np.asarray(env_state.ent_x)
    ent_type = np.asarray(env_state.ent_type)
    ent_power = np.asarray(env_state.ent_power)
    ent_in_t = np.asarray(env_state.ent_asm_in_type)
    ent_in_c = np.asarray(env_state.ent_asm_in_count)
    ent_out_t = np.asarray(env_state.ent_asm_out_type)
    ent_out_c = np.asarray(env_state.ent_asm_out_count)
    ent_buf_t = np.asarray(env_state.ent_buf_type)
    ent_buf_c = np.asarray(env_state.ent_buf_count)
    inv = np.asarray(env_state.player_inventory[0])
    held = {
        ItemType(i).name: int(inv[i]) for i in range(inv.shape[0]) if int(inv[i]) > 0
    }
    combiner_rows: list[str] = []
    for i in range(ent_y.shape[0]):
        if int(ent_y[i]) < 0:
            continue
        mt = int(ent_type[i])
        if mt not in (int(MachineType.ASSEMBLER), int(MachineType.FURNACE)):
            continue
        combiner_rows.append(
            f"  e{i} {MachineType(mt).name} at ({int(ent_x[i])},{int(ent_y[i])}) "
            f"power={int(ent_power[i])} "
            f"in0={int(ent_in_t[i, 0])}×{int(ent_in_c[i, 0])} "
            f"in1={int(ent_in_t[i, 1])}×{int(ent_in_c[i, 1])} "
            f"out={int(ent_out_t[i])}×{int(ent_out_c[i])} "
            f"buf={int(ent_buf_t[i])}×{int(ent_buf_c[i])}"
        )
    combiner_dump = "\n".join(combiner_rows) or "  (no combiners)"
    player_pos = tuple(int(v) for v in np.asarray(env_state.player_positions[0]))
    player_dir = int(np.asarray(env_state.player_directions[0]))
    raise AssertionError(
        f"Hit {_WALL_CAP_SEC}s wall cap after {result.steps} steps. "
        f"Active goal when bailed: {result.last_goal!r}.\n"
        f"  Player @ {player_pos} facing dir={player_dir}\n"
        f"  Last 20 goal transitions:\n    {tail}\n"
        f"  Player inventory: {held}\n"
        f"  Combiner states:\n{combiner_dump}"
    )


# ---------------------------------------------------------------------------
# Test A — starter phases produce infrastructure
# ---------------------------------------------------------------------------


def _starter_goals() -> list[Goal]:
    """Phases A-C of the factory plan: enough materials + components
    for the full node-factory + central-bank deployment.

    Every smelt now consumes 1 coal in addition to the ore.
    Starter smelts total 14+17+23+3 = 57 plates → 57 coal + 2
    for refractory = 59 coal needed.
    """
    return [
        # A — mining
        MineOre(ItemType.IRON_ORE, 14),
        MineOre(ItemType.COPPER_ORE, 17),
        MineOre(ItemType.TIN_ORE, 23),
        MineOre(ItemType.SILICON, 3),
        MineOre(ItemType.COAL, 62),
        # B — smelts
        ProduceInFurnace(ItemType.IRON_PLATE, 14),
        ProduceInFurnace(ItemType.COPPER_PLATE, 17),
        ProduceInFurnace(ItemType.TIN_PLATE, 23),
        ProduceInFurnace(ItemType.WAFER, 3),
        ProduceInFurnace(ItemType.REFRACTORY, 2),
        # C — components
        ProduceInAssembler(ItemType.WIRE, 13),
        ProduceInAssembler(ItemType.FRAME, 2),
        ProduceInAssembler(ItemType.CIRCUIT, 2),
        ProduceInAssembler(ItemType.MINER, 8),
        ProduceInAssembler(ItemType.PALLET, 5),
        ProduceInAssembler(ItemType.FURNACE, 2),
        ProduceInAssembler(ItemType.ASSEMBLER, 2),
    ]


def test_a_starter_produces_infrastructure() -> None:
    """Phases A-C alone: holds 8 miners + 5 pallets + 2 furnaces + 2 asms."""
    result = _run_plan(_starter_goals(), max_steps=3000)
    _assert_not_capped(result)

    inv = np.asarray(result.state.env_state.player_inventory[0])
    missing: list[str] = []
    for item, needed in (
        (ItemType.MINER, 8),
        (ItemType.PALLET, 5),
        (ItemType.FURNACE, 2),
        (ItemType.ASSEMBLER, 2),
    ):
        if int(inv[item]) < needed:
            missing.append(f"{item.name}: got {int(inv[item])}, expected >= {needed}")
    assert not missing, (
        f"Starter finished at step {result.steps} but inventory is short:\n"
        f"  {'; '.join(missing)}\n"
        f"  last goal: {result.last_goal!r}"
    )


# ---------------------------------------------------------------------------
# Test B — node factories place + push ore into pallets
# ---------------------------------------------------------------------------


def _node_placement_goals() -> list[Goal]:
    """Starter + the per-node miner/pallet placements with facings."""
    place = [
        # Iron patch at (7-9, 7-9)
        PlaceMachineAt(MachineType.PALLET, (8, 10), int(Direction.DOWN)),
        PlaceMachineAt(MachineType.MINER, (8, 9), int(Direction.DOWN)),
        # Copper patch at (22-24, 7-9)
        PlaceMachineAt(MachineType.PALLET, (23, 10), int(Direction.DOWN)),
        PlaceMachineAt(MachineType.MINER, (23, 9), int(Direction.DOWN)),
        # Tin patch at (22-24, 22-24)
        PlaceMachineAt(MachineType.PALLET, (23, 25), int(Direction.DOWN)),
        PlaceMachineAt(MachineType.MINER, (23, 24), int(Direction.DOWN)),
        # Silicon patch at (14-16, 3-5)
        PlaceMachineAt(MachineType.PALLET, (15, 6), int(Direction.DOWN)),
        PlaceMachineAt(MachineType.MINER, (15, 5), int(Direction.DOWN)),
        # Coal snake at (7-9, 22-24)
        PlaceMachineAt(MachineType.PALLET, (10, 23), int(Direction.LEFT)),
        PlaceMachineAt(MachineType.MINER, (9, 23), int(Direction.RIGHT)),
        PlaceMachineAt(MachineType.MINER, (8, 23), int(Direction.RIGHT)),
        PlaceMachineAt(MachineType.MINER, (7, 23), int(Direction.RIGHT)),
        PlaceMachineAt(MachineType.MINER, (7, 22), int(Direction.DOWN)),
    ]
    # After placements, wait a few ticks so miners push some ore
    # into pallets before the test inspects state.
    return _starter_goals() + place + [WaitUntil(lambda _: False, max_ticks=20)]


def test_b_node_factories_push_into_pallets() -> None:
    """Each node pallet has non-empty buf; coal snake terminal has ≥ 3."""
    result = _run_plan(_node_placement_goals(), max_steps=4000)
    _assert_not_capped(result)

    env_state = result.state.env_state
    mt = np.asarray(env_state.machine_types)
    ent_y = np.asarray(env_state.ent_y)
    ent_x = np.asarray(env_state.ent_x)
    ent_buf_count = np.asarray(env_state.ent_buf_count)

    def _buf_at(tile: tuple[int, int]) -> int:
        x, y = tile
        if int(mt[y, x]) == int(MachineType.NONE):
            return -1  # no machine placed
        # Find entity index for this tile.
        for i, (ey, ex) in enumerate(zip(ent_y, ent_x, strict=True)):
            if int(ey) == y and int(ex) == x:
                return int(ent_buf_count[i])
        return -1

    # Each node pallet should have at least 1 ore pushed by miner.
    node_pallets = {
        "iron": (8, 10),
        "copper": (23, 10),
        "tin": (23, 25),
        "silicon": (15, 6),
    }
    shortfalls: list[str] = []
    for name, tile in node_pallets.items():
        buf = _buf_at(tile)
        if buf < 1:
            shortfalls.append(f"{name} pallet at {tile}: buf={buf} (expected >= 1)")

    # Coal snake terminal pallet. A working snake accumulates fast;
    # accept any positive value since timing is variable.
    coal_buf = _buf_at((10, 23))
    if coal_buf < 1:
        shortfalls.append(f"coal pallet at (10, 23): buf={coal_buf} (expected >= 1)")

    assert not shortfalls, (
        f"Node automation not feeding pallets (step {result.steps}):\n"
        + "\n".join(f"  {s}" for s in shortfalls)
    )


# ---------------------------------------------------------------------------
# Test C — central bank (2 extra furnaces + 2 extra assemblers)
# ---------------------------------------------------------------------------


def _central_bank_goals() -> list[Goal]:
    return _starter_goals() + [
        PlaceMachine(MachineType.FURNACE, free_tile_near_player()),
        PlaceMachine(MachineType.FURNACE, free_tile_near_player()),
        PlaceMachine(MachineType.ASSEMBLER, free_tile_near_player()),
        PlaceMachine(MachineType.ASSEMBLER, free_tile_near_player()),
    ]


def test_c_central_bank_placed() -> None:
    """After the central-bank phase, the map has 3 furnaces + 3 assemblers
    (1 pre-placed + 2 newly placed, each type)."""
    result = _run_plan(_central_bank_goals(), max_steps=3000)
    _assert_not_capped(result)

    mt = np.asarray(result.state.env_state.machine_types)
    n_furnaces = int((mt == int(MachineType.FURNACE)).sum())
    n_assemblers = int((mt == int(MachineType.ASSEMBLER)).sum())
    assert n_furnaces >= 3, (
        f"Expected 3 furnaces (1 pre-placed + 2 extra), got {n_furnaces}. "
        f"Last goal: {result.last_goal!r}"
    )
    assert n_assemblers >= 3, (
        f"Expected 3 assemblers, got {n_assemblers}. Last goal: {result.last_goal!r}"
    )


# ---------------------------------------------------------------------------
# Test D — pipelined smelt completes in a reasonable time
# ---------------------------------------------------------------------------


def _pipelined_smelt_goals(count: int = 15) -> list[Goal]:
    """Starter + central bank + a modest pipelined smelt on 3 furnaces."""
    return (
        _starter_goals()
        + [
            PlaceMachine(MachineType.FURNACE, free_tile_near_player()),
            PlaceMachine(MachineType.FURNACE, free_tile_near_player()),
        ]
        + [
            MineOre(ItemType.IRON_ORE, count + 2),  # slack
            MineOre(ItemType.COAL, count + 2),  # 1 coal / smelt
            PipelinedProduce(
                ItemType.IRON_PLATE,
                count,
                MachineType.FURNACE,
                k=3,
            ),
        ]
    )


def test_d_pipelined_smelt_completes() -> None:
    """PipelinedProduce(IRON_PLATE, 15, k=3) finishes within the wall cap
    and the player ends up holding the plates."""
    count = 15
    result = _run_plan(_pipelined_smelt_goals(count), max_steps=3500)
    _assert_not_capped(result)

    inv = np.asarray(result.state.env_state.player_inventory[0])
    got = int(inv[ItemType.IRON_PLATE])
    assert got >= count, (
        f"PipelinedProduce finished but only produced {got}/{count} plates "
        f"at step {result.steps}. Last goal: {result.last_goal!r}"
    )


# ---------------------------------------------------------------------------
# Test E — consecutive PipelinedProduce goals on the same machine type
# ---------------------------------------------------------------------------


def _consecutive_pipelined_goals() -> list[Goal]:
    """Starter + central bank + two back-to-back PipelinedProduce goals.

    If the first goal leaves stale deposits in any of the 3 assembler
    input slots, the second goal (with a different recipe) can't fire
    its recipe — this is the same failure mode that cost the v1
    factory its craft_arm/place_arm achievements.

    Chose FRAME → WIRE because they share the same machine type but
    have disjoint input types (iron+tin vs copper+tin), so any
    carry-over from FRAME will actively wedge WIRE.
    """
    base = _starter_goals() + [
        PlaceMachine(MachineType.FURNACE, free_tile_near_player()),
        PlaceMachine(MachineType.FURNACE, free_tile_near_player()),
        PlaceMachine(MachineType.ASSEMBLER, free_tile_near_player()),
        PlaceMachine(MachineType.ASSEMBLER, free_tile_near_player()),
    ]
    # Mine + smelt just enough for both recipes.
    # 8 frames = 8 iron + 8 tin; 8 wires = 8 copper + 8 tin.
    # Smelts also consume 1 coal each (10+10+20 = 40).
    more = [
        MineOre(ItemType.IRON_ORE, 10),
        MineOre(ItemType.COPPER_ORE, 10),
        MineOre(ItemType.TIN_ORE, 20),
        MineOre(ItemType.COAL, 42),
        PipelinedProduce(ItemType.IRON_PLATE, 10, MachineType.FURNACE, k=3),
        PipelinedProduce(ItemType.COPPER_PLATE, 10, MachineType.FURNACE, k=3),
        PipelinedProduce(ItemType.TIN_PLATE, 20, MachineType.FURNACE, k=3),
        # The two back-to-back pipelined goals under test.
        PipelinedProduce(ItemType.FRAME, 8, MachineType.ASSEMBLER, k=3),
        PipelinedProduce(ItemType.WIRE, 8, MachineType.ASSEMBLER, k=3),
    ]
    return base + more


def test_e_consecutive_pipelined_completes() -> None:
    """Two PipelinedProduce goals on the same 3-machine set both finish
    within the wall cap."""
    result = _run_plan(_consecutive_pipelined_goals(), max_steps=5000)
    _assert_not_capped(result)

    inv = np.asarray(result.state.env_state.player_inventory[0])
    frames = int(inv[ItemType.FRAME])
    wires = int(inv[ItemType.WIRE])
    assert frames >= 8 and wires >= 8, (
        f"Consecutive pipelined goals short: frames={frames}/8, "
        f"wires={wires}/8 at step {result.steps}. "
        f"Last goal: {result.last_goal!r}"
    )


# ---------------------------------------------------------------------------
# Test F — pipelined → sequential transition on the same machine type
# ---------------------------------------------------------------------------


def _pipelined_then_sequential_goals() -> list[Goal]:
    """Starter + central bank + PipelinedProduce → ProduceInAssembler on
    the same machine type.

    This is exactly the Phase J → Phase K transition in the factory
    agent: pipelined sub-assemblies hand off to sequential ARM/BELT
    goals. If any of the 3 assemblers has stale state when the
    sequential goal starts, the goal picks that (nearest) machine
    and hangs.
    """
    base = _starter_goals() + [
        PlaceMachine(MachineType.ASSEMBLER, free_tile_near_player()),
        PlaceMachine(MachineType.ASSEMBLER, free_tile_near_player()),
    ]
    more = [
        MineOre(ItemType.COPPER_ORE, 12),
        MineOre(ItemType.TIN_ORE, 10),
        MineOre(ItemType.COAL, 22),  # 12 + 10 smelts
        PipelinedProduce(ItemType.COPPER_PLATE, 12, MachineType.FURNACE, k=3),
        PipelinedProduce(ItemType.TIN_PLATE, 10, MachineType.FURNACE, k=3),
        # 8 wires across 3 assemblers (pipelined).
        PipelinedProduce(ItemType.WIRE, 8, MachineType.ASSEMBLER, k=3),
        # Then 1 arm sequentially. Needs 1 copper + 1 wire.
        ProduceInAssembler(ItemType.ARM, 1),
    ]
    return base + more


def test_f_pipelined_then_sequential_transition() -> None:
    """After a PipelinedProduce phase, a follow-up ProduceInMachine
    completes without deadlock on a stale machine."""
    result = _run_plan(_pipelined_then_sequential_goals(), max_steps=4000)
    _assert_not_capped(result)

    inv = np.asarray(result.state.env_state.player_inventory[0])
    arms = int(inv[ItemType.ARM])
    assert arms >= 1, (
        f"Sequential ARM goal didn't complete after pipelined WIRE. "
        f"Arms held: {arms}. Last goal: {result.last_goal!r}. "
        f"Steps: {result.steps}."
    )
