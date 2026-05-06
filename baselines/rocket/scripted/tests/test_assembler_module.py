"""Tests for :class:`baselines.rocket.scripted.goals.BuildAssemblerModule`.

Module 3 of the advanced factory: a generic production cell that
hosts one assembler or furnace, two input pallets (north + west of
the center), an east arm, and an output pallet east of the arm.
The same goal is reused for every tier-2 intermediate (WIRE,
FRAME, WAFER) and for the 1-input REFRACTORY furnace recipe.

Three families of test:

1. **Placement test (2-input / assembler)** — run the goal, assert
   every entity (3 pallets + arm + assembler) lands at the
   expected tile with the expected facing.
2. **Bootstrap drain** — after the goal completes, the player's
   inventory has zero ASSEMBLER, ARM and PALLET left.
3. **Flow test** — pre-build a WIRE module via LevelBuilder,
   pre-load both input pallets with the recipe inputs (COPPER_PLATE
   + TIN_PLATE), run NOOPs, assert the output pallet accumulates
   WIRE.
4. **1-input variant** — run the goal with ``input_b_tile=None``,
   assert exactly two pallets (input_a + output) plus the arm and
   the center machine were placed and the bootstrap cost is one
   pallet lower.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from baselines.rocket.scripted import goals, skills
from baselines.rocket.scripted.world_model import decode_observation
from factoriax.benchmarks.rocket import (
    ROCKET_BLOCKED_ACTIONS,
    rocket_conditions,
)
from factoriax.constants import (
    Action,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.envs import FactoriaXEnv
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.levels import LevelBuilder, build_state
from factoriax.observations import global_array
from factoriax.state import EnvParams

pytestmark = pytest.mark.slow

_MAP_SIZE = 12
_SPAWN = (1, 1)

# Module geometry. center at (5, 5). Arm at (6, 5) facing RIGHT.
# Output pallet at (7, 5) east of the arm. Input pallet A north of
# center at (5, 4). Input pallet B west of center at (4, 5).
_CENTER = (5, 5)
_INPUT_A = (5, 4)
_INPUT_B = (4, 5)
_ARM = (6, 5)
_OUTPUT = (7, 5)

_JIT_STEP_CACHE: dict[tuple[int, int, int], object] = {}
_JIT_OBS_CACHE: dict[tuple[int, int, int], object] = {}


def _shared_jit_step(env_params: EnvParams):
    key = (env_params.map_width, env_params.map_height, env_params.max_timesteps)
    fn = _JIT_STEP_CACHE.get(key)
    if fn is None:
        env = ActionMaskWrapper(
            FactoriaXEnv(achievement_fn=rocket_conditions),
            ROCKET_BLOCKED_ACTIONS,
        )
        fn = jax.jit(env.step_env)
        _JIT_STEP_CACHE[key] = fn
    return fn


def _jit_obs(env_params: EnvParams):
    key = (env_params.map_width, env_params.map_height, env_params.max_timesteps)
    fn = _JIT_OBS_CACHE.get(key)
    if fn is None:
        fn = jax.jit(lambda s: global_array(s, env_params, 0))
        _JIT_OBS_CACHE[key] = fn
    return fn


def _view(state, env_params: EnvParams):
    obs = np.asarray(_jit_obs(env_params)(state))
    return decode_observation(
        obs,
        map_height=env_params.map_height,
        map_width=env_params.map_width,
        max_timesteps=env_params.max_timesteps,
    )


def _rollout(state, goal, jit_step, env_params, max_steps: int = 400):
    key = jax.random.PRNGKey(0)
    for _ in range(max_steps):
        view = _view(state, env_params)
        result, action = goal.step(view)
        if result is skills.Result.DONE:
            return state, "done"
        if result is skills.Result.FAIL:
            return state, "fail"
        assert action is not None
        key, sub = jax.random.split(key)
        _, state, _, _, _ = jit_step(sub, state, jnp.int32(int(action)), env_params)
    return state, "timeout"


def _build_module_level(
    *,
    prebuild_module: bool = False,
    prebuild_machine: int = int(MachineType.ASSEMBLER),
    prebuild_inputs: tuple[tuple[int, int], ...] = (),
    one_input: bool = False,
    max_timesteps: int = 400,
):
    """Build a 12x12 level for the module test.

    Args:
        prebuild_module: When True, place the module's machines at
            level-build time so the flow test can run NOOPs without
            depending on BuildAssemblerModule's placement code.
        prebuild_machine: ASSEMBLER or FURNACE for the center machine.
        prebuild_inputs: Tuples of ``(item_type, count)`` to preload
            into input pallet A then B, in order. Length must match
            the number of input pallets (1 if one_input else 2).
        one_input: When True, use the 1-input variant (no west pallet).

    Returns:
        ``(jit_step_fn, EnvState, env_params)``.
    """
    builder = LevelBuilder(_MAP_SIZE, _MAP_SIZE)
    builder.set_player_position(*_SPAWN)

    if prebuild_module:
        builder.place_machine(
            *_OUTPUT,
            int(MachineType.PALLET),
            int(Direction.DOWN),
        )
        # Input feeders are belts (facing into the assembler) under
        # the directional Phase 0; pallets here would never feed.
        builder.place_machine(
            *_INPUT_A,
            int(MachineType.CONVEYOR_BELT),
            int(Direction.DOWN),
        )
        if not one_input:
            builder.place_machine(
                *_INPUT_B,
                int(MachineType.CONVEYOR_BELT),
                int(Direction.RIGHT),
            )
        builder.place_machine(
            *_ARM,
            int(MachineType.ARM),
            int(Direction.RIGHT),
        )
        builder.place_machine(
            *_CENTER,
            int(prebuild_machine),
            int(Direction.DOWN),
        )
        # Preload inputs in order (A first, then B if any).
        input_tiles = (_INPUT_A,) if one_input else (_INPUT_A, _INPUT_B)
        for tile, (item_type, count) in zip(input_tiles, prebuild_inputs, strict=True):
            builder.set_machine_inventory(
                tile[0],
                tile[1],
                int(item_type),
                int(count),
            )

    level = builder.build("assembler_module_test")
    env_params = EnvParams(
        map_width=_MAP_SIZE,
        map_height=_MAP_SIZE,
        num_players=1,
        max_timesteps=max_timesteps,
    )
    state = build_state(level, env_params)

    inv = np.asarray(state.player_inventory).copy()
    inv[0, int(ItemType.ASSEMBLER)] = 1
    inv[0, int(ItemType.FURNACE)] = 1
    inv[0, int(ItemType.ARM)] = 1
    inv[0, int(ItemType.PALLET)] = 1
    inv[0, int(ItemType.CONVEYOR_BELT)] = 2
    state = state.replace(player_inventory=jnp.asarray(inv))

    return _shared_jit_step(env_params), state, env_params


def test_build_assembler_module_places_all_entities() -> None:
    """The 2-input goal places 3 pallets + arm + assembler at expected tiles."""
    jit_step, state, env_params = _build_module_level()
    goal = goals.BuildAssemblerModule(
        center_tile=_CENTER,
        center_machine=MachineType.ASSEMBLER,
        input_a_tile=_INPUT_A,
        input_b_tile=_INPUT_B,
        output_pallet_tile=_OUTPUT,
    )
    final_state, verdict = _rollout(state, goal, jit_step, env_params, max_steps=400)

    assert verdict == "done", f"got {verdict}"

    mt = np.asarray(final_state.machine_types)
    tile_entity = np.asarray(final_state.tile_entity)
    ent_direction = np.asarray(final_state.ent_direction)

    # Output pallet east-of-arm; inputs are feeder belts.
    assert mt[_OUTPUT[1], _OUTPUT[0]] == int(MachineType.PALLET), (
        f"expected pallet at {_OUTPUT}"
    )
    for tile in (_INPUT_A, _INPUT_B):
        assert mt[tile[1], tile[0]] == int(MachineType.CONVEYOR_BELT), (
            f"expected feeder belt at {tile}"
        )
    # Center machine = assembler facing DOWN.
    assert mt[_CENTER[1], _CENTER[0]] == int(MachineType.ASSEMBLER)
    center_eid = int(tile_entity[_CENTER[1], _CENTER[0]])
    assert center_eid >= 0
    assert int(ent_direction[center_eid]) == int(Direction.DOWN)
    # Arm facing RIGHT.
    assert mt[_ARM[1], _ARM[0]] == int(MachineType.ARM)
    arm_eid = int(tile_entity[_ARM[1], _ARM[0]])
    assert arm_eid >= 0
    assert int(ent_direction[arm_eid]) == int(Direction.RIGHT)


def test_build_assembler_module_consumes_bootstrap_inventory() -> None:
    """After building, the player has zero ASSEMBLER, ARM and PALLET left."""
    jit_step, state, env_params = _build_module_level()
    goal = goals.BuildAssemblerModule(
        center_tile=_CENTER,
        center_machine=MachineType.ASSEMBLER,
        input_a_tile=_INPUT_A,
        input_b_tile=_INPUT_B,
        output_pallet_tile=_OUTPUT,
    )
    final_state, verdict = _rollout(state, goal, jit_step, env_params, max_steps=400)
    assert verdict == "done"
    inv = np.asarray(final_state.player_inventory[0])
    assert int(inv[int(ItemType.ASSEMBLER)]) == 0
    assert int(inv[int(ItemType.ARM)]) == 0
    assert int(inv[int(ItemType.PALLET)]) == 0
    # Furnace was bootstrapped but not consumed by an assembler module.
    assert int(inv[int(ItemType.FURNACE)]) == 1


def test_assembler_module_produces_wire_when_fed_plates() -> None:
    """A pre-built assembler module + loaded input pallets yields WIRE.

    The WIRE recipe is 1 COPPER_PLATE + 1 TIN_PLATE -> 1 WIRE in 4
    ticks. Bypasses the goal under test by hand-placing the module
    at level-build time and pre-loading both input pallets so we can
    validate the output flow on its own. Each tick the assembler
    Phase 0 pulls one of each plate into ``ent_asm_in``; a recipe
    cycle takes 4 ticks; the east arm reads from ``ent_asm_out`` and
    pushes one WIRE into the output pallet per cycle. 60 ticks is
    enough for several handoffs.
    """
    jit_step, state, env_params = _build_module_level(
        prebuild_module=True,
        prebuild_machine=int(MachineType.ASSEMBLER),
        prebuild_inputs=(
            (int(ItemType.COPPER_PLATE), 1),
            (int(ItemType.TIN_PLATE), 1),
        ),
    )

    key = jax.random.PRNGKey(0)
    for _ in range(60):
        key, sub = jax.random.split(key)
        _, state, _, _, _ = jit_step(
            sub,
            state,
            jnp.int32(int(Action.NOOP)),
            env_params,
        )

    out_eid = int(state.tile_entity[_OUTPUT[1], _OUTPUT[0]])
    assert out_eid >= 0
    out_buf = int(state.ent_buf_count[out_eid])
    out_type = int(state.ent_buf_type[out_eid])
    assert out_type == int(ItemType.WIRE), (
        f"expected WIRE in output pallet, got ItemType={out_type}"
    )
    assert out_buf > 0, f"output empty after 60 ticks; buf={out_buf}"


def test_build_assembler_module_one_input_variant() -> None:
    """1-input variant places only 2 pallets + arm + furnace, leaves 1 PALLET unused."""
    jit_step, state, env_params = _build_module_level()
    goal = goals.BuildAssemblerModule(
        center_tile=_CENTER,
        center_machine=MachineType.FURNACE,
        input_a_tile=_INPUT_A,
        input_b_tile=None,
        output_pallet_tile=_OUTPUT,
    )
    final_state, verdict = _rollout(state, goal, jit_step, env_params, max_steps=400)

    assert verdict == "done", f"got {verdict}"

    mt = np.asarray(final_state.machine_types)
    # Output pallet + input_a feeder belt; west neighbour is dirt.
    assert mt[_INPUT_A[1], _INPUT_A[0]] == int(MachineType.CONVEYOR_BELT)
    assert mt[_OUTPUT[1], _OUTPUT[0]] == int(MachineType.PALLET)
    assert mt[_INPUT_B[1], _INPUT_B[0]] == int(MachineType.NONE), (
        "1-input variant must not place a west feeder belt"
    )
    # Furnace + arm.
    assert mt[_CENTER[1], _CENTER[0]] == int(MachineType.FURNACE)
    assert mt[_ARM[1], _ARM[0]] == int(MachineType.ARM)
    # One PALLET left in inventory (started with 1, used 1 for output);
    # one CONVEYOR_BELT used (started with 2, 1 left).
    inv = np.asarray(final_state.player_inventory[0])
    assert int(inv[int(ItemType.PALLET)]) == 0
    assert int(inv[int(ItemType.CONVEYOR_BELT)]) == 1
    assert int(inv[int(ItemType.FURNACE)]) == 0
    assert int(inv[int(ItemType.ARM)]) == 0


# ---------------------------------------------------------------------------
# Tier 3 recipes — proves BuildAssemblerModule generalises beyond WIRE.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("recipe_inputs", "expected_output", "wait_ticks"),
    [
        # MOTOR = FRAME + WIRE (assembler, 6 ticks)
        (
            ((int(ItemType.FRAME), 1), (int(ItemType.WIRE), 1)),
            int(ItemType.MOTOR),
            80,
        ),
        # SENSOR = CIRCUIT + WIRE (assembler, 6 ticks)
        (
            ((int(ItemType.CIRCUIT), 1), (int(ItemType.WIRE), 1)),
            int(ItemType.SENSOR),
            80,
        ),
        # CIRCUIT = COPPER_PLATE + WAFER (assembler, 4 ticks)
        (
            ((int(ItemType.COPPER_PLATE), 1), (int(ItemType.WAFER), 1)),
            int(ItemType.CIRCUIT),
            60,
        ),
    ],
    ids=["motor", "sensor", "circuit"],
)
def test_assembler_module_produces_tier3_recipes(
    recipe_inputs: tuple[tuple[int, int], tuple[int, int]],
    expected_output: int,
    wait_ticks: int,
) -> None:
    """The same module shape produces every 2-input assembler recipe.

    Proves :class:`BuildAssemblerModule` is recipe-agnostic — the
    engine's ``run_assemblers`` matches whichever recipe fits the
    inputs that land in ``ent_asm_in``, so MOTOR (FRAME+WIRE),
    SENSOR (CIRCUIT+WIRE) and CIRCUIT (COPPER_PLATE+WAFER) all flow
    through the exact same input-pallet -> assembler -> arm ->
    output-pallet topology with no goal changes. The wait window is
    longer than for WIRE because some tier-3 recipes have ``ticks=6``.
    """
    jit_step, state, env_params = _build_module_level(
        prebuild_module=True,
        prebuild_machine=int(MachineType.ASSEMBLER),
        prebuild_inputs=recipe_inputs,
    )

    key = jax.random.PRNGKey(0)
    for _ in range(wait_ticks):
        key, sub = jax.random.split(key)
        _, state, _, _, _ = jit_step(
            sub,
            state,
            jnp.int32(int(Action.NOOP)),
            env_params,
        )

    out_eid = int(state.tile_entity[_OUTPUT[1], _OUTPUT[0]])
    assert out_eid >= 0
    out_buf = int(state.ent_buf_count[out_eid])
    out_type = int(state.ent_buf_type[out_eid])
    assert out_type == expected_output, (
        f"expected {ItemType(expected_output).name} in output, got ItemType={out_type}"
    )
    assert out_buf > 0, f"output empty after {wait_ticks} ticks; buf={out_buf}"


# ---------------------------------------------------------------------------
# Composition — two adjacent modules built sequentially on the same map.
# ---------------------------------------------------------------------------


def test_two_assembler_modules_compose_without_collision() -> None:
    """Two BuildAssemblerModule calls on disjoint tiles both succeed.

    The advanced factory drops a chain of modules side by side along
    a bus. This test runs two BuildAssemblerModule goals back to back
    on a 16x16 map: module A is the canonical (5,5) layout from the
    other tests, module B sits four tiles further east. The point is
    to catch any cross-module interference — e.g. a stand tile used
    by module B that became unwalkable because of module A's
    placement.
    """
    map_size = 16
    builder = LevelBuilder(map_size, map_size)
    builder.set_player_position(1, 1)
    level = builder.build("two_modules_test")
    env_params = EnvParams(
        map_width=map_size,
        map_height=map_size,
        num_players=1,
        max_timesteps=600,
    )
    state = build_state(level, env_params)
    inv = np.asarray(state.player_inventory).copy()
    inv[0, int(ItemType.ASSEMBLER)] = 2
    inv[0, int(ItemType.ARM)] = 2
    inv[0, int(ItemType.PALLET)] = 2
    inv[0, int(ItemType.CONVEYOR_BELT)] = 4
    state = state.replace(player_inventory=jnp.asarray(inv))
    jit_step = _shared_jit_step(env_params)

    # Module A — the canonical layout.
    goal_a = goals.BuildAssemblerModule(
        center_tile=_CENTER,
        center_machine=MachineType.ASSEMBLER,
        input_a_tile=_INPUT_A,
        input_b_tile=_INPUT_B,
        output_pallet_tile=_OUTPUT,
    )
    state, verdict = _rollout(state, goal_a, jit_step, env_params, max_steps=400)
    assert verdict == "done", f"module A got {verdict}"

    # Module B — shifted east by 5 tiles. Center at (10, 5).
    center_b = (10, 5)
    input_a_b = (10, 4)
    input_b_b = (9, 5)
    output_b = (12, 5)
    arm_b = (11, 5)
    goal_b = goals.BuildAssemblerModule(
        center_tile=center_b,
        center_machine=MachineType.ASSEMBLER,
        input_a_tile=input_a_b,
        input_b_tile=input_b_b,
        output_pallet_tile=output_b,
    )
    state, verdict = _rollout(state, goal_b, jit_step, env_params, max_steps=400)
    assert verdict == "done", f"module B got {verdict}"

    mt = np.asarray(state.machine_types)
    # Both modules' centers + arms exist.
    assert mt[_CENTER[1], _CENTER[0]] == int(MachineType.ASSEMBLER)
    assert mt[_ARM[1], _ARM[0]] == int(MachineType.ARM)
    assert mt[center_b[1], center_b[0]] == int(MachineType.ASSEMBLER)
    assert mt[arm_b[1], arm_b[0]] == int(MachineType.ARM)
    # Both modules' output pallets exist.
    for tile in (_OUTPUT, output_b):
        assert mt[tile[1], tile[0]] == int(MachineType.PALLET), (
            f"missing pallet at {tile}"
        )
    # Both modules' input feeders are belts.
    for tile in (_INPUT_A, _INPUT_B, input_a_b, input_b_b):
        assert mt[tile[1], tile[0]] == int(MachineType.CONVEYOR_BELT), (
            f"missing feeder belt at {tile}"
        )
    # Bootstrap inventory fully consumed.
    inv = np.asarray(state.player_inventory[0])
    assert int(inv[int(ItemType.ASSEMBLER)]) == 0
    assert int(inv[int(ItemType.ARM)]) == 0
    assert int(inv[int(ItemType.PALLET)]) == 0
    assert int(inv[int(ItemType.CONVEYOR_BELT)]) == 0


# ---------------------------------------------------------------------------
# Tier 4 recipes — rocket sub-assemblies. Multi-quantity inputs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("recipe_inputs", "expected_output", "wait_ticks"),
    [
        # HULL = FRAME(2) + IRON_PLATE(2), 8 ticks. Pallets need
        # >=2 each so the assembler's Phase 0 can fill both
        # asm_in slots over two pull cycles before the recipe
        # idle gate satisfies.
        (
            ((int(ItemType.FRAME), 4), (int(ItemType.IRON_PLATE), 4)),
            int(ItemType.HULL),
            120,
        ),
        # ENGINE_UNIT = MOTOR(2) + WIRE(1), 8 ticks.
        (
            ((int(ItemType.MOTOR), 4), (int(ItemType.WIRE), 4)),
            int(ItemType.ENGINE_UNIT),
            120,
        ),
        # AVIONICS = CIRCUIT(2) + SENSOR(2), 10 ticks.
        (
            ((int(ItemType.CIRCUIT), 4), (int(ItemType.SENSOR), 4)),
            int(ItemType.AVIONICS),
            150,
        ),
        # ROCKET_CORE = ENGINE_UNIT(1) + AVIONICS(1), 10 ticks.
        (
            ((int(ItemType.ENGINE_UNIT), 4), (int(ItemType.AVIONICS), 4)),
            int(ItemType.ROCKET_CORE),
            150,
        ),
    ],
    ids=["hull", "engine_unit", "avionics", "rocket_core"],
)
def test_assembler_module_produces_tier4_recipes(
    recipe_inputs: tuple[tuple[int, int], tuple[int, int]],
    expected_output: int,
    wait_ticks: int,
) -> None:
    """The same module shape covers every tier-4 rocket sub-assembly.

    Tier-4 recipes (HULL, ENGINE_UNIT, AVIONICS, ROCKET_CORE) have
    multi-quantity inputs (typically 2 of each, plus a 10-tick recipe
    for AVIONICS / ROCKET_CORE). ``run_assemblers`` Phase 0 pulls
    one item per tick per direction so each ``asm_in`` slot fills
    over multiple pulls before the recipe idle gate satisfies.

    Pre-loads each input pallet with 4 (an over-provision so multiple
    cycles can complete inside the wait window) and asserts the
    expected output accumulates in the downstream pallet. The wait
    windows budget for: ``per_craft`` pull cycles + ``recipe_ticks``
    progress + arm-transfer + JIT noise.
    """
    jit_step, state, env_params = _build_module_level(
        prebuild_module=True,
        prebuild_machine=int(MachineType.ASSEMBLER),
        prebuild_inputs=recipe_inputs,
    )

    key = jax.random.PRNGKey(0)
    for _ in range(wait_ticks):
        key, sub = jax.random.split(key)
        _, state, _, _, _ = jit_step(
            sub,
            state,
            jnp.int32(int(Action.NOOP)),
            env_params,
        )

    out_eid = int(state.tile_entity[_OUTPUT[1], _OUTPUT[0]])
    assert out_eid >= 0
    out_buf = int(state.ent_buf_count[out_eid])
    out_type = int(state.ent_buf_type[out_eid])
    assert out_type == expected_output, (
        f"expected {ItemType(expected_output).name} in output, got ItemType={out_type}"
    )
    assert out_buf > 0, f"output empty after {wait_ticks} ticks; buf={out_buf}"


# ---------------------------------------------------------------------------
# Tier 5 — the rocket sink. ROCKET = HULL(6) + ROCKET_CORE(4), 300 ticks.
# ---------------------------------------------------------------------------


def test_assembler_module_produces_rocket_when_fed_subassemblies() -> None:
    """The rocket sink: a single assembler module crafts a ROCKET.

    ROCKET is the terminal recipe in the rocket benchmark — a single
    craft consumes 6 HULL + 4 ROCKET_CORE and runs for 300 ticks. It
    sits on top of the entire factory chain (ore -> plates ->
    intermediates -> sub-assemblies -> rocket) and is what the
    "factory wins" by producing.

    Engine timing: ``run_assemblers`` Phase 0 pulls one item per
    tick per direction. After 6 ticks the input slot holding HULL
    hits 6 (the limiting input — ROCKET only needs 4 ROCKET_CORE so
    the other slot is over-filled by then). Recipe starts at tick 6,
    runs 300 ticks, completes at ~tick 306, arm transfers to output
    pallet at ~tick 307. Allocates a 360-tick wait window for JIT
    noise + arm scheduling.

    Bumps max_timesteps so the env doesn't truncate mid-recipe.
    """
    jit_step, state, env_params = _build_module_level(
        prebuild_module=True,
        prebuild_machine=int(MachineType.ASSEMBLER),
        prebuild_inputs=(
            (int(ItemType.HULL), 8),
            (int(ItemType.ROCKET_CORE), 8),
        ),
        max_timesteps=500,
    )

    key = jax.random.PRNGKey(0)
    for _ in range(360):
        key, sub = jax.random.split(key)
        _, state, _, _, _ = jit_step(
            sub,
            state,
            jnp.int32(int(Action.NOOP)),
            env_params,
        )

    out_eid = int(state.tile_entity[_OUTPUT[1], _OUTPUT[0]])
    assert out_eid >= 0
    out_buf = int(state.ent_buf_count[out_eid])
    out_type = int(state.ent_buf_type[out_eid])
    assert out_type == int(ItemType.ROCKET), (
        f"expected ROCKET in output pallet, got ItemType={out_type}"
    )
    assert out_buf > 0, f"output empty after 360 ticks; buf={out_buf}"
