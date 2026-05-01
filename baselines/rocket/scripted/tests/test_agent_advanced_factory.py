"""End-to-end tests for the splitter-cell advanced-factory rocket agent.

Two tests:

- ``test_advanced_factory_goal_list_structural`` runs in milliseconds
  and just checks that ``build_advanced_factory_goals`` constructs
  cleanly with placement coordinates inside the 32x32 map AND emits
  the expected splitter / crossing counts (2 SPLITTERs — one each
  for iron and copper — and 1 CROSSING, the auto-inserted (21, 12)
  intersection between the copper coal trunk and the copper
  splitter's south output).
- ``test_advanced_factory_iron_cell_produces_plates`` is the load-
  bearing smoke test: runs the agent against the rocket benchmark
  env and asserts a PALLET ends up at (10, 10) (the iron cell's
  *manual stash*, north of the splitter) holding IRON_PLATE — the
  entire ore->furnace->arm->splitter->manual_stash chain.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from baselines.rocket.scripted.agent_advanced_factory import (
    build_advanced_factory_goals,
    make_advanced_factory_rocket_agent,
)
from baselines.rocket.scripted.goals import (
    MineOre,
    PlaceMachineAt,
    ProduceInMachine,
    WithdrawFromBusAt,
)
from factoriax.benchmarks.rocket import (
    NUM_ROCKET_ACHIEVEMENTS,
    ROCKET_ACHIEVEMENT_INFO,
    ROCKET_BLOCKED_ACTIONS,
    build_rocket_level,
    rocket_conditions,
)
from factoriax.constants import MAX_ACHIEVEMENTS, ItemType, MachineType
from factoriax.envs import FactoriaXEnv
from factoriax.envs.achievement_wrapper import AchievementState, AchievementWrapper
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.levels import build_state
from factoriax.observations import global_array
from factoriax.recipes import BASE_RECIPE_BOOK, RecipeBalance, RecipeOverride
from factoriax.state import EnvParams

_MAP_SIZE = 32
_IRON_MANUAL_STASH_TILE = (10, 10)
_IRON_SPLITTER_TILE = (10, 11)
_COPPER_SPLITTER_TILE = (21, 11)
_TIN_MANUAL_STASH_TILE = (25, 25)
_TIN_SPLITTER_TILE = (25, 26)

_EXPECTED_UNLOCKS: tuple[str, ...] = (
    "collect_iron",
    "collect_copper",
    "collect_tin",
    "collect_coal",
    "smelt_iron",
    "smelt_copper",
    "smelt_tin",
    "craft_wire",
    "craft_miner",
    "place_miner",
    "craft_furnace",
    "place_furnace",
    "automated_mining",
    "craft_belt",
    "place_belt",
    "craft_pallet",
    "place_pallet",
    "pallet_filled",
    "craft_arm",
    "place_arm",
    "first_assembly",
    "belt_network",
    "scaling_up",
    "industrialist",
)


def test_advanced_factory_goal_list_structural() -> None:
    """Goal list builds without errors and every placement is in-map.

    Also confirms the splitter-cell design: 4 SPLITTERs (one each
    for iron, copper, tin, silicon) and 0 CROSSINGs — the copper
    coal trunk is rerouted through row 13 so the splitter S output
    drops onto a sink PALLET at (21, 12) without any path crossing
    that tile.
    """
    goals = build_advanced_factory_goals()
    assert len(goals) > 0
    splitter_count = 0
    crossing_count = 0
    crossing_tiles: list[tuple[int, int]] = []
    for g in goals:
        if isinstance(g, PlaceMachineAt):
            x, y = g.target
            assert 0 <= x < _MAP_SIZE, f"x out of range for {g.name} target={g.target}"
            assert 0 <= y < _MAP_SIZE, f"y out of range for {g.name} target={g.target}"
            if g.machine_type == int(MachineType.SPLITTER):
                splitter_count += 1
            elif g.machine_type == int(MachineType.CROSSING):
                crossing_count += 1
                crossing_tiles.append(g.target)
    assert splitter_count == 4, (
        f"expected 1 SPLITTER per cell (iron + copper + tin + silicon = 4), "
        f"got {splitter_count}"
    )
    assert crossing_count == 0, (
        f"expected 0 CROSSINGs (copper coal trunk reroutes through row "
        f"13 so no path-collision occurs), got {crossing_count} at "
        f"{crossing_tiles}"
    )


def test_iterative_bootstrap_phase_order() -> None:
    """Verify the iterative bootstrap structure: iron + copper cells
    are fully built (both SPLITTERs placed) before any
    ``WithdrawFromBusAt`` goal targeting IRON_PLATE / COPPER_PLATE
    fires, and before any tin / silicon SPLITTER is placed.

    This is the load-bearing structural property of the iterative
    bootstrap: Phase 2's plate withdrawals only make sense if the
    iron + copper cells have already been producing plates for some
    time, which in turn requires their splitters to exist.
    """
    goals = build_advanced_factory_goals()

    iron_splitter_idx: int | None = None
    copper_splitter_idx: int | None = None
    tin_splitter_idx: int | None = None
    silicon_splitter_idx: int | None = None
    iron_withdraw_idx: int | None = None
    copper_withdraw_idx: int | None = None

    for i, g in enumerate(goals):
        if isinstance(g, PlaceMachineAt) and g.machine_type == int(
            MachineType.SPLITTER
        ):
            if g.target == _IRON_SPLITTER_TILE:
                iron_splitter_idx = i
            elif g.target == _COPPER_SPLITTER_TILE:
                copper_splitter_idx = i
            elif g.target == _TIN_SPLITTER_TILE:
                tin_splitter_idx = i
            else:
                silicon_splitter_idx = i
        elif isinstance(g, WithdrawFromBusAt):
            if g.item_type == int(ItemType.IRON_PLATE):
                iron_withdraw_idx = i
            elif g.item_type == int(ItemType.COPPER_PLATE):
                copper_withdraw_idx = i

    assert iron_splitter_idx is not None, "no iron splitter found"
    assert copper_splitter_idx is not None, "no copper splitter found"
    assert tin_splitter_idx is not None, "no tin splitter found"
    assert silicon_splitter_idx is not None, "no silicon splitter found"
    assert iron_withdraw_idx is not None, (
        "iterative bootstrap should withdraw IRON_PLATE from the "
        "running iron cell stash in Phase 2"
    )
    assert copper_withdraw_idx is not None, (
        "iterative bootstrap should withdraw COPPER_PLATE from the "
        "running copper cell stash in Phase 2"
    )

    assert iron_splitter_idx < iron_withdraw_idx, (
        f"iron cell must be built before its plate withdrawal: "
        f"splitter@{iron_splitter_idx}, withdraw@{iron_withdraw_idx}"
    )
    assert copper_splitter_idx < copper_withdraw_idx, (
        f"copper cell must be built before its plate withdrawal: "
        f"splitter@{copper_splitter_idx}, withdraw@{copper_withdraw_idx}"
    )
    assert iron_splitter_idx < tin_splitter_idx, (
        f"iron cell must be built before tin cell: "
        f"iron@{iron_splitter_idx}, tin@{tin_splitter_idx}"
    )
    assert copper_splitter_idx < silicon_splitter_idx, (
        f"copper cell must be built before silicon cell: "
        f"copper@{copper_splitter_idx}, silicon@{silicon_splitter_idx}"
    )


def _mine_count(goals: list, item: ItemType) -> int:
    """Sum the count across all MineOre(item, ...) goals in the list."""
    return sum(
        g.count for g in goals if isinstance(g, MineOre) and g.item_type == int(item)
    )


def _produce_count(goals: list, item: ItemType) -> int:
    """Sum the count across all ProduceInMachine goals targeting *item*.

    Both ProduceInFurnace and ProduceInAssembler return ProduceInMachine
    instances, so a single isinstance check covers both.
    """
    return sum(
        g.count
        for g in goals
        if isinstance(g, ProduceInMachine) and g.output_item == int(item)
    )


def test_balance_overlay_doubles_iron_ore_demand() -> None:
    """A book that doubles IRON_PLATE.input_counts should roughly
    double the agent's IRON_ORE mine count and the IRON_PLATE
    pre-smelt count, leaving every other ore count unchanged.
    """
    base_goals = build_advanced_factory_goals()
    base_iron_ore = _mine_count(base_goals, ItemType.IRON_ORE)
    base_iron_plate = _produce_count(base_goals, ItemType.IRON_PLATE)
    assert base_iron_ore > 0, "BOM produced no IRON_ORE goal in default book"

    balance = RecipeBalance(
        overrides=(
            (
                int(ItemType.IRON_PLATE),
                RecipeOverride(input_counts=(2, 1)),  # was (1, 1)
            ),
        )
    )
    tuned_book = BASE_RECIPE_BOOK.with_balance(balance)
    tuned_goals = build_advanced_factory_goals(book=tuned_book)
    tuned_iron_ore = _mine_count(tuned_goals, ItemType.IRON_ORE)

    # Mining demand: BOM = base_iron_plate * 2 (was * 1). Slack stays
    # constant so the relationship is exactly:
    #   tuned_iron_ore = (base_iron_ore - slack) * 2 + slack
    # The default IRON_ORE slack is 2.
    expected = (base_iron_ore - 2) * 2 + 2
    assert tuned_iron_ore == expected, (
        f"IRON_ORE demand did not double: base={base_iron_ore}, "
        f"tuned={tuned_iron_ore}, expected={expected}"
    )

    # COPPER_ORE / TIN_ORE demand should be unchanged.
    assert _mine_count(tuned_goals, ItemType.COPPER_ORE) == _mine_count(
        base_goals, ItemType.COPPER_ORE
    )
    assert _mine_count(tuned_goals, ItemType.TIN_ORE) == _mine_count(
        base_goals, ItemType.TIN_ORE
    )

    # IRON_PLATE pre-smelt count is the *number of plates produced*,
    # which is unchanged by an input-count tweak — the recipe still
    # yields 1 plate per cycle, just with 2 ore inputs instead of 1.
    assert _produce_count(tuned_goals, ItemType.IRON_PLATE) == base_iron_plate


def test_balance_overlay_higher_output_count_drops_smelts() -> None:
    """A book where IRON_PLATE.output_count=2 should halve (round up)
    the pre-smelt cycle count for plates AND the IRON_ORE / COAL
    demand, since one cycle now yields two plates from one ore + one
    coal.
    """
    base_goals = build_advanced_factory_goals()
    base_iron_plate = _produce_count(base_goals, ItemType.IRON_PLATE)
    base_iron_ore = _mine_count(base_goals, ItemType.IRON_ORE)

    balance = RecipeBalance(
        overrides=((int(ItemType.IRON_PLATE), RecipeOverride(output_count=2)),)
    )
    tuned_book = BASE_RECIPE_BOOK.with_balance(balance)
    tuned_goals = build_advanced_factory_goals(book=tuned_book)

    tuned_iron_plate = _produce_count(tuned_goals, ItemType.IRON_PLATE)
    tuned_iron_ore = _mine_count(tuned_goals, ItemType.IRON_ORE)

    # Plates *produced* must still cover the original demand —
    # production_schedule reports cycles * output_count, which for an
    # odd target rounds up. So the tuned plate count is either the
    # base count (if even) or base + 1 (if odd).
    assert tuned_iron_plate in (base_iron_plate, base_iron_plate + 1), (
        f"IRON_PLATE schedule out of expected range: base={base_iron_plate}, "
        f"tuned={tuned_iron_plate}"
    )

    # IRON_ORE demand: base = N plates * 1 ore + slack. tuned = ceil(N/2)
    # cycles * 1 ore + slack. With slack=2 and base N, expect:
    #   tuned_iron_ore = ceil(N/2) + 2
    n = base_iron_plate  # plates needed
    expected = -(-n // 2) + 2
    assert tuned_iron_ore == expected, (
        f"IRON_ORE demand did not halve: base={base_iron_ore}, "
        f"tuned={tuned_iron_ore}, expected={expected}"
    )


def test_slack_kwarg_overrides_default() -> None:
    """Passing ``slack={}`` mines exactly the BOM amount with no extra."""
    no_slack_goals = build_advanced_factory_goals(slack={})
    default_goals = build_advanced_factory_goals()

    no_slack_iron_ore = _mine_count(no_slack_goals, ItemType.IRON_ORE)
    default_iron_ore = _mine_count(default_goals, ItemType.IRON_ORE)
    # IRON_ORE only mined in Phase 1 (Phase 2 withdraws plates from
    # the running iron cell stash). Default slack adds 2.
    assert default_iron_ore - no_slack_iron_ore == 2

    no_slack_coal = _mine_count(no_slack_goals, ItemType.COAL)
    default_coal = _mine_count(default_goals, ItemType.COAL)
    # COAL is mined in *both* Phase 1 and Phase 2 (each phase smelts
    # its own plates / refractory + needs cell coal-trunk feed). Default
    # COAL slack=5 is applied to each phase's mine call independently,
    # so the delta is 2 * 5 = 10.
    assert default_coal - no_slack_coal == 10


@pytest.mark.slow
def test_advanced_factory_iron_cell_produces_plates() -> None:
    """Run the agent and verify the iron cell delivers plates to the
    manual stash through the new splitter design.

    The load-bearing assertion: at episode end, the entity at
    ``(10, 10)`` (the iron cell's *manual stash*, north of its
    splitter) is a PALLET whose buffer holds at least one
    IRON_PLATE. That tile is reachable only if (a) the coal trunk
    delivered coal to the buffer pallet at (8, 12) (via the rerouted
    network laid by ``_phase_b_belt_network``), (b) the furnace at
    (8, 11) auto-pulled both iron ore (from (8, 10)) and coal (from
    (8, 12)) and ran the IRON_PLATE recipe, (c) the arm at (9, 11)
    extracted the plate east into the splitter at (10, 11), and
    (d) the splitter fired its north output (manual stash) — which
    requires both the manual stash and the automation belt's
    downstream sink at (10, 13) to be receptive.
    """
    max_steps = 8000
    env_params = EnvParams(
        map_width=_MAP_SIZE,
        map_height=_MAP_SIZE,
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
    jit_obs = jax.jit(lambda s: global_array(s, env_params, 0))
    agent = make_advanced_factory_rocket_agent(env_params)

    key = jax.random.PRNGKey(0)
    last_state = state
    for _ in range(max_steps):
        obs = np.asarray(jit_obs(last_state.env_state))
        action = agent.act(obs)
        key, subkey = jax.random.split(key)
        _, last_state, _, done, _ = jit_step(
            subkey,
            last_state,
            jnp.int32(action),
            env_params,
        )
        if agent.is_done or bool(done):
            break

    env_state = last_state.env_state
    machine_types = np.asarray(env_state.machine_types)
    tile_entity = np.asarray(env_state.tile_entity)
    ent_buf_type = np.asarray(env_state.ent_buf_type)
    ent_buf_count = np.asarray(env_state.ent_buf_count)

    sx, sy = _IRON_SPLITTER_TILE
    assert int(machine_types[sy, sx]) == int(MachineType.SPLITTER), (
        f"iron splitter missing at {_IRON_SPLITTER_TILE}; "
        f"got machine_type={int(machine_types[sy, sx])}"
    )

    mx, my = _IRON_MANUAL_STASH_TILE
    assert int(machine_types[my, mx]) == int(MachineType.PALLET), (
        f"iron manual stash missing at {_IRON_MANUAL_STASH_TILE}; "
        f"got machine_type={int(machine_types[my, mx])}"
    )
    ent_idx = int(tile_entity[my, mx])
    assert ent_idx >= 0, "tile_entity has no entity at manual stash"
    assert int(ent_buf_count[ent_idx]) >= 1, (
        "iron manual stash at (10, 10) is empty — the splitter did not "
        "fire its north output. Check coal flow to (8, 12), furnace at "
        "(8, 11), arm at (9, 11), splitter at (10, 11), and the sink at "
        "(10, 13) (splitter requires both outputs receptive to fire)."
    )
    assert int(ent_buf_type[ent_idx]) == int(ItemType.IRON_PLATE), (
        f"manual stash holds wrong item: type={int(ent_buf_type[ent_idx])}"
    )

    # Tin cell must also have built and produced — this is the
    # iterative-bootstrap proof: tin cell construction in Phase 2
    # used IRON_PLATE / COPPER_PLATE withdrawn from the running
    # iron + copper cells, not freshly mined ore.
    tx, ty = _TIN_SPLITTER_TILE
    assert int(machine_types[ty, tx]) == int(MachineType.SPLITTER), (
        f"tin splitter missing at {_TIN_SPLITTER_TILE}; tin cell did "
        f"not build — Phase 2 likely failed to withdraw plates from "
        f"the running iron/copper cells. machine_type="
        f"{int(machine_types[ty, tx])}"
    )
    tmx, tmy = _TIN_MANUAL_STASH_TILE
    assert int(machine_types[tmy, tmx]) == int(MachineType.PALLET), (
        f"tin manual stash missing at {_TIN_MANUAL_STASH_TILE}; "
        f"got machine_type={int(machine_types[tmy, tmx])}"
    )

    mask = np.asarray(last_state.achievements_unlocked)[:NUM_ROCKET_ACHIEVEMENTS]
    ids = {info.id: i for i, info in enumerate(ROCKET_ACHIEVEMENT_INFO)}
    missing = [name for name in _EXPECTED_UNLOCKS if not mask[ids[name]]]
    assert not missing, f"Missing expected achievements: {missing}"
