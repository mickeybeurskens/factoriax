"""Rocket benchmark — achievement-based reward signal with the rocket as capstone.

The benchmark defines 38 achievements tiered Craftax-style (1/3/5/8 points,
max 140), ordered to teach the game's natural learning path: gather raw ore,
refine intermediates via a furnace and assembler, build and deploy the rest
of the factory, assemble the three rocket sub-components (hull, engine,
avionics), then launch the rocket.

Setup:
- A **furnace** and **assembler** are pre-placed adjacent to the player's
  spawn. The agent doesn't need to hand-craft the starter machinery — all
  crafting goes through these placed machines. Producing a *second*
  furnace/assembler via the main assembler is how the ``craft_furnace`` /
  ``craft_assembler`` achievements unlock.
- All direct player-crafting actions (``CRAFT_IRON_PLATE`` through
  ``CRAFT_ROCKET``) are **masked** — the env silently replaces them with
  ``NOOP``. Production must flow through the furnace / assembler pipeline.
- ``RocketBenchmark`` exposes ``achievement_fn`` and ``blocked_actions``
  attributes the :class:`~factoriax.benchmarks.runner.BenchmarkRunner`
  picks up automatically; callers stepping the env directly should wrap
  with :class:`~factoriax.envs.action_mask_wrapper.ActionMaskWrapper`
  themselves (see :data:`ROCKET_BLOCKED_ACTIONS`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from factoriax.achievements import AchievementInfo
from factoriax.benchmarks.core import BenchmarkLevel, LevelResult
from factoriax.constants import (
    MAX_ACHIEVEMENTS,
    Action,
    BlockType,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.levels import Level, LevelBuilder
from factoriax.rewards import achievement_reward
from factoriax.state import EnvParams, EnvState

# ---------------------------------------------------------------------------
# Achievement catalogue
# ---------------------------------------------------------------------------

ROCKET_ACHIEVEMENT_INFO: list[AchievementInfo] = [
    # Basic (1 pt) — raw gathering + simplest handcrafts.
    AchievementInfo("collect_iron", "Collect Iron", "Mine iron ore."),
    AchievementInfo("collect_copper", "Collect Copper", "Mine copper ore."),
    AchievementInfo("collect_tin", "Collect Tin", "Mine tin ore."),
    AchievementInfo(
        "collect_coal",
        "Collect Coal",
        "Mine coal — needed by every smelt.",
    ),
    AchievementInfo("collect_silicon", "Collect Silicon", "Mine silicon."),
    AchievementInfo("smelt_iron", "Smelt Iron", "Hold an iron plate."),
    AchievementInfo("smelt_copper", "Smelt Copper", "Hold a copper plate."),
    AchievementInfo("smelt_tin", "Smelt Tin", "Hold a tin plate."),
    AchievementInfo("smelt_wafer", "Smelt Wafer", "Hold a wafer."),
    AchievementInfo(
        "craft_wire",
        "Craft Wire",
        "Combine iron and copper into wire.",
    ),
    # Intermediate (3 pt) — deeper handcrafts + first machines.
    AchievementInfo(
        "craft_circuit",
        "Craft Circuit",
        "Combine copper and wafer.",
    ),
    AchievementInfo(
        "craft_frame",
        "Craft Frame",
        "Combine iron and tin plates.",
    ),
    AchievementInfo(
        "craft_motor",
        "Craft Motor",
        "Combine frame and wire.",
    ),
    AchievementInfo(
        "craft_sensor",
        "Craft Sensor",
        "Combine circuit and wire.",
    ),
    AchievementInfo("craft_miner", "Craft Miner", "Hold a miner in inventory."),
    AchievementInfo("place_miner", "Place Miner", "Place a miner on the map."),
    AchievementInfo(
        "craft_furnace",
        "Craft Furnace",
        "Hold a furnace in inventory.",
    ),
    AchievementInfo(
        "place_furnace",
        "Place Furnace",
        "Place a furnace on the map.",
    ),
    AchievementInfo(
        "automated_mining",
        "Automated Mining",
        "Have a placed miner produce ore.",
    ),
    AchievementInfo(
        "craft_belt",
        "Craft Belt",
        "Hold a conveyor belt in inventory.",
    ),
    AchievementInfo(
        "place_belt",
        "Place Belt",
        "Place a conveyor belt on the map.",
    ),
    # Advanced (5 pt) — logistics composition + assembler.
    AchievementInfo(
        "craft_pallet",
        "Craft Pallet",
        "Hold a pallet in inventory.",
    ),
    AchievementInfo(
        "place_pallet",
        "Place Pallet",
        "Place a pallet on the map.",
    ),
    AchievementInfo(
        "pallet_filled",
        "Pallet Filled",
        "Put an item in a pallet.",
    ),
    AchievementInfo("craft_arm", "Craft Arm", "Hold an arm in inventory."),
    AchievementInfo("place_arm", "Place Arm", "Place an arm on the map."),
    AchievementInfo(
        "craft_assembler",
        "Craft Assembler",
        "Hold an assembler in inventory.",
    ),
    AchievementInfo(
        "place_assembler",
        "Place Assembler",
        "Place an assembler on the map.",
    ),
    AchievementInfo(
        "first_assembly",
        "First Assembly",
        "Have a placed assembler produce output.",
    ),
    AchievementInfo(
        "belt_network",
        "Belt Network",
        "Place five conveyor belts.",
    ),
    AchievementInfo(
        "craft_hull",
        "Craft Hull",
        "Assemble a rocket hull (2 frame + 2 iron plate).",
    ),
    AchievementInfo(
        "craft_engine_unit",
        "Craft Engine Unit",
        "Assemble a rocket engine (2 motor + 1 wire).",
    ),
    AchievementInfo(
        "craft_avionics",
        "Craft Avionics",
        "Assemble an avionics package (2 circuit + 2 sensor).",
    ),
    AchievementInfo(
        "craft_rocket_core",
        "Craft Rocket Core",
        "Assemble a rocket core (engine + avionics).",
    ),
    # Very Advanced (8 pt) — scale + capstone.
    AchievementInfo(
        "scaling_up",
        "Scaling Up",
        "Place three miners at once.",
    ),
    AchievementInfo(
        "industrialist",
        "Industrialist",
        "Place ten machines in total.",
    ),
    AchievementInfo(
        "craft_rocket",
        "Craft Rocket",
        "Assemble a rocket into inventory.",
    ),
    AchievementInfo(
        "place_rocket",
        "Place Rocket",
        "Place the rocket — goal reached.",
    ),
]

NUM_ROCKET_ACHIEVEMENTS: int = len(ROCKET_ACHIEVEMENT_INFO)

# Tier layout: 10 Basic (1pt) + 11 Intermediate (3pt)
# + 13 Advanced (5pt) + 4 Very Advanced (8pt).
_TIER_WEIGHTS: list[int] = [1] * 10 + [3] * 11 + [5] * 13 + [8] * 4
assert len(_TIER_WEIGHTS) == NUM_ROCKET_ACHIEVEMENTS

ROCKET_ACHIEVEMENT_WEIGHTS: jax.Array = (
    jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.float32)
    .at[:NUM_ROCKET_ACHIEVEMENTS]
    .set(jnp.array(_TIER_WEIGHTS, dtype=jnp.float32))
)

MAX_ROCKET_SCORE: int = int(sum(_TIER_WEIGHTS))  # 140


# ---------------------------------------------------------------------------
# Condition helpers
# ---------------------------------------------------------------------------


def _holds_item(state: EnvState, item: int) -> jax.Array:
    """True when any player inventory contains at least one of *item*."""
    return jnp.sum(state.player_inventory[:, item]) >= 1


def _count_machines(state: EnvState, mt: int) -> jax.Array:
    """Count placed machines of a given type."""
    return jnp.sum(state.machine_types == mt)


def _any_entity_buf_nonempty(state: EnvState, mt: int) -> jax.Array:
    """True when any active entity of type *mt* has items in its buffer.

    Used for ``automated_mining`` (miner output) and ``pallet_filled``.
    """
    matches = (state.ent_type == mt) & (state.ent_y >= 0) & (state.ent_buf_count > 0)
    return jnp.any(matches)


def _any_assembler_has_output(state: EnvState) -> jax.Array:
    """True when any active assembler holds a recipe output.

    The engine parks completed output in ``ent_asm_out`` until a
    withdraw pulls it out (no buffer drain), so that's where we check.
    """
    matches = (
        (state.ent_type == MachineType.ASSEMBLER)
        & (state.ent_y >= 0)
        & (state.ent_asm_out_count > 0)
    )
    return jnp.any(matches)


# ---------------------------------------------------------------------------
# Condition function (JIT-pure)
# ---------------------------------------------------------------------------


def rocket_conditions(state: EnvState) -> jax.Array:
    """Compute the 38 rocket-benchmark achievement conditions.

    Every condition is a pure function of ``state``. The returned array
    is zero-padded to ``MAX_ACHIEVEMENTS`` so it plugs into
    :class:`~factoriax.envs.achievement_wrapper.AchievementWrapper`
    directly.

    Args:
        state: Current environment state.

    Returns:
        Boolean array of shape ``(MAX_ACHIEVEMENTS,)``.
    """
    total_machines = jnp.sum(state.machine_types != MachineType.NONE)

    conditions = jnp.array(
        [
            # Basic (1 pt).
            _holds_item(state, ItemType.IRON_ORE),
            _holds_item(state, ItemType.COPPER_ORE),
            _holds_item(state, ItemType.TIN_ORE),
            _holds_item(state, ItemType.COAL),
            _holds_item(state, ItemType.SILICON),
            _holds_item(state, ItemType.IRON_PLATE),
            _holds_item(state, ItemType.COPPER_PLATE),
            _holds_item(state, ItemType.TIN_PLATE),
            _holds_item(state, ItemType.WAFER),
            _holds_item(state, ItemType.WIRE),
            # Intermediate (3 pt).
            _holds_item(state, ItemType.CIRCUIT),
            _holds_item(state, ItemType.FRAME),
            _holds_item(state, ItemType.MOTOR),
            _holds_item(state, ItemType.SENSOR),
            _holds_item(state, ItemType.MINER),
            _count_machines(state, MachineType.MINER) >= 1,
            _holds_item(state, ItemType.FURNACE),
            _count_machines(state, MachineType.FURNACE) >= 1,
            _any_entity_buf_nonempty(state, MachineType.MINER),
            _holds_item(state, ItemType.CONVEYOR_BELT),
            _count_machines(state, MachineType.CONVEYOR_BELT) >= 1,
            # Advanced (5 pt).
            _holds_item(state, ItemType.PALLET),
            _count_machines(state, MachineType.PALLET) >= 1,
            _any_entity_buf_nonempty(state, MachineType.PALLET),
            _holds_item(state, ItemType.ARM),
            _count_machines(state, MachineType.ARM) >= 1,
            _holds_item(state, ItemType.ASSEMBLER),
            _count_machines(state, MachineType.ASSEMBLER) >= 1,
            _any_assembler_has_output(state),
            _count_machines(state, MachineType.CONVEYOR_BELT) >= 5,
            _holds_item(state, ItemType.HULL),
            _holds_item(state, ItemType.ENGINE_UNIT),
            _holds_item(state, ItemType.AVIONICS),
            _holds_item(state, ItemType.ROCKET_CORE),
            # Very Advanced (8 pt).
            _count_machines(state, MachineType.MINER) >= 3,
            total_machines >= 10,
            _holds_item(state, ItemType.ROCKET),
            _count_machines(state, MachineType.ROCKET) >= 1,
        ],
        dtype=jnp.bool_,
    )
    return jnp.concatenate(
        [
            conditions,
            jnp.zeros(MAX_ACHIEVEMENTS - NUM_ROCKET_ACHIEVEMENTS, dtype=jnp.bool_),
        ]
    )


# ---------------------------------------------------------------------------
# Reward function for training
# ---------------------------------------------------------------------------


def rocket_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Sparse reward for newly-unlocked rocket-benchmark achievements.

    Thin wrapper around :func:`factoriax.rewards.achievement_reward`
    bound to :data:`ROCKET_ACHIEVEMENT_WEIGHTS`. Assumes *prev_state*
    and *new_state* are :class:`AchievementState` instances — i.e. the
    env was wrapped with :class:`AchievementWrapper` using
    :func:`rocket_conditions`.

    Args:
        prev_state: State before the step.
        new_state: State after the step.
        params: Environment parameters (unused; interface uniformity).

    Returns:
        Scalar float32 reward.
    """
    return achievement_reward(
        prev_state, new_state, params, weights=ROCKET_ACHIEVEMENT_WEIGHTS
    )


# ---------------------------------------------------------------------------
# Level construction
# ---------------------------------------------------------------------------

# 32x32 map; player spawns at the centre with six 3x3 ore patches
# placed symmetrically at ~8 tiles radius. A furnace and an assembler
# are pre-placed immediately west / east of the spawn so the agent
# never has to hand-craft to get started. The sixth patch (LIMESTONE)
# was added so REFRACTORY can take two inputs (LIMESTONE + COAL)
# matching every other furnace recipe — see factoriax.recipes.
_MAP_SIZE: int = 32
_ORE_PATCH_SIZE: int = 3
# 280 per tile × 9 tiles per patch ≈ 2500 ore per patch. Enough for
# automated miners to run essentially the whole episode without
# depleting — the agent doesn't have to re-mine midway.
_ORE_RESOURCES_PER_TILE: int = 2800
# Coal is consumed by every smelt and every refractory craft, so the
# coal patch needs ~10x the throughput of any other ore. 28 000 per
# tile × 9 tiles per patch ≈ 250 000 coal per patch — enough to
# sustain a fully-automated factory through the whole rocket chain
# without refueling. Requires ``BLOCK_MAX_RESOURCES`` to be at least
# 28 000.
_COAL_RESOURCES_PER_TILE: int = 28000
_SPAWN: tuple[int, int] = (_MAP_SIZE // 2, _MAP_SIZE // 2)
_FURNACE_TILE: tuple[int, int] = (_SPAWN[0] - 1, _SPAWN[1])
_ASSEMBLER_TILE: tuple[int, int] = (_SPAWN[0] + 1, _SPAWN[1])
_PATCH_OFFSETS: list[tuple[int, int, BlockType]] = [
    # (x, y, block) — all coordinates are the top-left corner of the 3x3 patch.
    (7, 7, BlockType.IRON),
    (22, 7, BlockType.COPPER),
    (7, 22, BlockType.COAL),
    (22, 22, BlockType.TIN),
    (14, 3, BlockType.SILICON),
    # Limestone: south-centre, well clear of the iron/copper/tin/coal
    # patches and the spawn corridor. Covers (13..15, 28..30).
    (13, 28, BlockType.LIMESTONE),
]


def build_rocket_level() -> Level:
    """Construct the canonical 32x32 rocket benchmark level.

    Player spawns at :data:`_SPAWN`. Six 3x3 ore patches (iron, copper,
    coal, tin, silicon, limestone) sit at varying radii from spawn in
    a roughly-symmetric layout, each tile carrying
    :data:`_ORE_RESOURCES_PER_TILE` units (or
    :data:`_COAL_RESOURCES_PER_TILE` for coal) — plenty for a full
    rocket run. A furnace and an assembler are pre-placed one tile
    west and east of spawn respectively.

    Returns:
        Deterministic :class:`Level` used as the benchmark's only level.
    """
    builder = LevelBuilder(_MAP_SIZE, _MAP_SIZE)
    for x, y, block in _PATCH_OFFSETS:
        per_tile = (
            _COAL_RESOURCES_PER_TILE
            if block == BlockType.COAL
            else _ORE_RESOURCES_PER_TILE
        )
        builder.fill_rect(
            x,
            y,
            _ORE_PATCH_SIZE,
            _ORE_PATCH_SIZE,
            block,
            resources=per_tile,
        )
    builder.set_player_position(*_SPAWN)
    builder.place_machine(
        _FURNACE_TILE[0],
        _FURNACE_TILE[1],
        int(MachineType.FURNACE),
        direction=int(Direction.DOWN),
    )
    builder.place_machine(
        _ASSEMBLER_TILE[0],
        _ASSEMBLER_TILE[1],
        int(MachineType.ASSEMBLER),
        direction=int(Direction.DOWN),
    )
    return builder.build("rocket_v1")


# ---------------------------------------------------------------------------
# Action mask
# ---------------------------------------------------------------------------

# The rocket benchmark forbids all direct player crafting; production
# must flow through the pre-placed furnace / assembler. The mask covers
# every CRAFT_* action from IRON_PLATE through ROCKET (recipe actions
# 18–35).
ROCKET_BLOCKED_ACTIONS: frozenset[int] = frozenset(
    range(int(Action.CRAFT_IRON_PLATE), int(Action.CRAFT_ROCKET) + 1)
)


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------


class RocketBenchmark:
    """Achievement-based benchmark where the goal is placing a rocket.

    Implements the :class:`~factoriax.benchmarks.core.Benchmark`
    protocol. The ``achievement_fn`` attribute is read by
    :class:`~factoriax.benchmarks.runner.BenchmarkRunner`, which wraps
    the env so that ``LevelResult.achievements_unlocked`` is populated
    on each run. Scoring is done in :meth:`score` from the latched
    unlock mask — ``score_level`` returns a placeholder because the
    protocol hands it ``items_mined`` only, which is not enough to
    compute the weighted sum.
    """

    name: str = "rocket"
    num_players: int = 1
    achievement_fn = staticmethod(rocket_conditions)
    # Hand-craft actions are blocked for this benchmark; production
    # must flow through the pre-placed furnace + assembler. Callers
    # (e.g. :class:`BenchmarkRunner`) are expected to wrap the env with
    # :class:`~factoriax.envs.action_mask_wrapper.ActionMaskWrapper`
    # using this set.
    blocked_actions: frozenset[int] = ROCKET_BLOCKED_ACTIONS

    def levels(self) -> list[BenchmarkLevel]:
        """Return the single canonical rocket level."""
        # Step budget set at 8000 — enough headroom for a
        # factory-scale plan that parallelises smelts and assemblies
        # across multiple machines.
        params = EnvParams(
            max_timesteps=8000,
            map_width=_MAP_SIZE,
            map_height=_MAP_SIZE,
            num_players=1,
        )
        return [
            BenchmarkLevel(
                name="rocket_v1",
                description=(
                    "Rocket from ore patches on a 32x32 map. Furnace and "
                    "assembler are pre-placed next to spawn; hand-craft "
                    "actions are masked. Episode ends at T=8000; score "
                    "is the weighted sum of unlocked achievements."
                ),
                level=build_rocket_level(),
                env_params=params,
            )
        ]

    def score_level(
        self, bench_level: BenchmarkLevel, items_mined: dict[str, int]
    ) -> float:
        """Per-level placeholder score.

        The real scoring reads ``achievements_unlocked`` from
        :class:`LevelResult`, which is only accessible in :meth:`score`.
        Returning zero keeps the protocol satisfied; aggregate callers
        should use :meth:`score`.
        """
        return 0.0

    def score(self, level_results: list[LevelResult]) -> float:
        """Mean weighted-achievement score across levels.

        Reads each result's ``achievements_unlocked`` mask, weights it
        by :data:`ROCKET_ACHIEVEMENT_WEIGHTS`, and returns the mean.
        Results without an unlock mask contribute zero (defensive — the
        runner should always populate the mask for this benchmark).

        Args:
            level_results: Per-level outcomes from the runner.

        Returns:
            Mean score. Range ``[0, MAX_ROCKET_SCORE]`` for a
            single-level run; one level currently.
        """
        weights_np = jnp.asarray(ROCKET_ACHIEVEMENT_WEIGHTS)
        scores: list[float] = []
        for result in level_results:
            mask = result.achievements_unlocked
            if mask is None:
                scores.append(0.0)
                continue
            scores.append(float(jnp.sum(weights_np * jnp.asarray(mask))))
        if not scores:
            return 0.0
        return sum(scores) / len(scores)
