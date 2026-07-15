"""MinerBootstrap-v1 — sparse completion-only miner bootstrap.

One scenario: from an empty inventory on the 16x16 six-patch terrain
(:mod:`factoriax.engine.envs.common`), mine limestone and silicon,
craft six miners, and get all six producing. Reward is 1.0 exactly
once per episode, paid on the step that completes the task (which also
ends the episode); every other transition pays 0. Max score 1.

The 14-bit gate ladder over the mine -> craft -> place loop remains as
zero-weight *diagnostics*: the bits latch via
:func:`~factoriax.engine.envs.base.achievement_hook` so analysis can
see how far an episode got, but they pay nothing and are not part of
the observation, so the agent sees no difference.

This sparsity is deliberate. Standard PPO is expected to fail here from
scratch — that failure is the experimental baseline. The hypothesis
under test is that a *backward curriculum over start states* solves
what sparse-reward PPO cannot: training starts from nearly-done states
(place one last miner) and walks backwards to the pristine empty-
inventory start. The earlier forward curriculum over separate staged
envs (``MineOres-v1``, ``CraftMiners-v1``, ``PlaceMiners-v1``; see git
history) lost or tied against from-scratch PPO and was removed.

The scenario shares EasyRocket-v1's world and recipe table and defaults
to the same egocentric ``superficial_local`` obs with radius 7, so a
policy trained here transfers onto EasyRocket-v1 built with the same
obs kwargs without surgery.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from factoriax.engine.constants import (
    MAX_ACHIEVEMENTS,
    Direction,
    ItemType,
    Machine,
)
from factoriax.engine.envs.base import FactoriaxEnv, achievement_hook
from factoriax.engine.envs.common import (
    MAP_SIZE,
    ORE_RESOURCES_PER_TILE,
    PATCH_BLOCKS,
    count_machines,
    producing_miners,
    six_patch_terrain,
)
from factoriax.engine.envs.easy_rocket import EASY_ROCKET_RECIPE_TABLE
from factoriax.engine.placement import place_machine
from factoriax.engine.rewards import achievement_reward
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.tables import BLOCK_TO_ITEM_ARRAY

#: Episode budget. The scripted oracle in
#: ``tests/scenarios/test_miner_curriculum.py`` pins the actual solve
#: time well under it.
_MAX_TIMESTEPS: int = 300

#: Task target: six miners, one per ore patch.
_N_MINERS: int = 6

_Condition = Callable[[EnvState], jax.Array]


def _conditions_to_achievement_fn(
    conditions: tuple[_Condition, ...],
) -> Callable[[EnvState], jax.Array]:
    """Stack per-bit conditions and zero-pad to ``MAX_ACHIEVEMENTS``."""

    def achievement_fn(state: EnvState) -> jax.Array:
        bits = jnp.stack([condition(state) for condition in conditions])
        padding = jnp.zeros(MAX_ACHIEVEMENTS - bits.shape[0], dtype=jnp.bool_)
        return jnp.concatenate([bits, padding])

    return achievement_fn


def _mined_at_least(state: EnvState, item: int, count: int) -> jax.Array:
    """Cumulative mining counter for ``item`` has reached ``count``."""
    return state.items_mined[item] >= count


def _holds_at_least(state: EnvState, item: int, count: int) -> jax.Array:
    """Player inventories hold at least ``count`` of ``item``."""
    return jnp.sum(state.player_inventory[:, item]) >= count


def _producing_at_least(state: EnvState, count: int) -> jax.Array:
    """At least ``count`` placed miners have ore in their output buffer.

    A miner on dirt never fills its buffer, so this implies the miners
    sit on ore tiles.
    """
    producing, _ = producing_miners(state)
    return jnp.sum(producing) >= count


def _placed_at_least(state: EnvState, count: int) -> jax.Array:
    """At least ``count`` miners are placed on the map."""
    return count_machines(state, int(Machine.MINER)) >= count


#: 14 latched bits over the full mine -> craft -> place loop. Only the
#: last one pays (see the weights below); the rest are free diagnostics
#: for analysis — how far did an episode get before stalling:
#:
#: - mining (4): 1 and 6 of each miner ingredient, on the monotone
#:   ``items_mined`` counter (6 of each = materials for six miners);
#: - craft (1): hold a miner;
#: - placement (3): 1 / 3 / 6 miners placed;
#: - production (6): 1..6 producing miners.
_MINER_BOOTSTRAP_CONDITIONS: tuple[_Condition, ...] = (
    partial(_mined_at_least, item=int(ItemType.LIMESTONE), count=1),
    partial(_mined_at_least, item=int(ItemType.LIMESTONE), count=_N_MINERS),
    partial(_mined_at_least, item=int(ItemType.SILICON), count=1),
    partial(_mined_at_least, item=int(ItemType.SILICON), count=_N_MINERS),
    partial(_holds_at_least, item=int(ItemType.MINER), count=1),
    partial(_placed_at_least, count=1),
    partial(_placed_at_least, count=3),
    partial(_placed_at_least, count=_N_MINERS),
    *(partial(_producing_at_least, count=count) for count in range(1, _N_MINERS + 1)),
)

NUM_MINER_BOOTSTRAP_ACHIEVEMENTS: int = len(_MINER_BOOTSTRAP_CONDITIONS)

#: Completion-only: one episode pays at most 1.0.
MINER_BOOTSTRAP_MAX_SCORE: float = 1.0

miner_bootstrap_conditions = _conditions_to_achievement_fn(_MINER_BOOTSTRAP_CONDITIONS)

#: Only the final gate pays: reward 1.0 when six miners produce, once.
#: The other 13 conditions stay latched as free diagnostics.
MINER_BOOTSTRAP_ACHIEVEMENT_WEIGHTS: jax.Array = (
    jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.float32)
    .at[NUM_MINER_BOOTSTRAP_ACHIEVEMENTS - 1]
    .set(1.0)
)


def miner_bootstrap_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Sparse reward: 1.0 when the completion bit latches, else 0."""
    return achievement_reward(
        prev_state,
        new_state,
        params,
        weights=MINER_BOOTSTRAP_ACHIEVEMENT_WEIGHTS,
    )


def all_miners_producing(state: EnvState, params: EnvParams) -> jax.Array:
    """Episode-ending condition: six miners producing at once."""
    del params
    return _producing_at_least(state, _N_MINERS)


# ---------------------------------------------------------------------------
# Backward-curriculum start states
# ---------------------------------------------------------------------------


def _check_k(k: int) -> None:
    if not 1 <= k <= _N_MINERS:
        raise ValueError(f"k must be in 1..{_N_MINERS}, got {k}")


@dataclasses.dataclass(frozen=True)
class BootstrapStart:
    """Start-state spec for one backward-curriculum stage.

    The task decomposes as mine -> craft -> place, so the ladder undoes
    it in reverse: with k = 1..6 missing miners, ``place(k)`` (A_k)
    starts one action group from done, ``craft(k)`` (B_k) two, and
    ``mine(k)`` (C_k) three — ``mine(6)`` is the pristine task. All
    counts are Python-level statics baked into the env at build time.
    """

    n_placed: int
    n_miners: int
    n_materials: int

    @classmethod
    def place(cls, k: int) -> BootstrapStart:
        """A_k: 6-k miners producing, k miners in inventory — walk & place."""
        _check_k(k)
        return cls(n_placed=_N_MINERS - k, n_miners=k, n_materials=0)

    @classmethod
    def craft(cls, k: int) -> BootstrapStart:
        """B_k: 6-k producing, k limestone + k silicon — craft & place."""
        _check_k(k)
        return cls(n_placed=_N_MINERS - k, n_miners=0, n_materials=k)

    @classmethod
    def mine(cls, k: int) -> BootstrapStart:
        """C_k: 6-k producing, empty inventory — mine, craft & place."""
        _check_k(k)
        return cls(n_placed=_N_MINERS - k, n_miners=0, n_materials=0)


def _patch_corners(world: jax.Array) -> jax.Array:
    """``(6, 2)`` array of ``(y, x)`` patch corners, in PATCH_BLOCKS order.

    Each 2x2 patch is the only occurrence of its block type, so the
    first set tile in row-major order is its top-left corner. Fixed-size
    argmax per known block type keeps this jittable and vmappable.
    """
    width = world.shape[1]
    corners = []
    for block in PATCH_BLOCKS:
        flat = jnp.argmax((world == int(block)).reshape(-1))
        corners.append(jnp.stack([flat // width, flat % width]))
    return jnp.stack(corners)


def _install_producing_miner(
    state: EnvState, params: EnvParams, corner: jax.Array
) -> EnvState:
    """Install one producing miner on the patch corner at ``corner``.

    Teleports player 0 to the tile above the corner facing down and
    routes through :func:`~factoriax.engine.placement.place_machine`,
    so the installed entity matches action-path placement field for
    field (this consumes one miner from the inventory). The miner's
    output buffer is then seeded with one unit of its tile's ore, so
    it counts as producing from step 0.
    """
    cy, cx = corner[0], corner[1]
    state = state.replace(
        player_positions=state.player_positions.at[0].set(
            jnp.stack([cx, cy - 1]).astype(state.player_positions.dtype)
        ),
        player_directions=state.player_directions.at[0].set(
            jnp.asarray(int(Direction.DOWN), dtype=state.player_directions.dtype)
        ),
    )
    state = place_machine(state, params, 0, int(ItemType.MINER))
    idx = state.tile_entity[cy, cx]
    ore_item = BLOCK_TO_ITEM_ARRAY[state.map[cy, cx].astype(jnp.int32)]
    return state.replace(
        ent_buf_type=state.ent_buf_type.at[idx].set(ore_item.astype(jnp.int8)),
        ent_buf_count=state.ent_buf_count.at[idx].set(jnp.int16(1)),
    )


def apply_start(
    key: jax.Array, state: EnvState, params: EnvParams, start: BootstrapStart
) -> EnvState:
    """Transform a pristine reset state into ``start``'s stage state.

    Pure and jittable; wired into :func:`miner_bootstrap` as a reset
    hook and importable on its own for analysis. Pre-installs
    ``start.n_placed`` producing miners — one per ore patch, patches
    drawn from ``key`` so the free patch's ore type varies per episode
    (see :func:`_install_producing_miner`). Finally the player is
    restored and the inventory set to the stage's counts.
    """
    corners = _patch_corners(state.map)
    perm = jax.random.permutation(jax.random.fold_in(key, 1), len(PATCH_BLOCKS))

    orig_positions = state.player_positions
    orig_directions = state.player_directions

    stocked = state.player_inventory.at[:, int(ItemType.MINER)].set(start.n_placed)
    state = state.replace(player_inventory=stocked)
    for i in range(start.n_placed):
        state = _install_producing_miner(state, params, corners[perm[i]])

    inventory = state.player_inventory
    inventory = inventory.at[:, int(ItemType.MINER)].set(start.n_miners)
    inventory = inventory.at[:, int(ItemType.LIMESTONE)].set(start.n_materials)
    inventory = inventory.at[:, int(ItemType.SILICON)].set(start.n_materials)
    return state.replace(
        player_positions=orig_positions,
        player_directions=orig_directions,
        player_inventory=inventory,
    )


def miner_bootstrap(
    *,
    start: BootstrapStart | None = None,
    obs: str = "superficial_local",
    obs_radius: int = 7,
) -> tuple[FactoriaxEnv, EnvParams]:
    """Build the MinerBootstrap-v1 env.

    Empty inventory; mine limestone + silicon, craft six miners, and get
    all six producing on ore. Completion pays 1.0 and ends the episode;
    everything else pays 0. Max score 1.

    Parameters
    ----------
    start :
        Backward-curriculum stage state to reset into; ``None`` builds
        the pristine task (identical to ``BootstrapStart.mine(6)``).
        World, obs, actions, reward, and budget are the same for every
        stage — only the start state differs.
    obs :
        Observation variant passed to :class:`FactoriaxEnv`.
    obs_radius :
        Half-width of the egocentric local window (the default radius 7
        gives a 15×15 view on the 16×16 map); ignored for ``_global``
        variants.
    """
    reset_hooks = () if start is None else (partial(apply_start, start=start),)
    env = FactoriaxEnv(
        terrain_fn=six_patch_terrain,
        step_hooks=(achievement_hook(miner_bootstrap_conditions),),
        reset_hooks=reset_hooks,
        reward_fn=miner_bootstrap_reward,
        done_fn=all_miners_producing,
        obs=obs,
        obs_radius=obs_radius,
        map_width=MAP_SIZE,
        map_height=MAP_SIZE,
        num_players=1,
        max_machines=100,
    )
    params = EnvParams(
        max_timesteps=_MAX_TIMESTEPS,
        recipe_table=EASY_ROCKET_RECIPE_TABLE,
        base_resources=ORE_RESOURCES_PER_TILE,
    )
    return env, params
