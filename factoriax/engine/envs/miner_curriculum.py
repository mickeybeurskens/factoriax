"""MinerBootstrap-v1: a sparse task that pays only on completion.

The player starts with an empty inventory on the 16x16 six-patch terrain of
:mod:`factoriax.engine.envs.common`. The task is to mine limestone and silicon,
craft six miners, and put all six into production. The reward is 1.0 exactly
one time in an episode, on the step that completes the task. That step also
ends the episode. Every other step pays 0, and the highest score is 1.

The ladder of 14 gate bits over the mine, craft, and place loop stays as a
zero-weight diagnostic. The bits latch through
:func:`~factoriax.engine.envs.base.achievement_hook`, so analysis code can see
how far an episode reached. They pay nothing, and they are not part of the
observation, so the agent sees no difference.

This sparsity is deliberate. Standard PPO from scratch is expected to fail
here, and that failure is the experimental baseline. The hypothesis under test
is that a backward curriculum over start states solves what sparse-reward PPO
cannot. Training starts from states that are almost done, where the agent
places one last miner, and walks backwards to the empty-inventory start. An
earlier forward curriculum used three separate staged environments:
``MineOres-v1``, ``CraftMiners-v1``, and ``PlaceMiners-v1``. The git history
holds them. That curriculum lost against PPO from scratch or drew with it, so
it is gone.

This scenario shares the world and the recipe table of EasyRocket-v1. It
defaults to the same egocentric ``superficial_local`` observation with radius
7. A policy that trains here therefore transfers to an EasyRocket-v1 built with
the same observation arguments, with no change.
"""

from __future__ import annotations

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp

from factoriax.engine.achievements import (
    Achievement,
    achievement_fn,
    achievement_weights,
    count_machines,
    holds_item,
    max_score,
    mined_at_least,
)
from factoriax.engine.constants import (
    Direction,
    ItemType,
    Machine,
)
from factoriax.engine.envs.base import FactoriaxEnv, achievement_hook
from factoriax.engine.envs.common import (
    MAP_SIZE,
    ORE_RESOURCES_PER_TILE,
    PATCH_BLOCKS,
    producing_miners,
    six_patch_terrain,
)
from factoriax.engine.envs.easy_rocket import EASY_ROCKET_RECIPE_TABLE
from factoriax.engine.placement import place_machine
from factoriax.engine.rewards import achievement_reward
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.tables import BLOCK_TO_ITEM_ARRAY

#: Length of an episode. The scripted oracle in
#: ``tests/scenarios/test_miner_curriculum.py`` solves the task in far fewer
#: steps than this.
_MAX_TIMESTEPS: int = 300

#: The target of the task: six miners, one for each ore patch.
_N_MINERS: int = 6


def _producing_at_least(state: EnvState, count: int) -> jax.Array:
    """Test whether ``count`` placed miners or more hold ore in their buffer.

    A miner on dirt never fills its buffer, so a True result also means that
    those miners stand on ore tiles.
    """
    producing, _ = producing_miners(state)
    return jnp.sum(producing) >= count


def _placed_at_least(state: EnvState, count: int) -> jax.Array:
    """Test whether the map holds ``count`` placed miners or more."""
    return count_machines(state, int(Machine.MINER)) >= count


#: 14 latched bits over the whole mine, craft, and place loop. Only the last
#: bit pays. The other 13 carry weight 0 and serve as a diagnostic for
#: analysis: how far did an episode reach before it stopped. The bits are:
#:
#: - 4 mining bits: 1 and 6 of each miner ingredient, on the ``items_mined``
#:   counter, which never decreases. 6 of each is the material for six miners.
#: - 1 craft bit: the player holds a miner.
#: - 3 placement bits: 1, 3, and 6 miners placed.
#: - 6 production bits: 1 to 6 producing miners. The last one is the task.
MINER_BOOTSTRAP_ACHIEVEMENTS: tuple[Achievement, ...] = (
    Achievement(
        "mine_limestone",
        partial(mined_at_least, item=int(ItemType.LIMESTONE), count=1),
        name="Mine Limestone",
        weight=0.0,
    ),
    Achievement(
        "limestone_stocked",
        partial(mined_at_least, item=int(ItemType.LIMESTONE), count=_N_MINERS),
        name="Limestone Stocked",
        weight=0.0,
    ),
    Achievement(
        "mine_silicon",
        partial(mined_at_least, item=int(ItemType.SILICON), count=1),
        name="Mine Silicon",
        weight=0.0,
    ),
    Achievement(
        "silicon_stocked",
        partial(mined_at_least, item=int(ItemType.SILICON), count=_N_MINERS),
        name="Silicon Stocked",
        weight=0.0,
    ),
    Achievement(
        "craft_miner",
        partial(holds_item, item=int(ItemType.MINER)),
        name="Craft Miner",
        weight=0.0,
    ),
    *(
        Achievement(
            f"place_{count}_miners",
            partial(_placed_at_least, count=count),
            name=f"Place {count} Miners",
            weight=0.0,
        )
        for count in (1, 3, _N_MINERS)
    ),
    *(
        Achievement(
            f"producing_{count}_miners",
            partial(_producing_at_least, count=count),
            name=f"{count} Miners Producing",
            #: Only the final gate pays: 1.0 when six miners produce, once.
            weight=1.0 if count == _N_MINERS else 0.0,
        )
        for count in range(1, _N_MINERS + 1)
    ),
)

NUM_MINER_BOOTSTRAP_ACHIEVEMENTS: int = len(MINER_BOOTSTRAP_ACHIEVEMENTS)

#: Completion-only: one episode pays at most 1.0.
MINER_BOOTSTRAP_MAX_SCORE: float = max_score(MINER_BOOTSTRAP_ACHIEVEMENTS)

miner_bootstrap_conditions = achievement_fn(MINER_BOOTSTRAP_ACHIEVEMENTS)

MINER_BOOTSTRAP_ACHIEVEMENT_WEIGHTS: jax.Array = achievement_weights(
    MINER_BOOTSTRAP_ACHIEVEMENTS
)


def miner_bootstrap_reward(
    prev_state: EnvState, new_state: EnvState, params: EnvParams
) -> jax.Array:
    """Sparse reward: 1.0 on the step where the completion bit latches, else 0."""
    return achievement_reward(
        prev_state,
        new_state,
        params,
        weights=MINER_BOOTSTRAP_ACHIEVEMENT_WEIGHTS,
    )


def all_miners_producing(state: EnvState, params: EnvParams) -> jax.Array:
    """End condition of an episode: six miners produce at the same time."""
    del params
    return _producing_at_least(state, _N_MINERS)


# ---------------------------------------------------------------------------
# The start states of the backward curriculum
# ---------------------------------------------------------------------------


def _check_k(k: int) -> None:
    if not 1 <= k <= _N_MINERS:
        raise ValueError(f"k must be in 1..{_N_MINERS}, got {k}")


@dataclasses.dataclass(frozen=True)
class BootstrapStart:
    """Start-state spec for one stage of the backward curriculum.

    The task divides into mine, then craft, then place. The ladder undoes it in
    the reverse order. With ``k`` missing miners, from 1 to 6, ``place(k)``, or
    A_k, starts one group of actions from the end. ``craft(k)``, or B_k, starts
    two groups from the end, and ``mine(k)``, or C_k, starts three groups from
    the end. ``mine(6)`` is the full task. Every count is a Python static that
    the environment holds from build time.
    """

    n_placed: int
    n_miners: int
    n_materials: int
    partial_mine: bool = False

    @classmethod
    def place(cls, k: int) -> BootstrapStart:
        """A_k: 6-k miners produce, k miners in the inventory. Walk and place."""
        _check_k(k)
        return cls(n_placed=_N_MINERS - k, n_miners=k, n_materials=0)

    @classmethod
    def craft(cls, k: int) -> BootstrapStart:
        """B_k: 6-k produce, k limestone and k silicon. Craft and place."""
        _check_k(k)
        return cls(n_placed=_N_MINERS - k, n_miners=0, n_materials=k)

    @classmethod
    def mine_partial(cls, k: int) -> BootstrapStart:
        """B'_k: 6-k produce, k of one ore. Mine the other ore, craft, and place.

        The ore that the player holds is limestone or silicon, and each episode
        draws it with equal probability. The policy must therefore learn to
        mine both.
        """
        _check_k(k)
        return cls(
            n_placed=_N_MINERS - k,
            n_miners=0,
            n_materials=k,
            partial_mine=True,
        )

    @classmethod
    def mine(cls, k: int) -> BootstrapStart:
        """C_k: 6-k produce, empty inventory. Mine, craft, and place."""
        _check_k(k)
        return cls(n_placed=_N_MINERS - k, n_miners=0, n_materials=0)


def _patch_corners(world: jax.Array) -> jax.Array:
    """Return a ``(6, 2)`` array of ``(y, x)`` corners, in PATCH_BLOCKS order.

    Each 2x2 patch is the one place where its block type appears. The first
    tile of that type, in row-major order, is therefore its top-left corner.
    One argmax of fixed size for each known block type keeps the function
    jittable and vmappable.
    """
    width = world.shape[1]
    corners = []
    for block in PATCH_BLOCKS:
        flat = jnp.argmax((world == int(block)).reshape(-1))
        corners.append(jnp.stack([flat // width, flat % width]))
    return jnp.stack(corners)


def apply_start(
    key: jax.Array, state: EnvState, params: EnvParams, start: BootstrapStart
) -> EnvState:
    """Turn a plain reset state into the stage state that ``start`` describes.

    The function is pure and jittable. :func:`miner_bootstrap` uses it as a
    reset hook, and analysis code can import it on its own.

    It places ``start.n_placed`` miners, one on each ore patch. ``key`` selects
    the patches, so the ore type of the free patch changes with the episode.
    Each placement goes through
    :func:`~factoriax.engine.placement.place_machine`, and the function moves
    the player above each patch corner for that one call. The new entity
    therefore matches a placement from the action path, field for field.

    The function then puts one unit of the ore of its tile into the output
    buffer of each new miner, so the miner counts as a producer from step 0.
    Last, it puts the player back where it was and sets the inventory to the
    counts of the stage.
    """
    corners = _patch_corners(state.map)
    perm = jax.random.permutation(jax.random.fold_in(key, 1), len(PATCH_BLOCKS))

    orig_positions = state.player_positions
    orig_directions = state.player_directions

    inventory = state.player_inventory.at[:, int(ItemType.MINER)].set(start.n_placed)
    state = state.replace(player_inventory=inventory)
    for i in range(start.n_placed):
        cy, cx = corners[perm[i], 0], corners[perm[i], 1]
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
        state = state.replace(
            ent_buf_type=state.ent_buf_type.at[idx].set(ore_item.astype(jnp.int8)),
            ent_buf_count=state.ent_buf_count.at[idx].set(jnp.int16(1)),
        )

    inventory = state.player_inventory
    inventory = inventory.at[:, int(ItemType.MINER)].set(start.n_miners)
    n_limestone = jnp.asarray(start.n_materials, dtype=inventory.dtype)
    n_silicon = jnp.asarray(start.n_materials, dtype=inventory.dtype)
    if start.partial_mine:
        hold_limestone = jax.random.bernoulli(jax.random.fold_in(key, 2))
        n_limestone = jnp.where(hold_limestone, n_limestone, 0)
        n_silicon = jnp.where(hold_limestone, 0, n_silicon)
    inventory = inventory.at[:, int(ItemType.LIMESTONE)].set(n_limestone)
    inventory = inventory.at[:, int(ItemType.SILICON)].set(n_silicon)
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
    """Build the MinerBootstrap-v1 environment.

    The player starts with an empty inventory. The task is to mine limestone
    and silicon, craft six miners, and put all six into production on ore.
    Completion pays 1.0 and ends the episode. Every other step pays 0, and the
    highest score is 1.

    Parameters
    ----------
    start :
        Stage state of the backward curriculum to reset into. ``None`` builds
        the full task, which equals ``BootstrapStart.mine(6)``. The world, the
        observation, the actions, the reward, and the step budget are the same
        for every stage. Only the start state changes.
    obs :
        Observation variant. The function passes it to :class:`FactoriaxEnv`.
    obs_radius :
        Half-width of the egocentric local window. The default radius 7 gives a
        15x15 view on the 16x16 map. A ``_global`` variant ignores it.
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
