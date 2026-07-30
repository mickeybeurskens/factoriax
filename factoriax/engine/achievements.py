"""Achievement system for tracking player progress.

Achievements allow persistent logging of reached world states in an episode.
Once an achievement is unlocked it is tracked as part of the environment state.
It can be used to create a curriculum based on unlocks as done with Craftax,
or can serve to track progress towards a goal regardless of reward structure.

Achievement "bits" are tracked in the environment state. Each bit can be
assigned to a different achievement in different scenarios, so this module
holds no achievements of its own, only the pieces to declare a set:
:class:`Achievement` for one bit, :func:`achievement_fn` and
:func:`achievement_weights` to turn an ordered tuple of them into what the
env and the reward function need, plus the shared condition helpers.

Bit order is a wire format. ``achievements_unlocked`` is persisted into
recorded trajectories as a bare bool vector, and the index is the only
identifier that survives, so :mod:`factoriax.analysis` reads bits
positionally. Appending to a set is safe. Reordering or removing a bit
silently reinterprets every rollout recorded before the change.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from factoriax.engine.constants import MAX_ACHIEVEMENTS, ItemType, Machine
from factoriax.engine.state import EnvState

#: A single bit's predicate: pure, jittable, returns a scalar bool.
Condition = Callable[[EnvState], jax.Array]


@dataclass(frozen=True)
class Achievement:
    """One latched bit: what unlocks it, what to call it, what it pays.

    The condition sits next to its metadata so a scenario declares one
    ordered tuple. A bit's name cannot then drift from what actually unlocks
    it, and its weight cannot land on the wrong index.

    Attributes
    ----------
    id
        Stable identifier, unique within a set. Used to find a bit's index
        with :func:`index_of` when code needs one by name rather than
        position.
    condition
        Predicate evaluated every step. Latching is the env's job, so this
        answers "does the world satisfy it now", not "has it ever".
    name
        Human-readable label. Defaults to ``id``, which is what agent-facing
        scenarios want, since the string doubles as the metric key in
        training logs. Set it when a person reads it.
    hint
        Text for the interactive UI. Never read by the engine.
    weight
        Reward paid the step this bit first unlocks. Collected by
        :func:`achievement_weights`.
    """

    id: str
    condition: Condition
    name: str = ""
    hint: str = ""
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Default ``name`` to ``id`` when the caller left it blank."""
        if not self.name:
            object.__setattr__(self, "name", self.id)


def achievement_fn(
    achievements: Sequence[Achievement],
) -> Callable[[EnvState], jax.Array]:
    """Build the ``(EnvState) -> bool[MAX_ACHIEVEMENTS]`` evaluator.

    Suitable for :class:`~factoriax.engine.envs.base.FactoriaxEnv`'s
    ``achievement_fn`` argument. Conditions are stacked at trace time and
    zero-padded to the full slot budget, so every scenario shares one
    state shape no matter how many bits it declares.

    Parameters
    ----------
    achievements
        The set, in bit order. Position in this sequence is the bit index.

    Returns
    -------
    Callable[[EnvState], jax.Array]
        Evaluator returning a ``(MAX_ACHIEVEMENTS,)`` bool array. Bit ``i``
        is whether ``achievements[i]`` holds in the state passed, with the
        trailing slots always False. The env OR-folds the result into
        ``EnvState.achievements_unlocked``, so latching happens there and
        not here.

    Raises
    ------
    ValueError
        If the set is empty, exceeds ``MAX_ACHIEVEMENTS``, or repeats an id.
    """
    conditions = tuple(a.condition for a in _validated(achievements))
    n_padding = MAX_ACHIEVEMENTS - len(conditions)

    def evaluate(state: EnvState) -> jax.Array:
        bits = jnp.stack([condition(state) for condition in conditions])
        return jnp.concatenate([bits, jnp.zeros(n_padding, dtype=jnp.bool_)])

    return evaluate


def achievement_weights(achievements: Sequence[Achievement]) -> jax.Array:
    """Per-bit reward weights, zero-padded to ``(MAX_ACHIEVEMENTS,)``.

    Feeds :func:`factoriax.engine.rewards.achievement_reward`. Padding
    slots are zero so they can never contribute reward.

    Parameters
    ----------
    achievements
        The set, in the same bit order passed to :func:`achievement_fn`.

    Returns
    -------
    jax.Array
        Shape ``(MAX_ACHIEVEMENTS,)``, float32. Entry ``i`` is
        ``achievements[i].weight``.

    Raises
    ------
    ValueError
        If the set is empty, exceeds ``MAX_ACHIEVEMENTS``, or repeats an id.
    """
    items = _validated(achievements)
    declared = jnp.array([a.weight for a in items], dtype=jnp.float32)
    return jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.float32).at[: len(items)].set(declared)


def max_score(achievements: Sequence[Achievement]) -> float:
    """Total reward from unlocking every bit in the set exactly once.

    The ceiling for achievement reward over one episode, since a bit latches
    and so pays at most once. Unlike the other builders this runs no
    validation, so it will happily total a malformed set.
    """
    return float(sum(a.weight for a in achievements))


def index_of(achievements: Sequence[Achievement], achievement_id: str) -> int:
    """Bit index of ``achievement_id``.

    The index is the position in ``achievements``, which is what indexes
    ``EnvState.achievements_unlocked``. Use this rather than hardcoding a
    number, so a bit appended to the set does not silently shift a lookup.

    Raises
    ------
    KeyError
        If no achievement in the set carries that id.
    """
    for i, item in enumerate(achievements):
        if item.id == achievement_id:
            return i
    raise KeyError(achievement_id)


def _validated(achievements: Sequence[Achievement]) -> tuple[Achievement, ...]:
    """Return the set as a tuple, rejecting the ways it can be malformed."""
    items = tuple(achievements)
    if not items:
        raise ValueError("achievement set is empty")
    if len(items) > MAX_ACHIEVEMENTS:
        raise ValueError(
            f"{len(items)} achievements exceeds MAX_ACHIEVEMENTS ({MAX_ACHIEVEMENTS})"
        )
    ids = [a.id for a in items]
    duplicates = sorted({id_ for id_ in ids if ids.count(id_) > 1})
    if duplicates:
        raise ValueError(f"duplicate achievement ids: {duplicates}")
    return items


# ---------------------------------------------------------------------------
# Shared condition helpers
# ---------------------------------------------------------------------------


def count_total_items(state: EnvState, item_type: int) -> jax.Array:
    """Total quantity of ``item_type`` across all player inventories."""
    total: jax.Array = jnp.sum(state.player_inventory[:, item_type])
    return total


def count_machines(state: EnvState, machine_type: int) -> jax.Array:
    """Return how many placed machines of ``machine_type`` are on the map."""
    count: jax.Array = jnp.sum(state.machine_types == machine_type)
    return count


def holds_item(state: EnvState, item: int, count: int = 1) -> jax.Array:
    """Players hold at least ``count`` of ``item`` between them."""
    return count_total_items(state, item) >= count


def has_machines(state: EnvState, machine: int, count: int = 1) -> jax.Array:
    """At least ``count`` machines of type ``machine`` are placed."""
    return count_machines(state, machine) >= count


def total_machines(state: EnvState) -> jax.Array:
    """Return how many machines of any type are placed."""
    return jnp.sum(state.machine_types != Machine.NONE)


def mined_at_least(state: EnvState, item: int, count: int) -> jax.Array:
    """Cumulative mining counter for ``item`` has reached ``count``.

    Reads the monotone ``items_mined`` counter, so this latches on what
    was ever mined rather than what is currently held.
    """
    return state.items_mined[item] >= count


def any_buffer_nonempty(state: EnvState, machine: int) -> jax.Array:
    """Any placed machine of type ``machine`` has items in its buffer.

    Reads ``ent_buf``, which the simulation fills. A player can fill it too:
    a ``DEPOSIT_`` action writes ``ent_buf`` for any machine that is neither
    a miner nor a combiner, so a pallet, belt, splitter, or crossing can be
    loaded by hand. A miner is excluded from deposit and a combiner takes
    the item into its input slots instead, so only for those two does a set
    buffer bit mean the factory produced the item.
    """
    is_type = state.ent_type == machine
    is_active = state.ent_y >= 0
    has_items = state.ent_buf_count > 0
    return jnp.any(is_type & is_active & has_items)


def any_assembler_has_output(state: EnvState) -> jax.Array:
    """Any placed assembler has items in its output slot."""
    is_asm = state.ent_type == Machine.ASSEMBLER
    is_active = state.ent_y >= 0
    has_output = state.ent_asm_out_count > 0
    return jnp.any(is_asm & is_active & has_output)


def total_ore_mined(state: EnvState) -> jax.Array:
    """Cumulative count of coal, iron ore and copper ore ever mined."""
    return (
        state.items_mined[ItemType.COAL]
        + state.items_mined[ItemType.IRON_ORE]
        + state.items_mined[ItemType.COPPER_ORE]
    )
