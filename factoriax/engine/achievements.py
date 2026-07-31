"""Achievement bits that record the progress of a player.

An achievement records that the world reached a given state during an episode.
The environment state holds the bit after the unlock. A scenario can build a
curriculum on those unlocks, as Craftax does. A scenario can also read them to
follow progress towards a goal, whatever the reward function pays.

The environment state holds the achievement bits. Two scenarios can give the
same bit to two different achievements, so this module declares no achievements
of its own. It supplies the parts to declare a set:

- :class:`Achievement`, which holds one bit.
- :func:`achievement_fn` and :func:`achievement_weights`, which turn an ordered
  tuple of achievements into what the environment and the reward function read.
- The shared condition helpers at the end of this module.

The bit order is a wire format. A recorded trajectory holds
``achievements_unlocked`` as a plain bool vector, and the index is the only
identifier that survives. :mod:`factoriax.analysis` therefore reads the bits by
position. A new bit at the end of a set is safe.

CAUTION: Do not move a bit and do not remove one. Every rollout that was
recorded before the change then reads as a different set of achievements, and
nothing reports an error.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from factoriax.engine.constants import MAX_ACHIEVEMENTS, ItemType, Machine
from factoriax.engine.state import EnvState

#: Predicate for one bit. The function must be pure and jittable, and must
#: return a scalar bool.
Condition = Callable[[EnvState], jax.Array]


@dataclass(frozen=True)
class Achievement:
    """One latched bit, with its condition, its labels, and its reward.

    The condition sits next to its labels, so a scenario declares one ordered
    tuple. The name of a bit therefore cannot separate from the condition that
    unlocks it, and its weight cannot land on the wrong index.

    Attributes
    ----------
    id
        Stable identifier, unique inside one set. :func:`index_of` reads it to
        find the index of a bit, for code that holds a name and not a
        position.
    condition
        Predicate that the environment evaluates in every step. The
        environment does the latching, so this predicate answers "does the
        world satisfy the condition now", and not "did it ever satisfy the
        condition".
    name
        Label for a human reader. The default is ``id``. A scenario for an
        agent needs no other value, because the string is also the metric key
        in the training logs. Give a different value when a person reads the
        label.
    hint
        Text for the interactive UI. The engine never reads it.
    weight
        Reward for the step in which this bit first unlocks.
        :func:`achievement_weights` collects these values.
    """

    id: str
    condition: Condition
    name: str = ""
    hint: str = ""
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Set ``name`` to ``id`` when the caller gave no name."""
        if not self.name:
            object.__setattr__(self, "name", self.id)


def achievement_fn(
    achievements: Sequence[Achievement],
) -> Callable[[EnvState], jax.Array]:
    """Build the ``(EnvState) -> bool[MAX_ACHIEVEMENTS]`` evaluator.

    The result fits the ``achievement_fn`` argument of
    :class:`~factoriax.engine.envs.base.FactoriaxEnv`. The evaluator stacks the
    conditions at trace time and pads with zeros to the full number of slots.
    Every scenario therefore has the same state shape, whatever number of bits
    it declares.

    Parameters
    ----------
    achievements
        The set, in bit order. The position in this sequence is the bit index.

    Returns
    -------
    Callable[[EnvState], jax.Array]
        An evaluator that returns a ``(MAX_ACHIEVEMENTS,)`` bool array. Bit
        ``i`` says whether ``achievements[i]`` holds in the state that the
        caller passed. The slots at the end are always False. The environment
        folds the result into ``EnvState.achievements_unlocked`` with OR, so
        the latch happens there and not here.

    Raises
    ------
    ValueError
        If the set is empty, holds more than ``MAX_ACHIEVEMENTS`` entries, or
        repeats an id.
    """
    conditions = tuple(a.condition for a in _validated(achievements))
    n_padding = MAX_ACHIEVEMENTS - len(conditions)

    def evaluate(state: EnvState) -> jax.Array:
        bits = jnp.stack([condition(state) for condition in conditions])
        return jnp.concatenate([bits, jnp.zeros(n_padding, dtype=jnp.bool_)])

    return evaluate


def achievement_weights(achievements: Sequence[Achievement]) -> jax.Array:
    """Reward weight for each bit, with zeros to the length ``MAX_ACHIEVEMENTS``.

    The result feeds :func:`factoriax.engine.rewards.achievement_reward`. The
    padding slots hold zero, so they can never add reward.

    Parameters
    ----------
    achievements
        The set, in the same bit order that :func:`achievement_fn` received.

    Returns
    -------
    jax.Array
        Shape ``(MAX_ACHIEVEMENTS,)``, float32. Entry ``i`` is
        ``achievements[i].weight``.

    Raises
    ------
    ValueError
        If the set is empty, holds more than ``MAX_ACHIEVEMENTS`` entries, or
        repeats an id.
    """
    items = _validated(achievements)
    declared = jnp.array([a.weight for a in items], dtype=jnp.float32)
    return jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.float32).at[: len(items)].set(declared)


def max_score(achievements: Sequence[Achievement]) -> float:
    """Total reward for one unlock of every bit in the set.

    This is the maximum achievement reward for one episode, because a bit
    latches and pays one time at most. This function runs no validation, unlike
    the other builders here. It therefore also totals a malformed set.
    """
    return float(sum(a.weight for a in achievements))


def index_of(achievements: Sequence[Achievement], achievement_id: str) -> int:
    """Bit index of ``achievement_id``.

    The index is the position in ``achievements``, and that position indexes
    ``EnvState.achievements_unlocked``. Call this function instead of a number
    written into the code. A new bit at the end of the set then cannot move a
    lookup to the wrong index.

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
    """Return the set as a tuple, and refuse a set that is malformed."""
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
    """Total count of ``item_type`` in all player inventories."""
    total: jax.Array = jnp.sum(state.player_inventory[:, item_type])
    return total


def count_machines(state: EnvState, machine_type: int) -> jax.Array:
    """Return the number of placed machines of ``machine_type`` on the map."""
    count: jax.Array = jnp.sum(state.machine_types == machine_type)
    return count


def holds_item(state: EnvState, item: int, count: int = 1) -> jax.Array:
    """Test whether the players together hold ``count`` of ``item`` or more."""
    return count_total_items(state, item) >= count


def has_machines(state: EnvState, machine: int, count: int = 1) -> jax.Array:
    """Test whether the map holds ``count`` machines of type ``machine`` or more."""
    return count_machines(state, machine) >= count


def total_machines(state: EnvState) -> jax.Array:
    """Return the number of placed machines of all types."""
    return jnp.sum(state.machine_types != Machine.NONE)


def mined_at_least(state: EnvState, item: int, count: int) -> jax.Array:
    """Test whether the mining total for ``item`` is ``count`` or more.

    The function reads the ``items_mined`` counter, which never decreases. The
    result therefore follows the total mined amount, and not the amount that a
    player holds now.
    """
    return state.items_mined[item] >= count


def any_buffer_nonempty(state: EnvState, machine: int) -> jax.Array:
    """Test whether a placed machine of type ``machine`` holds items in its buffer.

    The function reads ``ent_buf``, which the simulation fills. A player can
    also fill it. A ``DEPOSIT_`` action writes ``ent_buf`` for every machine
    that is not a miner and has no input slots, so a player can load a pallet,
    a belt, a splitter, or a crossing by hand. A deposit never reaches the
    ``ent_buf`` of a miner, an assembler, a furnace, or a science lab. Only for
    those four kinds does a set buffer bit prove that the factory made the
    item.
    """
    is_type = state.ent_type == machine
    is_active = state.ent_y >= 0
    has_items = state.ent_buf_count > 0
    return jnp.any(is_type & is_active & has_items)


def any_assembler_has_output(state: EnvState) -> jax.Array:
    """Test whether a placed assembler holds items in its output slot."""
    is_asm = state.ent_type == Machine.ASSEMBLER
    is_active = state.ent_y >= 0
    has_output = state.ent_asm_out_count > 0
    return jnp.any(is_asm & is_active & has_output)


def total_ore_mined(state: EnvState) -> jax.Array:
    """Total count of coal, iron ore, and copper ore mined since the last reset."""
    return (
        state.items_mined[ItemType.COAL]
        + state.items_mined[ItemType.IRON_ORE]
        + state.items_mined[ItemType.COPPER_ORE]
    )
