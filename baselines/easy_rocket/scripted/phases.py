"""Phase drivers for the easy-rocket scripted agent.

Each :class:`Phase` is a stateful driver that emits one action per
tick and exposes a :meth:`success` predicate. The :class:`agent.ScriptedAgent`
calls :meth:`step` each tick and advances to the next phase when
:meth:`success` flips true.

The phases run in the fixed production order documented in
``script.md``:

1. :class:`PhasePlanLayout` — install the layout (no action emitted).
2. :class:`PhaseBootstrapMining` — hand-mine the ores needed to
   craft every manual miner.
3. :class:`PhasePlaceManualMiners` — place each manual miner.
4..N-1. ``PhaseSection`` (one per non-target automated assembler) —
   build that section's miners, belts, arm, assembler.
N. :class:`PhaseWaitAndPlaceRocket` — wait for the target item, then
   pick it up and place it.

Predicates evaluate against :class:`EnvState` only. Deadlines are
generous safety caps; the wait-and-check phase pattern adapts to the
real production rate.
"""

from __future__ import annotations

from baselines.easy_rocket.scripted.layout import FactoryLayout
from baselines.easy_rocket.scripted.skills import (
    craft_action,
    direction_to,
    face_action,
    is_adjacent,
    step_toward_adjacent,
)
from baselines.easy_rocket.scripted.state_reader import (
    OrePatch,
    inv_count,
    player_direction,
    player_pos,
)
from factoriax.constants import Action, BlockType, ItemType
from factoriax.state import EnvParams, EnvState

# Ore ItemType -> ore BlockType (mirror of layout.ORE_ITEM_TO_BLOCK,
# duplicated here to keep phases.py independent of layout internals).
_ORE_ITEM_TO_BLOCK: dict[int, int] = {
    int(ItemType.IRON_ORE): int(BlockType.IRON),
    int(ItemType.COPPER_ORE): int(BlockType.COPPER),
    int(ItemType.TIN_ORE): int(BlockType.TIN),
    int(ItemType.SILICON): int(BlockType.SILICON),
    int(ItemType.COAL): int(BlockType.COAL),
    int(ItemType.LIMESTONE): int(BlockType.LIMESTONE),
}


def _patch_for_ore(patches: list[OrePatch], ore_item: int) -> OrePatch | None:
    """Find the patch holding the given ore item."""
    block = _ORE_ITEM_TO_BLOCK.get(ore_item)
    if block is None:
        return None
    return next((p for p in patches if p.ore_block == block), None)


def _nearest_mineable_tile(state: EnvState, patch: OrePatch) -> tuple[int, int] | None:
    """Return the patch tile closest to the player that still has ore."""
    px, py = player_pos(state)
    candidates = [t for t in patch.tiles if int(state.block_resources[t[1], t[0]]) > 0]
    if not candidates:
        return None
    candidates.sort(key=lambda t: abs(t[0] - px) + abs(t[1] - py))
    return candidates[0]


def _walk_face_act(
    state: EnvState, target: tuple[int, int], terminal_action: int
) -> int:
    """Walk adjacent to ``target``, face it, then emit ``terminal_action``.

    Returns one action per call. The caller invokes per-tick until the
    terminal action is emitted (and the relevant predicate flips).
    """
    pos = player_pos(state)
    if not is_adjacent(pos, target):
        return step_toward_adjacent(state, target[0], target[1])
    desired_dir = direction_to(pos, target)
    if desired_dir is None:
        return int(Action.NOOP)
    if player_direction(state) != desired_dir:
        return face_action(desired_dir)
    return terminal_action


class Phase:
    """Base class for one phase of the scripted agent."""

    name: str
    deadline: int

    def __init__(self, layout: FactoryLayout, patches: list[OrePatch]) -> None:
        self.layout = layout
        self.patches = patches

    def step(self, state: EnvState, params: EnvParams) -> int:
        """Return one action for the current tick."""
        raise NotImplementedError

    def success(self, state: EnvState) -> bool:
        """True when the phase's exit predicate is satisfied."""
        raise NotImplementedError


class PhasePlanLayout(Phase):
    """Install the layout. Trivially done after construction."""

    name = "1: plan layout"
    deadline = 1

    def step(self, state: EnvState, params: EnvParams) -> int:
        del state, params
        return int(Action.NOOP)

    def success(self, state: EnvState) -> bool:
        del state
        return self.layout is not None and self.layout.valid


class PhaseBootstrapMining(Phase):
    """Hand-mine ores and craft every manual miner.

    The miner recipe takes 1 LIMESTONE + 1 SILICON. The phase
    interleaves mining and crafting: as soon as we have one of each
    in inventory, we craft a miner; otherwise we mine whichever ore
    is shorter.
    """

    name = "2: bootstrap mining + craft miners"
    deadline = 1000

    def __init__(self, layout: FactoryLayout, patches: list[OrePatch]) -> None:
        super().__init__(layout, patches)
        self._n_manual = sum(1 for m in layout.miners if m.role == "manual")
        # Inputs of the MINER recipe — read from the layout's manual
        # miners list. Hardcoding {LIMESTONE, SILICON} would couple
        # this phase to easy_rocket's specific recipe; instead we
        # derive from the recipe table on the fly when needed.
        self._miner_inputs: tuple[int, int] = (
            int(ItemType.LIMESTONE),
            int(ItemType.SILICON),
        )

    def step(self, state: EnvState, params: EnvParams) -> int:
        del params
        # If we already have enough miners crafted, the agent advances
        # on its next predicate check. Emit NOOP defensively.
        if inv_count(state, int(ItemType.MINER)) >= self._n_manual:
            return int(Action.NOOP)

        a, b = self._miner_inputs
        have_a = inv_count(state, a)
        have_b = inv_count(state, b)

        # If we can craft, do it.
        if have_a >= 1 and have_b >= 1:
            return craft_action(int(ItemType.MINER))

        # Otherwise mine whichever input is shorter.
        short = a if have_a <= have_b else b
        patch = _patch_for_ore(self.patches, short)
        if patch is None:
            return int(Action.NOOP)
        tile = _nearest_mineable_tile(state, patch)
        if tile is None:
            return int(Action.NOOP)
        return _walk_face_act(state, tile, int(Action.MINE))

    def success(self, state: EnvState) -> bool:
        return inv_count(state, int(ItemType.MINER)) >= self._n_manual


# Sentinel phase used while the rest are implemented — always done,
# always success. Lets the agent skeleton smoke-test against early
# phases without crashing on later ones being None.
class _PhaseTodo(Phase):
    """Placeholder for phases that aren't implemented yet."""

    name = "TODO"
    deadline = 1

    def step(self, state: EnvState, params: EnvParams) -> int:
        del state, params
        return int(Action.NOOP)

    def success(self, state: EnvState) -> bool:
        del state
        return True


def build_phases(layout: FactoryLayout, patches: list[OrePatch]) -> list[Phase]:
    """Build the ordered list of phases for the given layout.

    Currently returns Phase 1, Phase 2, and placeholder phases for
    3..7. Subsequent commits replace the placeholders with real
    drivers.
    """
    phases: list[Phase] = [
        PhasePlanLayout(layout, patches),
        PhaseBootstrapMining(layout, patches),
    ]
    # Phases 3..7 — placeholders for now.
    phases.extend(_PhaseTodo(layout, patches) for _ in range(5))
    return phases
