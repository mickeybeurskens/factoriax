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

import logging

import numpy as np

from baselines.easy_rocket.scripted.layout import (
    AssemblerPlan,
    BeltPlan,
    CrossingPlan,
    FactoryLayout,
    MinerPlan,
    PalletPlan,
)
from baselines.easy_rocket.scripted.skills import (
    craft_action,
    direction_to,
    face_action,
    is_adjacent,
    place_action,
    rotate_action,
    step_toward_adjacent,
)
from baselines.easy_rocket.scripted.state_reader import (
    OrePatch,
    ent_direction_at,
    entity_at,
    inv_count,
    machine_at,
    player_direction,
    player_pos,
)
from factoriax.constants import Action, BlockType, ItemType, MachineType
from factoriax.state import EnvParams, EnvState

# Recipe inputs map: crafted ItemType -> list of (input ItemType, count).
# Populated by :func:`build_phases` from the runtime recipe table.
RecipeInputs = dict[int, list[tuple[int, int]]]

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

    def debug_status(self, state: EnvState) -> str:
        """One-line summary of what the phase is currently doing.

        Default returns the phase name. Subclasses override to
        report current sub-step / target / blockers.
        """
        return self.name


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
    """Hand-mine ores and craft the bootstrap machines.

    The bootstrap is the manual miners plus one pallet each (the
    pallet buffers the miner's output). The phase batch-mines every
    ore the bootstrap recipes need — visiting each patch once — then
    crafts all miners and pallets. Mining strictly precedes crafting
    so the ore-gather never thrashes once crafting starts consuming.
    """

    name = "2: bootstrap mining + craft miners/pallets"
    deadline = 1000

    def __init__(
        self,
        layout: FactoryLayout,
        patches: list[OrePatch],
        recipe_inputs: RecipeInputs,
    ) -> None:
        super().__init__(layout, patches)
        self._recipe_inputs = recipe_inputs
        n_manual = sum(1 for m in layout.miners if m.role == "manual")
        n_pallets = len(layout.pallets)
        # Bootstrap bill of materials and the total ore it consumes.
        self._bom: dict[int, int] = {
            int(ItemType.MINER): n_manual,
            int(ItemType.PALLET): n_pallets,
        }
        self._ore_need: dict[int, int] = {}
        for item, count in self._bom.items():
            for inp_item, inp_count in recipe_inputs.get(item, []):
                self._ore_need[inp_item] = (
                    self._ore_need.get(inp_item, 0) + count * inp_count
                )

    def _mine(self, state: EnvState, ore_item: int) -> int:
        patch = _patch_for_ore(self.patches, ore_item)
        if patch is None:
            return int(Action.NOOP)
        tile = _nearest_mineable_tile(state, patch)
        if tile is None:
            return int(Action.NOOP)
        return _walk_face_act(state, tile, int(Action.MINE))

    def _can_craft(self, state: EnvState, item: int) -> bool:
        return all(
            inv_count(state, inp) >= cnt
            for inp, cnt in self._recipe_inputs.get(item, [])
        )

    def step(self, state: EnvState, params: EnvParams) -> int:
        del params
        if self.success(state):
            return int(Action.NOOP)

        # Phase A — gather ALL ore first (crafting hasn't started yet).
        crafted_total = sum(inv_count(state, item) for item in self._bom)
        if crafted_total == 0:
            for ore, need in self._ore_need.items():
                if inv_count(state, ore) < need:
                    return self._mine(state, ore)

        # Phase B — craft the bootstrap items.
        for item, count in self._bom.items():
            if inv_count(state, item) < count and self._can_craft(state, item):
                return craft_action(item)
        return int(Action.NOOP)

    def success(self, state: EnvState) -> bool:
        return all(inv_count(state, item) >= count for item, count in self._bom.items())


logger = logging.getLogger("easy_rocket_scripted.phases")


class _Target:
    """One machine to place + (optionally) orient.

    ``facing`` is the required ``ent_direction`` after placement, or
    ``None`` when facing is irrelevant (assemblers, manual miners).
    """

    __slots__ = ("pos", "item", "machine_type", "facing")

    def __init__(
        self,
        pos: tuple[int, int],
        item: int,
        machine_type: int,
        facing: int | None,
    ) -> None:
        self.pos = pos
        self.item = item
        self.machine_type = machine_type
        self.facing = facing


class _Placer:
    """Place + orient a list of :class:`_Target`s, verifying each.

    A target is *done* when a machine of the right type occupies its
    tile AND (if a facing is required) its ``ent_direction`` matches.
    Placement happens from any walkable adjacent tile; facing is then
    corrected with ``ROTATE_*`` (absolute set), retried up to
    ``MAX_ROTATE`` times before the target is logged as failed and
    skipped. Every transition is logged so a stalled build is
    diagnosable from the run output.
    """

    MAX_ROTATE = 4

    def __init__(self, targets: list[_Target], label: str) -> None:
        self._targets = targets
        self._label = label
        self._rotate_attempts: dict[tuple[int, int], int] = {}
        self._failed: set[tuple[int, int]] = set()
        self._logged_done: set[tuple[int, int]] = set()
        self._active: tuple[int, int] | None = None

    def _done(self, state: EnvState, t: _Target) -> bool:
        if machine_at(state, t.pos[0], t.pos[1]) != t.machine_type:
            return False
        if t.facing is None:
            return True
        return ent_direction_at(state, t.pos[0], t.pos[1]) == t.facing

    def all_done(self, state: EnvState) -> bool:
        return all(t.pos in self._failed or self._done(state, t) for t in self._targets)

    def pending(self, state: EnvState) -> list[str]:
        """Human-readable list of targets not yet done (for debug)."""
        out: list[str] = []
        for t in self._targets:
            if t.pos in self._failed or self._done(state, t):
                continue
            placed = machine_at(state, t.pos[0], t.pos[1]) == t.machine_type
            tag = "rotate" if placed else "place"
            out.append(f"{ItemType(t.item).name}@{t.pos}[{tag}]")
        return out

    def next_action(self, state: EnvState) -> int:
        for t in self._targets:
            if t.pos in self._failed:
                continue
            if self._done(state, t):
                if t.pos not in self._logged_done:
                    logger.info(
                        "[%s] done %s @ %s facing=%s",
                        self._label,
                        ItemType(t.item).name,
                        t.pos,
                        t.facing,
                    )
                    self._logged_done.add(t.pos)
                continue
            if self._active != t.pos:
                logger.info(
                    "[%s] working %s @ %s (want facing=%s)",
                    self._label,
                    ItemType(t.item).name,
                    t.pos,
                    t.facing,
                )
                self._active = t.pos

            # --- Placement phase ---
            if machine_at(state, t.pos[0], t.pos[1]) != t.machine_type:
                if inv_count(state, t.item) <= 0:
                    logger.warning(
                        "[%s] cannot place %s @ %s: 0 in inventory",
                        self._label,
                        ItemType(t.item).name,
                        t.pos,
                    )
                    return int(Action.NOOP)
                return _walk_face_act(state, t.pos, place_action(t.item))

            # --- Facing phase ---
            if t.facing is None:
                continue
            attempts = self._rotate_attempts.get(t.pos, 0)
            if attempts >= self.MAX_ROTATE:
                logger.error(
                    "[%s] rotate timeout %s @ %s: want %s got %s — skipping",
                    self._label,
                    ItemType(t.item).name,
                    t.pos,
                    t.facing,
                    ent_direction_at(state, t.pos[0], t.pos[1]),
                )
                self._failed.add(t.pos)
                continue
            ppos = player_pos(state)
            if not is_adjacent(ppos, t.pos):
                return step_toward_adjacent(state, t.pos[0], t.pos[1])
            desired = direction_to(ppos, t.pos)
            if desired is None:
                return int(Action.NOOP)
            if player_direction(state) != desired:
                return face_action(desired)
            self._rotate_attempts[t.pos] = attempts + 1
            logger.info(
                "[%s] rotate %s @ %s -> %s (attempt %d/%d)",
                self._label,
                ItemType(t.item).name,
                t.pos,
                t.facing,
                attempts + 1,
                self.MAX_ROTATE,
            )
            return rotate_action(t.facing)
        return int(Action.NOOP)


class PhasePlaceManualMiners(Phase):
    """Place each manual miner and its output pallet.

    The manual miner sits on an ore tile and faces its pallet (on an
    adjacent free tile), pushing mined ore into the pallet's large
    buffer. The agent later drains the pallet with WITHDRAW. Pallets
    are placed first so they exist before the miner is oriented at
    them (cosmetic ordering — rotation works regardless).
    """

    name = "3: place manual miners + pallets"
    deadline = 600

    def __init__(self, layout: FactoryLayout, patches: list[OrePatch]) -> None:
        super().__init__(layout, patches)
        targets: list[_Target] = []
        # Pallets first (no facing).
        for pal in layout.pallets:
            targets.append(
                _Target(pal.pos, int(ItemType.PALLET), int(MachineType.PALLET), None)
            )
        # Then manual miners, each facing its pallet.
        for m in layout.miners:
            if m.role != "manual":
                continue
            targets.append(
                _Target(m.pos, int(ItemType.MINER), int(MachineType.MINER), m.facing)
            )
        self._placer = _Placer(targets, label="phase3/manual-miners+pallets")

    def step(self, state: EnvState, params: EnvParams) -> int:
        del params
        return self._placer.next_action(state)

    def success(self, state: EnvState) -> bool:
        return self._placer.all_done(state)

    def debug_status(self, state: EnvState) -> str:
        pending = self._placer.pending(state)[:4]
        return f"{self.name} pos={player_pos(state)} pending={pending}"


def _withdraw_from_pallet_step(
    state: EnvState, pallets: tuple[PalletPlan, ...], wanted_ore: int
) -> int | None:
    """Walk + face + WITHDRAW a pallet holding ``wanted_ore``.

    Returns ``None`` if no pallet currently holds the wanted ore — the
    caller should NOOP that tick and check again next tick (the manual
    miner is still filling it).
    """
    for pal in pallets:
        if pal.source_ore != wanted_ore:
            continue
        eidx = entity_at(state, pal.pos[0], pal.pos[1])
        if eidx < 0:
            continue
        buf_type, buf_count = ent_buf_lookup(state, eidx)
        if buf_count > 0 and buf_type == wanted_ore:
            return _walk_face_act(state, pal.pos, int(Action.WITHDRAW))
    return None


# Re-export ent_buf with a local alias to avoid a name clash.
def ent_buf_lookup(state: EnvState, eidx: int) -> tuple[int, int]:
    from baselines.easy_rocket.scripted.state_reader import (  # noqa: PLC0415
        ent_buf,
    )

    return ent_buf(state, eidx)


class PhaseSection(Phase):
    """Build one automated assembler + its inputs (miners, belts, crossings).

    Used for the engine and hull sections (any non-target intermediate).
    Drives the wait-craft-place pattern: pick up ore from manual
    miners, hand-craft what's missing, place each entity in its
    layout-prescribed position.

    Predicate: every section entity exists at its layout tile AND
    the section assembler shows non-zero ``ent_asm_in_count`` for at
    least one of its inputs (the belt is actually flowing material).
    """

    deadline = 1200

    def __init__(
        self,
        layout: FactoryLayout,
        patches: list[OrePatch],
        section_output: int,
        recipe_inputs: RecipeInputs,
    ) -> None:
        super().__init__(layout, patches)
        self.section_output = section_output
        self.name = f"section: {ItemType(section_output).name}"
        self._recipe_inputs = recipe_inputs
        self._assemblers: list[AssemblerPlan] = [
            a for a in layout.assemblers if a.recipe_output == section_output
        ]
        self._miners: list[MinerPlan] = [
            m
            for m in layout.miners
            if m.role == "factory" and m.consumer_recipe_output == section_output
        ]
        self._belts: list[BeltPlan] = [
            b for b in layout.belts if b.consumer_recipe_output == section_output
        ]
        self._crossings: list[CrossingPlan] = [
            c for c in layout.crossings if c.consumer_recipe_output == section_output
        ]
        self._pallets: tuple[PalletPlan, ...] = layout.pallets
        # Build placement targets in placement order: assembler first
        # (facing irrelevant), belts downstream-first (reversed),
        # crossings, factory miners last. Belts/crossings/miners
        # require an exact facing so items actually flow.
        targets: list[_Target] = []
        for a in self._assemblers:
            targets.append(
                _Target(
                    a.pos, int(ItemType.ASSEMBLER), int(MachineType.ASSEMBLER), None
                )
            )
        for b in reversed(self._belts):
            targets.append(
                _Target(
                    b.pos,
                    int(ItemType.CONVEYOR_BELT),
                    int(MachineType.CONVEYOR_BELT),
                    b.facing,
                )
            )
        for c in self._crossings:
            targets.append(
                _Target(
                    c.pos,
                    int(ItemType.CROSSING),
                    int(MachineType.CROSSING),
                    c.ent_direction,
                )
            )
        for m in self._miners:
            targets.append(
                _Target(m.pos, int(ItemType.MINER), int(MachineType.MINER), m.facing)
            )
        self._targets = targets
        self._placer = _Placer(
            targets, label=f"section/{ItemType(section_output).name}"
        )

    def _items_still_to_craft(self, state: EnvState) -> dict[int, int]:
        """How many of each item still need crafting.

        Counts only targets whose machine is not yet placed, minus
        what's already in inventory. This avoids re-crafting items as
        placement consumes them (the earlier 'extra miner' bug).
        """
        need: dict[int, int] = {}
        for t in self._targets:
            if machine_at(state, t.pos[0], t.pos[1]) != t.machine_type:
                need[t.item] = need.get(t.item, 0) + 1
        return {
            item: max(0, cnt - inv_count(state, item)) for item, cnt in need.items()
        }

    def _can_craft(self, state: EnvState, item: int) -> bool:
        for inp_item, inp_count in self._recipe_inputs.get(item, []):
            if inv_count(state, inp_item) < inp_count:
                return False
        return True

    def step(self, state: EnvState, params: EnvParams) -> int:
        del params
        if self._placer.all_done(state):
            return int(Action.NOOP)

        # Craft any item still short for an unplaced target.
        remaining = self._items_still_to_craft(state)
        for item, count in remaining.items():
            if count > 0 and self._can_craft(state, item):
                return craft_action(item)
        # Need ore: withdraw from a manual miner holding a missing input.
        for item, count in remaining.items():
            if count <= 0:
                continue
            for inp_item, inp_count in self._recipe_inputs.get(item, []):
                if inv_count(state, inp_item) < inp_count:
                    action = _withdraw_from_pallet_step(state, self._pallets, inp_item)
                    if action is not None:
                        return action
        # Everything craftable is crafted — place + orient.
        return self._placer.next_action(state)

    def success(self, state: EnvState) -> bool:
        if not self._placer.all_done(state):
            return False
        # Confirm the section assembler is actually receiving inputs —
        # the belt is flowing, not just placed.
        asm = self._assemblers[0]
        eidx = entity_at(state, asm.pos[0], asm.pos[1])
        if eidx < 0:
            return False
        in_counts = np.asarray(state.ent_asm_in_count)[eidx]
        return bool((in_counts > 0).any())

    def debug_status(self, state: EnvState) -> str:
        remaining = {
            ItemType(k).name: v
            for k, v in self._items_still_to_craft(state).items()
            if v > 0
        }
        asm = self._assemblers[0]
        eidx = entity_at(state, asm.pos[0], asm.pos[1])
        flow = np.asarray(state.ent_asm_in_count)[eidx].tolist() if eidx >= 0 else None
        return (
            f"{self.name} pos={player_pos(state)} to_craft={remaining} "
            f"pending={self._placer.pending(state)[:4]} asm_in={flow}"
        )


class PhaseRocketSection(Phase):
    """Build the rocket section: arms, ROCKET assembler, output belts."""

    name = "6: rocket section"
    deadline = 1200

    def __init__(
        self,
        layout: FactoryLayout,
        patches: list[OrePatch],
        target: int,
        recipe_inputs: RecipeInputs,
    ) -> None:
        super().__init__(layout, patches)
        self._target = target
        self._recipe_inputs = recipe_inputs
        self._assemblers: list[AssemblerPlan] = [
            a for a in layout.assemblers if a.recipe_output == target
        ]
        # ALL arms get placed here (they drain non-target assemblers
        # to feed the target assembler).
        self._arms = list(layout.arms)
        # Output belts route into the target assembler.
        self._belts: list[BeltPlan] = [
            b for b in layout.belts if b.consumer_recipe_output == target
        ]
        self._crossings: list[CrossingPlan] = [
            c for c in layout.crossings if c.consumer_recipe_output == target
        ]
        self._pallets: tuple[PalletPlan, ...] = layout.pallets
        # Targets in placement order: assembler, output belts
        # downstream-first (reversed), crossings, arms last (an arm
        # sits upstream of its output belt, so placing it last keeps
        # that belt's tile clear).
        targets: list[_Target] = []
        for a in self._assemblers:
            targets.append(
                _Target(
                    a.pos, int(ItemType.ASSEMBLER), int(MachineType.ASSEMBLER), None
                )
            )
        for b in reversed(self._belts):
            targets.append(
                _Target(
                    b.pos,
                    int(ItemType.CONVEYOR_BELT),
                    int(MachineType.CONVEYOR_BELT),
                    b.facing,
                )
            )
        for c in self._crossings:
            targets.append(
                _Target(
                    c.pos,
                    int(ItemType.CROSSING),
                    int(MachineType.CROSSING),
                    c.ent_direction,
                )
            )
        for arm in self._arms:
            targets.append(
                _Target(arm.pos, int(ItemType.ARM), int(MachineType.ARM), arm.facing)
            )
        self._targets = targets
        self._placer = _Placer(targets, label="section/ROCKET")

    def _items_still_to_craft(self, state: EnvState) -> dict[int, int]:
        need: dict[int, int] = {}
        for t in self._targets:
            if machine_at(state, t.pos[0], t.pos[1]) != t.machine_type:
                need[t.item] = need.get(t.item, 0) + 1
        return {
            item: max(0, cnt - inv_count(state, item)) for item, cnt in need.items()
        }

    def _can_craft(self, state: EnvState, item: int) -> bool:
        for inp_item, inp_count in self._recipe_inputs.get(item, []):
            if inv_count(state, inp_item) < inp_count:
                return False
        return True

    def step(self, state: EnvState, params: EnvParams) -> int:
        del params
        if self._placer.all_done(state):
            return int(Action.NOOP)
        remaining = self._items_still_to_craft(state)
        for item, count in remaining.items():
            if count > 0 and self._can_craft(state, item):
                return craft_action(item)
        for item, count in remaining.items():
            if count <= 0:
                continue
            for inp_item, inp_count in self._recipe_inputs.get(item, []):
                if inv_count(state, inp_item) < inp_count:
                    action = _withdraw_from_pallet_step(state, self._pallets, inp_item)
                    if action is not None:
                        return action
        return self._placer.next_action(state)

    def success(self, state: EnvState) -> bool:
        return self._placer.all_done(state)

    def debug_status(self, state: EnvState) -> str:
        remaining = {
            ItemType(k).name: v
            for k, v in self._items_still_to_craft(state).items()
            if v > 0
        }
        return (
            f"{self.name} pos={player_pos(state)} to_craft={remaining} "
            f"pending={self._placer.pending(state)[:4]}"
        )


class PhaseWaitAndPlaceRocket(Phase):
    """Wait for a ROCKET item to be produced, then place it.

    The ROCKET assembler produces a ``ROCKET`` item (which is
    placeable). The agent picks it up from the assembler and places
    it at the layout's reserved rocket tile.
    """

    name = "7: wait + place rocket"
    deadline = 2000

    def __init__(
        self, layout: FactoryLayout, patches: list[OrePatch], target: int
    ) -> None:
        super().__init__(layout, patches)
        self._target = target
        self._rocket_assemblers = [
            a for a in layout.assemblers if a.recipe_output == target
        ]

    def step(self, state: EnvState, params: EnvParams) -> int:
        del params
        # If a ROCKET machine is already on the map, we're done.
        if self.success(state):
            return int(Action.NOOP)
        # If we have a ROCKET item in inventory, place it.
        if inv_count(state, int(ItemType.ROCKET)) > 0:
            return _walk_face_act(
                state, self.layout.rocket_tile, place_action(int(ItemType.ROCKET))
            )
        # Otherwise: pick up from the ROCKET assembler if it has
        # output ready (in ent_asm_out OR ent_buf — the engine moves
        # output to buf each tick).
        if not self._rocket_assemblers:
            return int(Action.NOOP)
        asm = self._rocket_assemblers[0]
        eidx = entity_at(state, asm.pos[0], asm.pos[1])
        if eidx < 0:
            return int(Action.NOOP)
        out_count = int(np.asarray(state.ent_asm_out_count)[eidx])
        buf_type, buf_count = ent_buf_lookup(state, eidx)
        if out_count > 0 or (buf_count > 0 and buf_type == int(ItemType.ROCKET)):
            return _walk_face_act(state, asm.pos, int(Action.WITHDRAW))
        return int(Action.NOOP)

    def success(self, state: EnvState) -> bool:
        # Mirror easy_rocket's rocket_placed condition.
        return bool((np.asarray(state.machine_types) == int(MachineType.ROCKET)).any())


def _recipe_inputs_from_table(recipe_table: object) -> RecipeInputs:
    """Project a RecipeTable into {output_item: [(input_item, count), ...]}."""
    out_to_recipe = np.asarray(recipe_table.output_to_recipe)
    input_items = np.asarray(recipe_table.input_items)
    input_counts = np.asarray(recipe_table.input_counts)
    result: RecipeInputs = {}
    for item_id in range(out_to_recipe.size):
        ridx = int(out_to_recipe[item_id])
        if ridx < 0:
            continue
        inputs = [
            (int(input_items[ridx, i]), int(input_counts[ridx, i]))
            for i in range(input_items.shape[1])
            if int(input_counts[ridx, i]) > 0
        ]
        result[item_id] = inputs
    return result


def build_phases(
    layout: FactoryLayout,
    patches: list[OrePatch],
    recipe_table: object,
    target: int,
) -> list[Phase]:
    """Build the ordered list of phases for the given layout.

    Phases 4..N-2 are one ``PhaseSection`` per non-target automated
    assembler, in topological (DAG) order. The penultimate phase
    builds the rocket section. The final phase waits for the rocket
    item and places it.
    """
    recipe_inputs = _recipe_inputs_from_table(recipe_table)
    phases: list[Phase] = [
        PhasePlanLayout(layout, patches),
        PhaseBootstrapMining(layout, patches, recipe_inputs),
        PhasePlaceManualMiners(layout, patches),
    ]
    non_target_outputs = [
        a.recipe_output for a in layout.assemblers if a.recipe_output != target
    ]
    for i, output in enumerate(non_target_outputs):
        section = PhaseSection(layout, patches, output, recipe_inputs)
        section.name = f"{4 + i}: section {ItemType(output).name}"
        phases.append(section)
    phases.append(PhaseRocketSection(layout, patches, target, recipe_inputs))
    phases[-1].name = f"{4 + len(non_target_outputs)}: rocket section"
    phases.append(PhaseWaitAndPlaceRocket(layout, patches, target))
    phases[-1].name = f"{5 + len(non_target_outputs)}: wait + place rocket"
    return phases
