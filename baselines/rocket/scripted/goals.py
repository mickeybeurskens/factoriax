"""High-level goals for the scripted rocket agent.

A :class:`Goal` composes :mod:`skills` to achieve a concrete
in-world outcome (e.g. "have 10 iron ore in inventory", "place a
miner on an ore tile"). Goals are finite-state machines; the
:mod:`planner` layer sequences them through the phases from
``docs/rocket_scripted_agent.md``.

Each goal exposes the same shape as a skill:

    goal.step(view: WorldView) -> (Result, int | None)

Goals may swap out their internal skill when one completes (e.g.
after arriving at a patch, replace ``NavigateAdjacent`` with
``FaceAndInteract`` to mine). They carry enough state to replan
on the fly — for instance if an ore patch depletes, ``MineOre``
picks the next nearest patch without giving up.
"""

from __future__ import annotations

from collections.abc import Callable

from factoriax.constants import Action, ItemType, MachineType

from .skills import (
    EmitOnce,
    FaceAndInteract,
    Result,
    Skill,
    StandOnAndAct,
    StepReturn,
)
from .world_model import (
    WorldView,
    place_action,
)

# Signature for "pick a location" callbacks used by :class:`PlaceMachine`.
LocationPredicate = Callable[[WorldView], tuple[int, int] | None]


class Goal:
    """Base class for high-level goals."""

    name: str = "Goal"

    def step(self, view: WorldView) -> StepReturn:  # pragma: no cover - abstract
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Resource-gathering
# ---------------------------------------------------------------------------


class MineOre(Goal):
    """Mine ore until the player holds at least *count* of *item_type*.

    Replans each tick: reselects the nearest ore patch from the current
    observation, so a depleted patch or a new blocker doesn't strand the
    agent.
    """

    name = "MineOre"

    def __init__(self, item_type: int | ItemType, count: int) -> None:
        self.item_type = int(item_type)
        self.count = count
        self._active: Skill | None = None
        self._active_target: tuple[int, int] | None = None

    def step(self, view: WorldView) -> StepReturn:
        if view.player.held(self.item_type) >= self.count:
            return Result.DONE, None

        # Pick the nearest patch by BFS distance, approximated via
        # Manhattan distance from the player.
        px, py = view.player.pos
        tiles = view.ore_tiles(self.item_type)
        if not tiles:
            return Result.FAIL, None
        tiles.sort(key=lambda t: abs(t[0] - px) + abs(t[1] - py))

        # If we already picked a target and it's still productive, keep it.
        if (
            self._active_target is not None
            and view.block_resources[self._active_target[1], self._active_target[0]] > 0
            and self._active is not None
        ):
            result, action = self._active.step(view)
            if result is Result.DONE:
                # Finished one mine cycle. Clear the skill; next tick we
                # start another (same patch, or a new nearest if depleted).
                self._active = None
                # Fall through to create a fresh skill below — don't
                # waste a NOOP when we could emit the next move or mine.
            elif result is Result.FAIL:
                self._active = None
                self._active_target = None
                return Result.RUNNING, int(Action.NOOP)
            else:
                return Result.RUNNING, action

        # Select (or re-select) a patch and start a fresh StandOnAndAct.
        self._active_target = tiles[0]
        self._active = StandOnAndAct(self._active_target, int(Action.MINE))
        return self._active.step(view)


# ---------------------------------------------------------------------------
# Crafting
# ---------------------------------------------------------------------------


_ITEM_TO_CRAFT_ACTION: dict[int, int] = {
    int(ItemType.IRON_PLATE): int(Action.CRAFT_IRON_PLATE),
    int(ItemType.COPPER_PLATE): int(Action.CRAFT_COPPER_PLATE),
    int(ItemType.TIN_PLATE): int(Action.CRAFT_TIN_PLATE),
    int(ItemType.WAFER): int(Action.CRAFT_WAFER),
    int(ItemType.FRAME): int(Action.CRAFT_FRAME),
    int(ItemType.CIRCUIT): int(Action.CRAFT_CIRCUIT),
    int(ItemType.WIRE): int(Action.CRAFT_WIRE),
    int(ItemType.MOTOR): int(Action.CRAFT_MOTOR),
    int(ItemType.SENSOR): int(Action.CRAFT_SENSOR),
    int(ItemType.CONVEYOR_BELT): int(Action.CRAFT_BELT),
    int(ItemType.MINER): int(Action.CRAFT_MINER),
    int(ItemType.ASSEMBLER): int(Action.CRAFT_ASSEMBLER),
    int(ItemType.PALLET): int(Action.CRAFT_PALLET),
    int(ItemType.ARM): int(Action.CRAFT_ARM),
    int(ItemType.FURNACE): int(Action.CRAFT_FURNACE),
    int(ItemType.BASIC_SCIENCE_PACK): int(Action.CRAFT_BASIC_SCIENCE),
    int(ItemType.ADVANCED_SCIENCE_PACK): int(Action.CRAFT_ADV_SCIENCE),
    int(ItemType.ROCKET): int(Action.CRAFT_ROCKET),
}


class CraftItem(Goal):
    """Emit ``CRAFT_<item>`` until inventory holds ≥ *count* of *item_type*.

    Crafting is direction-agnostic — no navigation needed. If the
    player's count doesn't increase after an attempt it means
    ingredients are missing; the goal reports FAIL so the planner can
    schedule more gathering.
    """

    name = "CraftItem"

    def __init__(self, item_type: int | ItemType, count: int) -> None:
        self.item_type = int(item_type)
        if self.item_type not in _ITEM_TO_CRAFT_ACTION:
            raise ValueError(
                f"item_type {ItemType(self.item_type).name} is not craftable",
            )
        self.count = count
        self._last_held: int | None = None
        self._pending: EmitOnce | None = None

    def step(self, view: WorldView) -> StepReturn:
        held = view.player.held(self.item_type)
        if held >= self.count:
            return Result.DONE, None

        # If we already emitted a craft action last tick and inventory
        # didn't grow, the recipe failed — ingredients missing.
        if self._last_held is not None and held <= self._last_held:
            return Result.FAIL, None

        # Fire off another craft attempt.
        self._pending = EmitOnce(_ITEM_TO_CRAFT_ACTION[self.item_type])
        self._last_held = held
        result, action = self._pending.step(view)
        return result, action


# ---------------------------------------------------------------------------
# Machine placement
# ---------------------------------------------------------------------------


def _first_ore_tile(
    view: WorldView, item_type: int | ItemType
) -> tuple[int, int] | None:
    """Nearest ore tile of the requested type, or None."""
    tiles = view.ore_tiles(item_type)
    if not tiles:
        return None
    px, py = view.player.pos
    tiles.sort(key=lambda t: abs(t[0] - px) + abs(t[1] - py))
    return tiles[0]


def on_ore(item_type: int | ItemType) -> LocationPredicate:
    """Predicate: pick the nearest ore tile of *item_type*.

    Miners placed on ore tiles auto-extract from them; the engine
    accepts a ``PLACE_MINER`` action that targets an ore tile even
    though the tile isn't "walkable" in the usual sense.
    """

    def picker(view: WorldView) -> tuple[int, int] | None:
        return _first_ore_tile(view, item_type)

    return picker


def free_tile_near_player() -> LocationPredicate:
    """Predicate: pick the nearest empty walkable tile adjacent to the player.

    Good default for placements that don't need specific adjacency —
    furnaces, extra belts, pallets in a scaling-up pile.
    """

    def picker(view: WorldView) -> tuple[int, int] | None:
        px, py = view.player.pos
        # The tile the player is ON is blocked by the player; look at
        # the four neighbors and pick any that is walkable with no
        # machine on it.
        for nx, ny in view.adjacent_tiles((px, py)):
            if view.walkable[ny, nx] and view.machine_type[ny, nx] == 0:
                return (nx, ny)
        return None

    return picker


class PlaceMachine(Goal):
    """Place a machine of *machine_type* at a tile chosen by *predicate*.

    Fails fast if the predicate returns ``None`` (no valid location
    exists right now) or if the player doesn't hold the placeable item.
    """

    name = "PlaceMachine"

    def __init__(
        self,
        machine_type: int | MachineType,
        predicate: LocationPredicate,
    ) -> None:
        self.machine_type = int(machine_type)
        self.item_type = _machine_to_item(self.machine_type)
        self.predicate = predicate
        self._active: FaceAndInteract | None = None
        self._start_count: int | None = None

    def step(self, view: WorldView) -> StepReturn:
        # Detect success: machine count went up, our inventory went down.
        if self._start_count is not None and view.total_machines() > self._start_count:
            return Result.DONE, None

        if view.player.held(self.item_type) < 1:
            return Result.FAIL, None

        if self._active is None:
            target = self.predicate(view)
            if target is None:
                return Result.FAIL, None
            self._start_count = view.total_machines()
            self._active = FaceAndInteract(
                target,
                place_action(self.machine_type),
            )

        result, action = self._active.step(view)
        if result is Result.DONE:
            # Let the next tick verify via the total_machines check.
            self._active = None
            return Result.RUNNING, int(Action.NOOP)
        if result is Result.FAIL:
            self._active = None
            return Result.FAIL, None
        return Result.RUNNING, action


def _machine_to_item(machine_type: int) -> int:
    """Item type corresponding to a placeable machine."""
    return {
        int(MachineType.MINER): int(ItemType.MINER),
        int(MachineType.PALLET): int(ItemType.PALLET),
        int(MachineType.CONVEYOR_BELT): int(ItemType.CONVEYOR_BELT),
        int(MachineType.ASSEMBLER): int(ItemType.ASSEMBLER),
        int(MachineType.ARM): int(ItemType.ARM),
        int(MachineType.ROCKET): int(ItemType.ROCKET),
        int(MachineType.FURNACE): int(ItemType.FURNACE),
    }[machine_type]
