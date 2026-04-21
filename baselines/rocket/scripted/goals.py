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

import numpy as np

from factoriax.constants import Action, ItemType, MachineType
from factoriax.recipes import RECIPES

from .skills import (
    FaceAndInteract,
    PlaceAt,
    Result,
    Skill,
    StandOnAndAct,
    StepReturn,
)
from .world_model import (
    WorldView,
    place_action,
    withdraw_action,
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
    """Emit ``CRAFT_<item>`` up to *count* times (or until satisfied).

    The goal is forgiving by design: if ingredients run out partway,
    the agent keeps attempting for a small buffer of extra ticks and
    then moves on. This trades an ideal-world guarantee ("exactly
    *count* produced") for robustness against resource shortages —
    downstream goals that need the item will FAIL on their own and
    the planner will skip them.
    """

    name = "CraftItem"

    def __init__(
        self,
        item_type: int | ItemType,
        count: int,
        extra_attempts: int = 2,
    ) -> None:
        self.item_type = int(item_type)
        if self.item_type not in _ITEM_TO_CRAFT_ACTION:
            raise ValueError(
                f"item_type {ItemType(self.item_type).name} is not craftable",
            )
        self.count = count
        self._attempts_remaining = count + extra_attempts
        self._started_held: int | None = None

    def step(self, view: WorldView) -> StepReturn:
        held = view.player.held(self.item_type)
        if self._started_held is None:
            self._started_held = held
        if held >= self.count:
            return Result.DONE, None
        if self._attempts_remaining <= 0:
            # Out of attempts; either we made some progress or the
            # recipe is unsatisfiable. Either way the planner moves on.
            return Result.DONE, None
        self._attempts_remaining -= 1
        return Result.RUNNING, _ITEM_TO_CRAFT_ACTION[self.item_type]


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

    Filters out tiles that already carry a machine so chained
    :class:`PlaceMachine` calls can cover a patch instead of
    repeatedly aiming at the same tile.

    Miners placed on ore tiles auto-extract from them; the engine
    accepts a ``PLACE_MINER`` action that targets an ore tile even
    though the tile isn't "walkable" in the usual sense.
    """

    def picker(view: WorldView) -> tuple[int, int] | None:
        tiles = view.ore_tiles(item_type)
        if not tiles:
            return None
        px, py = view.player.pos
        unoccupied = [t for t in tiles if view.machine_type[t[1], t[0]] == 0]
        if not unoccupied:
            return None
        unoccupied.sort(key=lambda t: abs(t[0] - px) + abs(t[1] - py))
        return unoccupied[0]

    return picker


def free_tile_near_player(
    max_radius: int = 8,
    min_distance: int = 2,
) -> LocationPredicate:
    """Predicate: closest walkable tile that holds no machine.

    Picks a tile at least ``min_distance`` Manhattan away from the
    player and at most ``max_radius``. The minimum distance guards
    against the player boxing themselves in with machines on their
    4-neighbors — after placing, the player's immediate neighbors
    include the targeted tile as a new machine, but the tile they were
    standing on (one-tile-removed from the target) remains walkable.
    Without this buffer, sequential placements on each of the four
    neighbors strand the player on an island.
    """

    def picker(view: WorldView) -> tuple[int, int] | None:
        px, py = view.player.pos
        empty = view.walkable & (view.machine_type == 0)
        empty_np = empty.copy()
        empty_np[py, px] = False
        if not empty_np.any():
            return None
        ys, xs = np.nonzero(empty_np)
        dists = np.abs(xs - px) + np.abs(ys - py)
        band = (dists >= min_distance) & (dists <= max_radius)
        if not band.any():
            # Fall back to any empty tile if the preferred band is
            # unavailable (e.g. map is almost full).
            band = dists <= max_radius
        if not band.any():
            return None
        xs = xs[band]
        ys = ys[band]
        dists = dists[band]
        idx = int(np.argmin(dists))
        return (int(xs[idx]), int(ys[idx]))

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


class PlaceMachineAt(Goal):
    """Place a machine at a specific tile facing a specific direction.

    Reuses :class:`PlaceAt` under the hood: navigates the player to
    ``target - unit_vec(facing)``, turns to ``facing``, then emits
    ``PLACE_*``. This is how the factory agent aligns miners so
    their per-tick push lands in the adjacent pallet instead of a
    dead tile.
    """

    name = "PlaceMachineAt"

    def __init__(
        self,
        machine_type: int | MachineType,
        target: tuple[int, int],
        facing: int,
    ) -> None:
        self.machine_type = int(machine_type)
        self.item_type = _machine_to_item(self.machine_type)
        self.target = target
        self.facing = int(facing)
        self._active: PlaceAt | None = None
        self._start_count: int | None = None

    def step(self, view: WorldView) -> StepReturn:
        if self._start_count is not None and view.total_machines() > self._start_count:
            return Result.DONE, None

        if view.player.held(self.item_type) < 1:
            return Result.FAIL, None

        if self._active is None:
            self._start_count = view.total_machines()
            self._active = PlaceAt(
                self.target,
                self.facing,
                place_action(self.machine_type),
            )

        result, action = self._active.step(view)
        if result is Result.DONE:
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


# ---------------------------------------------------------------------------
# Interaction with placed machines
# ---------------------------------------------------------------------------


class DepositInto(Goal):
    """Deposit one unit of *item_type* into the nearest *machine_type*.

    Used for loading pallets (to unlock ``pallet_filled``), loading
    assemblers with inputs, or feeding a furnace. The goal succeeds
    after a single successful DEPOSIT; repeat the goal to load more.
    """

    name = "DepositInto"

    def __init__(
        self,
        machine_type: int | MachineType,
        item_type: int | ItemType,
    ) -> None:
        self.machine_type = int(machine_type)
        self.item_type = int(item_type)
        self._active: FaceAndInteract | None = None
        self._start_inv: int | None = None

    def step(self, view: WorldView) -> StepReturn:
        current_inv = view.player.held(self.item_type)

        if self._start_inv is not None and current_inv < self._start_inv:
            # Inventory went down → deposit succeeded.
            return Result.DONE, None

        if current_inv < 1:
            return Result.FAIL, None

        if self._active is None:
            tiles = view.tiles_with_machine(self.machine_type)
            if not tiles:
                return Result.FAIL, None
            px, py = view.player.pos
            tiles.sort(key=lambda t: abs(t[0] - px) + abs(t[1] - py))
            target = tiles[0]
            self._start_inv = current_inv
            from .world_model import deposit_action  # noqa: PLC0415

            self._active = FaceAndInteract(target, deposit_action(self.item_type))

        result, action = self._active.step(view)
        if result is Result.DONE:
            # FaceAndInteract finished emitting its action; next tick will
            # detect the inventory drop and report DONE itself.
            self._active = None
            return Result.RUNNING, int(Action.NOOP)
        if result is Result.FAIL:
            self._active = None
            return Result.FAIL, None
        return Result.RUNNING, action


class WaitUntil(Goal):
    """Emit ``NOOP`` until *predicate* returns True or the budget expires.

    Useful for waiting on timed game events — ``automated_mining``
    fires one tick after a miner is placed on ore; ``first_assembly``
    fires after the assembler's recipe timer completes.
    """

    name = "WaitUntil"

    def __init__(
        self,
        predicate: Callable[[WorldView], bool],
        max_ticks: int,
    ) -> None:
        self.predicate = predicate
        self.max_ticks = max_ticks
        self._ticks = 0

    def step(self, view: WorldView) -> StepReturn:
        if self.predicate(view):
            return Result.DONE, None
        if self._ticks >= self.max_ticks:
            return Result.FAIL, None
        self._ticks += 1
        return Result.RUNNING, int(Action.NOOP)


# ---------------------------------------------------------------------------
# Machine-based production (furnace / assembler)
# ---------------------------------------------------------------------------


class Wait(Goal):
    """Emit ``NOOP`` for exactly *ticks* then report DONE.

    Used between deposit and withdraw inside a machine production
    cycle so the recipe has time to complete.
    """

    name = "Wait"

    def __init__(self, ticks: int) -> None:
        self.ticks = ticks
        self._elapsed = 0

    def step(self, view: WorldView) -> StepReturn:  # noqa: ARG002
        if self._elapsed >= self.ticks:
            return Result.DONE, None
        self._elapsed += 1
        return Result.RUNNING, int(Action.NOOP)


class WithdrawFrom(Goal):
    """Withdraw one unit of *item_type* from the nearest *machine_type*.

    Retries up to *max_attempts* times if the first withdraw doesn't
    actually move anything (e.g. recipe hasn't finished yet). Succeeds
    when the player's inventory of *item_type* grows.
    """

    name = "WithdrawFrom"

    def __init__(
        self,
        machine_type: int | MachineType,
        item_type: int | ItemType,
        max_attempts: int = 6,
    ) -> None:
        self.machine_type = int(machine_type)
        self.item_type = int(item_type)
        self.max_attempts = max_attempts
        self._active: FaceAndInteract | None = None
        self._start_inv: int | None = None
        self._attempts = 0

    def step(self, view: WorldView) -> StepReturn:
        current_inv = view.player.held(self.item_type)
        if self._start_inv is not None and current_inv > self._start_inv:
            return Result.DONE, None

        if self._active is not None:
            result, action = self._active.step(view)
            if result is Result.DONE:
                # Emitted the WITHDRAW action; next tick we check inventory.
                self._active = None
                return Result.RUNNING, int(Action.NOOP)
            if result is Result.FAIL:
                self._active = None
                return Result.FAIL, None
            return Result.RUNNING, action

        if self._attempts >= self.max_attempts:
            return Result.FAIL, None

        tiles = view.tiles_with_machine(self.machine_type)
        if not tiles:
            return Result.FAIL, None
        px, py = view.player.pos
        tiles.sort(key=lambda t: abs(t[0] - px) + abs(t[1] - py))
        target = tiles[0]
        if self._start_inv is None:
            self._start_inv = current_inv
        self._attempts += 1
        self._active = FaceAndInteract(target, withdraw_action())
        return self._active.step(view)


_FURNACE_OUTPUTS: frozenset[int] = frozenset(
    {
        int(ItemType.IRON_PLATE),
        int(ItemType.COPPER_PLATE),
        int(ItemType.TIN_PLATE),
        int(ItemType.WAFER),
        int(ItemType.REFRACTORY),
    }
)


def _find_recipe(output_item: int) -> dict | None:
    """Look up the recipe that produces *output_item*."""
    for r in RECIPES:
        if int(r["output"]) == int(output_item):
            return r
    return None


def _default_machine_for(output_item: int) -> int:
    """Decide which machine processes this recipe (furnace vs. assembler)."""
    if int(output_item) in _FURNACE_OUTPUTS:
        return int(MachineType.FURNACE)
    return int(MachineType.ASSEMBLER)


class ProduceInMachine(Goal):
    """Run a recipe on a placed machine until inventory holds *count* outputs.

    One production cycle = deposit every input (respecting per-input
    quantities) → wait for the recipe's ticks → withdraw one output.
    Inputs are not pre-batched: each cycle deposits exactly one
    recipe's worth. This is slower but makes the FSM simple and the
    inventory delta trivial to track.

    Use the :func:`ProduceInFurnace` / :func:`ProduceInAssembler`
    helpers below for the common case — they auto-pick the right
    machine type based on the recipe.
    """

    name = "ProduceInMachine"

    def __init__(
        self,
        output_item: int | ItemType,
        count: int,
        machine_type: int | MachineType | None = None,
    ) -> None:
        self.output_item = int(output_item)
        self.count = count
        recipe = _find_recipe(self.output_item)
        if recipe is None:
            raise ValueError(
                f"no recipe produces {ItemType(self.output_item).name}",
            )
        self.machine_type = (
            int(machine_type)
            if machine_type is not None
            else _default_machine_for(self.output_item)
        )
        self.recipe_inputs: list[tuple[int, int]] = [
            (int(it), int(q)) for it, q in recipe["inputs"]
        ]
        self.wait_ticks: int = int(recipe["ticks"]) + 3

        self._sub: Goal | None = None
        self._phase: str = "deposit"  # "deposit" | "wait" | "withdraw"
        self._deposit_input_idx: int = 0
        self._deposit_count: int = 0
        self._wait_elapsed: int = 0

    def step(self, view: WorldView) -> StepReturn:
        if view.player.held(self.output_item) >= self.count:
            return Result.DONE, None

        if self._sub is not None:
            result, action = self._sub.step(view)
            if result is Result.DONE:
                self._sub = None
                return Result.RUNNING, int(Action.NOOP)
            if result is Result.FAIL:
                self._sub = None
                return Result.FAIL, None
            return Result.RUNNING, action

        if self._phase == "deposit":
            return self._step_deposit(view)
        if self._phase == "wait":
            return self._step_wait()
        if self._phase == "withdraw":
            return self._step_withdraw(view)
        raise AssertionError(f"unknown phase: {self._phase}")

    def _step_deposit(self, view: WorldView) -> StepReturn:
        input_item, required = self.recipe_inputs[self._deposit_input_idx]
        if self._deposit_count < required:
            if view.player.held(input_item) < 1:
                return Result.FAIL, None
            self._deposit_count += 1
            self._sub = DepositInto(self.machine_type, input_item)
            return self._sub.step(view)
        # All copies of this input are deposited; move to next input.
        self._deposit_input_idx += 1
        self._deposit_count = 0
        if self._deposit_input_idx >= len(self.recipe_inputs):
            # All inputs deposited — start the wait.
            self._phase = "wait"
            self._wait_elapsed = 0
        return Result.RUNNING, int(Action.NOOP)

    def _step_wait(self) -> StepReturn:
        if self._wait_elapsed >= self.wait_ticks:
            self._phase = "withdraw"
            return Result.RUNNING, int(Action.NOOP)
        self._wait_elapsed += 1
        return Result.RUNNING, int(Action.NOOP)

    def _step_withdraw(self, view: WorldView) -> StepReturn:
        # Reset the phase before delegating to withdraw; when the
        # withdraw sub-goal finishes we'll loop back to "deposit" for
        # the next production cycle (or exit via the held >= count
        # check at the top of step()).
        self._sub = WithdrawFrom(self.machine_type, self.output_item)
        self._phase = "deposit"
        self._deposit_input_idx = 0
        self._deposit_count = 0
        return self._sub.step(view)


def ProduceInFurnace(  # noqa: N802 - factory mirrors class-style instantiation
    output_item: int | ItemType,
    count: int,
) -> ProduceInMachine:
    """Produce *count* of *output_item* via the nearest furnace."""
    return ProduceInMachine(output_item, count, int(MachineType.FURNACE))


def ProduceInAssembler(  # noqa: N802 - factory mirrors class-style instantiation
    output_item: int | ItemType,
    count: int,
) -> ProduceInMachine:
    """Produce *count* of *output_item* via the nearest assembler."""
    return ProduceInMachine(output_item, count, int(MachineType.ASSEMBLER))


# ---------------------------------------------------------------------------
# Pipelined production across multiple machines
# ---------------------------------------------------------------------------


class PipelinedProduce(Goal):
    """Produce ``count`` of ``output_item`` by rotating through *k* machines.

    Unlike :class:`ProduceInMachine`, which parks the agent at one
    machine during its ``Wait`` phase, this goal keeps *k* machines
    running in parallel. On each tick the agent picks the highest-
    priority action across the k-nearest matching machines:

    1. **Withdraw** from any machine whose output slot matches
       ``output_item`` (the buffer channel of the obs exposes asm_out
       one tick after the recipe completes).
    2. **Deposit** the next unmet input into the machine with the
       fewest deposits so far, provided the player still holds that
       ingredient.
    3. Otherwise emit ``NOOP`` — a recipe is in flight and no deposit
       is pending.

    Per-machine state is tracked in a small dict keyed by tile; it
    resets when the agent withdraws (the cycle has ended and the next
    deposit starts a fresh batch).

    Reuses :class:`FaceAndInteract` (via the ``_active`` sub-skill)
    for navigation + action emission, matching how
    :class:`DepositInto` / :class:`WithdrawFrom` already work.
    """

    name = "PipelinedProduce"

    def __init__(
        self,
        output_item: int | ItemType,
        count: int,
        machine_type: int | MachineType,
        k: int = 3,
    ) -> None:
        self.output_item = int(output_item)
        self.count = count
        self.machine_type = int(machine_type)
        self.k = max(1, k)
        recipe = _find_recipe(self.output_item)
        if recipe is None:
            raise ValueError(
                f"no recipe produces {ItemType(self.output_item).name}",
            )
        # Flatten (item, count) pairs into a deposit sequence so each
        # element corresponds to one DEPOSIT_* action. Frame: iron +
        # tin → [iron, tin]. Hull: 2 frame + 2 iron → [frame, frame,
        # iron, iron].
        self._deposit_sequence: list[int] = []
        for item_t, qty in recipe["inputs"]:
            self._deposit_sequence.extend([int(item_t)] * int(qty))

        self._active: FaceAndInteract | None = None
        # Per-machine deposit progress. Keyed by ``(x, y)`` tile.
        self._deposits: dict[tuple[int, int], int] = {}

    def step(self, view: WorldView) -> StepReturn:
        # Exit as soon as the target is met. _deposits tracking can
        # drift out of sync with physical state (Phase 0 pulls,
        # adjacent-combiner interactions) so waiting for "pending
        # cycles to drain" risks deadlock. Any leftover output is
        # cleaned up by the NEXT goal's broadened priority-1 scan.
        if view.player.held(self.output_item) >= self.count:
            return Result.DONE, None

        if self._active is not None:
            result, action = self._active.step(view)
            if result is Result.RUNNING:
                return Result.RUNNING, action
            self._active = None
            # fall through — pick the next action this same tick.

        tiles = view.tiles_with_machine(self.machine_type)
        if not tiles:
            return Result.FAIL, None
        px, py = view.player.pos
        tiles.sort(key=lambda t: abs(t[0] - px) + abs(t[1] - py))
        candidates = tiles[: self.k]

        # Priority 1: withdraw ANY non-empty output slot on a
        # candidate machine. Broadened from "only our output_item"
        # so that leftovers from a previous goal (e.g. a FRAME still
        # sitting in asm_out when we start a WIRE goal) don't block
        # the idle gate on the next cycle.
        for tile in candidates:
            tx, ty = tile
            if int(view.buffer_type[ty, tx]) != 0:
                self._deposits.pop(tile, None)
                self._active = FaceAndInteract(tile, int(Action.WITHDRAW))
                return self._active.step(view)

        # Priority 2: deposit the next input into the NEAREST
        # machine that can accept one. Walks to a partial cycle
        # first if the nearest machine has one; otherwise starts
        # a fresh cycle at the nearest idle machine.
        need = len(self._deposit_sequence)
        best_tile: tuple[int, int] | None = None
        for tile in candidates:  # already sorted by distance
            progress = self._deposits.get(tile, 0)
            if progress >= need:
                continue
            next_item = self._deposit_sequence[progress]
            if view.player.held(next_item) < 1:
                continue
            best_tile = tile
            break

        if best_tile is not None:
            progress = self._deposits.get(best_tile, 0)
            next_item = self._deposit_sequence[progress]
            self._deposits[best_tile] = progress + 1
            self._active = FaceAndInteract(
                best_tile,
                deposit_action_for(next_item),
            )
            return self._active.step(view)

        # Nothing to do this tick: cooking in progress, no output
        # ready yet, no deposits to make. Wait.
        return Result.RUNNING, int(Action.NOOP)


def deposit_action_for(item_type: int) -> int:
    """Resolve an ItemType to its DEPOSIT_* action id."""
    from .world_model import deposit_action

    return deposit_action(item_type)
