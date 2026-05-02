"""Advanced-factory rocket agent — v2 incremental rebuild.

Currently at **M3** of the v2 plan: bootstrap-only. The agent
hand-mines a small starter inventory, smelts each plate at the
pre-placed furnace at (15, 16), and crafts a single WIRE batch
through the pre-placed assembler at (17, 16). No automation is
placed yet; M4-M15 layer cells, belts, splitters, and arms onto
this skeleton.

Why a thin starting point matters: every phase of the build relies
on the BOM math and the navigation skills working against the v2
map (coal column at x=0, 2x2 ore patches on cols 3-4). M3 proves
that exact flow without any of the geometry that comes later.
Achievement floor at M3 is around the dozen-mark — every mining
and smelting unlock plus the first assembler output and a single
crafted intermediate. Subsequent milestones layer cells on top
without touching this prefix.

Recipe flexibility. Mining and smelting quantities are not
hand-typed; they're derived from
:func:`~baselines.rocket.scripted.recipe_planning.bill_of_materials`
applied to the targets the next phase needs. Tuning a recipe via
``BASE_RECIPE_BOOK.with_balance(...)`` and passing the new book
through to :func:`build_advanced_factory_goals` automatically
re-sizes the goal list — geometry stays put, quantities flex.
"""

from __future__ import annotations

from factoriax.constants import ItemType
from factoriax.recipes import BASE_RECIPE_BOOK, RecipeBook
from factoriax.state import EnvParams

from .agent import ScriptedAgent
from .goals import (
    Goal,
    MineOre,
    ProduceInAssembler,
    ProduceInFurnace,
    Wait,
)
from .planner import Planner
from .recipe_planning import bill_of_materials, sum_inventories

# ---------------------------------------------------------------------------
# Bootstrap targets and slack
# ---------------------------------------------------------------------------

# At M3 the agent's only goal is to demonstrate the new map's
# geometry works for basic mining / smelting / crafting. The
# starter targets are a single tiny batch of each plate and a small
# WIRE craft — just enough to unlock the matching achievements
# (collect_*, smelt_*, craft_wire, first_assembly) and prove the
# navigation skills find every patch and the pre-placed machines.
_M3_PLATE_TARGETS: dict[int, int] = {
    int(ItemType.IRON_PLATE): 2,
    int(ItemType.COPPER_PLATE): 2,
    int(ItemType.TIN_PLATE): 2,
    int(ItemType.WAFER): 2,
}

_M3_CRAFT_TARGETS: dict[int, int] = {
    int(ItemType.WIRE): 2,
}

# Slack added on top of the BOM-derived raw-resource demand. Absorbs
# ``ProduceInMachine``'s deposit-then-wait ordering quirks where the
# engine's auto-pull may consume an ore / coal unit between two of
# the player's deposits. Tuned by hand from end-to-end runs in v1;
# kept the same here to give M4+ room when they layer on more
# smelts.
_DEFAULT_SLACK: dict[int, int] = {
    int(ItemType.IRON_ORE): 2,
    int(ItemType.COPPER_ORE): 2,
    int(ItemType.TIN_ORE): 2,
    int(ItemType.SILICON): 2,
    int(ItemType.COAL): 5,
}


# ---------------------------------------------------------------------------
# Phase 0 — bootstrap (hand-mine + smelt + craft at pre-placed)
# ---------------------------------------------------------------------------


def _phase_0(book: RecipeBook, slack: dict[int, int]) -> list[Goal]:
    """Hand-mine + smelt + craft a starter batch through pre-placed F+A.

    The plate and craft targets in :data:`_M3_PLATE_TARGETS` and
    :data:`_M3_CRAFT_TARGETS` get rolled back to leaves via
    :func:`bill_of_materials`, then ``slack`` is added per leaf so
    the deposit-then-wait race in :class:`ProduceInMachine` doesn't
    starve a smelt cycle.
    """
    bom = bill_of_materials(
        sum_inventories(_M3_PLATE_TARGETS, _M3_CRAFT_TARGETS),
        book,
    )
    for item, qty in slack.items():
        bom[int(item)] = bom.get(int(item), 0) + int(qty)

    mine_goals: list[Goal] = [MineOre(item, qty) for item, qty in sorted(bom.items())]
    smelt_goals: list[Goal] = [
        ProduceInFurnace(item, qty, book=book)
        for item, qty in sorted(_M3_PLATE_TARGETS.items())
    ]
    craft_goals: list[Goal] = [
        ProduceInAssembler(item, qty, book=book)
        for item, qty in sorted(_M3_CRAFT_TARGETS.items())
    ]
    return [*mine_goals, *smelt_goals, *craft_goals]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_advanced_factory_goals(
    book: RecipeBook = BASE_RECIPE_BOOK,
    slack: dict[int, int] | None = None,
) -> list[Goal]:
    """Build the flat goal list for the advanced-factory rocket agent.

    Currently M3-only: hand-mine + pre-smelt + craft WIRE. M4
    appends the iron smelter cell + coal trunk; M5+ extend Phase 1
    and Phase 2 onto this prefix without touching it.

    Args:
        book: :class:`~factoriax.recipes.RecipeBook` whose recipes
            drive the BOM and production schedule. Defaults to the
            shipped :data:`~factoriax.recipes.BASE_RECIPE_BOOK`;
            pass a tuned book (e.g.
            ``BASE_RECIPE_BOOK.with_balance(...)``) to make the
            agent's mine / smelt / craft quantities track the
            balance overlay.
        slack: Per-leaf-resource slack added on top of the BOM
            count. Defaults to :data:`_DEFAULT_SLACK`. Set to an
            empty dict to mine exactly the BOM amount (useful for
            tests that want to fail fast on a starvation bug).

    Returns:
        Flat list of :class:`~baselines.rocket.scripted.goals.Goal`
        instances ready for the planner.
    """
    if slack is None:
        slack = _DEFAULT_SLACK
    return [
        *_phase_0(book, slack),
        # Trailing wait. The pre-placed assembler at (17, 16) takes
        # a few ticks to drain its ent_asm_out after the last
        # ProduceInAssembler returns; this wait lets the player's
        # inventory show the WIRE so achievement-tracking reports a
        # stable end state.
        Wait(50),
    ]


def make_advanced_factory_rocket_agent(
    env_params: EnvParams,
    book: RecipeBook = BASE_RECIPE_BOOK,
    slack: dict[int, int] | None = None,
) -> ScriptedAgent:
    """Construct the advanced-factory scripted agent.

    Args:
        env_params: Environment parameters (passed to the planner).
        book: :class:`~factoriax.recipes.RecipeBook` driving the
            bootstrap quantities. Pass a tuned book to explore
            balance changes.
        slack: Per-leaf-resource slack on top of the BOM. See
            :func:`build_advanced_factory_goals`.
    """
    return ScriptedAgent(
        env_params,
        Planner(build_advanced_factory_goals(book=book, slack=slack)),
    )
