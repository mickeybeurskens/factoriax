"""Recipe definitions for the FactoriaX environment.

Single source of truth for recipe data. Two layers:

- :class:`Recipe` is a frozen dataclass holding one recipe's identity
  (output item, input pairs, machine type) and balance numbers
  (input counts via the ``inputs`` tuple, ``output_count``, ``ticks``).
- :data:`BASE_RECIPES` is the canonical tuple of :class:`Recipe`
  instances shipped with the engine. The five JAX arrays consumed by
  :mod:`factoriax.crafting` and :mod:`factoriax.machines` (outputs,
  input items, input counts, output counts, ticks, machine type) are
  derived projections of this tuple.

Recipe shapes:

- **Furnace recipes** take 1 or 2 input types. As of the LIMESTONE
  addition every shipped furnace recipe is 2-input (the four plate
  smelts pair their ore with COAL, REFRACTORY pairs LIMESTONE with
  COAL). The 1-input branch is still supported by the projection —
  the unused slot is padded with ``(EMPTY, 0)`` so the Phase 3
  matcher in ``run_combiners`` can treat every recipe as a 2-slot
  lookup.
- **Assembler recipes** take exactly 2 input types, and the
  (unordered) pair is unique across the table at the same machine
  type. Uniqueness is the invariant that lets Phase 3 forward-match
  deterministically; see ``tests/test_recipes.py``.

Player crafting consumes materials from inventory and produces
``output_count`` output items instantly. Combiners use ``ticks`` as
the production delay and the same ``output_count`` for the cycle's
deposit. The output slot can't be overwritten, so combiner
throughput is bounded by withdrawal.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from factoriax.constants import ItemType, MachineType

# ---------------------------------------------------------------------------
# Recipe dataclass — the canonical per-recipe record
# ---------------------------------------------------------------------------


# Outputs that run on a FURNACE entity. Everything else runs on an
# ASSEMBLER. This curated set is the recipe-identity decision for the
# machine_type axis; tuning it is a code change, not a config.
_FURNACE_OUTPUTS: frozenset[int] = frozenset(
    {
        int(ItemType.IRON_PLATE),
        int(ItemType.COPPER_PLATE),
        int(ItemType.TIN_PLATE),
        int(ItemType.WAFER),
        int(ItemType.REFRACTORY),
    }
)


@dataclass(frozen=True)
class Recipe:
    """One recipe's full record: identity plus default balance.

    Identity (``output``, ``inputs``-types, ``machine_type``) is fixed
    in the canonical ``BASE_RECIPES`` tuple; tuning it requires a code
    change. Balance numbers (input counts via ``inputs``-amounts,
    ``output_count``, ``ticks``) are construction-time-tunable through
    the upcoming :class:`RecipeBalance` overlay.

    Attributes:
        output: ``ItemType`` integer this recipe produces.
        inputs: One or two ``(ItemType, count)`` pairs. The matcher
            in :func:`factoriax.machines.run_assemblers` requires
            unordered uniqueness of the input *type-set* per machine
            type — two recipes with the same input items on the same
            machine would race the deterministic forward-match.
        ticks: Combiner cycle length in ticks. Player crafting is
            instant and ignores this field.
        output_count: How many output items are deposited per
            completed cycle (or per player craft). Defaults to 1.
        name: Human-readable display name for renderer / logging.
            Falls back to the ItemType repr if empty.
    """

    output: int
    inputs: tuple[tuple[int, int], ...]
    ticks: int
    output_count: int = 1
    name: str = ""

    @property
    def machine_type(self) -> int:
        """Combiner type that runs this recipe — derived from output."""
        return (
            int(MachineType.FURNACE)
            if self.output in _FURNACE_OUTPUTS
            else int(MachineType.ASSEMBLER)
        )


# ---------------------------------------------------------------------------
# BASE_RECIPES — single source of truth
# ---------------------------------------------------------------------------

# Recipe order MATTERS: positions 0–20 are the recipes that map to
# ``Action.CRAFT_IRON_PLATE`` … ``Action.CRAFT_CROSSING`` via
# ``CRAFT_ACTION_TO_RECIPE = jnp.arange(NUM_RECIPES)`` (the dispatcher
# in ``execute_action`` uses ``action - CRAFT_BASE`` directly). Positions
# 21+ are machine-only (no CRAFT action) — refractory plus the four
# rocket sub-assemblies.
BASE_RECIPES: tuple[Recipe, ...] = (
    # -------- CRAFT-addressable slots 0–17 --------
    # Furnace smelts — coal is consumed as fuel for every plate
    # (1 ore + 1 coal → 1 plate). Refractory pairs LIMESTONE with
    # COAL so every furnace recipe is shaped the same — two inputs
    # with COAL as fuel — and downstream belt logistics never have
    # to special-case a single-input outlier.
    Recipe(
        output=int(ItemType.IRON_PLATE),
        inputs=((int(ItemType.IRON_ORE), 1), (int(ItemType.COAL), 1)),
        ticks=2,
        name="Iron Plate",
    ),
    Recipe(
        output=int(ItemType.COPPER_PLATE),
        inputs=((int(ItemType.COPPER_ORE), 1), (int(ItemType.COAL), 1)),
        ticks=2,
        name="Copper Plate",
    ),
    Recipe(
        output=int(ItemType.TIN_PLATE),
        inputs=((int(ItemType.TIN_ORE), 1), (int(ItemType.COAL), 1)),
        ticks=2,
        name="Tin Plate",
    ),
    Recipe(
        output=int(ItemType.WAFER),
        inputs=((int(ItemType.SILICON), 1), (int(ItemType.COAL), 1)),
        ticks=2,
        name="Wafer",
    ),
    # Assembler: base intermediates.
    Recipe(
        output=int(ItemType.FRAME),
        inputs=((int(ItemType.IRON_PLATE), 1), (int(ItemType.TIN_PLATE), 1)),
        ticks=4,
        name="Frame",
    ),
    Recipe(
        output=int(ItemType.CIRCUIT),
        inputs=((int(ItemType.COPPER_PLATE), 1), (int(ItemType.WAFER), 1)),
        ticks=4,
        name="Circuit",
    ),
    Recipe(
        output=int(ItemType.WIRE),
        inputs=((int(ItemType.COPPER_PLATE), 1), (int(ItemType.TIN_PLATE), 1)),
        ticks=4,
        name="Wire",
    ),
    # Assembler: components.
    Recipe(
        output=int(ItemType.MOTOR),
        inputs=((int(ItemType.FRAME), 1), (int(ItemType.WIRE), 1)),
        ticks=6,
        name="Motor",
    ),
    Recipe(
        output=int(ItemType.SENSOR),
        inputs=((int(ItemType.CIRCUIT), 1), (int(ItemType.WIRE), 1)),
        ticks=6,
        name="Sensor",
    ),
    # Assembler: logistics machines (cheap). Order matches the
    # CRAFT_* enum: belt, miner, assembler, pallet, arm, furnace.
    Recipe(
        output=int(ItemType.CONVEYOR_BELT),
        inputs=((int(ItemType.IRON_PLATE), 1), (int(ItemType.COPPER_PLATE), 1)),
        ticks=4,
        name="Conveyor Belt",
    ),
    Recipe(
        output=int(ItemType.MINER),
        inputs=((int(ItemType.IRON_PLATE), 1), (int(ItemType.WIRE), 1)),
        ticks=6,
        name="Miner",
    ),
    Recipe(
        output=int(ItemType.ASSEMBLER),
        inputs=((int(ItemType.FRAME), 1), (int(ItemType.CIRCUIT), 1)),
        ticks=8,
        name="Assembler",
    ),
    Recipe(
        output=int(ItemType.PALLET),
        inputs=((int(ItemType.TIN_PLATE), 1), (int(ItemType.WIRE), 1)),
        ticks=4,
        name="Pallet",
    ),
    Recipe(
        output=int(ItemType.ARM),
        inputs=((int(ItemType.COPPER_PLATE), 1), (int(ItemType.WIRE), 1)),
        ticks=4,
        name="Arm",
    ),
    Recipe(
        output=int(ItemType.FURNACE),
        inputs=((int(ItemType.IRON_PLATE), 1), (int(ItemType.REFRACTORY), 1)),
        ticks=6,
        name="Furnace",
    ),
    # Assembler: science packs.
    Recipe(
        output=int(ItemType.BASIC_SCIENCE_PACK),
        inputs=((int(ItemType.MOTOR), 1), (int(ItemType.TIN_PLATE), 1)),
        ticks=8,
        name="Basic Science Pack",
    ),
    Recipe(
        output=int(ItemType.ADVANCED_SCIENCE_PACK),
        inputs=((int(ItemType.SENSOR), 1), (int(ItemType.WAFER), 1)),
        ticks=8,
        name="Advanced Science Pack",
    ),
    # Assembler: capstone.
    Recipe(
        output=int(ItemType.ROCKET),
        inputs=((int(ItemType.HULL), 6), (int(ItemType.ROCKET_CORE), 4)),
        ticks=300,
        name="Rocket",
    ),
    # Assembler: science lab (pairs with CRAFT_SCIENCE_LAB). Inputs
    # chosen from an unused pair so the recipe-uniqueness invariant
    # (see tests/test_recipes.py) holds.
    Recipe(
        output=int(ItemType.SCIENCE_LAB),
        inputs=((int(ItemType.CIRCUIT), 2), (int(ItemType.MOTOR), 2)),
        ticks=8,
        name="Science Lab",
    ),
    # Belt-network pieces — same logistical tier as CONVEYOR_BELT.
    # Pair each with COAL so the type-sets {TIN_PLATE, COAL} and
    # {COPPER_PLATE, COAL} stay unique (the four ore+COAL pairs are
    # all FURNACE-gated, so they don't collide on the ASSEMBLER side).
    Recipe(
        output=int(ItemType.SPLITTER),
        inputs=((int(ItemType.TIN_PLATE), 1), (int(ItemType.COAL), 1)),
        ticks=4,
        name="Splitter",
    ),
    Recipe(
        output=int(ItemType.CROSSING),
        inputs=((int(ItemType.COPPER_PLATE), 1), (int(ItemType.COAL), 1)),
        ticks=4,
        name="Crossing",
    ),
    # -------- Machine-only slots 21+ (no CRAFT action) --------
    # Furnace half-fab — limestone calcined with coal heat. Two
    # inputs, so the (input-type-set) uniqueness invariant in
    # tests/test_recipes.py still holds and the recipe shares the
    # same 2-slot shape as every plate smelt.
    Recipe(
        output=int(ItemType.REFRACTORY),
        inputs=((int(ItemType.LIMESTONE), 1), (int(ItemType.COAL), 1)),
        ticks=4,
        name="Refractory",
    ),
    # Rocket sub-assemblies — the four-tier convergence.
    Recipe(
        output=int(ItemType.HULL),
        inputs=((int(ItemType.FRAME), 2), (int(ItemType.IRON_PLATE), 2)),
        ticks=8,
        name="Hull",
    ),
    Recipe(
        output=int(ItemType.ENGINE_UNIT),
        inputs=((int(ItemType.MOTOR), 2), (int(ItemType.WIRE), 1)),
        ticks=8,
        name="Engine Unit",
    ),
    Recipe(
        output=int(ItemType.AVIONICS),
        inputs=((int(ItemType.CIRCUIT), 2), (int(ItemType.SENSOR), 2)),
        ticks=10,
        name="Avionics",
    ),
    Recipe(
        output=int(ItemType.ROCKET_CORE),
        inputs=((int(ItemType.ENGINE_UNIT), 1), (int(ItemType.AVIONICS), 1)),
        ticks=10,
        name="Rocket Core",
    ),
)


NUM_RECIPES: int = len(BASE_RECIPES)
MAX_RECIPE_INPUTS: int = max(len(r.inputs) for r in BASE_RECIPES)

RECIPE_NAMES: list[str] = [r.name for r in BASE_RECIPES]


# ---------------------------------------------------------------------------
# Derived JAX arrays — projected from BASE_RECIPES
# ---------------------------------------------------------------------------

# Per-recipe machine-type gate: smelting recipes run on FURNACE
# entities, everything else on ASSEMBLER entities.
RECIPE_MACHINE_TYPE: jnp.ndarray = jnp.array(
    [r.machine_type for r in BASE_RECIPES],
    dtype=jnp.int32,
)

RECIPE_OUTPUTS: jnp.ndarray = jnp.array(
    [r.output for r in BASE_RECIPES],
    dtype=jnp.int32,
)
# Per-recipe output count — number of items deposited per completed
# cycle. Defaults to 1 for every shipped recipe; settable per-recipe
# for balance tuning.
RECIPE_OUTPUT_COUNTS: jnp.ndarray = jnp.array(
    [r.output_count for r in BASE_RECIPES],
    dtype=jnp.int32,
)
RECIPE_TICKS: jnp.ndarray = jnp.array(
    [r.ticks for r in BASE_RECIPES],
    dtype=jnp.int32,
)
# 1-input recipes pad the missing slot with (EMPTY, 0) so the
# Phase 3 matcher in ``run_combiners`` can treat every recipe as a
# 2-slot lookup without branching on recipe arity.
RECIPE_INPUT_ITEMS: jnp.ndarray = jnp.array(
    [
        [item for item, _ in r.inputs]
        + [int(ItemType.EMPTY)] * (MAX_RECIPE_INPUTS - len(r.inputs))
        for r in BASE_RECIPES
    ],
    dtype=jnp.int32,
)
RECIPE_INPUT_COUNTS: jnp.ndarray = jnp.array(
    [
        [count for _, count in r.inputs] + [0] * (MAX_RECIPE_INPUTS - len(r.inputs))
        for r in BASE_RECIPES
    ],
    dtype=jnp.int32,
)

# Reverse lookup: ItemType -> recipe index (-1 if not an output).
OUTPUT_TO_RECIPE: jnp.ndarray = jnp.full(
    len(ItemType),
    -1,
    dtype=jnp.int32,
)
for _i, _r in enumerate(BASE_RECIPES):
    OUTPUT_TO_RECIPE = OUTPUT_TO_RECIPE.at[_r.output].set(_i)

# Maps CRAFT action offset to recipe index (same order as BASE_RECIPES).
CRAFT_ACTION_TO_RECIPE: jnp.ndarray = jnp.arange(
    NUM_RECIPES,
    dtype=jnp.int32,
)
