"""Recipe definitions for the FactoriaX environment.

Unified recipe table used by both player crafting (instant) and
combiner auto-crafting (timed). Two shapes:

- **Furnace recipes** take exactly 1 input type. The recipe table
  pads the unused slot with ``(EMPTY, 0)`` so the Phase 3 matcher in
  ``run_combiners`` can treat every recipe as a 2-slot lookup.
- **Assembler recipes** take exactly 2 input types, and the
  (unordered) pair is unique across the table. Uniqueness is the
  invariant that lets Phase 3 forward-match deterministically; see
  ``tests/test_recipes.py``.

The ``RECIPE_MACHINE_TYPE`` array gates each recipe to its owning
machine type. Player crafting consumes materials from inventory and
produces one output item instantly. Combiners use the ``ticks``
field as a production delay — plus the output slot can't be
overwritten, so throughput is bounded by withdrawal.
"""

from __future__ import annotations

from typing import TypedDict

import jax.numpy as jnp

from factoriax.constants import ItemType, MachineType


class _Recipe(TypedDict):
    """Recipe dictionary with output item, input list, and tick count."""

    output: int
    inputs: list[tuple[int, int]]
    ticks: int


# ---------------------------------------------------------------------------
# Unified recipe table — single source of truth
# ---------------------------------------------------------------------------

# Recipe order MATTERS: positions 0–18 are the recipes that map to
# ``Action.CRAFT_IRON_PLATE`` … ``Action.CRAFT_SCIENCE_LAB`` via
# ``CRAFT_ACTION_TO_RECIPE = jnp.arange(19)``. Positions 19+ are
# machine-only (no CRAFT action) — refractory plus the four new
# rocket sub-assemblies.
RECIPES: list[_Recipe] = [
    # -------- CRAFT-addressable slots 0–17 --------
    # Furnace smelts — coal is consumed as fuel for every plate
    # (1 ore + 1 coal → 1 plate). Refractory stays as a 1-input
    # recipe (coal only) so coal can still be smelted standalone.
    {
        "output": ItemType.IRON_PLATE,
        "inputs": [(ItemType.IRON_ORE, 1), (ItemType.COAL, 1)],
        "ticks": 2,
    },
    {
        "output": ItemType.COPPER_PLATE,
        "inputs": [(ItemType.COPPER_ORE, 1), (ItemType.COAL, 1)],
        "ticks": 2,
    },
    {
        "output": ItemType.TIN_PLATE,
        "inputs": [(ItemType.TIN_ORE, 1), (ItemType.COAL, 1)],
        "ticks": 2,
    },
    {
        "output": ItemType.WAFER,
        "inputs": [(ItemType.SILICON, 1), (ItemType.COAL, 1)],
        "ticks": 2,
    },
    # Assembler: base intermediates.
    {
        "output": ItemType.FRAME,
        "inputs": [(ItemType.IRON_PLATE, 1), (ItemType.TIN_PLATE, 1)],
        "ticks": 4,
    },
    {
        "output": ItemType.CIRCUIT,
        "inputs": [(ItemType.COPPER_PLATE, 1), (ItemType.WAFER, 1)],
        "ticks": 4,
    },
    {
        "output": ItemType.WIRE,
        "inputs": [(ItemType.COPPER_PLATE, 1), (ItemType.TIN_PLATE, 1)],
        "ticks": 4,
    },
    # Assembler: components.
    {
        "output": ItemType.MOTOR,
        "inputs": [(ItemType.FRAME, 1), (ItemType.WIRE, 1)],
        "ticks": 6,
    },
    {
        "output": ItemType.SENSOR,
        "inputs": [(ItemType.CIRCUIT, 1), (ItemType.WIRE, 1)],
        "ticks": 6,
    },
    # Assembler: logistics machines (cheap). Order matches the
    # CRAFT_* enum: belt, miner, assembler, pallet, arm, furnace.
    {
        "output": ItemType.CONVEYOR_BELT,
        "inputs": [(ItemType.IRON_PLATE, 1), (ItemType.COPPER_PLATE, 1)],
        "ticks": 4,
    },
    {
        "output": ItemType.MINER,
        "inputs": [(ItemType.IRON_PLATE, 1), (ItemType.WIRE, 1)],
        "ticks": 6,
    },
    {
        "output": ItemType.ASSEMBLER,
        "inputs": [(ItemType.FRAME, 1), (ItemType.CIRCUIT, 1)],
        "ticks": 8,
    },
    {
        "output": ItemType.PALLET,
        "inputs": [(ItemType.TIN_PLATE, 1), (ItemType.WIRE, 1)],
        "ticks": 4,
    },
    {
        "output": ItemType.ARM,
        "inputs": [(ItemType.COPPER_PLATE, 1), (ItemType.WIRE, 1)],
        "ticks": 4,
    },
    {
        "output": ItemType.FURNACE,
        "inputs": [(ItemType.IRON_PLATE, 1), (ItemType.REFRACTORY, 1)],
        "ticks": 6,
    },
    # Assembler: science packs.
    {
        "output": ItemType.BASIC_SCIENCE_PACK,
        "inputs": [(ItemType.MOTOR, 1), (ItemType.TIN_PLATE, 1)],
        "ticks": 8,
    },
    {
        "output": ItemType.ADVANCED_SCIENCE_PACK,
        "inputs": [(ItemType.SENSOR, 1), (ItemType.WAFER, 1)],
        "ticks": 8,
    },
    # Assembler: capstone.
    {
        "output": ItemType.ROCKET,
        "inputs": [(ItemType.HULL, 6), (ItemType.ROCKET_CORE, 4)],
        "ticks": 300,
    },
    # Assembler: science lab (pairs with CRAFT_SCIENCE_LAB). Inputs
    # chosen from an unused pair so the recipe-uniqueness invariant
    # (see tests/test_recipes.py) holds.
    {
        "output": ItemType.SCIENCE_LAB,
        "inputs": [(ItemType.CIRCUIT, 2), (ItemType.MOTOR, 2)],
        "ticks": 8,
    },
    # -------- Machine-only slots 19+ (no CRAFT action) --------
    # Furnace half-fab (keeps FURNACE recipe's input type-set unique).
    {
        "output": ItemType.REFRACTORY,
        "inputs": [(ItemType.COAL, 1)],
        "ticks": 4,
    },
    # Rocket sub-assemblies — the four-tier convergence.
    {
        "output": ItemType.HULL,
        "inputs": [(ItemType.FRAME, 2), (ItemType.IRON_PLATE, 2)],
        "ticks": 8,
    },
    {
        "output": ItemType.ENGINE_UNIT,
        "inputs": [(ItemType.MOTOR, 2), (ItemType.WIRE, 1)],
        "ticks": 8,
    },
    {
        "output": ItemType.AVIONICS,
        "inputs": [(ItemType.CIRCUIT, 2), (ItemType.SENSOR, 2)],
        "ticks": 10,
    },
    {
        "output": ItemType.ROCKET_CORE,
        "inputs": [(ItemType.ENGINE_UNIT, 1), (ItemType.AVIONICS, 1)],
        "ticks": 10,
    },
]

NUM_RECIPES: int = len(RECIPES)
MAX_RECIPE_INPUTS: int = max(len(r["inputs"]) for r in RECIPES)

RECIPE_NAMES: list[str] = [
    "Iron Plate",
    "Copper Plate",
    "Tin Plate",
    "Wafer",
    "Frame",
    "Circuit",
    "Wire",
    "Motor",
    "Sensor",
    "Conveyor Belt",
    "Miner",
    "Assembler",
    "Pallet",
    "Arm",
    "Furnace",
    "Basic Science Pack",
    "Advanced Science Pack",
    "Rocket",
    "Science Lab",
    "Refractory",
    "Hull",
    "Engine Unit",
    "Avionics",
    "Rocket Core",
]

# ---------------------------------------------------------------------------
# Derived JAX arrays — single source of truth from the dicts above.
# ---------------------------------------------------------------------------

# Per-recipe machine-type gate: smelting recipes run on FURNACE
# entities, everything else on ASSEMBLER entities.
_FURNACE_OUTPUTS: frozenset[int] = frozenset(
    {
        int(ItemType.IRON_PLATE),
        int(ItemType.COPPER_PLATE),
        int(ItemType.TIN_PLATE),
        int(ItemType.WAFER),
        int(ItemType.REFRACTORY),
    }
)
RECIPE_MACHINE_TYPE: jnp.ndarray = jnp.array(
    [
        int(MachineType.FURNACE)
        if r["output"] in _FURNACE_OUTPUTS
        else int(MachineType.ASSEMBLER)
        for r in RECIPES
    ],
    dtype=jnp.int32,
)

RECIPE_OUTPUTS: jnp.ndarray = jnp.array(
    [r["output"] for r in RECIPES],
    dtype=jnp.int32,
)
RECIPE_TICKS: jnp.ndarray = jnp.array(
    [r["ticks"] for r in RECIPES],
    dtype=jnp.int32,
)
# 1-input recipes pad the missing slot with (EMPTY, 0) so the
# Phase 3 matcher in ``run_combiners`` can treat every recipe as a
# 2-slot lookup without branching on recipe arity.
RECIPE_INPUT_ITEMS: jnp.ndarray = jnp.array(
    [
        [item for item, _ in r["inputs"]]
        + [ItemType.EMPTY] * (MAX_RECIPE_INPUTS - len(r["inputs"]))
        for r in RECIPES
    ],
    dtype=jnp.int32,
)
RECIPE_INPUT_COUNTS: jnp.ndarray = jnp.array(
    [
        [count for _, count in r["inputs"]]
        + [0] * (MAX_RECIPE_INPUTS - len(r["inputs"]))
        for r in RECIPES
    ],
    dtype=jnp.int32,
)

# Reverse lookup: ItemType -> recipe index (-1 if not an output).
OUTPUT_TO_RECIPE: jnp.ndarray = jnp.full(
    len(ItemType),
    -1,
    dtype=jnp.int32,
)
for _i, _r in enumerate(RECIPES):
    OUTPUT_TO_RECIPE = OUTPUT_TO_RECIPE.at[_r["output"]].set(_i)

# Maps CRAFT action offset to recipe index (same order as RECIPES).
CRAFT_ACTION_TO_RECIPE: jnp.ndarray = jnp.arange(
    NUM_RECIPES,
    dtype=jnp.int32,
)
