"""Recipe definitions for the FactoriaX environment.

Unified recipe table used by both player crafting (instant) and assembler
auto-crafting (timed). Each recipe has a unique input type-set: the
assembler determines what to produce from what goes in, with no manual
recipe selection needed.

Player crafting consumes materials from inventory and produces one output
item instantly. Assemblers use the ``ticks`` field as a production delay.
"""

from __future__ import annotations

from typing import TypedDict

import jax.numpy as jnp

from factoriax.constants import ItemType


class _Recipe(TypedDict):
    """Recipe dictionary with output item, input list, and tick count."""

    output: int
    inputs: list[tuple[int, int]]
    ticks: int


# ---------------------------------------------------------------------------
# Unified recipe table — single source of truth
# ---------------------------------------------------------------------------

RECIPES: list[_Recipe] = [
    # Plates (tier 0.5 — smelting)
    {
        "output": ItemType.IRON_PLATE,
        "inputs": [(ItemType.IRON_ORE, 2)],
        "ticks": 2,
    },
    {
        "output": ItemType.COPPER_PLATE,
        "inputs": [(ItemType.COPPER_ORE, 2)],
        "ticks": 2,
    },
    {
        "output": ItemType.TIN_PLATE,
        "inputs": [(ItemType.TIN_ORE, 2)],
        "ticks": 2,
    },
    {
        "output": ItemType.WAFER,
        "inputs": [(ItemType.SILICON, 2)],
        "ticks": 2,
    },
    # Intermediates (tier 1)
    {
        "output": ItemType.STEEL,
        "inputs": [(ItemType.IRON_PLATE, 2), (ItemType.TIN_PLATE, 1)],
        "ticks": 4,
    },
    {
        "output": ItemType.CIRCUIT,
        "inputs": [(ItemType.COPPER_PLATE, 1), (ItemType.WAFER, 1)],
        "ticks": 4,
    },
    {
        "output": ItemType.WIRE,
        "inputs": [(ItemType.IRON_PLATE, 1), (ItemType.COPPER_PLATE, 2)],
        "ticks": 4,
    },
    # Components (tier 2)
    {
        "output": ItemType.MOTOR,
        "inputs": [(ItemType.STEEL, 1), (ItemType.WIRE, 1)],
        "ticks": 6,
    },
    {
        "output": ItemType.SENSOR,
        "inputs": [(ItemType.CIRCUIT, 1), (ItemType.WIRE, 1)],
        "ticks": 6,
    },
    # Machines
    {
        "output": ItemType.CONVEYOR_BELT,
        "inputs": [(ItemType.MOTOR, 1), (ItemType.IRON_PLATE, 2)],
        "ticks": 4,
    },
    {
        "output": ItemType.MINER,
        "inputs": [(ItemType.SENSOR, 1), (ItemType.STEEL, 2)],
        "ticks": 6,
    },
    {
        "output": ItemType.ASSEMBLER,
        "inputs": [(ItemType.SENSOR, 1), (ItemType.CIRCUIT, 2)],
        "ticks": 6,
    },
    {
        "output": ItemType.PALLET,
        "inputs": [(ItemType.STEEL, 2), (ItemType.TIN_PLATE, 1)],
        "ticks": 4,
    },
    # Science packs
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
    # Goal
    {
        "output": ItemType.ROCKET,
        "inputs": [(ItemType.MOTOR, 2), (ItemType.SENSOR, 2)],
        "ticks": 100,
    },
]

NUM_RECIPES: int = len(RECIPES)
MAX_RECIPE_INPUTS: int = max(len(r["inputs"]) for r in RECIPES)

RECIPE_NAMES: list[str] = [
    "Iron Plate",
    "Copper Plate",
    "Tin Plate",
    "Wafer",
    "Steel",
    "Circuit",
    "Wire",
    "Motor",
    "Sensor",
    "Conveyor Belt",
    "Miner",
    "Assembler",
    "Pallet",
    "Basic Science Pack",
    "Advanced Science Pack",
    "Rocket",
]

# ---------------------------------------------------------------------------
# Derived JAX arrays — single source of truth from the dicts above.
# ---------------------------------------------------------------------------

RECIPE_OUTPUTS: jnp.ndarray = jnp.array(
    [r["output"] for r in RECIPES], dtype=jnp.int32,
)
RECIPE_TICKS: jnp.ndarray = jnp.array(
    [r["ticks"] for r in RECIPES], dtype=jnp.int32,
)
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
    len(ItemType), -1, dtype=jnp.int32,
)
for _i, _r in enumerate(RECIPES):
    OUTPUT_TO_RECIPE = OUTPUT_TO_RECIPE.at[_r["output"]].set(_i)

# Maps CRAFT action offset to recipe index (same order as RECIPES).
CRAFT_ACTION_TO_RECIPE: jnp.ndarray = jnp.arange(
    NUM_RECIPES, dtype=jnp.int32,
)

# ---------------------------------------------------------------------------
# Backward-compat aliases (will be removed in Stage 2+)
# ---------------------------------------------------------------------------

MAX_ASSEMBLER_STACK_SIZE: int = 1000
NUM_ASSEMBLER_RECIPES: int = NUM_RECIPES
MAX_ASSEMBLER_RECIPE_INPUTS: int = MAX_RECIPE_INPUTS
ASSEMBLER_RECIPES = RECIPES
ASSEMBLER_RECIPE_NAMES: list[str] = RECIPE_NAMES
ASSEMBLER_RECIPE_OUTPUTS: jnp.ndarray = RECIPE_OUTPUTS
ASSEMBLER_RECIPE_TICKS: jnp.ndarray = RECIPE_TICKS
ASSEMBLER_RECIPE_INPUT_ITEMS: jnp.ndarray = RECIPE_INPUT_ITEMS
ASSEMBLER_RECIPE_INPUT_COUNTS: jnp.ndarray = RECIPE_INPUT_COUNTS
