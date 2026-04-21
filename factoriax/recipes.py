"""Recipe definitions for the FactoriaX environment.

Unified recipe table used by both player crafting (instant) and
combiner auto-crafting (timed). Each recipe has a unique input
type-set: the machine determines what to produce from what goes in,
with no manual recipe selection needed. The ``RECIPE_MACHINE_TYPE``
array gates each recipe to its owning machine type — smelting
recipes run on furnaces, everything else on assemblers.

Player crafting consumes materials from inventory and produces one
output item instantly. Combiners use the ``ticks`` field as a
production delay.
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

RECIPES: list[_Recipe] = [
    # Plates (tier 0.5 — smelting, coal acts as a reductant)
    {
        "output": ItemType.IRON_PLATE,
        "inputs": [(ItemType.IRON_ORE, 2), (ItemType.COAL, 1)],
        "ticks": 2,
    },
    {
        "output": ItemType.COPPER_PLATE,
        "inputs": [(ItemType.COPPER_ORE, 2), (ItemType.COAL, 1)],
        "ticks": 2,
    },
    {
        "output": ItemType.TIN_PLATE,
        "inputs": [(ItemType.TIN_ORE, 2), (ItemType.COAL, 1)],
        "ticks": 2,
    },
    {
        "output": ItemType.WAFER,
        "inputs": [(ItemType.SILICON, 2), (ItemType.COAL, 1)],
        "ticks": 2,
    },
    # Intermediates (tier 1)
    {
        "output": ItemType.FRAME,
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
        "inputs": [(ItemType.FRAME, 1), (ItemType.WIRE, 1)],
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
        "inputs": [(ItemType.SENSOR, 1), (ItemType.FRAME, 2)],
        "ticks": 6,
    },
    {
        "output": ItemType.ASSEMBLER,
        "inputs": [(ItemType.SENSOR, 1), (ItemType.CIRCUIT, 2)],
        "ticks": 6,
    },
    {
        "output": ItemType.PALLET,
        "inputs": [(ItemType.FRAME, 2), (ItemType.TIN_PLATE, 1)],
        "ticks": 4,
    },
    {
        "output": ItemType.ARM,
        "inputs": [(ItemType.WIRE, 1), (ItemType.IRON_PLATE, 1)],
        "ticks": 4,
    },
    {
        "output": ItemType.FURNACE,
        "inputs": [(ItemType.IRON_PLATE, 1), (ItemType.REFRACTORY, 1)],
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
    # Furnace half-fab (separate recipe so FURNACE no longer shares its
    # input type-set with WIRE — makes recipe matching unambiguous).
    {
        "output": ItemType.REFRACTORY,
        "inputs": [(ItemType.TIN_PLATE, 1), (ItemType.COAL, 1)],
        "ticks": 4,
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
    "Refractory",
]

# ---------------------------------------------------------------------------
# Derived JAX arrays — single source of truth from the dicts above.
# ---------------------------------------------------------------------------

# Per-recipe machine-type gate: first 4 smelting recipes run on
# FURNACE entities, all others on ASSEMBLER entities. The engine
# shares a single code path — the gate only restricts which recipes
# each machine type can match during Phase 3.
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
