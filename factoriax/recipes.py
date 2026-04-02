"""Recipe definitions for the FactoriaX environment.

All recipes are defined once as Python dicts. JAX arrays used by the
simulation (``RECIPE_OUTPUTS``, ``RECIPE_TICKS``, etc.) are derived
programmatically from these dicts, eliminating manual duplication.

Player-craftable recipes
------------------------
Players craft these via direct actions (e.g. ``CRAFT_MINER``). Each
recipe consumes materials from the player's inventory and produces one
output item. Crafting is instant by default (ticks=0). Set ticks > 0
to introduce a countdown delay for research purposes.

Assembler recipes
-----------------
Assemblers (placed machines) run these automatically when their input
slots hold enough materials. Products include rocket components and
the rocket itself.
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
# Player-craftable recipes
# ---------------------------------------------------------------------------

RECIPES: list[_Recipe] = [
    {
        "output": ItemType.MINER,
        "inputs": [(ItemType.COPPER, 5), (ItemType.IRON, 5)],
        "ticks": 0,
    },
    {
        "output": ItemType.CHEST,
        "inputs": [(ItemType.IRON, 5)],
        "ticks": 0,
    },
    {
        "output": ItemType.CONVEYOR_BELT,
        "inputs": [(ItemType.IRON, 1)],
        "ticks": 0,
    },
    {
        "output": ItemType.ARM,
        "inputs": [(ItemType.IRON, 5), (ItemType.COPPER, 1)],
        "ticks": 0,
    },
    {
        "output": ItemType.ASSEMBLER,
        "inputs": [(ItemType.IRON, 10), (ItemType.COPPER, 5)],
        "ticks": 0,
    },
]

NUM_RECIPES: int = len(RECIPES)
MAX_RECIPE_INPUTS: int = max(len(r["inputs"]) for r in RECIPES)

RECIPE_NAMES: list[str] = ["Miner", "Chest", "Conveyor Belt", "Arm", "Assembler"]

# Derived JAX arrays — single source of truth from the dicts above.
RECIPE_OUTPUTS: jnp.ndarray = jnp.array(
    [r["output"] for r in RECIPES], dtype=jnp.int32
)
RECIPE_TICKS: jnp.ndarray = jnp.array(
    [r["ticks"] for r in RECIPES], dtype=jnp.int32
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

# ---------------------------------------------------------------------------
# Assembler recipes
# ---------------------------------------------------------------------------

MAX_ASSEMBLER_STACK_SIZE: int = 1000

ASSEMBLER_RECIPES: list[_Recipe] = [
    {
        "output": ItemType.HULL,
        "inputs": [(ItemType.IRON, 5)],
        "ticks": 4,
    },
    {
        "output": ItemType.FUEL_PACK,
        "inputs": [(ItemType.COPPER, 3), (ItemType.COAL, 2)],
        "ticks": 6,
    },
    {
        "output": ItemType.ROCKET,
        "inputs": [(ItemType.HULL, 50), (ItemType.FUEL_PACK, 20)],
        "ticks": 100,
    },
    {
        "output": ItemType.BASIC_SCIENCE_PACK,
        "inputs": [(ItemType.IRON, 1), (ItemType.COPPER, 1)],
        "ticks": 4,
    },
    {
        "output": ItemType.FUEL_SCIENCE_PACK,
        "inputs": [(ItemType.IRON, 1), (ItemType.COAL, 1)],
        "ticks": 4,
    },
    {
        "output": ItemType.ADVANCED_SCIENCE_PACK,
        "inputs": [(ItemType.HULL, 1), (ItemType.FUEL_PACK, 1)],
        "ticks": 8,
    },
]

NUM_ASSEMBLER_RECIPES: int = len(ASSEMBLER_RECIPES)
MAX_ASSEMBLER_RECIPE_INPUTS: int = max(
    len(r["inputs"]) for r in ASSEMBLER_RECIPES
)

ASSEMBLER_RECIPE_NAMES: list[str] = [
    "Hull",
    "Fuel Pack",
    "Rocket",
    "Basic Science Pack",
    "Fuel Science Pack",
    "Advanced Science Pack",
]

ASSEMBLER_RECIPE_OUTPUTS: jnp.ndarray = jnp.array(
    [r["output"] for r in ASSEMBLER_RECIPES], dtype=jnp.int32
)
ASSEMBLER_RECIPE_TICKS: jnp.ndarray = jnp.array(
    [r["ticks"] for r in ASSEMBLER_RECIPES], dtype=jnp.int32
)
ASSEMBLER_RECIPE_INPUT_ITEMS: jnp.ndarray = jnp.array(
    [
        [item for item, _ in r["inputs"]]
        + [ItemType.EMPTY] * (MAX_ASSEMBLER_RECIPE_INPUTS - len(r["inputs"]))
        for r in ASSEMBLER_RECIPES
    ],
    dtype=jnp.int32,
)
ASSEMBLER_RECIPE_INPUT_COUNTS: jnp.ndarray = jnp.array(
    [
        [count for _, count in r["inputs"]]
        + [0] * (MAX_ASSEMBLER_RECIPE_INPUTS - len(r["inputs"]))
        for r in ASSEMBLER_RECIPES
    ],
    dtype=jnp.int32,
)
