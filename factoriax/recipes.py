"""Recipe definitions for the FactoriaX environment.

All recipes are defined once as Python dicts. JAX arrays used by the
simulation (``RECIPE_OUTPUTS``, ``RECIPE_TICKS``, etc.) are derived
programmatically from these dicts, eliminating manual duplication.

Player-craftable recipes
------------------------
Players craft these by hand via the CRAFT action. Each recipe consumes
materials from the player's inventory and produces one output item after
a fixed number of ticks.

Assembler recipes
-----------------
Assemblers (placed machines) run these automatically when their input
slots hold enough materials. Products include rocket components and
the rocket itself.
"""

from __future__ import annotations

import jax.numpy as jnp

from factoriax.constants import ItemType

# ---------------------------------------------------------------------------
# Player-craftable recipes
# ---------------------------------------------------------------------------

RECIPES: list[dict] = [
    {
        "output": ItemType.MINER,
        "inputs": [(ItemType.COPPER, 5), (ItemType.IRON, 5)],
        "ticks": 3,
    },
    {
        "output": ItemType.CHEST,
        "inputs": [(ItemType.IRON, 5)],
        "ticks": 2,
    },
    {
        "output": ItemType.CONVEYOR_BELT,
        "inputs": [(ItemType.IRON, 1)],
        "ticks": 1,
    },
    {
        "output": ItemType.ARM,
        "inputs": [(ItemType.IRON, 5), (ItemType.COPPER, 1)],
        "ticks": 5,
    },
    {
        "output": ItemType.ASSEMBLER,
        "inputs": [(ItemType.IRON, 10), (ItemType.COPPER, 5)],
        "ticks": 5,
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

ASSEMBLER_RECIPES: list[dict] = [
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
]

NUM_ASSEMBLER_RECIPES: int = len(ASSEMBLER_RECIPES)
MAX_ASSEMBLER_RECIPE_INPUTS: int = max(
    len(r["inputs"]) for r in ASSEMBLER_RECIPES
)

ASSEMBLER_RECIPE_NAMES: list[str] = ["Hull", "Fuel Pack", "Rocket"]

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
