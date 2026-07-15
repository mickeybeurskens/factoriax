"""Recipe definitions for the FactoriaX environment.

Single source of truth for recipe data. Two layers:

- :class:`Recipe` is a frozen dataclass holding one recipe's identity
  (output item, input pairs, machine type) and balance numbers
  (input counts via the ``inputs`` tuple, ``output_count``, ``ticks``).
- :data:`BASE_RECIPES` is the canonical tuple of :class:`Recipe`
  instances shipped with the engine. The JAX arrays consumed by
  :mod:`factoriax.engine.crafting` and :mod:`factoriax.engine.machines` are
  derived projections of this tuple.

Furnace recipes take 1 or 2 input types; the projection pads the unused
slot with ``(EMPTY, 0)`` so every recipe is a 2-slot lookup. Assembler
recipes take exactly 2 input types, and the unordered input pair is
unique per machine type — the invariant that lets the combiner
forward-match dispatch deterministically (see ``tests/test_recipes.py``).

Player crafting consumes inventory and produces ``output_count`` items
instantly; combiners use ``ticks`` as the production delay. The output
slot can't be overwritten, so combiner throughput is bounded by
withdrawal.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from flax import struct

from factoriax.engine.constants import ItemType, Machine

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
    ``output_count``, ``ticks``) are tunable through the
    :class:`RecipeBalance` overlay.
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
            int(Machine.FURNACE)
            if self.output in _FURNACE_OUTPUTS
            else int(Machine.ASSEMBLER)
        )


# ---------------------------------------------------------------------------
# RecipeOverride / RecipeBalance — sparse balance overlay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecipeOverride:
    """Per-recipe balance override.

    Each field is optional; ``None`` means "keep the default from the
    base :class:`Recipe`". Identity (output item, input item *types*,
    machine type) is not tunable through this class — only the
    balance numbers. To rewire item flow, modify the base
    :data:`BASE_RECIPES` tuple.
    """

    input_counts: tuple[int, ...] | None = None
    output_count: int | None = None
    ticks: int | None = None


@dataclass(frozen=True)
class RecipeBalance:
    """Sparse balance overlay keyed by output ``ItemType``.

    A :class:`RecipeBalance` is the user-facing config for tuning a
    game without forking the recipe table. Construct one with the
    overrides you want, then apply it via
    :meth:`RecipeBook.with_balance`:

    The overrides are stored as a tuple of pairs (rather than a dict)
    so :class:`RecipeBalance` is hashable and safe to share across
    JIT cache keys; lookups are linear but the overrides list is
    typically tiny (single-digit entries) and only consulted at
    book construction time, not in the hot path.

    Returns
    -------

    >>> balance = RecipeBalance(overrides=(
    ...     (int(ItemType.IRON_PLATE), RecipeOverride(ticks=1)),
    ...     (int(ItemType.WIRE), RecipeOverride(output_count=2)),
    ... ))
    >>> book = BASE_RECIPE_BOOK.with_balance(balance)
    >>> table = RecipeTable.from_book(book)
    >>> params = EnvParams(recipe_table=table)
    """

    overrides: tuple[tuple[int, RecipeOverride], ...] = ()

    def __post_init__(self) -> None:
        seen: set[int] = set()
        for output, _ in self.overrides:
            if output in seen:
                raise ValueError(
                    f"RecipeBalance has duplicate override for output "
                    f"{ItemType(output).name}; each output may appear "
                    f"at most once."
                )
            seen.add(output)

    def get(self, output_item: int) -> RecipeOverride | None:
        """

        Returns
        -------
        type
        """
        for output, override in self.overrides:
            if output == output_item:
                return override
        return None


# ---------------------------------------------------------------------------
# RecipeBook — validated tuple of Recipe records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecipeBook:
    """Validated bundle of :class:`Recipe` records.

    Wraps a tuple of recipes and enforces two structural invariants
    that downstream JAX kernels rely on:

    1. **Unique outputs** — each ``ItemType`` appears as the output of
       at most one recipe. The reverse-lookup ``OUTPUT_TO_RECIPE`` is a
       single-valued mapping; two recipes producing the same item would
       race the deterministic forward-match in
       :func:`factoriax.engine.machines.run_assemblers` and would also break
       :func:`factoriax.engine.crafting.craft_recipe`'s yield calculation.
    2. **Unique input pair per (machine_type, arity)** — no two recipes
       on the same combiner type at the same arity share an unordered
       input type-set. This is the gate the Phase 3 forward-match in
       ``run_combiners`` uses to dispatch a buffer pair to a single
       recipe slot. Different arities are allowed because the matcher
       gates 1-input recipes on slot-emptiness; different machines are
       always allowed because the matcher already partitions by
       machine.

    Both invariants are checked at construction time so a bad book
    fails fast with a named offender rather than producing wrong
    arrays at runtime.

    Raises
    ------
    ValueError
        If two recipes share an output, or if two recipes
        on the same machine type at the same arity share an
        unordered input type-set.
    """

    recipes: tuple[Recipe, ...]

    def __post_init__(self) -> None:
        seen_outputs: dict[int, int] = {}
        for idx, recipe in enumerate(self.recipes):
            if recipe.output in seen_outputs:
                raise ValueError(
                    f"Duplicate recipe output {ItemType(recipe.output).name}: "
                    f"recipes {seen_outputs[recipe.output]} and {idx} both "
                    f"produce it. Each ItemType must have at most one recipe."
                )
            seen_outputs[recipe.output] = idx

        seen_keys: dict[tuple[int, int, frozenset[int]], int] = {}
        for idx, recipe in enumerate(self.recipes):
            key = (
                recipe.machine_type,
                len(recipe.inputs),
                frozenset(int(item) for item, _ in recipe.inputs),
            )
            if key in seen_keys:
                other_idx = seen_keys[key]
                other = self.recipes[other_idx]
                input_names = ", ".join(ItemType(item).name for item in sorted(key[2]))
                raise ValueError(
                    f"Recipe {idx} ({ItemType(recipe.output).name}, "
                    f"{recipe.name!r}) and recipe {other_idx} "
                    f"({ItemType(other.output).name}, {other.name!r}) both "
                    f"consume the input set {{{input_names}}} as "
                    f"{key[1]}-input recipes on machine "
                    f"{Machine(key[0]).name}. The forward-match in "
                    f"run_combiners dispatches a machine's input buffers to a "
                    f"recipe by input item-types alone (counts are ignored), "
                    f"so it cannot tell these two apart. Give one recipe a "
                    f"distinct input item-type set, or a different number of "
                    f"input types."
                )
            seen_keys[key] = idx

    def with_balance(self, balance: RecipeBalance) -> RecipeBook:
        """Apply a balance overlay, returning a new validated book.

        Identity (output items, machine type, recipe order, input
        item types) is preserved — only the per-recipe balance numbers
        (input counts, ``output_count``, ``ticks``) are tunable.
        Recipes whose ``output`` does not appear in ``balance`` are
        passed through unchanged. The returned :class:`RecipeBook` is
        re-validated through the standard ``__post_init__`` checks so
        the uniqueness invariants still hold; balance overlays cannot
        introduce duplicate outputs or input pairs because they only
        touch counts/ticks.

        Parameters
        ----------
        balance :
            Sparse :class:`RecipeBalance` keyed by output
            ``ItemType``. Overrides for outputs that don't exist
            in this book are silently ignored.

        Returns
        -------
        New
            class:`RecipeBook` with the overrides applied.

        Raises
        ------
        ValueError
            If a :class:`RecipeOverride` specifies an
            ``input_counts`` tuple whose length does not match the
            target recipe's input count, or if any count / ticks
            value is negative.
        """
        if not balance.overrides:
            return self

        new_recipes: list[Recipe] = []
        applied: set[int] = set()
        for recipe in self.recipes:
            override = balance.get(recipe.output)
            if override is None:
                new_recipes.append(recipe)
                continue
            applied.add(recipe.output)

            new_inputs = recipe.inputs
            if override.input_counts is not None:
                if len(override.input_counts) != len(recipe.inputs):
                    raise ValueError(
                        f"RecipeOverride for "
                        f"{ItemType(recipe.output).name} specifies "
                        f"input_counts of length "
                        f"{len(override.input_counts)} but the recipe has "
                        f"{len(recipe.inputs)} inputs."
                    )
                for count in override.input_counts:
                    if count < 0:
                        raise ValueError(
                            f"RecipeOverride for "
                            f"{ItemType(recipe.output).name} has a "
                            f"negative input count {count}."
                        )
                new_inputs = tuple(
                    (item, override.input_counts[i])
                    for i, (item, _) in enumerate(recipe.inputs)
                )

            new_output_count = (
                recipe.output_count
                if override.output_count is None
                else override.output_count
            )
            new_ticks = recipe.ticks if override.ticks is None else override.ticks
            if new_output_count < 0:
                raise ValueError(
                    f"RecipeOverride for {ItemType(recipe.output).name} "
                    f"has a negative output_count {new_output_count}."
                )
            if new_ticks < 0:
                raise ValueError(
                    f"RecipeOverride for {ItemType(recipe.output).name} "
                    f"has a negative ticks value {new_ticks}."
                )

            new_recipes.append(
                Recipe(
                    output=recipe.output,
                    inputs=new_inputs,
                    ticks=new_ticks,
                    output_count=new_output_count,
                    name=recipe.name,
                )
            )

        return RecipeBook(recipes=tuple(new_recipes))


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
    # Furnace smelts — 1 ore + 1 coal (fuel) -> 1 plate. Every furnace
    # recipe is 2-input so logistics never special-case a 1-input outlier.
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
        output=int(ItemType.TIER1_SCIENCE_PACK),
        inputs=((int(ItemType.MOTOR), 1), (int(ItemType.TIN_PLATE), 1)),
        ticks=8,
        name="Tier 1 Science Pack",
    ),
    Recipe(
        output=int(ItemType.TIER2_SCIENCE_PACK),
        inputs=((int(ItemType.SENSOR), 1), (int(ItemType.WAFER), 1)),
        ticks=8,
        name="Tier 2 Science Pack",
    ),
    Recipe(
        output=int(ItemType.TIER3_SCIENCE_PACK),
        inputs=((int(ItemType.TIER2_SCIENCE_PACK), 1), (int(ItemType.SENSOR), 1)),
        ticks=8,
        name="Tier 3 Science Pack",
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
    # Belt-network pieces. Paired with COAL to keep their assembler
    # input type-sets unique (the ore+COAL smelts are FURNACE-gated).
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
    # Furnace half-fab — limestone + coal, 2-input like every smelt.
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
# RecipeTable — JAX-friendly projection of a RecipeBook
# ---------------------------------------------------------------------------


class RecipeTable(struct.PyTreeNode):  # type: ignore[no-untyped-call]
    """Stacked JAX arrays projected from a :class:`RecipeBook`.

    Holds every per-recipe number the engine consumes inside JIT'd
    kernels (combiner cycle matching, crafting yield, action dispatch)
    as a single PyTree leaf set. Stored on :class:`EnvParams` so the
    table flows through every JIT call without being baked into the
    XLA graph as a Python global — this keeps balance numbers
    runtime-tunable without per-tweak recompilation (shapes are stable
    across all balance overlays since :class:`RecipeBook` fixes the
    recipe count and arity at construction time).
    """

    outputs: jnp.ndarray
    output_counts: jnp.ndarray
    input_items: jnp.ndarray
    input_counts: jnp.ndarray
    ticks: jnp.ndarray
    machine_type: jnp.ndarray
    output_to_recipe: jnp.ndarray
    craft_action_to_recipe: jnp.ndarray

    @classmethod
    def from_book(cls, book: RecipeBook) -> RecipeTable:
        """Project a :class:`RecipeBook` into stacked JAX arrays.

        The book has already validated uniqueness, so this method is
        a pure shape-and-dtype projection — no further checks. Pads
        1-input recipes with ``(EMPTY, 0)`` so every recipe row has
        the same arity (``MAX_RECIPE_INPUTS``).

        Parameters
        ----------
        book :
            Validated :class:`RecipeBook`.

        Returns
        -------
        New
            class:`RecipeTable` containing the projected arrays.
        """
        recipes = book.recipes
        n = len(recipes)
        max_inputs = max(len(r.inputs) for r in recipes)

        outputs = jnp.array([r.output for r in recipes], dtype=jnp.int32)
        output_counts = jnp.array([r.output_count for r in recipes], dtype=jnp.int32)
        ticks = jnp.array([r.ticks for r in recipes], dtype=jnp.int32)
        machine_type = jnp.array([r.machine_type for r in recipes], dtype=jnp.int32)
        input_items = jnp.array(
            [
                [item for item, _ in r.inputs]
                + [int(ItemType.EMPTY)] * (max_inputs - len(r.inputs))
                for r in recipes
            ],
            dtype=jnp.int32,
        )
        input_counts = jnp.array(
            [
                [count for _, count in r.inputs] + [0] * (max_inputs - len(r.inputs))
                for r in recipes
            ],
            dtype=jnp.int32,
        )
        output_to_recipe = jnp.full(len(ItemType), -1, dtype=jnp.int32)
        for i, r in enumerate(recipes):
            output_to_recipe = output_to_recipe.at[r.output].set(i)
        craft_action_to_recipe = jnp.arange(n, dtype=jnp.int32)
        return cls(
            outputs=outputs,
            output_counts=output_counts,
            input_items=input_items,
            input_counts=input_counts,
            ticks=ticks,
            machine_type=machine_type,
            output_to_recipe=output_to_recipe,
            craft_action_to_recipe=craft_action_to_recipe,
        )


# ---------------------------------------------------------------------------
# Default book + table — module globals projected from BASE_RECIPES
# ---------------------------------------------------------------------------

#: Canonical :class:`RecipeBook` shipped with the engine.
BASE_RECIPE_BOOK: RecipeBook = RecipeBook(recipes=BASE_RECIPES)

#: Default :class:`RecipeTable` derived from :data:`BASE_RECIPE_BOOK`.
DEFAULT_RECIPE_TABLE: RecipeTable = RecipeTable.from_book(BASE_RECIPE_BOOK)

# Deprecated module-level aliases of the default table's fields. Prefer
# ``params.recipe_table.*``; kept until all call sites are migrated, then remove.
RECIPE_OUTPUTS: jnp.ndarray = DEFAULT_RECIPE_TABLE.outputs
RECIPE_OUTPUT_COUNTS: jnp.ndarray = DEFAULT_RECIPE_TABLE.output_counts
RECIPE_INPUT_ITEMS: jnp.ndarray = DEFAULT_RECIPE_TABLE.input_items
RECIPE_INPUT_COUNTS: jnp.ndarray = DEFAULT_RECIPE_TABLE.input_counts
RECIPE_TICKS: jnp.ndarray = DEFAULT_RECIPE_TABLE.ticks
RECIPE_MACHINE_TYPE: jnp.ndarray = DEFAULT_RECIPE_TABLE.machine_type
OUTPUT_TO_RECIPE: jnp.ndarray = DEFAULT_RECIPE_TABLE.output_to_recipe
CRAFT_ACTION_TO_RECIPE: jnp.ndarray = DEFAULT_RECIPE_TABLE.craft_action_to_recipe
