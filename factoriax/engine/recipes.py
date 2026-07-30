"""Recipe definitions for the FactoriaX environment.

Recipe data sits in three layers:

- :class:`Recipe` is one record: the item it produces, the items it consumes,
  and how long it takes.
- :class:`RecipeBook` is a validated tuple of records. It enforces the rules
  the engine's recipe matching depends on.
- :class:`RecipeTable` projects a book into stacked JAX arrays, which is what
  JIT'd engine code reads.

:data:`BASE_RECIPES` is the tuple shipped with the engine and
:data:`DEFAULT_RECIPE_TABLE` its projection. A scenario may ship its own book,
so engine code sizes loops from the table it is given rather than from
:data:`NUM_RECIPES`.

A recipe runs on a furnace or an assembler, decided by the output item rather
than declared per recipe. Recipes take 1 or 2 input types and the projection
pads the unused slot with ``(EMPTY, 0)``, so every row is a two-slot lookup.

A player crafting a recipe consumes inventory and gets the output in the same
step. A combiner takes ``ticks`` steps instead, and its output slot cannot be
overwritten, so its throughput is bounded by how fast the output is withdrawn.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from flax import struct

from factoriax.engine.constants import ItemType, Machine

# ---------------------------------------------------------------------------
# Recipe dataclass: the canonical per-recipe record
# ---------------------------------------------------------------------------


# Items smelted on a FURNACE. Anything else is built on an ASSEMBLER. This set
# is what decides a recipe's machine, so it is recipe identity: changing it is
# a code change, not config.
_FURNACE_OUTPUTS: frozenset[int] = frozenset(
    {
        int(ItemType.IRON_PLATE),
        int(ItemType.COPPER_PLATE),
        int(ItemType.TIN_PLATE),
        int(ItemType.WAFER),
        int(ItemType.REFRACTORY),
    }
)

#: Most distinct input item types one recipe may consume. Not a limit on
#: amounts: a recipe may ask for any count of each type, so Hull consumes four
#: items, 2 frames and 2 iron plates, and still counts as two inputs.
#:
#: A fixed engine limit, not a property of :data:`BASE_RECIPES`, and it applies
#: to hand crafting as much as to machines. It is 2 because
#: ``EnvState.ent_asm_in_type`` gives a combiner two input slots; a player's
#: inventory could feed a wider recipe, but the engine commits to one width so
#: both paths read the same rows. Raising it means widening those state arrays
#: and generalising the combiner matcher, which checks two slots in both
#: orderings by hand. :class:`RecipeBook` rejects anything wider.
MAX_RECIPE_INPUTS: int = 2


@dataclass(frozen=True)
class Recipe:
    """One recipe: the item it produces and what it takes to produce it.

    A recipe's identity is its output item, its input item types, and the
    machine type those imply. Identity is fixed in :data:`BASE_RECIPES` and
    changing it is a code change. The numbers are balance, and
    :class:`RecipeBalance` overrides them without touching identity.

    Attributes
    ----------
    output
        ``ItemType`` value this recipe produces. Unique within a
        :class:`RecipeBook`, which is what lets
        :attr:`RecipeTable.output_to_recipe` map an item back to one recipe.
    inputs
        ``(item_type, count)`` pairs consumed per craft, one or two of them.
        Order pairs each count with its item and fixes the order a
        :class:`RecipeOverride` must list ``input_counts`` in. It does not
        decide which combiner input slot an item lands in: the matcher
        accepts either slot ordering.
    ticks
        Steps a combiner takes to finish one craft. Player crafting ignores
        this and completes in the same step.
    output_count
        Items produced per craft.
    name
        Label for the UI. Not an identifier; the engine addresses a recipe by
        its position in the book.
    """

    output: int
    inputs: tuple[tuple[int, int], ...]
    ticks: int
    output_count: int = 1
    name: str = ""

    @property
    def machine_type(self) -> int:
        """Return the combiner kind that runs this recipe.

        Derived from the output item rather than stored, so a recipe cannot
        declare a machine that disagrees with what it produces.

        Returns
        -------
        int
            ``Machine.FURNACE`` for the smelted outputs, ``Machine.ASSEMBLER``
            for everything else.
        """
        return (
            int(Machine.FURNACE)
            if self.output in _FURNACE_OUTPUTS
            else int(Machine.ASSEMBLER)
        )


# ---------------------------------------------------------------------------
# RecipeOverride and RecipeBalance: the balance overlay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecipeOverride:
    """Balance numbers to replace on one recipe.

    ``None`` in a field keeps the base recipe's value. Identity, meaning the
    output item, the input item types, and the machine, is not tunable here.
    To rewire item flow, edit :data:`BASE_RECIPES`.

    Attributes
    ----------
    input_counts
        Replacement count per input, in the base recipe's input order. Its
        length must match that recipe's input count.
    output_count
        Replacement number of items produced per craft.
    ticks
        Replacement combiner craft duration, in steps.
    """

    input_counts: tuple[int, ...] | None = None
    output_count: int | None = None
    ticks: int | None = None


@dataclass(frozen=True)
class RecipeBalance:
    """Balance overrides for a whole game, keyed by output item.

    This is the config a scenario tunes without forking the recipe table.
    Build one, apply it with :meth:`RecipeBook.with_balance`, then project the
    result into a :class:`RecipeTable`.

    Overrides are a tuple of pairs rather than a dict so the class stays
    hashable and safe to share across JIT cache keys. Lookup is linear, which
    costs nothing here: the list is small and it is read when a book is built,
    never in the hot path.

    Attributes
    ----------
    overrides
        ``(output_item, override)`` pairs. An output may appear at most once.
        An override for an item no recipe in the book produces is ignored.

    Examples
    --------
    >>> from factoriax.engine.constants import ItemType
    >>> from factoriax.engine.recipes import (
    ...     BASE_RECIPE_BOOK,
    ...     RecipeBalance,
    ...     RecipeOverride,
    ...     RecipeTable,
    ... )
    >>> balance = RecipeBalance(
    ...     overrides=((int(ItemType.IRON_PLATE), RecipeOverride(ticks=1)),)
    ... )
    >>> table = RecipeTable.from_book(BASE_RECIPE_BOOK.with_balance(balance))
    >>> int(table.ticks[0])
    1
    """

    overrides: tuple[tuple[int, RecipeOverride], ...] = ()

    def __post_init__(self) -> None:
        """Reject a duplicate override for the same output item.

        Raises
        ------
        ValueError
            If an output item appears in :attr:`overrides` more than once,
            which would leave the winning override decided by tuple order.
        """
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
        """Find the override that applies to one recipe.

        Parameters
        ----------
        output_item
            ``ItemType`` value of the recipe's output.

        Returns
        -------
        RecipeOverride or None
            The matching override, or ``None`` when this balance leaves that
            recipe on its base numbers.
        """
        for output, override in self.overrides:
            if output == output_item:
                return override
        return None


# ---------------------------------------------------------------------------
# RecipeBook: validated tuple of Recipe records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecipeBook:
    """A tuple of recipes, validated against the rules the engine relies on.

    Construction checks four rules, so a bad book fails immediately and names
    the offender instead of producing arrays that misbehave at runtime:

    1. **Input arity.** Every recipe takes 1 to :data:`MAX_RECIPE_INPUTS`
       input types. A combiner has that many input slots in ``EnvState``, so
       the engine cannot feed a wider recipe.
    2. **No repeated input item.** A recipe names each input item at most
       once. :func:`factoriax.engine.crafting.can_afford_recipe` checks each
       slot on its own, so a repeat would read as affordable on one slot's
       worth and then craft the player's count below zero.
    3. **Unique outputs.** Each item is produced by at most one recipe.
       :attr:`RecipeTable.output_to_recipe` maps an item back to a single
       recipe, and two recipes making the same item would also break the yield
       calculation in :func:`factoriax.engine.crafting.craft_recipe`.
    4. **Unique input set per machine and arity.** No two recipes on the same
       combiner kind, taking the same number of inputs, consume the same
       unordered set of input item types. The combiner matcher picks a recipe
       by input item types; a recipe's counts only gate whether the match
       fires, so counts cannot tell two recipes apart and such a pair would be
       ambiguous. Two recipes may share an input set at different arities,
       because the matcher gates 1-input recipes on the second slot being
       empty, and across machine kinds, because it partitions by machine
       first.

    Attributes
    ----------
    recipes
        The validated records, in order. A recipe's position is its id, which
        indexes every array in :class:`RecipeTable`.

    Raises
    ------
    ValueError
        If any of the four rules above is broken.
    """

    recipes: tuple[Recipe, ...]

    def __post_init__(self) -> None:
        """Check the construction rules, naming the first offender.

        Raises
        ------
        ValueError
            If a recipe's input arity falls outside 1 to
            :data:`MAX_RECIPE_INPUTS`, if a recipe names the same input item
            twice, if two recipes produce the same item, or if two recipes on
            one machine kind share an input item-type set at the same arity.
        """
        for idx, recipe in enumerate(self.recipes):
            arity = len(recipe.inputs)
            if not 1 <= arity <= MAX_RECIPE_INPUTS:
                raise ValueError(
                    f"Recipe {idx} ({ItemType(recipe.output).name}, "
                    f"{recipe.name!r}) has {arity} input types; a recipe must "
                    f"have 1 to {MAX_RECIPE_INPUTS}. A combiner holds "
                    f"{MAX_RECIPE_INPUTS} input slots in EnvState, so the "
                    f"engine cannot feed a wider recipe."
                )

            seen_inputs: set[int] = set()
            for item, _ in recipe.inputs:
                if item in seen_inputs:
                    raise ValueError(
                        f"Recipe {idx} ({ItemType(recipe.output).name}, "
                        f"{recipe.name!r}) names {ItemType(item).name} twice. "
                        f"craft_recipe checks and subtracts each input slot "
                        f"on its own, so a repeated item reads as affordable "
                        f"on one slot's worth and crafts into a negative "
                        f"count. Combine the amounts into one slot."
                    )
                seen_inputs.add(item)

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
        """Apply balance overrides, returning a new validated book.

        Recipe order, output items, machine types, and input item types carry
        over unchanged; only counts and ``ticks`` move. A recipe whose output
        the balance does not mention passes through as is. The new book runs
        the same construction checks, which still hold because overrides touch
        no part of a recipe's identity.

        Parameters
        ----------
        balance
            Overrides keyed by output item. An override for an item this book
            does not produce is ignored rather than reported.

        Returns
        -------
        RecipeBook
            A new book with the overrides applied. Returns the same book
            unchanged when ``balance`` holds no overrides.

        Raises
        ------
        ValueError
            If an override's ``input_counts`` length does not match the target
            recipe's input count, or if a resulting input count,
            ``output_count``, or ``ticks`` is negative.
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
# BASE_RECIPES: the recipes shipped with the engine
# ---------------------------------------------------------------------------

#: The recipes shipped with the engine, in the order that defines their ids.
#:
#: Order is internal. ``execute_action`` maps a ``CRAFT_`` action to an item
#: and looks that item up in ``output_to_recipe``, so a recipe's position
#: never reaches the action space and reordering this tuple does not change
#: what a craft action means. Every recipe here has a matching ``CRAFT_``
#: action; a scenario's own book need not.
BASE_RECIPES: tuple[Recipe, ...] = (
    # Furnace smelts: 1 ore + 1 coal (fuel) -> 1 plate. Every furnace
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
    # Furnace half-fab: limestone + coal, 2-input like every smelt.
    Recipe(
        output=int(ItemType.REFRACTORY),
        inputs=((int(ItemType.LIMESTONE), 1), (int(ItemType.COAL), 1)),
        ticks=4,
        name="Refractory",
    ),
    # Rocket sub-assemblies: the four-tier convergence.
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


#: Number of recipes in :data:`BASE_RECIPES`. Scenarios ship their own books,
#: so engine code sizes loops from ``table.outputs.shape[0]`` instead.
NUM_RECIPES: int = len(BASE_RECIPES)

#: Names of :data:`BASE_RECIPES`. A scenario table carries its own in
#: :attr:`RecipeTable.names`.
RECIPE_NAMES: list[str] = [r.name for r in BASE_RECIPES]


# ---------------------------------------------------------------------------
# RecipeTable: JAX-friendly projection of a RecipeBook
# ---------------------------------------------------------------------------


class RecipeTable(struct.PyTreeNode):  # type: ignore[no-untyped-call]
    """Stacked JAX arrays projected from a :class:`RecipeBook`.

    Holds every per-recipe number the engine reads inside JIT'd code: combiner
    cycle matching, crafting yield, and action dispatch. It rides on
    :class:`~factoriax.engine.state.EnvParams`, so the numbers reach traced
    code as PyTree leaves instead of Python globals baked into the XLA graph.
    That is what makes balance tunable at runtime without recompiling per
    tweak: :class:`RecipeBook` fixes the recipe count and arity at
    construction, so every balance overlay produces identical shapes.

    Rows are indexed by recipe id, which is a recipe's position in the book.
    ``n`` below is the recipe count.

    Attributes
    ----------
    outputs
        Item produced per recipe. Shape ``(n,)``, int32.
    output_counts
        Items produced per craft. Shape ``(n,)``, int32.
    input_items
        Input item per slot. Shape ``(n, MAX_RECIPE_INPUTS)``, int32. A recipe
        with one input pads the unused slot with ``ItemType.EMPTY``.
    input_counts
        Items consumed per slot, aligned with :attr:`input_items`. Shape
        ``(n, MAX_RECIPE_INPUTS)``, int32. A padded slot holds 0.
    ticks
        Steps a combiner needs per craft. Shape ``(n,)``, int32.
    machine_type
        ``Machine`` value that runs each recipe. Shape ``(n,)``, int32.
    output_to_recipe
        Recipe id that produces each item, or ``-1`` for an item no recipe
        produces. Shape ``(len(ItemType),)``, int32. Indexed by item, not by
        recipe, so it is the one array here that is not ``n`` long.
    names
        Label per recipe id, parallel to :attr:`outputs`. Static rather than a
        traced leaf, so the UI can label a scenario's own recipes.
    """

    outputs: jnp.ndarray
    output_counts: jnp.ndarray
    input_items: jnp.ndarray
    input_counts: jnp.ndarray
    ticks: jnp.ndarray
    machine_type: jnp.ndarray
    output_to_recipe: jnp.ndarray
    names: tuple[str, ...] = struct.field(pytree_node=False, default=())

    @classmethod
    def from_book(cls, book: RecipeBook) -> RecipeTable:
        """Project a :class:`RecipeBook` into stacked JAX arrays.

        The book validated itself on construction, so this is a shape and
        dtype projection with no further checks. Rows keep the book's recipe
        order, and a 1-input recipe is padded with ``(EMPTY, 0)`` so every row
        has :data:`MAX_RECIPE_INPUTS` input slots.

        Parameters
        ----------
        book
            A validated book. Its recipe count and arity fix the shapes of
            every array on the result.

        Returns
        -------
        RecipeTable
            The projected arrays, plus the recipe names as static data.
        """
        recipes = book.recipes
        # The engine limit, not max(len(r.inputs)): run_assemblers always
        # reads slot 1, and an out-of-bounds read would clamp onto slot 0.
        max_inputs = MAX_RECIPE_INPUTS

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
        return cls(
            outputs=outputs,
            output_counts=output_counts,
            input_items=input_items,
            input_counts=input_counts,
            ticks=ticks,
            machine_type=machine_type,
            output_to_recipe=output_to_recipe,
            names=tuple(r.name for r in recipes),
        )


# ---------------------------------------------------------------------------
# Default book and table, projected from BASE_RECIPES
# ---------------------------------------------------------------------------

#: Canonical :class:`RecipeBook` shipped with the engine.
BASE_RECIPE_BOOK: RecipeBook = RecipeBook(recipes=BASE_RECIPES)

#: Default :class:`RecipeTable` derived from :data:`BASE_RECIPE_BOOK`.
DEFAULT_RECIPE_TABLE: RecipeTable = RecipeTable.from_book(BASE_RECIPE_BOOK)

# Aliases of the default table's fields, kept for call sites that have not
# moved to ``params.recipe_table``. They read BASE_RECIPES, so they ignore a
# scenario's own book and its balance overrides. Prefer the table on EnvParams.
RECIPE_OUTPUTS: jnp.ndarray = DEFAULT_RECIPE_TABLE.outputs
RECIPE_OUTPUT_COUNTS: jnp.ndarray = DEFAULT_RECIPE_TABLE.output_counts
RECIPE_INPUT_ITEMS: jnp.ndarray = DEFAULT_RECIPE_TABLE.input_items
RECIPE_INPUT_COUNTS: jnp.ndarray = DEFAULT_RECIPE_TABLE.input_counts
RECIPE_TICKS: jnp.ndarray = DEFAULT_RECIPE_TABLE.ticks
RECIPE_MACHINE_TYPE: jnp.ndarray = DEFAULT_RECIPE_TABLE.machine_type
OUTPUT_TO_RECIPE: jnp.ndarray = DEFAULT_RECIPE_TABLE.output_to_recipe
