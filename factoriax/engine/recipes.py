"""Recipe definitions for the Factoriax environment.

Recipe data sits in three layers:

- :class:`Recipe` is one record. It holds the item that the recipe produces,
  the items that it consumes, and the time that it takes.
- :class:`RecipeBook` is a validated tuple of records. It enforces the rules
  that the recipe matcher of the engine depends on.
- :class:`RecipeTable` turns a book into stacked JAX arrays, which is what
  JIT-compiled engine code reads.

:data:`BASE_RECIPES` is the tuple that comes with the engine, and
:data:`DEFAULT_RECIPE_TABLE` is its projection. A scenario can supply its own
book, so engine code takes the size of its loops from the table that it
receives, and not from :data:`NUM_RECIPES`.

A recipe runs on a furnace or on an assembler. The output item decides which
one, and no recipe declares a machine. A recipe takes one or two input types.
The projection pads an unused slot with ``(EMPTY, 0)``, so every row is a
two-slot lookup.

A player who crafts a recipe pays from the inventory and gets the output in the
same step. An assembler or a furnace takes ``ticks`` steps instead. Nothing can
write over its output slot, so the rate at which something withdraws that
output limits its throughput.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from flax import struct

from factoriax.engine.constants import ItemType, Machine

# ---------------------------------------------------------------------------
# Recipe dataclass: the canonical per-recipe record
# ---------------------------------------------------------------------------


# Items that a FURNACE smelts. An ASSEMBLER builds every other item. This set
# decides the machine of a recipe, so it is part of recipe identity. A change
# here is a code change, not a balance change.
_FURNACE_OUTPUTS: frozenset[int] = frozenset(
    {
        int(ItemType.IRON_PLATE),
        int(ItemType.COPPER_PLATE),
        int(ItemType.TIN_PLATE),
        int(ItemType.WAFER),
        int(ItemType.REFRACTORY),
    }
)

#: Largest number of different input item types in one recipe. This is not a
#: limit on amounts. A recipe can ask for any count of each type, so Hull
#: consumes four items, 2 frames and 2 iron plates, and still has two inputs.
#:
#: This is a fixed engine limit, not a property of :data:`BASE_RECIPES`, and it
#: applies to a hand craft as much as to a machine. The value is 2 because
#: ``EnvState.ent_asm_in_type`` gives a machine two input slots. The inventory
#: of a player can feed a wider recipe, but the engine holds both paths to one
#: width, so both read the same rows. A larger value needs wider state arrays
#: and a more general recipe matcher, which today tests two slots in both
#: orders by hand. :class:`RecipeBook` refuses a wider recipe.
MAX_RECIPE_INPUTS: int = 2


@dataclass(frozen=True)
class Recipe:
    """One recipe: the item that it produces, and the cost to produce it.

    The identity of a recipe is its output item, its input item types, and the
    machine type that those imply. :data:`BASE_RECIPES` fixes that identity,
    and a change to it is a code change. The numbers are balance, and
    :class:`RecipeBalance` overrides them and leaves the identity alone.

    Attributes
    ----------
    output
        ``ItemType`` value that this recipe produces. The value is unique
        inside a :class:`RecipeBook`, which lets
        :attr:`RecipeTable.output_to_recipe` map an item back to one recipe.
    inputs
        ``(item_type, count)`` pairs that one craft consumes, one or two of
        them. The order joins each count to its item, and it fixes the order
        in which a :class:`RecipeOverride` must list ``input_counts``. The
        order does not decide which input slot holds an item, because the
        matcher accepts both slot orders.
    ticks
        Steps that an assembler or a furnace takes to finish one craft. A hand
        craft ignores this field and finishes in the same step.
    output_count
        Items that one craft produces.
    name
        Label for the UI. This is not an identifier. The engine addresses a
        recipe by its position in the book.
    """

    output: int
    inputs: tuple[tuple[int, int], ...]
    ticks: int
    output_count: int = 1
    name: str = ""

    @property
    def machine_type(self) -> int:
        """Return the machine kind that runs this recipe.

        The value comes from the output item, and nothing stores it. A recipe
        therefore cannot declare a machine that disagrees with its output.

        Returns
        -------
        int
            ``Machine.FURNACE`` for the smelted outputs, ``Machine.ASSEMBLER``
            for every other output.
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
    """Balance numbers that replace the numbers of one recipe.

    A field that holds ``None`` keeps the value of the base recipe. This class
    cannot tune the identity of a recipe: the output item, the input item
    types, and the machine. To change the flow of items, edit
    :data:`BASE_RECIPES`.

    Attributes
    ----------
    input_counts
        New count for each input, in the input order of the base recipe. The
        length must match the number of inputs of that recipe.
    output_count
        New number of items that one craft produces.
    ticks
        New craft time on a machine, in steps.
    """

    input_counts: tuple[int, ...] | None = None
    output_count: int | None = None
    ticks: int | None = None


@dataclass(frozen=True)
class RecipeBalance:
    """Balance overrides for a whole game, with the output item as the key.

    A scenario tunes this object and does not copy the recipe table. Build one,
    apply it with :meth:`RecipeBook.with_balance`, then turn the result into a
    :class:`RecipeTable`.

    The overrides are a tuple of pairs and not a dict, so the class stays
    hashable and safe to share across JIT cache keys. A lookup is linear, and
    that costs nothing here. The list is small, and code reads it when it
    builds a book, never in the hot path.

    Attributes
    ----------
    overrides
        ``(output_item, override)`` pairs. An output can appear one time at
        most. The class ignores an override for an item that no recipe in the
        book produces.

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
            If an output item appears in :attr:`overrides` more than one time.
            The tuple order alone then decides which override wins.
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
            ``ItemType`` value that the recipe produces.

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
    """A tuple of recipes, validated against the rules that the engine needs.

    Construction tests four rules. A bad book therefore fails at once and names
    the recipe at fault, and never builds arrays that fail later at runtime:

    1. **Input arity.** Every recipe takes 1 to :data:`MAX_RECIPE_INPUTS`
       input types. A machine holds that number of input slots in
       ``EnvState``, so the engine cannot feed a wider recipe.
    2. **No repeated input item.** A recipe names each input item one time at
       most. :func:`factoriax.engine.crafting.can_afford_recipe` tests each
       slot on its own. With a repeated item, one slot alone can pass the
       test, and the craft then takes the count of the player below zero.
    3. **Unique outputs.** One recipe at most produces each item.
       :attr:`RecipeTable.output_to_recipe` maps an item back to one recipe.
       Two recipes for the same item also break the yield arithmetic in
       :func:`factoriax.engine.crafting.craft_recipe`.
    4. **Unique input set for each machine and arity.** No two recipes on the
       same machine kind, with the same number of inputs, consume the same
       unordered set of input item types. The recipe matcher selects a recipe
       by input item types alone. The counts of a recipe only gate whether the
       match fires, so counts cannot separate two recipes, and such a pair is
       ambiguous. Two recipes can share an input set at different arities,
       because the matcher gates a 1-input recipe on an empty second slot.
       Two recipes can also share one across machine kinds, because the
       matcher divides by machine first.

    Attributes
    ----------
    recipes
        The validated records, in order. The position of a recipe is its id,
        and that id indexes every array in :class:`RecipeTable`.

    Raises
    ------
    ValueError
        If a recipe breaks one of the four rules above.
    """

    recipes: tuple[Recipe, ...]

    def __post_init__(self) -> None:
        """Test the construction rules, and name the first recipe at fault.

        Raises
        ------
        ValueError
            If the input arity of a recipe falls outside 1 to
            :data:`MAX_RECIPE_INPUTS`, if a recipe names the same input item
            two times, if two recipes produce the same item, or if two recipes
            on one machine kind share an input item-type set at the same
            arity.
        """
        for idx, recipe in enumerate(self.recipes):
            arity = len(recipe.inputs)
            if not 1 <= arity <= MAX_RECIPE_INPUTS:
                raise ValueError(
                    f"Recipe {idx} ({ItemType(recipe.output).name}, "
                    f"{recipe.name!r}) has {arity} input types; a recipe must "
                    f"have 1 to {MAX_RECIPE_INPUTS}. A machine holds "
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
                    f"run_assemblers dispatches a machine's input buffers to a "
                    f"recipe by input item-types alone (counts are ignored), "
                    f"so it cannot tell these two apart. Give one recipe a "
                    f"distinct input item-type set, or a different number of "
                    f"input types."
                )
            seen_keys[key] = idx

    def with_balance(self, balance: RecipeBalance) -> RecipeBook:
        """Apply balance overrides, and return a new validated book.

        The recipe order, the output items, the machine types, and the input
        item types all pass through unchanged. Only the counts and ``ticks``
        change. A recipe whose output the balance does not name passes through
        as it is. The new book runs the same construction tests, and those
        tests still hold, because an override touches no part of the identity
        of a recipe.

        Parameters
        ----------
        balance
            Overrides, with the output item as the key. The method ignores an
            override for an item that this book does not produce, and reports
            nothing.

        Returns
        -------
        RecipeBook
            A new book with the overrides applied. The method returns the same
            book unchanged when ``balance`` holds no overrides.

        Raises
        ------
        ValueError
            If the length of ``input_counts`` in an override does not match the
            number of inputs of the target recipe, or if an input count, an
            ``output_count``, or a ``ticks`` value comes out negative.
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

#: The recipes that come with the engine, in the order that gives them their
#: ids.
#:
#: The order is internal. ``execute_action`` maps a ``CRAFT_`` action to an item
#: and finds that item in ``output_to_recipe``. The position of a recipe
#: therefore never reaches the action space, and a new order for this tuple does
#: not change the meaning of a craft action. Every recipe here has a matching
#: ``CRAFT_`` action. The book of a scenario does not have to do the same.
BASE_RECIPES: tuple[Recipe, ...] = (
    # Furnace smelts: 1 ore + 1 coal (fuel) -> 1 plate. Every furnace recipe
    # takes 2 inputs, so logistics code has no 1-input special case.
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
    # Assembler: the cheap logistics machines. The order matches the CRAFT_*
    # enum: belt, miner, assembler, pallet, arm, furnace.
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
    # Assembler: science lab, the pair of CRAFT_SCIENCE_LAB. The inputs are an
    # unused pair, so the recipe stays unique (see tests/test_recipes.py).
    Recipe(
        output=int(ItemType.SCIENCE_LAB),
        inputs=((int(ItemType.CIRCUIT), 2), (int(ItemType.MOTOR), 2)),
        ticks=8,
        name="Science Lab",
    ),
    # Belt-network parts. COAL is the second input, which keeps their assembler
    # input sets unique. The ore+COAL smelts run on a FURNACE.
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
    # Furnace half-fabricate: limestone + coal, 2 inputs like every smelt.
    Recipe(
        output=int(ItemType.REFRACTORY),
        inputs=((int(ItemType.LIMESTONE), 1), (int(ItemType.COAL), 1)),
        ticks=4,
        name="Refractory",
    ),
    # Rocket sub-assemblies: where the four tiers come together.
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


#: Number of recipes in :data:`BASE_RECIPES`. A scenario supplies its own book,
#: so engine code takes the size of its loops from ``table.outputs.shape[0]``.
NUM_RECIPES: int = len(BASE_RECIPES)

#: Names of :data:`BASE_RECIPES`. The table of a scenario carries its own names
#: in :attr:`RecipeTable.names`.
RECIPE_NAMES: list[str] = [r.name for r in BASE_RECIPES]


# ---------------------------------------------------------------------------
# RecipeTable: JAX-friendly projection of a RecipeBook
# ---------------------------------------------------------------------------


class RecipeTable(struct.PyTreeNode):  # type: ignore[no-untyped-call]
    """Stacked JAX arrays that come from a :class:`RecipeBook`.

    This class holds every per-recipe number that the engine reads inside
    JIT-compiled code: the machine cycle match, the craft yield, and the
    action dispatch. It travels on
    :class:`~factoriax.engine.state.EnvParams`, so the numbers reach
    traced code as PyTree leaves, and not as Python globals inside the XLA
    graph. Balance is therefore tunable at runtime with no new compile.
    :class:`RecipeBook` fixes the recipe count and the arity at construction,
    so every balance overlay gives the same shapes.

    A recipe id indexes the rows, and that id is the position of the recipe in
    the book. ``n`` below is the number of recipes.

    Attributes
    ----------
    outputs
        Item that each recipe produces. Shape ``(n,)``, int32.
    output_counts
        Items that one craft produces. Shape ``(n,)``, int32.
    input_items
        Input item in each slot. Shape ``(n, MAX_RECIPE_INPUTS)``, int32. A
        recipe with one input pads the unused slot with ``ItemType.EMPTY``.
    input_counts
        Items that each slot consumes, in the same order as
        :attr:`input_items`. Shape ``(n, MAX_RECIPE_INPUTS)``, int32. A padded
        slot holds 0.
    ticks
        Steps that an assembler or a furnace needs for one craft. Shape
        ``(n,)``, int32.
    machine_type
        ``Machine`` value that runs each recipe. Shape ``(n,)``, int32.
    output_to_recipe
        Recipe id that produces each item, or ``-1`` for an item that no recipe
        produces. Shape ``(len(ItemType),)``, int32. An item indexes this
        array, not a recipe, so it is the one array here with a length that is
        not ``n``.
    names
        Label for each recipe id, in the same order as :attr:`outputs`. This
        field is static and not a traced leaf, so the UI can label the recipes
        of a scenario.
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
        """Turn a :class:`RecipeBook` into stacked JAX arrays.

        The book validated itself at construction, so this method changes only
        shapes and dtypes and runs no further tests. The rows keep the recipe
        order of the book. A 1-input recipe gets the padding ``(EMPTY, 0)``, so
        every row has :data:`MAX_RECIPE_INPUTS` input slots.

        Parameters
        ----------
        book
            A validated book. Its recipe count and its arity fix the shapes of
            every array in the result.

        Returns
        -------
        RecipeTable
            The stacked arrays, and the recipe names as static data.
        """
        recipes = book.recipes
        # Use the engine limit, not max(len(r.inputs)): run_assemblers always
        # reads slot 1, and a read past the end clamps onto slot 0.
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

#: The :class:`RecipeBook` that comes with the engine.
BASE_RECIPE_BOOK: RecipeBook = RecipeBook(recipes=BASE_RECIPES)

#: Default :class:`RecipeTable`, made from :data:`BASE_RECIPE_BOOK`.
DEFAULT_RECIPE_TABLE: RecipeTable = RecipeTable.from_book(BASE_RECIPE_BOOK)

# Aliases of the fields of the default table, for call sites that still do not
# read ``params.recipe_table``. They read BASE_RECIPES, so they ignore the book
# of a scenario and its balance overrides. Read the table on EnvParams instead.
RECIPE_OUTPUTS: jnp.ndarray = DEFAULT_RECIPE_TABLE.outputs
RECIPE_OUTPUT_COUNTS: jnp.ndarray = DEFAULT_RECIPE_TABLE.output_counts
RECIPE_INPUT_ITEMS: jnp.ndarray = DEFAULT_RECIPE_TABLE.input_items
RECIPE_INPUT_COUNTS: jnp.ndarray = DEFAULT_RECIPE_TABLE.input_counts
RECIPE_TICKS: jnp.ndarray = DEFAULT_RECIPE_TABLE.ticks
RECIPE_MACHINE_TYPE: jnp.ndarray = DEFAULT_RECIPE_TABLE.machine_type
OUTPUT_TO_RECIPE: jnp.ndarray = DEFAULT_RECIPE_TABLE.output_to_recipe
