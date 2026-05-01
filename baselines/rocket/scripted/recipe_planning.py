"""Bill-of-materials and production scheduling for scripted agents.

These helpers walk the recipe graph to convert a *target inventory*
(what the agent wants to have placed at end of bootstrap) into:

- a leaf-resource demand (raw ores / coal / limestone) — used to size
  ``MineOre`` goals;
- a topologically ordered crafting schedule — used to emit
  ``ProduceInFurnace`` / ``ProduceInAssembler`` goals in the right
  order, with intermediates (e.g. WIRE for MINER) sized to cover
  every consumer.

These functions are agent-strategy concerns and do not appear inside
the JIT'd engine. They consume :class:`~factoriax.recipes.RecipeBook`
because that is the engine's source of truth for recipe identity and
balance, but produce only Python data — no JAX arrays.

The :func:`bill_of_materials` ``output_count`` ceiling is the
load-bearing detail for balance overlays: a recipe that yields 2
outputs per cycle but is invoked for an odd target count rounds up,
because running an extra cycle is the only way to satisfy the
request, and the agent's bootstrap math has to budget for the extra
ore + coal that extra cycle consumes.
"""

from __future__ import annotations

from factoriax.constants import ItemType
from factoriax.recipes import BASE_RECIPE_BOOK, Recipe, RecipeBook


def bill_of_materials(
    targets: dict[int, int],
    book: RecipeBook = BASE_RECIPE_BOOK,
) -> dict[int, int]:
    """Roll a target item set back to its leaf-resource demand.

    Walks the recipe DAG depth-first from each target. For each
    ``(item, qty)``:

    - If ``item`` has a recipe in ``book``, compute the number of
      cycles needed (``ceil(qty / output_count)``) and recurse on
      every input scaled by that cycle count.
    - Otherwise, ``item`` is a leaf (an ore, COAL, LIMESTONE, ...) —
      add ``qty`` to the leaves dict.

    Cycle detection uses a path-stack: revisiting an item that is
    already on the recursion stack raises ``ValueError`` naming the
    cycle. The recipe books shipped with the engine are DAGs by
    design (Phase 3 forward-match would not terminate otherwise),
    so the check exists for synthetic / user-authored books.

    Args:
        targets: Required item counts, ``{ItemType: qty}``. Items
            already in this dict that are also recipe outputs are
            still rolled back to leaves (the dict is the *target
            inventory at end of bootstrap*, not the raw shopping
            list).
        book: :class:`RecipeBook` to walk. Defaults to
            :data:`~factoriax.recipes.BASE_RECIPE_BOOK`; pass a
            tuned book obtained via
            ``BASE_RECIPE_BOOK.with_balance(...)`` to make the BOM
            track a balance overlay.

    Returns:
        ``{leaf_item: qty}`` — the raw resources required.

    Raises:
        ValueError: If the recipe graph rooted at any target
            contains a cycle.
    """
    by_output: dict[int, Recipe] = {r.output: r for r in book.recipes}
    leaves: dict[int, int] = {}

    def visit(item: int, qty: int, stack: tuple[int, ...]) -> None:
        if qty <= 0:
            return
        if item in stack:
            cycle = [ItemType(i).name for i in stack] + [ItemType(item).name]
            raise ValueError(
                f"Recipe cycle detected: {' -> '.join(cycle)}. "
                f"BOM walk would not terminate."
            )
        recipe = by_output.get(item)
        if recipe is None:
            leaves[item] = leaves.get(item, 0) + qty
            return
        cycles = -(-qty // recipe.output_count)  # ceil division
        new_stack = stack + (item,)
        for input_item, input_qty in recipe.inputs:
            visit(int(input_item), int(input_qty) * cycles, new_stack)

    for item, qty in targets.items():
        visit(int(item), int(qty), ())
    return leaves


def production_schedule(
    targets: dict[int, int],
    book: RecipeBook = BASE_RECIPE_BOOK,
) -> list[tuple[int, int, int]]:
    """Topologically ordered crafting schedule covering ``targets``.

    Returns a list of ``(item, qty, machine_type)`` tuples in an
    order such that every entry's recipe inputs are produced (or
    are leaves) by an earlier entry. The list contains every
    recipe-output item that must be crafted to reach ``targets``,
    intermediates included — e.g. a target of ``MINER`` pulls in
    a separate ``WIRE`` entry sized to cover the WIRE input
    consumed across all the MINER cycles.

    The agent uses this to emit ``ProduceInFurnace`` /
    ``ProduceInAssembler`` goals in the right order without
    enumerating intermediates by hand. ``qty`` reports the *number
    of items produced* (cycles × ``output_count``), which is what
    the agent's goals consume; for a recipe with ``output_count=2``
    asked to produce an odd target, this rounds up to the next even
    number.

    Args:
        targets: Required item counts, ``{ItemType: qty}``. Same
            semantics as :func:`bill_of_materials`.
        book: :class:`RecipeBook` to walk. Defaults to
            :data:`~factoriax.recipes.BASE_RECIPE_BOOK`.

    Returns:
        Ordered ``[(item, qty, machine_type), ...]``. Items not
        in ``book`` (raw resources) are not included; mine them
        via the :func:`bill_of_materials` result.

    Raises:
        ValueError: If the recipe graph contains a cycle.
    """
    by_output: dict[int, Recipe] = {r.output: r for r in book.recipes}
    needed: dict[int, int] = {}

    def visit(item: int, qty: int, stack: tuple[int, ...]) -> None:
        if qty <= 0:
            return
        if item in stack:
            cycle = [ItemType(i).name for i in stack] + [ItemType(item).name]
            raise ValueError(
                f"Recipe cycle detected: {' -> '.join(cycle)}. "
                f"Production schedule would not terminate."
            )
        recipe = by_output.get(item)
        if recipe is None:
            return
        cycles = -(-qty // recipe.output_count)
        produced = cycles * recipe.output_count
        needed[item] = needed.get(item, 0) + produced
        new_stack = stack + (item,)
        for input_item, input_qty in recipe.inputs:
            visit(int(input_item), int(input_qty) * cycles, new_stack)

    for item, qty in targets.items():
        visit(int(item), int(qty), ())

    # Kahn's algorithm over the produced subgraph: order items so
    # an item's recipe inputs come earlier. Items not in ``needed``
    # (leaves) are not in the graph.
    in_degree: dict[int, int] = {item: 0 for item in needed}
    edges: dict[int, list[int]] = {item: [] for item in needed}
    for item in needed:
        recipe = by_output[item]
        for input_item, _ in recipe.inputs:
            iitem = int(input_item)
            if iitem in needed:
                edges[iitem].append(item)
                in_degree[item] += 1

    # Sort frontier by item id so the output order is deterministic
    # across Python builds (dict iteration is insertion-ordered, but
    # explicit ID sort makes the schedule trivially testable).
    frontier = sorted(item for item, deg in in_degree.items() if deg == 0)
    ordered: list[int] = []
    while frontier:
        item = frontier.pop(0)
        ordered.append(item)
        for consumer in edges[item]:
            in_degree[consumer] -= 1
            if in_degree[consumer] == 0:
                frontier.append(consumer)
        frontier.sort()

    if len(ordered) != len(needed):
        raise ValueError(
            "Production schedule failed: residual in-degrees imply a "
            "cycle in the recipe subgraph."
        )

    return [(item, needed[item], by_output[item].machine_type) for item in ordered]


def schedule_qty(
    targets: dict[int, int],
    item: int,
    book: RecipeBook = BASE_RECIPE_BOOK,
) -> int:
    """Return the produced qty for *item* in the schedule, or 0.

    Convenience wrapper around :func:`production_schedule` for the
    common case of "how many WIRE / IRON_PLATE / etc. does this
    target set imply?". Returns 0 if *item* is not in the schedule
    (either a leaf resource or not consumed at all).

    Args:
        targets: Required item counts.
        item: ``ItemType`` to look up in the schedule.
        book: :class:`RecipeBook` driving the schedule.

    Returns:
        The qty for *item* in the schedule (cycles * output_count),
        or 0 if absent.
    """
    for s_item, qty, _ in production_schedule(targets, book):
        if s_item == int(item):
            return qty
    return 0


def book_without_recipes_for(
    book: RecipeBook,
    items: set[int],
) -> RecipeBook:
    """Return a new book with the recipes producing *items* dropped.

    Used to mark certain outputs as "leaves" for the purposes of
    :func:`bill_of_materials` and :func:`production_schedule` —
    e.g. when the agent will withdraw those items from already-
    running cells rather than craft them itself. The dropped
    recipes are not consumed by the BOM walk, so any inputs they
    would have required (ore, coal) drop out of the leaf demand.

    Args:
        book: Source :class:`RecipeBook`.
        items: Output ``ItemType`` integers whose recipes should
            be removed.

    Returns:
        New :class:`RecipeBook` with the matching recipes dropped.
        The remaining recipes are re-validated through the standard
        ``__post_init__`` checks.
    """
    return RecipeBook(
        recipes=tuple(r for r in book.recipes if r.output not in items),
    )


def sum_inventories(*inventories: dict[int, int]) -> dict[int, int]:
    """Sum any number of ``{item: count}`` dicts into one.

    The component-inventory helpers in
    :mod:`baselines.rocket.scripted.goals` (``smelter_cell_inventory``,
    ``belt_network_inventory``, ``coal_feed_inventory``,
    ``ore_node_inventory``) each return a small dict; agents typically
    sum 4–6 of them to build the ``targets`` argument for
    :func:`bill_of_materials` / :func:`production_schedule`. This
    helper is just ``functools.reduce(merge, ...)`` with a
    descriptive name.

    Args:
        *inventories: Any number of ``{item: count}`` dicts.

    Returns:
        Combined dict where each item's count is the sum across
        all inputs.
    """
    total: dict[int, int] = {}
    for inv in inventories:
        for item, qty in inv.items():
            total[int(item)] = total.get(int(item), 0) + int(qty)
    return total


def scale_inventory(inventory: dict[int, int], factor: int) -> dict[int, int]:
    """Multiply every count in an inventory by ``factor``.

    Used to scale a per-component cost (e.g. one smelter cell) up
    to the total for several copies (e.g. four smelter cells).

    Args:
        inventory: ``{item: count}``.
        factor: Multiplier (must be non-negative).

    Returns:
        A new dict with each count multiplied by ``factor``.

    Raises:
        ValueError: If ``factor`` is negative.
    """
    if factor < 0:
        raise ValueError(f"scale_inventory factor must be >= 0; got {factor}")
    return {int(item): int(qty) * factor for item, qty in inventory.items()}
