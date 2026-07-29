"""Generic layered-DAG layout primitives.

Two complementary functions:

* :func:`assign_tiers` — longest-path tier assignment. Source nodes
  (those with no inputs) sit at tier 0; every other node sits one tier
  above the deepest input.
* :func:`order_within_tiers` — Sugiyama-style barycenter sweep that
  reorders nodes within each tier to reduce edge crossings.

Both work on any acyclic ``{node: inputs}`` adjacency map, so the same
code can drive the recipe DAG and any other layered visualisations.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

_SENTINEL: float = 1e9


def assign_tiers(adjacency: Mapping[str, Iterable[str]]) -> dict[str, int]:
    """Longest-path tier assignment for an acyclic ``{output: inputs}`` map.

    Parameters
    ----------
    adjacency :
        ``{node: iterable_of_input_nodes}``. Nodes that act
        only as inputs (sources) need not appear as keys; they are
        discovered while walking the recipes.
    adjacency: Mapping[str :

    Iterable[str]] :


    Returns
    -------
    ``{node
        tier}`` for every node reachable from the adjacency
    ``{node
        tier}`` for every node reachable from the adjacency
        map. Sources land at tier 0; every other node sits one tier
    ``{node
        tier}`` for every node reachable from the adjacency
        map. Sources land at tier 0; every other node sits one tier
        above the deepest input.

    Raises
    ------
    ValueError
        When a cycle is detected.

    """
    cache: dict[str, int] = {}
    visiting: set[str] = set()

    def tier_of(node: str) -> int:
        """

        Parameters
        ----------
        node: str :


        Returns
        -------

        """
        if node in cache:
            return cache[node]
        if node in visiting:
            raise ValueError(f"Cycle detected at node {node!r}")
        inputs = list(adjacency.get(node, ()))
        if not inputs:
            cache[node] = 0
            return 0
        visiting.add(node)
        depth = 1 + max(tier_of(inp) for inp in inputs)
        visiting.discard(node)
        cache[node] = depth
        return depth

    for output, inputs in adjacency.items():
        tier_of(output)
        for inp in inputs:
            tier_of(inp)
    return cache


def order_within_tiers(
    adjacency: Mapping[str, Iterable[str]],
    tiers: Mapping[str, int],
    iterations: int = 8,
) -> dict[str, int]:
    """Sugiyama barycenter sweep that reduces edge crossings.

    Each tier's nodes are reordered so every node sits near the mean
    position of its neighbours in the adjacent tier. The sweep
    alternates direction; eight iterations is the standard default
    from the Sugiyama framework.

    Parameters
    ----------
    adjacency :
        same ``{output: inputs}`` map used for tier assignment.
    tiers :
        ``{node: tier}`` from :func:`assign_tiers`.
    iterations :
        number of forward-backward passes.

    Returns
    -------
    dict
        ``{node: row}`` giving each node's 0-indexed row within its tier.

    """
    predecessors: dict[str, list[str]] = defaultdict(list)
    successors: dict[str, list[str]] = defaultdict(list)
    for output, inputs in adjacency.items():
        for inp in inputs:
            predecessors[output].append(inp)
            successors[inp].append(output)

    by_tier: dict[int, list[str]] = defaultdict(list)
    for node, tier in tiers.items():
        by_tier[tier].append(node)
    for nodes in by_tier.values():
        nodes.sort()

    def position(node: str, tier: int) -> int:
        """

        Parameters
        ----------
        node: str :

        tier: int :


        Returns
        -------

        """
        return by_tier[tier].index(node)

    def neighbour_mean(node: str, tier: int, lookup: dict[str, list[str]]) -> float:
        """

        Parameters
        ----------
        node: str :

        tier: int :

        lookup: dict[str :

        list[str]] :


        Returns
        -------

        """
        ps = [position(n, tier) for n in lookup[node] if tiers.get(n) == tier]
        return sum(ps) / len(ps) if ps else _SENTINEL

    for _ in range(iterations):
        for tier in sorted(by_tier):
            if tier == 0:
                continue
            prev_tier = tier - 1
            by_tier[tier].sort(
                key=lambda node, t=prev_tier: neighbour_mean(node, t, predecessors)
            )
        for tier in sorted(by_tier, reverse=True):
            next_tier = tier + 1
            if next_tier not in by_tier:
                continue
            by_tier[tier].sort(
                key=lambda node, t=next_tier: neighbour_mean(node, t, successors)
            )

    rows: dict[str, int] = {}
    for nodes in by_tier.values():
        for index, node in enumerate(nodes):
            rows[node] = index
    return rows
