"""Place the nodes of an acyclic graph on a layered grid.

Both functions take the same map, ``{node: inputs}``, and give each
node one number. Together those two numbers are a grid position.

* :func:`assign_tiers` gives each node a tier, which becomes the
  column. A node with no inputs sits at tier 0. Every other node sits
  one tier after its deepest input.
* :func:`order_within_tiers` gives each node a row inside its tier,
  which becomes the slot in that column. It runs a Sugiyama barycenter
  sweep, which reduces the number of edges that cross.

Neither function knows about recipes. The recipe diagram is the only
caller today, but any layered drawing can use them.

A tier is a longest-path layer, and not a shortest-path one. An item
reachable by a two-step chain and by a one-step chain sits at tier 2.
With shortest paths it would sit at tier 1, level with a node that
feeds it, and that edge would then have to run backwards.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

#: Stand-in mean row for a node with no neighbor in the tier beside
#: it. Larger than any real row, so such a node sorts to the end.
_SENTINEL: float = 1e9


def assign_tiers(adjacency: Mapping[str, Iterable[str]]) -> dict[str, int]:
    """Give every node a tier, by the longest path from a source.

    Parameters
    ----------
    adjacency :
        ``{node: inputs}``. A node with no inputs of its own does not
        need a key of its own. The walk finds such a node in the input
        lists and gives it tier 0.

    Returns
    -------
    dict
        ``{node: tier}`` for every node named anywhere in the map,
        whether as a key or only as an input. A source sits at tier 0.
        Every other node sits one tier after its deepest input. An
        empty map gives an empty result.

    Raises
    ------
    ValueError
        When the map holds a cycle. The message names one node on that
        cycle. A node that lists itself as an input counts as a cycle.

    Notes
    -----
    The walk recurses once for each step of a chain, so a chain longer
    than the Python recursion limit raises ``RecursionError`` and not
    ``ValueError``.
    """
    cache: dict[str, int] = {}
    visiting: set[str] = set()

    def tier_of(node: str) -> int:
        """Return the tier of one node, and cache it.

        The set ``visiting`` holds the nodes on the current chain. A
        node that appears twice on one chain is a cycle.

        Parameters
        ----------
        node :
            The node to place.

        Returns
        -------
        int
            The tier of ``node``, counted from 0 at a source.
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
    """Give every node a row inside its tier, to reduce crossings.

    Each pass sorts the nodes of a tier by the mean row of their
    neighbors in the tier beside it. A node with no neighbor in that
    tier sorts to the end. The sweep runs forward and then backward,
    because a node has neighbors on both sides.

    This is a heuristic. It has no optimal answer to reach, and more
    passes do not always give fewer crossings.

    Parameters
    ----------
    adjacency :
        The same ``{node: inputs}`` map given to :func:`assign_tiers`.
    tiers :
        ``{node: tier}`` from :func:`assign_tiers`. Every node in
        ``adjacency`` must have an entry.
    iterations :
        How many forward and backward pairs of passes to run. Eight is
        the usual default of the Sugiyama method.

    Returns
    -------
    dict
        ``{node: row}``. Rows start at 0 in each tier and count up
        with no gaps. Position code multiplies the row by a fixed
        step, so a gap leaves an empty band in the figure.

    Notes
    -----
    The result depends only on the arguments. Each pass starts from a
    sorted node list, and not from dictionary order. Two runs in
    different processes therefore give the same rows, and a committed
    figure does not change between rebuilds.
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
        """Return the current row of ``node`` inside ``tier``.

        Parameters
        ----------
        node :
            The node to find.
        tier :
            The tier that holds it.

        Returns
        -------
        int
            The 0-based row.
        """
        # A linear scan of the tier. Tiers here hold tens of nodes, so
        # this stays cheaper than keeping a second index in step.
        return by_tier[tier].index(node)

    def neighbour_mean(node: str, tier: int, lookup: dict[str, list[str]]) -> float:
        """Return the mean row of the neighbors of ``node`` in ``tier``.

        Parameters
        ----------
        node :
            The node whose neighbors to average.
        tier :
            The tier to look in. Neighbors in any other tier are
            skipped.
        lookup :
            Either the predecessor map or the successor map, which
            selects the direction of the sweep.

        Returns
        -------
        float
            The mean row, or :data:`_SENTINEL` when ``node`` has no
            neighbor in ``tier``. The sentinel is large, so such a node
            sorts to the end of its tier.
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
