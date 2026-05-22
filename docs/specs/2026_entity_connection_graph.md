# Spec: Entity Connection Graph

## Status

**Drafted, not yet planned.** Captures an idea raised during
`2026_easy_rocket_scenario.md` review: a first-class graph over the
placed entities would unlock achievements, agent observations,
scripted-agent queries, and renderer overlays that today require
hand-rolled adjacency walks. Picked up when one of those consumers
becomes a blocker — most immediately, the easy-rocket scenario's
belt-network achievements (#8, #10, #11, #12 in
`2026_easy_rocket_scenario.md`).

## Objective

Expose the current factory layout as a directed graph in `EnvState`,
computed once per step from the existing entity arrays, so that
condition functions, observation functions, and scripted agents can
ask graph-shaped questions ("is assembler C's input fed from
assembler A's output?", "how many machines downstream of this miner
have buffered items?") without each consumer re-implementing
belt-walk logic against the raw entity arrays.

Today every belt-aware computation in the codebase
(`belts.py`, the assembler input-routing in `machines.py`, the
rocket-scenario condition helpers) operates against raw `ent_x` /
`ent_y` / `ent_type` arrays and reasons about adjacency one tile at a
time. Two costs follow from that: (1) every new consumer pays the
same cost again, and (2) the implementations drift apart in subtle
ways (which direction is "input", which is "output", what counts as
"connected"). A single graph projection paid once per step removes
both costs.

### Non-goals

- Replacing `belts.py`'s tick-level item-transport simulation. The
  graph is a topology projection, not a physics engine. Belt tick
  logic continues to live where it is.
- Mutating the entity arrays. The graph is a derived view computed
  from `state` and discarded; entity state stays canonical.
- General-purpose JAX graph algorithms. We need a small fixed set
  of queries (adjacency, k-hop reachability, in/out neighbours);
  anything more is out of scope.

## Concept

Nodes are placed entities that produce, consume, or route items:

- `MINER` — source node, output direction determined by facing.
- `FURNACE`, `ASSEMBLER` — transformer node with up to two input
  ports and one output port.
- `SPLITTER` — router node with one input port and two output ports.
- `CROSSING` — router node carrying two perpendicular item streams
  past each other.
- `PALLET`, `ARM` — buffering / lifting nodes (input + output
  conceptually but degenerate ports).
- `ROCKET` — sink node.

Edges are `CONVEYOR_BELT` entities, oriented by belt facing. An edge
goes from the tile feeding into the belt's tail to the tile the
belt's head feeds into. A chain of belts in series collapses into a
single logical edge between the producing node and the consuming
node, with the belt-count and aggregate `buf_count` retained as edge
attributes (so an "is the chain carrying any items right now?" query
is constant-time).

The graph is recomputed each step in a JIT-pure pass over the entity
arrays. Shapes are bounded by the entity-array capacity, so XLA
caches stay warm across steps.

## Touched files

```
factoriax/graph.py                            # New module. Graph
                                              # projection from EnvState,
                                              # plus the small fixed set
                                              # of query primitives.
factoriax/state.py                            # EnvState gains a graph
                                              # PyTree leaf (or the
                                              # graph is computed
                                              # on-demand — open
                                              # question).
tests/test_graph.py                           # New file. Unit tests for
                                              # the projection and each
                                              # query primitive.
```

No consumer-side wiring in this spec. Consumers (easy-rocket
achievements, scripted agents, renderer overlay) take dependencies
on `factoriax.graph` in their own specs.

## Query primitives the graph must support

The minimum surface that unblocks the easy-rocket achievements:

- `is_belt_feeding(state, src_ent_idx, dst_ent_idx) -> bool` — true
  when there is a belt chain from `src` to `dst` and at least one
  item is somewhere on that chain (any belt with `buf_count > 0`).
- `out_neighbours(state, ent_idx) -> set[int]` — set of entity
  indices reachable from `ent_idx` along one logical edge (i.e. one
  belt chain).
- `distinct_upstream_machines(state, ent_idx, machine_type) -> int`
  — count of distinct entities of a given machine type that feed
  `ent_idx` along one logical edge each.

The plan phase will likely add more, but these three are the
load-bearing ones for the easy-rocket spec.

## Open questions

- **EnvState leaf vs on-demand projection.** Caching the graph as an
  `EnvState` field grows the state PyTree but avoids recomputation
  across queries within a step. Recomputing on demand keeps state
  lean but costs more when multiple consumers ask graph questions.
  Pick once we know how many consumers there are.
- **Edge granularity.** Whether one logical edge per belt chain (as
  proposed) is the right abstraction, or whether routers
  (`SPLITTER`, `CROSSING`) should always be node-boundaries, even
  mid-chain. Splitter routing is part-stateful (the splitter
  alternates outputs), which complicates the "this edge carries
  items from A to B" semantics. Default: routers are always
  node-boundaries.
- **Capacity bounds.** Need to confirm the entity-array capacity
  used by the engine is generous enough that the graph projection
  doesn't need its own resizing logic.
