# Spec: Machine vs Hand-Craft Efficiency Multipliers

## Status

**Deferred.** Spun out of `2026_easy_rocket_scenario.md` so that
scenario can ship without an engine touch. Pick this up when a
scenario actually needs the efficiency gradient between hand crafting
and placed combiners (the original easy-rocket motivation), or when
the broader balance-tuning surface needs per-recipe knobs that today
require forking the recipe book.

## Objective

Add two per-recipe balance knobs so a placed combiner produces more
output per cycle and finishes its cycles faster than a player
hand-crafting the same recipe. The knobs are configured through the
standard `RecipeBalance` overlay (same path scenarios already use to
tune `output_count`, `ticks`, `input_counts`) and flow into the JIT'd
engine through `EnvParams.recipe_table`.

The motivating use case: a scenario wants the agent to *prefer* using
the pre-placed machines without masking the hand-craft actions, so the
incentive is in the balance numbers rather than the action space. Two
multipliers expose enough surface to express "machines are Nx faster
and yield Mx more per cycle" cleanly, with neutral defaults so every
existing scenario reproduces today's behaviour bit-for-bit.

### Non-goals

- Changing the hand-craft action set or its dispatch (still
  instantaneous, still consumes inventory inputs, still bounded by
  `Recipe.output_count`).
- Per-machine-instance balance (e.g. "tier-2 assembler is faster than
  tier-1"). The knobs are per-recipe, applied uniformly to every
  combiner running that recipe.
- Touching `factoriax/crafting.craft_recipe`. Hand crafting reads the
  base `output_count` / `ticks` fields unchanged.

## Current state (verified on the `graph` branch)

A grep of `factoriax/recipes.py`, `factoriax/crafting.py`, and
`factoriax/machines.py` confirms:

- `Recipe` has no machine-vs-hand fields. `output_count` and `ticks`
  are a single set of balance numbers.
- `RecipeOverride` mirrors `Recipe`'s shape and likewise has none.
- `RecipeTable` projects only the unified arrays `output_counts` and
  `ticks`. Both the crafting path
  (`crafting.craft_recipe`) and the combiner path
  (`machines.run_combiners`) index into the same arrays.

So this is net-new functionality, not a refactor.

## Design

Two new fields on `Recipe`, both defaulting to neutral values that
reproduce today's behaviour exactly:

```python
@dataclass(frozen=True)
class Recipe:
    output: int
    inputs: tuple[tuple[int, int], ...]
    ticks: int
    output_count: int = 1
    name: str = ""
    machine_output_multiplier: int = 1   # NEW
    machine_speed_multiplier: int = 1    # NEW
```

A placed combiner running this recipe deposits
`output_count * machine_output_multiplier` items per completed cycle
and finishes its cycle in
`max(1, ticks // machine_speed_multiplier)` ticks. Hand crafting
ignores both fields and keeps using the base `output_count` (already
instant, so the speed multiplier doesn't apply).

`RecipeOverride` gets the same two fields as `Optional[int]`.
`RecipeBook.with_balance` propagates them with the same
negative-value validation the existing fields receive.

`RecipeTable` grows two derived arrays — `machine_output_counts` and
`machine_ticks` — computed in `RecipeTable.from_book` so the JIT path
sees them as separate PyTree leaves. The existing `output_counts` and
`ticks` arrays stay unchanged and continue to feed the hand-craft
path.

`machines.run_combiners` reads from `machine_output_counts` and
`machine_ticks` instead of the base arrays. This is the *only* engine
edit; the change is a one-line index swap per array and is gated by
the "Ask first" boundary on JIT engine touches.

## Touched files

```
factoriax/recipes.py                       # New fields on Recipe, RecipeOverride,
                                           # RecipeTable; propagation in
                                           # RecipeBook.with_balance and
                                           # RecipeTable.from_book.
factoriax/machines.py                      # run_combiners reads the new arrays.
tests/test_recipes.py                      # New unit tests (see below).
tests/test_machines.py                     # New parity tests (see below).
```

No scenario code changes in this spec. Scenarios that want to use the
multipliers do so via a `RecipeBalance` overlay; that wiring belongs
in their own specs.

## Testing strategy

All tests run as fast unit tests against pure functions and the
recipe-table projection. No env stepping.

1. **Default neutrality** — building `RecipeTable.from_book` on the
   stock `BASE_RECIPE_BOOK` produces `machine_output_counts` and
   `machine_ticks` arrays that are element-wise equal to the legacy
   `output_counts` and `ticks` arrays. Guarantees the additive change
   doesn't drift existing scenarios.
2. **Override propagation** — a `RecipeBalance` setting
   `machine_output_multiplier=3` and `machine_speed_multiplier=2` on
   one recipe produces the expected derived array values for that
   recipe (and no others), while leaving the hand-craft `output_counts`
   and `ticks` for that same recipe unchanged.
3. **Speed floor** — a recipe with `ticks=1` and
   `machine_speed_multiplier=4` projects to `machine_ticks = 1` (not
   zero). The `max(1, ...)` floor is enforced.
4. **Negative validation** — `RecipeOverride` rejects negative values
   for either multiplier at `RecipeBook.with_balance` time, with a
   named error message.
5. **Combiner parity** — a focused test on `machines.run_combiners`:
   running a recipe with neutral multipliers reproduces the existing
   per-cycle output exactly (regression guard for the index swap);
   running the same recipe with non-neutral multipliers produces the
   scaled per-cycle output with the scaled cycle length.

## Boundaries

### Always do

- Keep `crafting.craft_recipe` reading the base `output_count` field.
  Hand crafting is not affected by these knobs.
- Default both multipliers to 1 so every existing book + table +
  scenario reproduces today's behaviour without code changes.
- Validate negative multipliers at book-construction time, not at
  runtime in the JIT path.

### Ask first

- The single edit to `machines.run_combiners`. This is the spec's only
  JIT engine touch. Flag at plan-review time and do not land without
  human approval.

### Never do

- Introduce machine-instance-level balance. Per-recipe multipliers
  are the agreed scope.
- Bake multiplier values into Python globals. They flow through
  `EnvParams.recipe_table` like every other balance number.

## Open questions

- Should `machine_speed_multiplier` be a float (e.g. 1.5x) rather than
  an int? Today's `ticks` is an int and the JIT path floor-divides;
  keeping the multiplier integer preserves the dtype story but limits
  granularity. Defer until a real scenario hits the limit.
