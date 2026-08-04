# Build a custom scenario

This page is a stub. It will show how to place a level by hand and register
it as a scenario.

## Task

You want a map, a machine layout, or a reward mix that the five built-in
scenarios do not give you.

## Core elements

- The level editor (`python -m factoriax.playground`, then Editor) places
  ore, machines, and the player on a grid, and writes the result to a level
  file.
- `factoriax.engine.levels` loads a level file into the arrays that
  `EnvState` needs.
- A scenario is a factory function of the form `() -> (env, params)`.
  `factoriax.engine.envs.registry` maps a version id, such as `Mining-v1`,
  to a `ScenarioSpec` that holds this factory plus a display name and
  description.
- `factoriax.engine.rewards` and `factoriax.engine.achievements` supply the
  reward function and the achievement set a new scenario can pick from or
  extend.

## Steps to write

1. Build a small level in the editor and save it.
2. Load the level and wrap it in a scenario factory.
3. Register the scenario and confirm `factoriax.make("<id>")` returns it.
