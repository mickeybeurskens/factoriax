# Scenarios and curriculum

This page is a stub. It will explain the scenario registry and how
achievements and rewards build a curriculum.

## Core elements

- Five scenarios carry an id: `Mining-v1`, `MinerBootstrap-v1`,
  `ScienceTiers-v1`, `EasyRocket-v1`, and `Rocket-v1`.
- A scenario is a factory function, `() -> (env, params)`.
  `factoriax.engine.envs.registry` maps each version id to a `ScenarioSpec`
  that holds the factory, a display name, and a description.
  `factoriax.make("<id>")` resolves the id and applies the observation
  settings and the auto-reset wrapper the caller asked for.
  `list_scenarios()` returns every registered spec.
- An achievement is a bit that records that the world reached a state during
  an episode, such as [Craftax]()-style unlocks. The environment state holds the
  bit after it unlocks. A scenario can read the bits to track progress, or
  build a curriculum from them, independent of what the reward function
  pays. The bit order is a wire format: a recorded trajectory stores
  `achievements_unlocked` as a plain bool vector, addressed by position, so
  a set can only grow at the end.
- A reward function reads the state before and after one step and returns a
  scalar. A **sparse** reward pays only on a target event, such as an ore
  mined; it is honest and hard to learn from. A **dense** reward adds
  shaping terms that pay for progress toward a goal; it is easier to learn
  from and easier to get wrong, because the agent optimizes the terms it is
  paid for, not the goal behind them.

## Questions this page should answer

- What separates `Mining-v1` from `MinerBootstrap-v1` as a training target?
- When should a new scenario reuse an existing reward function instead of
  writing a dense one?
- What breaks if an achievement bit is removed or reordered after a
  trajectory has been recorded?
