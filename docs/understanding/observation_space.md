# The observation space

This page is a stub. It will explain the observation profiles and when to
pick each one.

## Core elements

- Two profiles: **x_ray** sees through machines to their slot contents and
  the ore under each tile. **superficial** sees only what a player would see
  from outside.
- Two view extents: **global** flattens the whole map, in absolute
  coordinates. **local** takes a window centered on the player, and is
  egocentric and translation-invariant.
- The four top-level builders are the profile-extent pairs: `x_ray_global`,
  `x_ray_local`, `superficial_global`, `superficial_local`.
- `global_*` fits a function approximator that reads the spatial channels
  directly, such as a CNN; a flat MLP usually cannot find the player in this
  encoding without help. `local_*` fits an MLP better, because the player is
  always at the center.
- An affordability bit, one per item, says whether the player can craft that
  item right now. The bit belongs to the item, not to a recipe, so the agent
  reads what an action does before it takes the action.

## Questions this page should answer

- Which profile and extent should a new PPO run start with?
- Why does `superficial_local` outperform `superficial_global` on a flat MLP?
- What does the `rgb` observation return, and why can it not run under `jit`?
