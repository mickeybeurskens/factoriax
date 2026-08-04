# State, step, and compilation

This page is a stub. It will explain the state representation and why it
compiles to one XLA graph.

## Core elements

- `EnvState` and `EnvParams` are `flax.struct.PyTreeNode` classes. Every
  field is a JAX array, so both pass through `jax.jit` and `jax.vmap` with no
  extra registration.
- The engine holds machine state as entity lists, not as one grid per
  attribute. An entity id indexes every `ent_` array. The capacity is fixed
  when a level is built, so array shapes stay static during an episode,
  whatever the player builds.
- `reset_env` and `step_env` are pure functions: they take state in and
  return new state, with no hidden mutation and no global variables. This is
  the property that lets `jax.lax.scan` compile a full rollout and
  `jax.vmap` run a batch of environments in one compiled kernel, with no
  Python in the inner loop.

## Questions this page should answer

- Why an entity list instead of a per-attribute grid?
- What breaks if a field is not a JAX array?
- Why does a fixed entity capacity matter for `jit`?
