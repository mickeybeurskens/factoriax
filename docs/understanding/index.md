# Understanding Factoriax

This section explains the design behind Factoriax. It gives background, not
steps. If you want to act, go to {doc}`../start-here/index` for a first
result or {doc}`../training/index` for a specific task.

## In this section

- **{doc}`state_and_step`**: why the state is a set of JAX arrays, and what
  that buys you under `jit` and `vmap`.
- **{doc}`observation_space`**: the vector and image observation profiles,
  and when to use each one.
- **{doc}`action_design`**: the three rules behind the 87-action space.
- **{doc}`scenarios_and_curriculum`**: the five built-in scenarios, the
  registry that names them, and how achievements and rewards drive a
  curriculum.
