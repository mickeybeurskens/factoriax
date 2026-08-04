# Training models using Factoriax

This section holds task-based guides. Each page assumes you finished
{doc}`../start-here/index`, and starts from a specific task instead of from
zero. Pages here do not teach concepts from scratch; they link to
{doc}`../understanding/index` for the reasoning behind a design.

## In this section

- **{doc}`train_ppo_mining`**: run a full PPO training loop on `Mining-v1`
  with PureJaxRL, and read the learning curve.
- **{doc}`record_and_replay_a_rollout`**: capture a training or play session
  as a `Trajectory`, and load it back for analysis or video.
- **{doc}`build_a_custom_scenario`**: place machines and ore with the level
  editor, and register the result as a scenario with its own id.
