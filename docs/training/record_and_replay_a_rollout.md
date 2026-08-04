# Record and replay a rollout

This page is a stub. It will show how to capture a training or play session
and load it back.

## Task

You have a training loop or a play session, and you want to save it, then
load it back for analysis or for a video.

## Core elements

- `factoriax.analysis.trajectory.Trajectory` holds a batch of episodes. Each
  field is an array shaped `(episode, timestep, ...)`. Only `actions` is
  required; every other field is `None` when not recorded.
- `factoriax.analysis.recorder.RolloutRecorder` collects the arrays a
  training loop already returns, with one `record()` call per iteration and
  one `finish()` call at the end. It does not change `collect_fn`, and it
  does not run under `jit`.
- `Trajectory.save` and `Trajectory.load` round-trip through a compressed
  `.npz` file.
- `factoriax.analysis.video.compose_frame_with_inventory` and `write_video`
  turn a loaded trajectory into an MP4.

## Steps to write

1. Add a `RolloutRecorder` to the PPO loop from
   {doc}`train_ppo_mining`.
2. Save the resulting `Trajectory` to disk.
3. Load it back and render one episode to video.
