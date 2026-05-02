"""Run scripted rocket agents and compare their episode timings.

All agents share the same rocket benchmark level (pre-placed
furnace + assembler, five ore patches). The difference is the plan:

- ``naive``   — :mod:`baselines.rocket.scripted.agent` runs every
  recipe serially through the pre-placed machines.
- ``factory`` — :mod:`baselines.rocket.scripted.agent_factory`
  spends a starter phase to place two extra furnaces and two extra
  assemblers, then uses :class:`PipelinedProduce` to rotate bulk
  work across the 3-machine batteries.
- ``advanced_factory`` —
  :mod:`baselines.rocket.scripted.agent_advanced_factory` builds a
  full :class:`BuildSmelterCell` on every non-coal patch so each
  patch auto-mines + auto-smelts its own plates locally; the agent
  then ferries coal in and uses :class:`CraftFromBus` to chain the
  bus-pallet plates into rocket-chain intermediates.

All agents are deterministic, so the comparison is reproducible
from a single seed.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from baselines.rocket.scripted.agent import make_scripted_rocket_agent
from baselines.rocket.scripted.agent_advanced_factory import (
    make_advanced_factory_rocket_agent,
)
from baselines.rocket.scripted.agent_factory import make_factory_rocket_agent
from factoriax.analysis.eval import EvalRollout, generate_eval_plots
from factoriax.analysis.video import (
    compose_frame_with_inventory,
    write_video_streaming,
)
from factoriax.benchmarks.rocket import (
    NUM_ROCKET_ACHIEVEMENTS,
    ROCKET_ACHIEVEMENT_INFO,
    ROCKET_ACHIEVEMENT_WEIGHTS,
    ROCKET_BLOCKED_ACTIONS,
    ROCKET_RECIPE_BOOK,
    ROCKET_RECIPE_TABLE,
    build_rocket_level,
    rocket_conditions,
)
from factoriax.constants import MAX_ACHIEVEMENTS, Action
from factoriax.envs import FactoriaXEnv
from factoriax.envs.achievement_wrapper import AchievementState, AchievementWrapper
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.levels import build_state
from factoriax.observations import global_array
from factoriax.state import EnvParams

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Per-agent episode summary.

    ``rollout`` is populated when the caller asks for offline
    analysis artifacts (video / plots); it carries the per-step
    ``EnvState`` snapshots needed to render the map+inventory video
    and compute the trajectory-level plots.
    """

    label: str
    steps: int
    wall_sec: float
    score: float
    unlocked: int
    unlock_step: np.ndarray  # shape (NUM_ROCKET_ACHIEVEMENTS,), int32
    rollout: EvalRollout | None = None


def _run(
    label: str,
    agent_factory,
    max_steps: int,
    seed: int,
    wandb_run: Any | None = None,
    log_every: int = 10,
    collect_rollout: bool = False,
) -> RunResult:
    """Drive one agent through the rocket benchmark env.

    Args:
        label: Agent label used for log lines and wandb tags.
        agent_factory: Callable ``(env_params) -> ScriptedAgent``.
        max_steps: Episode budget.
        seed: JAX PRNG seed.
        wandb_run: Optional active ``wandb.Run`` to log per-step metrics
            into. When ``None``, the run is silent. Achievement-unlock
            ticks are always logged when they happen, regardless of
            ``log_every``.
        log_every: Sample period for the rolling per-step metrics
            (cumulative score, achievement count). Defaults to every
            10 ticks to keep the wandb queue light over an 8k–16k
            step run.
        collect_rollout: When true, snapshot the inner ``EnvState`` and
            achievement mask at every tick so the caller can render a
            map+inventory video and the per-trajectory analysis plots.
            ~20 KB per state at 32x32, so a full episode is on the
            order of 100 MB — small compared to the rendered frames
            themselves.
    """
    env_params = EnvParams(
        map_width=32,
        map_height=32,
        num_players=1,
        max_timesteps=max_steps,
        recipe_table=ROCKET_RECIPE_TABLE,
    )
    level = build_rocket_level()
    env_state = build_state(level, env_params)
    state = AchievementState(
        env_state=env_state,
        achievements_unlocked=jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_),
    )
    env = ActionMaskWrapper(
        AchievementWrapper(FactoriaXEnv(), rocket_conditions),
        ROCKET_BLOCKED_ACTIONS,
    )
    jit_step = jax.jit(env.step_env)
    # JIT the obs too — without this, every tick pays ~50ms for the
    # 10-channel scatter-based global_array.
    jit_obs = jax.jit(lambda s: global_array(s, env_params, 0))
    agent = agent_factory(env_params)

    unlock_step = np.full((NUM_ROCKET_ACHIEVEMENTS,), -1, dtype=np.int32)
    weights = np.asarray(ROCKET_ACHIEVEMENT_WEIGHTS)[:NUM_ROCKET_ACHIEVEMENTS]
    key = jax.random.PRNGKey(seed)
    t0 = time.perf_counter()

    env_states_log: list[Any] = []
    actions_log: list[int] = []
    ach_log: list[np.ndarray] = []
    if collect_rollout:
        env_states_log.append(state.env_state)
        ach_log.append(np.asarray(state.achievements_unlocked))

    for t in range(max_steps):
        obs = np.asarray(jit_obs(state.env_state))
        action = agent.act(obs)
        key, subkey = jax.random.split(key)
        _, state, _, done, _ = jit_step(
            subkey,
            state,
            jnp.int32(action),
            env_params,
        )
        mask = np.asarray(state.achievements_unlocked)[:NUM_ROCKET_ACHIEVEMENTS]
        newly = (unlock_step < 0) & mask
        unlock_step[newly] = t
        any_unlock = bool(newly.any())

        if wandb_run is not None and (any_unlock or t % log_every == 0):
            step_reward = float(np.sum(weights * newly))
            cum_score = float(np.sum(weights * mask))
            wandb_run.log(
                {
                    "scripted/step_reward": step_reward,
                    "scripted/cum_score": cum_score,
                    "scripted/cum_achievements": int(mask.sum()),
                    "scripted/action": int(action),
                },
                step=t,
            )

        if collect_rollout:
            env_states_log.append(state.env_state)
            actions_log.append(int(action))
            ach_log.append(np.asarray(state.achievements_unlocked))

        if agent.is_done or bool(done):
            break

    wall = time.perf_counter() - t0
    final_mask = np.asarray(state.achievements_unlocked)[:NUM_ROCKET_ACHIEVEMENTS]
    score = float(np.sum(weights * final_mask))

    # Surface verify diagnostics at end of run. ScriptedAgent owns its
    # planner; pre-existing agents that don't expose ``planner``
    # silently skip this block.
    diag = getattr(getattr(agent, "planner", None), "verify_diagnostic", None)
    if diag is not None:
        print(f"[{label}] {diag.format()}")

    rollout: EvalRollout | None = None
    if collect_rollout:
        rollout = EvalRollout(
            frames=None,
            actions=np.asarray(actions_log, dtype=np.int32),
            env_states=env_states_log,
            ach_per_step=np.stack(ach_log, axis=0),
        )

    return RunResult(
        label=label,
        steps=t + 1,
        wall_sec=wall,
        score=score,
        unlocked=int(final_mask.sum()),
        unlock_step=unlock_step,
        rollout=rollout,
    )


def _print_summary(results: list[RunResult]) -> None:
    """Print a side-by-side comparison of the runs."""
    print()
    print("=" * 72)
    header = f"{'':<24}" + "".join(f"{r.label:>16}" for r in results)
    print(header)
    print("-" * 72)
    rows = [
        ("Total ticks", [str(r.steps) for r in results]),
        ("Achievements", [f"{r.unlocked}/{NUM_ROCKET_ACHIEVEMENTS}" for r in results]),
        ("Score", [f"{r.score:.0f}" for r in results]),
        ("Wall clock", [f"{r.wall_sec:.1f}s" for r in results]),
    ]
    for name, vals in rows:
        print(f"{name:<24}" + "".join(f"{v:>16}" for v in vals))
    print("=" * 72)

    # Achievement-by-achievement unlock timing.
    print()
    print("Unlock ticks (earlier is better). ``-`` = not unlocked.")
    print(f"{'Achievement':<24}" + "".join(f"{r.label:>16}" for r in results))
    print("-" * 72)
    for i, info in enumerate(ROCKET_ACHIEVEMENT_INFO):
        cells = []
        for r in results:
            t = int(r.unlock_step[i])
            cells.append(f"{t:>16}" if t >= 0 else f"{'-':>16}")
        print(f"{info.id:<24}" + "".join(cells))


def _finalize_artifacts(
    label: str,
    result: RunResult,
    out_dir: Path,
    video_fps: int,
    block_pixel_size: int,
    wandb_run: Any,
) -> None:
    """Render the rollout video, generate plots, and upload to wandb.

    Streams frames straight to MP4 via
    :func:`factoriax.analysis.video.write_video_streaming` so peak RAM
    stays bounded to a single rendered frame even on 6000-step
    episodes. Plots are produced by
    :func:`factoriax.analysis.eval.generate_eval_plots` so the layout
    matches what the rocket PPO eval pipeline produces.
    """
    import wandb  # noqa: PLC0415  # type: ignore[import-untyped]

    rollout = result.rollout
    if rollout is None:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = out_dir / f"scripted_{label}_rollout.mp4"
    written = 0
    try:
        written = write_video_streaming(
            video_path,
            (
                compose_frame_with_inventory(s, block_pixel_size=block_pixel_size)
                for s in rollout.env_states
            ),
            fps=video_fps,
        )
        logger.info(
            "Saved rollout video: %s (%d frames)",
            video_path,
            written,
        )
    except ImportError:
        logger.error("imageio[ffmpeg] missing; rollout video skipped.")
        video_path = None  # type: ignore[assignment]

    plot_paths = generate_eval_plots(
        rollout,
        out_dir,
        achievement_labels=[info.id for info in ROCKET_ACHIEVEMENT_INFO],
        num_achievements=NUM_ROCKET_ACHIEVEMENTS,
        title_prefix=f"Scripted {label}",
    )
    for name, path in plot_paths.items():
        logger.info("Saved %s plot: %s", name, path)

    run_id = getattr(wandb_run, "id", "run")

    if video_path is not None:
        vid_artifact = wandb.Artifact(
            f"scripted-{label}-video-{run_id}",
            type="video",
        )
        vid_artifact.add_file(str(video_path))
        wandb_run.log_artifact(vid_artifact)
        wandb_run.log(
            {
                "final/rollout": wandb.Video(
                    str(video_path),
                    fps=video_fps,
                    format="mp4",
                ),
            },
        )

    if plot_paths:
        plots_artifact = wandb.Artifact(
            f"scripted-{label}-plots-{run_id}",
            type="analysis",
        )
        inline: dict[str, Any] = {}
        for name, path in plot_paths.items():
            plots_artifact.add_file(str(path))
            inline[f"final/plots/{name}"] = wandb.Image(str(path))
        wandb_run.log_artifact(plots_artifact)
        wandb_run.log(inline)


def _run_with_wandb(
    label: str,
    agent_factory,
    max_steps: int,
    seed: int,
    wandb_project: str,
    wandb_run_name: str | None,
    log_every: int,
    out_dir_root: Path,
    video_fps: int,
    block_pixel_size: int,
    save_video: bool,
) -> RunResult:
    """Run a scripted agent inside a fresh wandb run.

    One wandb run per agent so they show up as comparable rows in
    the project view. Uploads:

    - per-step ``scripted/*`` curves (logged inside :func:`_run`),
    - a ``summary/unlocks_table`` with per-achievement unlock ticks,
    - the map+inventory rollout video as an artifact + inline
      ``final/rollout``,
    - the inventory-over-time, action-counts, and achievement-timing
      PNGs as an artifact + inline ``final/plots/*`` images.
    """
    import wandb  # noqa: PLC0415  # type: ignore[import-untyped]

    run = wandb.init(
        project=wandb_project,
        name=wandb_run_name or f"scripted_{label}",
        reinit=True,
        config={
            "agent": label,
            "max_steps": max_steps,
            "seed": seed,
            "level": "rocket",
            "video_fps": video_fps,
            "block_pixel_size": block_pixel_size,
        },
        tags=["rocket", "scripted", label],
    )
    try:
        result = _run(
            label,
            agent_factory,
            max_steps,
            seed,
            wandb_run=run,
            log_every=log_every,
            collect_rollout=save_video,
        )

        unlock_table = wandb.Table(
            columns=["index", "achievement", "weight", "unlock_step"],
        )
        for i, info in enumerate(ROCKET_ACHIEVEMENT_INFO):
            t = int(result.unlock_step[i])
            unlock_table.add_data(
                i,
                info.id,
                int(ROCKET_ACHIEVEMENT_WEIGHTS[i]),
                t if t >= 0 else -1,
            )
        run.log(
            {
                "summary/total_ticks": result.steps,
                "summary/wall_sec": result.wall_sec,
                "summary/score": result.score,
                "summary/unlocked": result.unlocked,
                "summary/max_achievements": NUM_ROCKET_ACHIEVEMENTS,
                "summary/unlocks_table": unlock_table,
            },
        )
        run.summary["total_ticks"] = result.steps
        run.summary["wall_sec"] = result.wall_sec
        run.summary["score"] = result.score
        run.summary["unlocked"] = result.unlocked

        if save_video and result.rollout is not None:
            _finalize_artifacts(
                label,
                result,
                out_dir_root / f"scripted_{label}",
                video_fps=video_fps,
                block_pixel_size=block_pixel_size,
                wandb_run=run,
            )
    finally:
        run.finish()
    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare the naive and factory rocket agents.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=16000,
        help="Episode budget (default matches the benchmark).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Log per-step metrics and the per-achievement unlock "
        "table to Weights & Biases (one run per agent).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="factoriax_rocket",
        help="W&B project name (matches the rocket PPO default).",
    )
    parser.add_argument(
        "--wandb-run-prefix",
        type=str,
        default=None,
        help="Optional prefix for the wandb run names. The agent label "
        "is appended (e.g. prefix=foo gives foo_naive / foo_factory).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Sample period for rolling per-step metrics. "
        "Achievement unlocks are always logged.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs") / "scripted_rocket",
        help="Where to write the rollout video and analysis PNGs "
        "before they are uploaded to wandb.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=30,
        help="Frame rate of the saved MP4 (matches PPO default).",
    )
    parser.add_argument(
        "--block-pixel-size",
        type=int,
        default=16,
        help="Tile size in the rendered map frame.",
    )
    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Skip rendering the rollout video and analysis plots. "
        "Per-step metrics are still logged when --use-wandb is set.",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=["naive", "factory", "advanced_factory"],
        default=None,
        help="Subset of agents to run. Defaults to all three. "
        "Example: --agents advanced_factory.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    _ = Action  # symbol check
    save_video = not args.no_save_video

    def _go(label: str, factory) -> RunResult:
        if not args.use_wandb:
            return _run(
                label,
                factory,
                args.max_steps,
                args.seed,
                collect_rollout=save_video,
            )
        run_name = (
            f"{args.wandb_run_prefix}_{label}"
            if args.wandb_run_prefix
            else f"scripted_{label}"
        )
        return _run_with_wandb(
            label,
            factory,
            args.max_steps,
            args.seed,
            wandb_project=args.wandb_project,
            wandb_run_name=run_name,
            log_every=args.log_every,
            out_dir_root=args.out_dir,
            video_fps=args.video_fps,
            block_pixel_size=args.block_pixel_size,
            save_video=save_video,
        )

    # advanced_factory's BOM math must use the same recipe book the
    # engine uses, otherwise its mine / smelt / craft quantities will
    # mis-size against the rebalanced engine.
    def _advanced_factory(env_params: EnvParams) -> Any:
        return make_advanced_factory_rocket_agent(env_params, book=ROCKET_RECIPE_BOOK)

    all_agents: dict[str, Any] = {
        "naive": make_scripted_rocket_agent,
        "factory": make_factory_rocket_agent,
        "advanced_factory": _advanced_factory,
    }
    selected = args.agents or list(all_agents.keys())

    results: list[RunResult] = []
    for label in selected:
        print(f"Running {label.upper()} agent...")
        result = _go(label, all_agents[label])
        print(
            f"  -> {result.steps} ticks, "
            f"{result.unlocked}/{NUM_ROCKET_ACHIEVEMENTS} unlocked.",
        )
        results.append(result)

    _print_summary(results)


if __name__ == "__main__":
    main()
