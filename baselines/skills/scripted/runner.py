"""Run scripted baselines through the skills curriculum and log to wandb.

Drives each scripted policy through its corresponding
:class:`SkillsBenchmark` level, renders an mp4 of the rollout via
:class:`~factoriax.engine.jax_renderer.JaxRenderer`, generates action +
inventory analysis plots, and uploads everything to a single wandb
run per level. Mirrors the eval-pass pattern in
``baselines/skills/train_ppo.py``.

Wandb conventions (from the spec/plan):

- Project: ``factoriax_skills_benchmark``
- Tags: ``skills``, ``eval``, ``<level_name>``, ``<git_sha_short>``,
  ``scripted``

Usage::

    uv run python -m baselines.skills.scripted.runner --level all
    uv run python -m baselines.skills.scripted.runner --level navigate
    uv run python -m baselines.skills.scripted.runner --level all \
        --no-wandb          # local-only, video + plots saved to runs/

Run when the GPU is free (no training in progress).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from baselines.skills.scripted import SCRIPTED_POLICIES, ScriptedPolicy
from factoriax.analysis.video import compose_frame_with_inventory, write_video
from factoriax.engine.levels import build_state
from factoriax.engine.state import EnvState
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.scenarios.skills import SkillsBenchmark, skills_conditions

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("scripted_runner")

WANDB_PROJECT = "factoriax_skills_benchmark"


@dataclass
class RolloutOutcome:
    """Outcome of running one scripted policy through one level."""

    level_name: str
    solved: bool
    timesteps_used: int
    score: float
    states: list[EnvState]
    actions: list[int]


def _git_sha_short() -> str:
    """Best-effort git SHA short. Falls back to 'nogit' if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def run_level(
    level_idx: int,
    policy: ScriptedPolicy,
    seed: int = 0,
) -> RolloutOutcome:
    """Drive *policy* through ``SkillsBenchmark.levels()[level_idx]``.

    Captures the full trajectory of states + actions for downstream
    rendering. Uses :class:`ActionMaskWrapper` with the level's mask
    so the rollout matches what the runner would do in evaluation.

    Args:
        level_idx: Index into ``SkillsBenchmark().levels()``.
        policy: Scripted policy implementing ``ScriptedPolicy``.
        seed: PRNG seed for ``step_env``.

    Returns:
        :class:`RolloutOutcome` with score, solve flag, and recorded
        states/actions for video rendering.
    """
    scenario = SkillsBenchmark()
    scenario_level = scenario.levels()[level_idx]
    params = scenario_level.env_params

    inner = FactoriaXEnv(achievement_fn=skills_conditions)
    blocked = scenario_level.blocked_actions or frozenset()
    env = ActionMaskWrapper(inner, tuple(blocked)) if blocked else inner
    jit_step = jax.jit(env.step_env)

    state: EnvState = build_state(scenario_level.level, params)
    rng = jax.random.PRNGKey(seed)

    states: list[EnvState] = [state]
    actions: list[int] = []
    solved = False
    timesteps = 0

    for t in range(params.max_timesteps):
        action = policy(state, params)
        rng, subkey = jax.random.split(rng)
        _o, state, _r, _d, _i = jit_step(subkey, state, action, params)
        actions.append(int(action))
        states.append(state)
        timesteps = t + 1
        if bool(jnp.asarray(state.achievements_unlocked)[level_idx]):
            solved = True
            break

    max_t = params.max_timesteps
    score = (max_t - timesteps + 1) / max_t if solved else 0.0
    return RolloutOutcome(
        level_name=scenario_level.name,
        solved=solved,
        timesteps_used=timesteps,
        score=score,
        states=states,
        actions=actions,
    )


def render_video(
    outcome: RolloutOutcome,
    out_dir: Path,
    fps: int = 2,
    tile_px: int = 32,
) -> Path | None:
    """Write an mp4 of the rollout with an inventory side-panel.

    Reuses :func:`compose_frame_with_inventory` so the map+inventory
    composition matches what the agent debugger and PPO eval pipeline
    produce. Returns the path on success, ``None`` if imageio is
    unavailable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [
        compose_frame_with_inventory(s, block_pixel_size=tile_px)
        for s in outcome.states
    ]
    mp4_path = out_dir / f"{outcome.level_name}.mp4"
    try:
        write_video(mp4_path, frames, fps=fps)
    except ImportError:
        logger.error("imageio[ffmpeg] not available; skipping video.")
        return None
    logger.info("Saved video: %s (%d frames)", mp4_path, len(outcome.states))
    return mp4_path


def render_plots(
    outcome: RolloutOutcome,
    out_dir: Path,
) -> dict[str, Path]:
    """Render an inline trajectory-trace plot for *outcome*.

    The factoriax analysis module's plot helpers
    (:func:`plot_trajectory_trace`, :func:`plot_inventory`) have
    pre-existing shape bugs with single-player trajectories built
    via :func:`states_to_trajectory` — they expect 3D arrays but get
    4D. Rather than work around them, this function builds a simple
    matplotlib trajectory plot directly from
    ``state.player_positions``. Inventory composition is visible in
    the video frames already.

    Returns a dict mapping plot name to saved path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: dict[str, Path] = {}
    figs: dict[str, Any] = {}

    map_w = int(outcome.states[0].map.shape[1])
    map_h = int(outcome.states[0].map.shape[0])

    positions = np.asarray(
        [np.asarray(s.player_positions[0]) for s in outcome.states]
    )  # (T, 2)
    fig_tr, ax_tr = plt.subplots(figsize=(6, 6))
    n = positions.shape[0]
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, n))
    ax_tr.plot(
        positions[:, 0],
        positions[:, 1],
        color="gray",
        alpha=0.3,
        linewidth=1,
        zorder=1,
    )
    ax_tr.scatter(positions[:, 0], positions[:, 1], c=colors, s=40, zorder=2)
    ax_tr.scatter(
        positions[0, 0],
        positions[0, 1],
        marker="o",
        s=180,
        facecolors="none",
        edgecolors="green",
        linewidths=2,
        label="start",
    )
    ax_tr.scatter(
        positions[-1, 0],
        positions[-1, 1],
        marker="*",
        s=240,
        c="red",
        label="end",
    )
    ax_tr.set_xlim(-0.5, map_w - 0.5)
    ax_tr.set_ylim(map_h - 0.5, -0.5)  # invert y for screen coords
    ax_tr.set_aspect("equal")
    ax_tr.grid(True, alpha=0.3)
    ax_tr.set_title(f"{outcome.level_name} -- trajectory (scripted)")
    ax_tr.legend(loc="upper left")
    figs["trajectory"] = fig_tr

    for plot_name, fig in figs.items():
        path = out_dir / f"{outcome.level_name}_{plot_name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths[plot_name] = path

    return paths


def upload_to_wandb(
    outcome: RolloutOutcome,
    mp4_path: Path | None,
    plot_paths: dict[str, Path],
    sha_short: str,
    seed: int,
    project: str,
) -> Any:
    """Initialise a wandb run, upload artifacts, and return the run.

    Tags follow the spec/plan conventions:
    ``skills``, ``eval``, ``<level_name>``, ``<sha>``, ``scripted``.

    Returns:
        The live wandb run, or ``None`` if wandb isn't installed.
    """
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed; skipping upload.")
        return None

    tags = ["skills", "eval", outcome.level_name, sha_short, "scripted"]
    run = wandb.init(
        project=project,
        name=f"scripted_{outcome.level_name}_{sha_short}_{seed}",
        config={
            "level": outcome.level_name,
            "seed": seed,
            "git_sha": sha_short,
            "policy": "scripted",
        },
        tags=tags,
        reinit=True,
    )

    log_data: dict[str, object] = {
        "eval/score": outcome.score,
        "eval/solved": int(outcome.solved),
        "eval/timesteps_used": outcome.timesteps_used,
    }
    if mp4_path is not None:
        log_data[f"videos/{outcome.level_name}"] = wandb.Video(
            str(mp4_path), format="mp4"
        )
    for plot_name, path in plot_paths.items():
        log_data[f"plots/{plot_name}"] = wandb.Image(str(path))

    run.log(log_data)
    logger.info("Uploaded to wandb: %s", run.url)
    return run


def run_one(
    level_name: str,
    seed: int,
    sha_short: str,
    out_root: Path,
    use_wandb: bool,
    project: str,
    fps: int = 2,
) -> RolloutOutcome:
    """Run + render + (optionally) upload one level. Returns the outcome."""
    scenario = SkillsBenchmark()
    level_names = [bl.name for bl in scenario.levels()]
    if level_name not in level_names:
        raise ValueError(f"Unknown level {level_name!r}. Available: {level_names}")
    level_idx = level_names.index(level_name)

    if level_name not in SCRIPTED_POLICIES:
        raise ValueError(
            f"No scripted policy registered for {level_name!r}. "
            f"Available: {sorted(SCRIPTED_POLICIES)}"
        )
    policy = SCRIPTED_POLICIES[level_name]

    logger.info("Running %s (seed=%d)...", level_name, seed)
    t0 = time.perf_counter()
    outcome = run_level(level_idx, policy, seed=seed)
    elapsed = time.perf_counter() - t0
    logger.info(
        "  %s: solved=%s  ticks=%d  score=%.3f  rollout=%.2fs",
        level_name,
        outcome.solved,
        outcome.timesteps_used,
        outcome.score,
        elapsed,
    )

    out_dir = out_root / f"scripted_{level_name}"
    mp4_path = render_video(outcome, out_dir, fps=fps)
    plot_paths = render_plots(outcome, out_dir)

    run = None
    if use_wandb:
        run = upload_to_wandb(
            outcome=outcome,
            mp4_path=mp4_path,
            plot_paths=plot_paths,
            sha_short=sha_short,
            seed=seed,
            project=project,
        )
    if run is not None:
        run.finish()

    return outcome


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level",
        default="all",
        help="Level name (e.g. navigate, mine, ...) or 'all' for the curriculum.",
    )
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed (default: 0).")
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=True,
        help="Log to wandb (default: True).",
    )
    parser.add_argument(
        "--no-wandb",
        dest="use_wandb",
        action="store_false",
        help="Disable wandb logging — useful for local-only smoke runs.",
    )
    parser.add_argument(
        "--wandb-project",
        default=WANDB_PROJECT,
        help=f"Wandb project name (default: {WANDB_PROJECT}).",
    )
    parser.add_argument(
        "--out-root",
        default="runs",
        help="Directory for video + plot artifacts (default: runs/).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Video framerate (default: 2; gives 0.5s per tick).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    sha_short = _git_sha_short()
    out_root = Path(args.out_root)

    scenario = SkillsBenchmark()
    if args.level == "all":
        level_names = [bl.name for bl in scenario.levels()]
    else:
        level_names = [args.level]

    if not args.use_wandb:
        logger.warning("wandb disabled — results saved locally only.")

    summary: list[RolloutOutcome] = []
    for name in level_names:
        outcome = run_one(
            level_name=name,
            seed=args.seed,
            sha_short=sha_short,
            out_root=out_root,
            use_wandb=args.use_wandb,
            project=args.wandb_project,
            fps=args.fps,
        )
        summary.append(outcome)

    logger.info("=" * 60)
    logger.info("%-16s %-8s %-8s %s", "level", "solved", "ticks", "score")
    logger.info("-" * 60)
    for o in summary:
        logger.info(
            "%-16s %-8s %-8d %.3f",
            o.level_name,
            "yes" if o.solved else "no",
            o.timesteps_used,
            o.score,
        )
    agg = sum(o.score for o in summary) / len(summary) if summary else 0.0
    logger.info("-" * 60)
    logger.info("%-16s %-8s %-8s %.3f", "AGGREGATE", "", "", agg)


if __name__ == "__main__":
    main()
