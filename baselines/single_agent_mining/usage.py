"""Usage example for SingleAgentMiningBenchmark.

Demonstrates the full evaluation loop with a random policy. Run directly::

    python -m baselines.single_agent_mining.usage
    python -m baselines.single_agent_mining.usage --use-wandb
    python -m baselines.single_agent_mining.usage --use-wandb --wandb-project my-project

The only thing you need to change to evaluate your own agent is the policy
function. It receives a JAX float32 observation array and must return a JAX
integer action scalar in [0, 11].
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import jax

from factoriax.benchmarks.runner import BenchmarkRunner
from factoriax.benchmarks.single_agent_mining.analysis import (
    log_to_wandb,
    plot_action_distribution,
    plot_level_scores,
    plot_resource_breakdown,
    render_level_video,
    save_mp4,
)
from factoriax.benchmarks.single_agent_mining.benchmark import (
    SingleAgentMiningBenchmark,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _make_random_policy() -> ...:
    """Factory that returns a fresh random policy with its own PRNG state."""
    key = jax.random.PRNGKey(0)

    def policy(obs: jax.Array) -> jax.Array:
        nonlocal key
        key, subkey = jax.random.split(key)
        return jax.random.randint(subkey, shape=(), minval=0, maxval=12)

    return policy


def main() -> None:
    """Run a random policy through the single-agent mining benchmark."""
    p = argparse.ArgumentParser(description="Single-agent mining benchmark — random policy")
    p.add_argument("--use-wandb", action="store_true", help="Log results to Weights & Biases.")
    p.add_argument("--wandb-project", type=str, default="factoriax-benchmarks")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    benchmark = SingleAgentMiningBenchmark()
    runner = BenchmarkRunner(seed=args.seed)
    result = runner.run(benchmark, policies=[_make_random_policy()])

    print(f"Benchmark : {result.benchmark_name}")
    print(f"Aggregate : {result.aggregate_score:.2f}")
    print()
    for lr in result.level_results:
        coal = lr.items_mined.get("coal", 0)
        iron = lr.items_mined.get("iron", 0)
        copper = lr.items_mined.get("copper", 0)
        print(
            f"  {lr.level_name:<28}  score={lr.weighted_score:6.1f}"
            f"  coal={coal:3d}  iron={iron:3d}  copper={copper:3d}"
            f"  steps={lr.timesteps_used}"
        )

    fig_scores = plot_level_scores(result)
    fig_breakdown = plot_resource_breakdown(result)
    fig_actions = plot_action_distribution(result)

    fig_scores.savefig("benchmark_scores.png", dpi=120, bbox_inches="tight")
    fig_breakdown.savefig("benchmark_breakdown.png", dpi=120, bbox_inches="tight")
    fig_actions.savefig("benchmark_actions.png", dpi=120, bbox_inches="tight")
    logger.info("Plots saved: benchmark_scores.png, benchmark_breakdown.png, benchmark_actions.png")

    # Render one video per level. Each level gets a fresh policy so the
    # videos are independent of the scoring run above.
    videos: dict[str, Path] = {}
    for bl in benchmark.levels():
        frames = render_level_video(bl, _make_random_policy(), seed=args.seed)
        path = Path(f"{bl.name}.mp4")
        save_mp4(frames, path)
        videos[bl.name] = path
        logger.info("Video saved: %s  (%d frames)", path, len(frames))

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb  # type: ignore[import-untyped]
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={"policy": "random", "seed": args.seed},
            )
        except ImportError:
            logger.error("wandb not found. Install with: uv add wandb")

    log_to_wandb(result, wandb_run, videos=videos)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
