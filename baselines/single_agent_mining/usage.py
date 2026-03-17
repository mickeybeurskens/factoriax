"""Usage example for SingleAgentMiningBenchmark.

Demonstrates the full evaluation loop with a random policy. Run directly::

    python -m baselines.single_agent_mining.usage

The only thing you need to change to evaluate your own agent is the policy
function. It receives a JAX float32 observation array and must return a JAX
integer action scalar in [0, 11].
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from factoriax.benchmarks.runner import BenchmarkRunner
from factoriax.benchmarks.single_agent_mining.benchmark import SingleAgentMiningBenchmark
from factoriax.benchmarks.single_agent_mining.analysis import (
    plot_action_distribution,
    plot_level_scores,
    plot_resource_breakdown,
)


def main() -> None:
    """Run a random policy through the single-agent mining benchmark."""
    key = jax.random.PRNGKey(0)

    # Replace this function with your own policy.
    # obs: float32 JAX array of shape (obs_dim,)
    # return: integer JAX scalar in [0, 11]
    def random_policy(obs: jax.Array) -> jax.Array:
        nonlocal key
        key, subkey = jax.random.split(key)
        return jax.random.randint(subkey, shape=(), minval=0, maxval=12)

    benchmark = SingleAgentMiningBenchmark()
    runner = BenchmarkRunner(seed=42)
    result = runner.run(benchmark, policies=[random_policy])

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

    # Optionally inspect the figures.
    fig_scores = plot_level_scores(result)
    fig_breakdown = plot_resource_breakdown(result)
    fig_actions = plot_action_distribution(result)

    # Save locally — comment out if you only want the printed summary.
    fig_scores.savefig("benchmark_scores.png", dpi=120, bbox_inches="tight")
    fig_breakdown.savefig("benchmark_breakdown.png", dpi=120, bbox_inches="tight")
    fig_actions.savefig("benchmark_actions.png", dpi=120, bbox_inches="tight")
    print("\nPlots saved: benchmark_scores.png, benchmark_breakdown.png, benchmark_actions.png")


if __name__ == "__main__":
    main()
