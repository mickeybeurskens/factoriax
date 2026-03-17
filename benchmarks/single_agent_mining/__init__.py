"""Single-agent mining benchmark.

Five levels of progressively more difficult mining challenges. The agent
has 200 ticks to collect as many resources as possible. Resources have
different values: copper (3 pts) > iron (2 pts) > coal (1 pt). The
benchmark score is the mean weighted score across all five levels.
"""

from benchmarks.single_agent_mining.benchmark import SingleAgentMiningBenchmark

__all__ = ["SingleAgentMiningBenchmark"]
