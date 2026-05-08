"""Scripted baselines for the skills challenge curriculum.

Each level in :class:`SkillsBenchmark` ships with a tiny, deterministic
scripted policy that solves it. Three jobs:

1. **Solvability proof** — if a hand-written policy can solve the
   level, the achievement condition is reachable from the level's
   starting state under its action mask.
2. **Score floor for RL** — RL agents should at least match the
   scripted policy's time-weighted score before the level is
   considered "learned".
3. **Tutorial by example** — each scripted file is short enough to
   read in one sitting and shows how to consume the engine API.

Scripted policies are *state-readers* — they peek at :class:`EnvState`
directly. They are baselines, not agents under evaluation. The signature
``Callable[[EnvState, EnvParams], jax.Array]`` keeps this explicit; the
test harness drives the env step-by-step rather than going through
:class:`BenchmarkRunner`'s obs-only interface.
"""

from __future__ import annotations

from collections.abc import Callable

import jax

from baselines.skills.scripted.mine import mine_policy
from baselines.skills.scripted.navigate import navigate_policy
from factoriax.state import EnvParams, EnvState

#: Function signature every scripted policy implements.
ScriptedPolicy = Callable[[EnvState, EnvParams], jax.Array]

#: Registry mapping skill (level) name to its scripted policy. Each
#: Phase L slice appends one entry. Test harnesses look up by level
#: name to dispatch the right policy at evaluation time.
SCRIPTED_POLICIES: dict[str, ScriptedPolicy] = {
    "navigate": navigate_policy,
    "mine": mine_policy,
}

__all__ = [
    "SCRIPTED_POLICIES",
    "ScriptedPolicy",
    "mine_policy",
    "navigate_policy",
]
