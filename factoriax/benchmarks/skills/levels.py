"""Level builders for the skills challenge curriculum.

Each builder is a deterministic function of a ``seed`` argument
returning ``(Level, EnvParams, frozenset[int])`` — level geometry,
runtime parameters, and the per-level action mask. Seed ``0`` is the
canonical evaluation layout used by :class:`SkillsBenchmark`; arbitrary
seeds are available for training-time variation.

Achievement conditions are designed to be **layout-invariant** so the
seed only varies spawn / source positions / distractors, not the target
the agent must hit. See the spec for details.

Builders are added one at a time as Phase L slices land. The skeleton
file ships empty.
"""

from __future__ import annotations

# Per-skill builders go here, one function per slice:
#
#   def build_navigate_level(seed: int = 0) -> tuple[Level, EnvParams, frozenset[int]]:
#       ...
#
# The skeleton intentionally exports nothing — :func:`SkillsBenchmark.levels`
# returns an empty list until L.1 lands.
