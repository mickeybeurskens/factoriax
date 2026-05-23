"""Stateful scripted agent for the easy-rocket scenario.

Drives the seven-phase production order from ``script.md``. The agent
is a callable that returns one action per tick. It plans the factory
layout from the initial state at construction, then walks through the
phases, advancing when each phase's success predicate flips true.

Halts fail-fast on the first phase whose deadline elapses without
success. The halt status is recorded on the agent so the runner can
report which phase failed.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from baselines.easy_rocket.scripted.layout import FactoryLayout, plan_factory
from baselines.easy_rocket.scripted.phases import Phase, build_phases
from baselines.easy_rocket.scripted.state_reader import find_patches
from factoriax.constants import Action
from factoriax.recipes import RecipeTable
from factoriax.state import EnvParams, EnvState


@dataclasses.dataclass
class AgentReport:
    """Final status the runner can print or log.

    Attributes:
        halted: True if the agent stopped before completing phase 7.
        halt_reason: Free-text explanation; empty when ``halted=False``.
        phase_outcomes: One entry per phase actually attempted, in
            order. Each is ``(name, success, ticks_used)``.
    """

    halted: bool = False
    halt_reason: str = ""
    phase_outcomes: list[tuple[str, bool, int]] = dataclasses.field(
        default_factory=list
    )


class ScriptedAgent:
    """Phase-orchestrating policy for the easy-rocket scenario."""

    def __init__(
        self,
        initial_state: EnvState,
        recipe_table: RecipeTable,
        target_item: int | None = None,
    ) -> None:
        """Plan the factory and prime the phase list.

        Args:
            initial_state: First :class:`EnvState` the agent will see.
                The layout is planned from this state's map.
            recipe_table: Recipe table that the env uses. The layout
                planner walks the DAG from ``target_item``.
            target_item: Item the factory should produce. Defaults to
                ``ItemType.ROCKET`` via :func:`plan_factory`.
        """
        if target_item is None:
            self.layout: FactoryLayout = plan_factory(initial_state, recipe_table)
        else:
            self.layout = plan_factory(initial_state, recipe_table, target=target_item)
        self.patches = find_patches(initial_state)
        self.phases: list[Phase] = build_phases(self.layout, self.patches)
        self.current_phase: int = 0
        self.phase_start_tick: int = 0
        self.report: AgentReport = AgentReport()
        if not self.layout.valid:
            self.report.halted = True
            self.report.halt_reason = f"layout planning failed: {self.layout.error}"

    def __call__(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Return the next action for ``state``."""
        if self.report.halted:
            return jnp.asarray(int(Action.NOOP), dtype=jnp.int32)
        if self.current_phase >= len(self.phases):
            return jnp.asarray(int(Action.NOOP), dtype=jnp.int32)

        tick = int(state.timestep)
        phase = self.phases[self.current_phase]

        # Has this phase succeeded? Advance.
        if phase.success(state):
            self.report.phase_outcomes.append(
                (phase.name, True, tick - self.phase_start_tick)
            )
            self.current_phase += 1
            self.phase_start_tick = tick
            if self.current_phase >= len(self.phases):
                return jnp.asarray(int(Action.NOOP), dtype=jnp.int32)
            phase = self.phases[self.current_phase]
            if phase.success(state):
                # Edge case: next phase already done at advance time.
                return self.__call__(state, params)

        # Deadline exceeded?
        if tick - self.phase_start_tick > phase.deadline:
            self.report.halted = True
            self.report.halt_reason = (
                f"phase '{phase.name}' deadline exceeded ({phase.deadline} ticks)"
            )
            self.report.phase_outcomes.append(
                (phase.name, False, tick - self.phase_start_tick)
            )
            return jnp.asarray(int(Action.NOOP), dtype=jnp.int32)

        return jnp.asarray(phase.step(state, params), dtype=jnp.int32)
