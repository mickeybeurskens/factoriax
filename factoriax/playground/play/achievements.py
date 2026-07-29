"""Free-play achievement ladder for the interactive client.

A tutorial progression that walks a human player from hand-mining raw ore
to launching a rocket, one mechanic at a time. The hints are keyboard
instructions, so this ladder is specific to the playground — research
scenarios declare their own sets next to the env that uses them, and none
of them share this one.

Not registered as a scenario: free play builds its env directly in
:mod:`factoriax.playground.play.main`, so the ladder never enters the
engine or the scenario registry.
"""

from __future__ import annotations

from functools import partial

import jax

from factoriax.engine.achievements import (
    Achievement,
    achievement_fn,
    any_assembler_has_output,
    any_buffer_nonempty,
    count_total_items,
    has_machines,
    holds_item,
    total_machines,
    total_ore_mined,
)
from factoriax.engine.constants import ItemType, Machine
from factoriax.engine.state import EnvState


def _holds_any_machine_item(state: EnvState) -> jax.Array:
    """Player holds a placeable machine — i.e. something was crafted."""
    held = (
        count_total_items(state, int(ItemType.MINER))
        + count_total_items(state, int(ItemType.PALLET))
        + count_total_items(state, int(ItemType.CONVEYOR_BELT))
    )
    return held >= 1


def _arm_and_pallet_placed(state: EnvState) -> jax.Array:
    """Both an arm and a pallet are on the map — the pair moves items."""
    return has_machines(state, int(Machine.ARM)) & has_machines(
        state, int(Machine.PALLET)
    )


def _holds_any_science(state: EnvState) -> jax.Array:
    """Player holds a science pack of either tier."""
    packs = count_total_items(
        state, int(ItemType.TIER1_SCIENCE_PACK)
    ) + count_total_items(state, int(ItemType.TIER2_SCIENCE_PACK))
    return packs >= 1


#: The ladder, in unlock order. Index is the persisted bit position.
FREE_PLAY_ACHIEVEMENTS: tuple[Achievement, ...] = (
    Achievement(
        id="first_ore",
        name="First Ore",
        condition=lambda state: total_ore_mined(state) >= 1,
        hint="Walk onto an ore tile and press SPACE to mine.",
    ),
    Achievement(
        id="stockpile",
        name="Stockpile",
        condition=lambda state: total_ore_mined(state) >= 10,
        hint="Keep mining until you have 10 ore total.",
    ),
    Achievement(
        id="apprentice_engineer",
        name="Apprentice Engineer",
        condition=_holds_any_machine_item,
        hint="Open inventory (I), select a recipe, and craft it (E).",
    ),
    Achievement(
        id="breaking_ground",
        name="Breaking Ground",
        condition=lambda state: total_machines(state) >= 1,
        hint="Select a machine in your hotbar and press E to place it.",
    ),
    Achievement(
        id="coal_gathered",
        name="Coal Gathered",
        condition=partial(holds_item, item=int(ItemType.COAL)),
        hint="Walk onto a coal tile and press SPACE to mine a piece of coal.",
    ),
    Achievement(
        id="automated_mining",
        name="Automated Mining",
        condition=partial(any_buffer_nonempty, machine=int(Machine.MINER)),
        hint="Place a miner on an ore tile and wait for it to produce.",
    ),
    Achievement(
        id="moving_parts",
        name="Moving Parts",
        condition=_arm_and_pallet_placed,
        hint="Craft and place both an arm and a pallet.",
    ),
    Achievement(
        id="first_pipeline",
        name="First Pipeline",
        condition=partial(any_buffer_nonempty, machine=int(Machine.PALLET)),
        hint="Use an arm to move miner output into a pallet.",
    ),
    Achievement(
        id="belt_network",
        name="Belt Network",
        condition=partial(has_machines, machine=int(Machine.CONVEYOR_BELT), count=5),
        hint="Craft and place at least 5 conveyor belts.",
    ),
    Achievement(
        id="scaling_up",
        name="Scaling Up",
        condition=partial(has_machines, machine=int(Machine.MINER), count=3),
        hint="Have 3 miners placed on the map at the same time.",
    ),
    Achievement(
        id="industrialist",
        name="Industrialist",
        condition=lambda state: total_machines(state) >= 10,
        hint="Place 10 machines of any type on the map.",
    ),
    Achievement(
        id="assembler_crafted",
        name="Assembler Crafted",
        condition=partial(holds_item, item=int(ItemType.ASSEMBLER)),
        hint="Open inventory (I), select the Assembler recipe, and craft it.",
    ),
    Achievement(
        id="assembly_line",
        name="Assembly Line",
        condition=partial(has_machines, machine=int(Machine.ASSEMBLER)),
        hint="Place an assembler on the map.",
    ),
    Achievement(
        id="first_assembly",
        name="First Assembly",
        condition=any_assembler_has_output,
        hint="Set a recipe on your assembler (Q) and feed it inputs.",
    ),
    Achievement(
        id="rocket_complete",
        name="Rocket Complete",
        condition=partial(has_machines, machine=int(Machine.ROCKET)),
        hint="Craft a rocket in an assembler and place it on the map.",
    ),
    Achievement(
        id="first_science",
        name="First Science",
        condition=_holds_any_science,
        hint="Produce a science pack in an assembler.",
    ),
    Achievement(
        id="advanced_science",
        name="Advanced Science",
        condition=partial(holds_item, item=int(ItemType.TIER2_SCIENCE_PACK)),
        hint="Produce an advanced science pack.",
    ),
)

#: Bound evaluator for :class:`~factoriax.engine.envs.base.FactoriaxEnv`.
free_play_conditions = achievement_fn(FREE_PLAY_ACHIEVEMENTS)
