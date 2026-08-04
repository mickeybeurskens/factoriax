# Action design

The action set has 87 discrete actions, and it follows three rules.

## One action, one outcome

A player crafts a miner with the single action `CRAFT_MINER`. It does not
cycle through a recipe list and then press a craft key. No action in the 87
hides a sequence of other actions.

## The observation tells the agent what works

An affordability bit for each item says whether the player can craft that
item now. The agent reads what an action does before it takes that action.
The bit belongs to an item, not to a recipe. See
{doc}`observation_space` for where this bit lives in the observation.

## Context does not change an action

A craft takes items from the inventory and adds the result. A mine adds an
item, and a place takes one. There is no separate crafting mode that changes
what a key does.

## Why this matters for training

A policy network maps one observation to one action distribution over a
fixed 87-way space. Because context never changes what an action does, the
network does not need to track a hidden mode to interpret its own output.
