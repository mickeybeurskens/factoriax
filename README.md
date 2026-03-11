# FactoriaX

Multi-agent coordination for building complex infrastructure is an open problem. Factoriax tests the ability of agents to handle logistics, task decomposition, and temporal dependencies in an open ended challenge setting inspired by the game Factorio. The goal of the challenge is to create and expand a factory with one or multiple agents, with the core goal being to automate more and more of the process through building machines and infrastructure. As the agents progress they can do research to unlock more machines, creating a multiple intertwining progression paths and dependencies throughout the game.

To make the environment as accesible as possible for reinforcement learning the entire environment is built to run on the GPU through JAX. This allows for many more training runs in a given time relative to CPU based simulations.

