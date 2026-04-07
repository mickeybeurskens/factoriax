"""FactoriaX Agent Debugger -- interactive step-through debugging for RL agents.

Spawn a pygame window from code to step through a live JAX environment
as either a human player or an AI agent, with reward and cost charts
updating in real time.

Quick start::

    from factoriax.agentdebugger import Debugger
    debugger = Debugger(env, params, state, policy=my_policy)
    debugger.run()
"""

from factoriax.agentdebugger.main import Debugger

__all__ = ["Debugger"]
