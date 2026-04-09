"""FactoriaX Agent Debugger -- interactive debugging and trajectory replay.

**Live stepping** from code::

    from factoriax.agentdebugger import Debugger
    debugger = Debugger(env, params, state, policy=my_policy)
    debugger.run()

**Trajectory replay** from the command line::

    python -m factoriax.agentdebugger rollout.npz

Or from code::

    from factoriax.agentdebugger import Debugger
    debugger = Debugger.from_trajectory("rollout.npz")
    debugger.run()
"""

from factoriax.agentdebugger.main import Debugger

__all__ = ["Debugger"]
