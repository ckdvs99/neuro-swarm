"""
Core components of the neuro-swarm system.

- agent: The NeuroAgent - minimal, complete
- substrate: Stigmergic communication layer
- rhythm: Temporal dynamics
"""

from .agent import NeuroAgent, AgentState, AgentConfig

__all__ = ["NeuroAgent", "AgentState", "AgentConfig"]
