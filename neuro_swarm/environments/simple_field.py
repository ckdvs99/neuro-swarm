"""
environments/simple_field.py

2D continuous space for basic experiments.

A simple world for simple beginnings.
Watch the system before changing it.

Inspired by:
- Reynolds boids simulation
- Particle physics sandboxes
- Cellular automata spaces
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import numpy as np

from neuro_swarm.core.agent import NeuroAgent, AgentState, AgentConfig
from neuro_swarm.core.substrate import Substrate, SubstrateConfig


@dataclass
class FieldConfig:
    """Configuration for the simple field environment."""
    bounds: Tuple[float, float] = (-50.0, 50.0)  # World boundaries
    wrap_edges: bool = True                       # Toroidal topology
    friction: float = 0.98                        # Velocity decay
    substrate_enabled: bool = True                # Enable stigmergy


class SimpleField:
    """
    2D continuous environment for swarm experiments.

    Features:
    - Bounded or toroidal space
    - Optional substrate for stigmergy
    - Neighbor queries (respecting locality)
    - Step-based simulation

    Principles embodied:
    - Observation before intervention (Principle 6)
    - The medium is the message (Principle 3) via substrate
    """

    def __init__(
        self,
        config: Optional[FieldConfig] = None,
        substrate_config: Optional[SubstrateConfig] = None
    ):
        self.config = config or FieldConfig()
        self.agents: Dict[str, NeuroAgent] = {}
        self.time = 0

        # Initialize substrate if enabled
        if self.config.substrate_enabled:
            sub_config = substrate_config or SubstrateConfig(
                world_bounds=self.config.bounds
            )
            self.substrate = Substrate(sub_config)
        else:
            self.substrate = None

    def add_agent(
        self,
        agent_id: str,
        position: Optional[np.ndarray] = None,
        agent_config: Optional[AgentConfig] = None
    ) -> NeuroAgent:
        """Add an agent to the field."""
        if position is None:
            # Random position within bounds
            position = np.random.uniform(
                self.config.bounds[0],
                self.config.bounds[1],
                size=2
            )

        agent = NeuroAgent(agent_id, position, agent_config)
        self.agents[agent_id] = agent
        return agent

    def remove_agent(self, agent_id: str) -> Optional[NeuroAgent]:
        """Remove an agent from the field."""
        return self.agents.pop(agent_id, None)

    def get_neighbors(
        self,
        agent_id: str,
        radius: Optional[float] = None,
        limit: int = 7
    ) -> List[AgentState]:
        """
        Get neighboring agent states.

        Respects locality - only returns up to `limit` nearest neighbors.
        Optionally filters by radius.
        """
        if agent_id not in self.agents:
            return []

        agent = self.agents[agent_id]
        neighbors = []

        for other_id, other in self.agents.items():
            if other_id == agent_id:
                continue

            distance = agent.distance_to(other)

            if radius is not None and distance > radius:
                continue

            neighbors.append((distance, other.state))

        # Sort by distance, take closest
        neighbors.sort(key=lambda x: x[0])
        return [state for _, state in neighbors[:limit]]

    def step(self) -> None:
        """
        Advance simulation by one time step.

        For each agent:
        1. Perceive neighbors and substrate
        2. Update internal state
        3. Act (move)
        4. Deposit trace (if substrate enabled)
        5. Apply physics (friction, boundaries)
        """
        self.time += 1

        # Phase 1: Perception
        perceptions = {}
        for agent_id, agent in self.agents.items():
            neighbor_states = self.get_neighbors(
                agent_id,
                limit=agent.config.neighbor_limit
            )
            perceptions[agent_id] = agent.perceive(neighbor_states, self.substrate)

        # Phase 2: Update
        for agent_id, agent in self.agents.items():
            agent.update(perceptions[agent_id])

        # Phase 3: Act
        for agent in self.agents.values():
            agent.act()

        # Phase 4: Deposit traces
        if self.substrate:
            for agent in self.agents.values():
                agent.deposit(self.substrate)
            self.substrate.step()

        # Phase 5: Physics
        for agent in self.agents.values():
            self._apply_physics(agent)

    def _apply_physics(self, agent: NeuroAgent) -> None:
        """Apply friction and boundary conditions."""
        # Friction
        agent.state.velocity *= self.config.friction

        # Boundaries
        if self.config.wrap_edges:
            # Toroidal wrapping
            bounds_range = self.config.bounds[1] - self.config.bounds[0]
            agent.state.position = (
                (agent.state.position - self.config.bounds[0]) % bounds_range
                + self.config.bounds[0]
            )
        else:
            # Hard boundaries with bounce
            for i in range(2):
                if agent.state.position[i] < self.config.bounds[0]:
                    agent.state.position[i] = self.config.bounds[0]
                    agent.state.velocity[i] *= -0.5
                elif agent.state.position[i] > self.config.bounds[1]:
                    agent.state.position[i] = self.config.bounds[1]
                    agent.state.velocity[i] *= -0.5

    def get_state_snapshot(self) -> Dict[str, AgentState]:
        """Get current state of all agents."""
        return {aid: agent.state for aid, agent in self.agents.items()}

    def get_positions(self) -> np.ndarray:
        """Get positions of all agents as array."""
        return np.array([a.state.position for a in self.agents.values()])

    def get_velocities(self) -> np.ndarray:
        """Get velocities of all agents as array."""
        return np.array([a.state.velocity for a in self.agents.values()])

    def get_energies(self) -> np.ndarray:
        """Get energy levels of all agents."""
        return np.array([a.state.energy for a in self.agents.values()])

    def __repr__(self) -> str:
        return (
            f"SimpleField(agents={len(self.agents)}, "
            f"time={self.time}, "
            f"substrate={'on' if self.substrate else 'off'})"
        )
