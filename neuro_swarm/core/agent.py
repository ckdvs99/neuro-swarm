"""
core/agent.py

An agent should be like a haiku:
constrained, complete, resonant.

Inspired by:
- Starling flocking (local neighbor attention)
- Neural refractory periods (rhythm)  
- Cortical selective attention (gating)
- Ant stigmergy (environment as memory)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional
import numpy as np

if TYPE_CHECKING:
    from .substrate import Substrate


@dataclass
class AgentState:
    """
    What an agent IS at this moment.
    
    Not what it does. Not what it knows.
    Just its current being.
    """
    position: np.ndarray          # Where in space
    velocity: np.ndarray          # Momentum, tendency, direction
    internal: np.ndarray          # Hidden state - the SSM core
    energy: float = 1.0           # Capacity for action
    age: int = 0                  # Time steps lived
    
    def __post_init__(self):
        # Ensure arrays are proper numpy arrays
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        self.internal = np.asarray(self.internal, dtype=np.float64)


@dataclass 
class AgentConfig:
    """
    The unchanging nature of an agent.
    Set at birth, honored throughout life.
    """
    state_dim: int = 8                    # Internal state dimensionality
    neighbor_limit: int = 7               # Biological constant (Miller's law)
    energy_decay: float = 0.05            # Cost of action
    energy_recovery: float = 0.15         # Benefit of rest
    rest_threshold: float = 0.1           # When rest becomes mandatory
    memory_persistence: float = 0.9       # How much state persists (SSM A matrix analog)
    observation_weight: float = 0.1       # How much new info enters (SSM B matrix analog)
    

class NeuroAgent:
    """
    A single agent in the swarm.
    
    Principles embodied:
    - Locality: knows only nearby neighbors
    - Rhythm: must rest when depleted
    - Stigmergy: reads/writes environmental traces
    - Selectivity: gates what information enters state
    """
    
    def __init__(
        self, 
        agent_id: str, 
        position: Optional[np.ndarray] = None,
        config: Optional[AgentConfig] = None
    ):
        self.id = agent_id
        self.config = config or AgentConfig()
        
        # Initialize state
        init_pos = position if position is not None else np.random.randn(2) * 2
        self.state = AgentState(
            position=init_pos,
            velocity=np.zeros(2),
            internal=np.zeros(self.config.state_dim)
        )
        
        # Neighbor tracking (IDs only - we don't own their state)
        self.neighbor_ids: List[str] = []
        
        # History for observation (optional, for studies)
        self.history: List[AgentState] = []
        self.record_history = False
        
    # ==================== Core Loop ====================
    
    def perceive(
        self, 
        neighbor_states: List[AgentState], 
        substrate: Optional[Substrate] = None
    ) -> np.ndarray:
        """
        Gather information from the world.
        
        Two channels:
        1. Direct sensing of neighbors (limited by neighbor_limit)
        2. Reading stigmergic traces from substrate
        
        Returns: observation vector
        """
        # Respect locality - only perceive up to neighbor_limit
        visible_neighbors = neighbor_states[:self.config.neighbor_limit]
        
        # Compose neighbor information
        if not visible_neighbors:
            neighbor_obs = np.zeros(6)
        else:
            positions = np.array([n.position for n in visible_neighbors])
            velocities = np.array([n.velocity for n in visible_neighbors])
            
            # Relative positions (egocentric frame)
            relative_pos = positions - self.state.position
            
            neighbor_obs = np.concatenate([
                relative_pos.mean(axis=0),    # Center of neighbors (2,)
                relative_pos.std(axis=0),     # Spread (2,)
                velocities.mean(axis=0),      # Average heading (2,)
            ])
        
        # Read substrate trace
        if substrate is not None:
            trace = substrate.read(self.state.position)
        else:
            trace = np.zeros(4)
            
        # Compose full observation
        observation = np.concatenate([neighbor_obs, trace])
        
        return observation
    
    def update(self, observation: np.ndarray) -> None:
        """
        Update internal state based on observation.
        
        This is the heart of the SSM-inspired dynamics:
        x_{t+1} = A * x_t + B * (gated_observation)
        
        Where the gating is SELECTIVE - not all information enters.
        Restraint is intelligence.
        """
        self.state.age += 1
        
        # Check if rest is mandatory
        if self.state.energy < self.config.rest_threshold:
            self._rest()
            return
            
        # === Selective Gating ===
        # What deserves attention? Surprise relative to expectation.
        salience = self._compute_salience(observation)
        gated_obs = observation * salience
        
        # === State Space Update ===
        # x_{t+1} = A * x_t + B * u_t
        # A = memory_persistence (diagonal, for now)
        # B = observation_weight
        
        # Ensure observation fits state dimension
        obs_contribution = np.zeros(self.config.state_dim)
        obs_len = min(len(gated_obs), self.config.state_dim)
        obs_contribution[:obs_len] = gated_obs[:obs_len]
        
        self.state.internal = (
            self.config.memory_persistence * self.state.internal +
            self.config.observation_weight * obs_contribution
        )
        
        # Bound internal state (biological plausibility)
        self.state.internal = np.clip(self.state.internal, -5, 5)
        
        # Action has cost
        self.state.energy -= self.config.energy_decay
        
        # Record if studying
        if self.record_history:
            self._record()
    
    def act(self) -> np.ndarray:
        """
        Produce behavior based on internal state.
        
        The action IS the message (like the waggle dance).
        Others can observe our behavior and learn from it.
        
        Returns: velocity/heading change
        """
        if self.state.energy < self.config.rest_threshold:
            return np.zeros(2)  # Resting agents don't move
            
        # Behavior emerges from internal state
        # First two dimensions encode heading preference
        heading = self.state.internal[:2]
        
        # Magnitude modulated by energy (tired agents move slowly)
        magnitude = np.clip(self.state.energy, 0, 1)
        
        # Smooth velocity update (momentum)
        desired_velocity = heading * magnitude * 0.5
        self.state.velocity = 0.8 * self.state.velocity + 0.2 * desired_velocity
        
        # Limit speed
        speed = np.linalg.norm(self.state.velocity)
        if speed > 1.0:
            self.state.velocity = self.state.velocity / speed
            
        # Update position
        self.state.position = self.state.position + self.state.velocity
        
        return self.state.velocity
    
    def deposit(self, substrate: Substrate) -> None:
        """
        Leave a trace in the environment (stigmergy).
        
        What mark do we leave? A function of our internal state.
        Others will read this trace and be influenced.
        """
        trace = self._compute_trace()
        substrate.write(self.state.position, trace)
    
    # ==================== Internal Mechanisms ====================
    
    def _compute_salience(self, observation: np.ndarray) -> np.ndarray:
        """
        Selective attention: what information deserves to enter state?
        
        Based on surprise - deviation from what internal state predicts.
        High surprise = high salience = gate opens.
        Low surprise = low salience = gate stays closed.
        """
        # Use internal state as implicit prediction
        expected = self.state.internal[:len(observation)]
        if len(expected) < len(observation):
            expected = np.pad(expected, (0, len(observation) - len(expected)))
            
        # Surprise as absolute deviation
        surprise = np.abs(observation - expected)
        
        # Soft gating via tanh (bounded 0-1 after shift)
        salience = np.tanh(surprise)
        
        return salience
    
    def _compute_trace(self) -> np.ndarray:
        """
        What mark do we leave in the substrate?
        
        A bounded, smoothed version of internal state.
        Limited to 4 dimensions (substrate constraint).
        """
        trace_raw = self.state.internal[:4]
        if len(trace_raw) < 4:
            trace_raw = np.pad(trace_raw, (0, 4 - len(trace_raw)))
        return np.tanh(trace_raw)
    
    def _rest(self) -> None:
        """
        Refractory period. Even neurons need this.
        
        During rest:
        - Energy recovers
        - State persists but doesn't update
        - No action taken
        """
        self.state.energy = min(1.0, self.state.energy + self.config.energy_recovery)
        
        if self.record_history:
            self._record()
    
    def _record(self) -> None:
        """Record current state for later analysis."""
        # Deep copy to avoid reference issues
        snapshot = AgentState(
            position=self.state.position.copy(),
            velocity=self.state.velocity.copy(),
            internal=self.state.internal.copy(),
            energy=self.state.energy,
            age=self.state.age
        )
        self.history.append(snapshot)
    
    # ==================== Utilities ====================
    
    def distance_to(self, other: NeuroAgent) -> float:
        """Euclidean distance to another agent."""
        return np.linalg.norm(self.state.position - other.state.position)
    
    def __repr__(self) -> str:
        return (
            f"NeuroAgent(id={self.id}, "
            f"pos=[{self.state.position[0]:.2f}, {self.state.position[1]:.2f}], "
            f"energy={self.state.energy:.2f}, "
            f"age={self.state.age})"
        )
