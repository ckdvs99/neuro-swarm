"""
core/agent_brain.py

Integration of distilled models with the NeuroAgent architecture.

This module extends NeuroAgent to optionally use distilled neural brains
for decision-making, while preserving the core SSM-inspired dynamics.

Two modes of operation:
1. Pure SSM mode: Original NeuroAgent behavior (state + gating)
2. Brain-augmented mode: Distilled model provides high-level guidance

The brain doesn't replace the agent's dynamics - it augments them.
Like the relationship between cortex and brainstem.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Dict, Any, List
import logging

import numpy as np

from .agent import NeuroAgent, AgentState, AgentConfig

if TYPE_CHECKING:
    from neuro_swarm.distillation.inference import AgentBrain

logger = logging.getLogger(__name__)


@dataclass
class BrainConfig:
    """Configuration for brain-augmented agents"""
    
    # Brain type: 'triumvirate', 'paladin', 'chaos', 'swarm_unit', or None
    brain_type: Optional[str] = None
    
    # Path to distilled model
    model_path: Optional[str] = None
    
    # How much the brain influences decisions (0-1)
    # 0 = pure SSM dynamics, 1 = pure brain output
    brain_influence: float = 0.7
    
    # Brain decision frequency (every N steps)
    decision_interval: int = 10
    
    # Fall back to SSM if brain confidence < threshold
    confidence_threshold: float = 0.3


class BrainAugmentedAgent(NeuroAgent):
    """
    NeuroAgent with optional distilled brain for high-level decisions.
    
    The brain provides strategic guidance while the underlying SSM
    dynamics handle reactive behavior. Think of it as:
    
    - Brain: "Go toward that cluster" (strategic)
    - SSM dynamics: "Avoid that obstacle, maintain rhythm" (reactive)
    
    Principles embodied:
    - Hierarchical sovereignty (Principle 7) - brain guides, SSM executes
    - Balance over power (Principle 2) - blend of neural and rule-based
    """
    
    def __init__(
        self,
        agent_id: str,
        position: Optional[np.ndarray] = None,
        config: Optional[AgentConfig] = None,
        brain_config: Optional[BrainConfig] = None,
    ):
        super().__init__(agent_id, position, config)
        
        self.brain_config = brain_config or BrainConfig()
        self.brain: Optional[AgentBrain] = None
        self.last_brain_decision: Optional[Dict[str, Any]] = None
        self.steps_since_brain_decision = 0
        
        # Load brain if configured
        if self.brain_config.model_path:
            self._load_brain()
    
    def _load_brain(self):
        """Load the distilled brain model"""
        from neuro_swarm.distillation.inference import (
            DistilledAgentBrain,
            MambaAgentBrain,
            InferenceConfig,
        )
        
        brain_type = self.brain_config.brain_type or "swarm_unit"
        model_path = self.brain_config.model_path
        
        logger.info(f"Loading {brain_type} brain from {model_path}")
        
        if brain_type == "swarm_unit":
            self.brain = MambaAgentBrain(model_path=model_path)
        else:
            config = InferenceConfig(model_path=model_path)
            self.brain = DistilledAgentBrain(config, agent_tier=brain_type)
        
        logger.info(f"Brain loaded for agent {self.id}")
    
    def _get_observation_dict(
        self,
        neighbor_states: List[AgentState],
    ) -> Dict[str, Any]:
        """Convert current state to observation dict for brain"""
        return {
            "position": self.state.position.tolist(),
            "velocity": self.state.velocity.tolist(),
            "energy": self.state.energy,
            "phase": getattr(self.state, 'phase', 0),
            "age": self.state.age,
            "internal_state": self.state.internal.tolist(),
        }
    
    def _get_context_dict(
        self,
        neighbor_states: List[AgentState],
    ) -> Dict[str, Any]:
        """Build context dict from neighbor states"""
        neighbors = []
        for n in neighbor_states[:self.config.neighbor_limit]:
            rel_pos = n.position - self.state.position
            neighbors.append({
                "relative_position": rel_pos.tolist(),
                "velocity": n.velocity.tolist(),
                "energy": n.energy,
            })
        
        return {
            "neighbors": neighbors,
            "num_neighbors": len(neighbors),
        }
    
    def _should_consult_brain(self) -> bool:
        """Determine if we should query the brain this step"""
        if self.brain is None:
            return False
        
        # Check decision interval
        if self.steps_since_brain_decision < self.brain_config.decision_interval:
            return False
        
        return True
    
    def _apply_brain_decision(self, decision: Dict[str, Any]) -> np.ndarray:
        """Convert brain decision to internal state modification"""
        # Brain output format depends on type
        if "direction" in decision:
            # Swarm unit style output
            direction = np.array(decision["direction"])
            intensity = decision.get("intensity", 0.5)
            
            # Scale by brain influence
            return direction * intensity * self.brain_config.brain_influence
        
        elif "action" in decision:
            # Triumvirate/Paladin style output
            # Parse the action string to determine effect
            action = decision.get("action", "")
            
            # Simple parsing - in practice this would be more sophisticated
            if "move_toward" in action:
                # Extract target from action
                return np.random.randn(2) * 0.5  # Placeholder
            elif "defensive" in action:
                return np.zeros(2)  # Hold position
            else:
                return np.zeros(2)
        
        return np.zeros(2)
    
    def update(self, observation: np.ndarray) -> None:
        """
        Extended update that optionally consults brain.
        
        The brain provides high-level guidance that's blended
        with the underlying SSM dynamics.
        """
        self.steps_since_brain_decision += 1
        
        # Call parent update for SSM dynamics
        super().update(observation)
        
        # Optionally consult brain for guidance
        if self._should_consult_brain():
            # We need neighbor states for context - stored from perceive
            neighbor_states = getattr(self, '_last_neighbors', [])
            
            obs_dict = self._get_observation_dict(neighbor_states)
            ctx_dict = self._get_context_dict(neighbor_states)
            
            try:
                decision = self.brain.decide(obs_dict, ctx_dict)
                
                # Check confidence threshold
                confidence = decision.get("confidence", 1.0)
                if confidence >= self.brain_config.confidence_threshold:
                    self.last_brain_decision = decision
                    
                    # Blend brain guidance into internal state
                    brain_guidance = self._apply_brain_decision(decision)
                    blend = self.brain_config.brain_influence
                    
                    self.state.internal[:2] = (
                        (1 - blend) * self.state.internal[:2] +
                        blend * brain_guidance
                    )
                
                self.steps_since_brain_decision = 0
                
            except Exception as e:
                logger.warning(f"Brain decision failed for {self.id}: {e}")
    
    def perceive(
        self,
        neighbor_states: List[AgentState],
        substrate = None,
    ) -> np.ndarray:
        """Extended perceive that stores neighbors for brain context"""
        self._last_neighbors = neighbor_states
        return super().perceive(neighbor_states, substrate)
    
    def get_brain_stats(self) -> Dict[str, Any]:
        """Get statistics about brain usage"""
        return {
            "has_brain": self.brain is not None,
            "brain_type": self.brain_config.brain_type,
            "brain_influence": self.brain_config.brain_influence,
            "last_decision": self.last_brain_decision,
            "steps_since_decision": self.steps_since_brain_decision,
        }
    
    def __repr__(self) -> str:
        brain_str = f", brain={self.brain_config.brain_type}" if self.brain else ""
        return (
            f"BrainAugmentedAgent(id={self.id}, "
            f"pos=[{self.state.position[0]:.2f}, {self.state.position[1]:.2f}], "
            f"energy={self.state.energy:.2f}"
            f"{brain_str})"
        )


class TriumvirateAgent(BrainAugmentedAgent):
    """
    Triumvirate coordinator agent with distilled reasoning capabilities.
    
    The highest tier in the agent hierarchy - capable of multi-agent
    reasoning and strategic coordination.
    """
    
    def __init__(
        self,
        agent_id: str,
        position: Optional[np.ndarray] = None,
        model_path: Optional[str] = None,
    ):
        # Triumvirate-specific config
        agent_config = AgentConfig(
            energy_decay=0.03,      # Efficient
            memory_persistence=0.95, # Strong memory
            neighbor_limit=7,
        )
        
        brain_config = BrainConfig(
            brain_type="triumvirate",
            model_path=model_path,
            brain_influence=0.8,    # High brain influence
            decision_interval=5,    # Frequent strategic decisions
            confidence_threshold=0.5,
        )
        
        super().__init__(agent_id, position, agent_config, brain_config)


class PaladinAgent(BrainAugmentedAgent):
    """
    Defensive agent specialized in threat detection and protection.
    
    Uses distilled pattern recognition for anomaly detection.
    """
    
    def __init__(
        self,
        agent_id: str,
        position: Optional[np.ndarray] = None,
        model_path: Optional[str] = None,
    ):
        agent_config = AgentConfig(
            energy_decay=0.04,
            memory_persistence=0.90,
            observation_weight=0.15,  # Sensitive to observations
        )
        
        brain_config = BrainConfig(
            brain_type="paladin",
            model_path=model_path,
            brain_influence=0.75,
            decision_interval=3,     # Quick threat response
            confidence_threshold=0.4,
        )
        
        super().__init__(agent_id, position, agent_config, brain_config)


class ChaosAgent(BrainAugmentedAgent):
    """
    Adversarial agent for red team simulation.
    
    Uses distilled creativity for vulnerability probing.
    """
    
    def __init__(
        self,
        agent_id: str,
        position: Optional[np.ndarray] = None,
        model_path: Optional[str] = None,
    ):
        agent_config = AgentConfig(
            energy_decay=0.06,       # Higher activity cost
            memory_persistence=0.85,  # Adapts quickly
            observation_weight=0.2,   # Highly reactive
        )
        
        brain_config = BrainConfig(
            brain_type="chaos",
            model_path=model_path,
            brain_influence=0.7,
            decision_interval=7,
            confidence_threshold=0.3,
        )
        
        super().__init__(agent_id, position, agent_config, brain_config)


class SwarmUnitAgent(BrainAugmentedAgent):
    """
    Basic swarm unit with fast Mamba-based decisions.
    
    Designed for large swarms with sub-millisecond inference.
    """
    
    def __init__(
        self,
        agent_id: str,
        position: Optional[np.ndarray] = None,
        model_path: Optional[str] = None,
    ):
        agent_config = AgentConfig(
            state_dim=8,
            energy_decay=0.05,
            memory_persistence=0.9,
        )
        
        brain_config = BrainConfig(
            brain_type="swarm_unit",
            model_path=model_path,
            brain_influence=0.6,
            decision_interval=1,     # Every step for fast response
            confidence_threshold=0.2,
        )
        
        super().__init__(agent_id, position, agent_config, brain_config)


def create_agent(
    agent_id: str,
    agent_type: str,
    position: Optional[np.ndarray] = None,
    model_path: Optional[str] = None,
) -> BrainAugmentedAgent:
    """
    Factory function to create agents of different types.
    
    Args:
        agent_id: Unique identifier
        agent_type: 'triumvirate', 'paladin', 'chaos', or 'swarm_unit'
        position: Initial position
        model_path: Path to distilled model (optional)
    
    Returns:
        Configured agent instance
    """
    agent_classes = {
        "triumvirate": TriumvirateAgent,
        "paladin": PaladinAgent,
        "chaos": ChaosAgent,
        "swarm_unit": SwarmUnitAgent,
    }
    
    if agent_type not in agent_classes:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agent_classes[agent_type](
        agent_id=agent_id,
        position=position,
        model_path=model_path,
    )
