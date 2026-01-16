"""
protocols/tension.py

Opposing forces for balance.

A system in balance is antifragile.
Inhibition is as important as excitation.
Rest is as important as action.

The Paladin needs the Chaos agent. The Chaos agent needs the Paladin.
Without tension, there is no stability—only stagnation.

Inspired by:
- Excitatory/inhibitory balance in cortex
- Predator-prey dynamics
- Yin-yang philosophy
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import numpy as np


class ForceType(Enum):
    """Types of forces in the tension system."""
    COHESION = "cohesion"       # Pull together
    SEPARATION = "separation"   # Push apart
    ALIGNMENT = "alignment"     # Match direction
    EXPLORATION = "exploration" # Seek novelty
    EXPLOITATION = "exploitation"  # Use known
    PRESERVATION = "preservation"  # Maintain state
    DISRUPTION = "disruption"   # Change state


@dataclass
class Force:
    """A single force acting on an agent."""
    force_type: ForceType
    direction: np.ndarray
    magnitude: float
    source_id: Optional[str] = None


class TensionResolver(ABC):
    """
    Resolves competing forces into balanced action.

    Principles embodied:
    - Balance over power (Principle 2)
    - Every force has a counterforce
    """

    @abstractmethod
    def resolve(self, forces: List[Force]) -> np.ndarray:
        """Resolve multiple forces into a single action vector."""
        pass


class LinearTension(TensionResolver):
    """
    Simple weighted sum of forces.

    Each force type has a weight; resolution is linear combination.
    """

    def __init__(self, weights: Optional[dict] = None):
        self.weights = weights or {
            ForceType.COHESION: 1.0,
            ForceType.SEPARATION: 1.5,
            ForceType.ALIGNMENT: 1.0,
            ForceType.EXPLORATION: 0.5,
            ForceType.EXPLOITATION: 0.5,
            ForceType.PRESERVATION: 0.8,
            ForceType.DISRUPTION: 0.3,
        }

    def resolve(self, forces: List[Force]) -> np.ndarray:
        """Weighted sum of force vectors."""
        if not forces:
            return np.zeros(2)

        result = np.zeros_like(forces[0].direction)
        total_weight = 0.0

        for force in forces:
            weight = self.weights.get(force.force_type, 1.0) * force.magnitude
            result += weight * force.direction
            total_weight += weight

        if total_weight > 0:
            result /= total_weight

        return result


class DynamicTension(TensionResolver):
    """
    Context-dependent force resolution.

    Weights adapt based on agent state and environment.

    Implements Principle 2: Balance over power — the system
    dynamically balances forces based on context.
    """

    def __init__(self):
        self.base_weights = {
            ForceType.COHESION: 1.0,
            ForceType.SEPARATION: 1.5,
            ForceType.ALIGNMENT: 1.0,
            ForceType.EXPLORATION: 0.5,
            ForceType.EXPLOITATION: 0.5,
            ForceType.PRESERVATION: 0.8,
            ForceType.DISRUPTION: 0.3,
        }

    def _compute_dynamic_weights(
        self,
        energy: float,
        threat_level: float
    ) -> Dict[ForceType, float]:
        """
        Compute context-dependent weights.

        State-dependent modulation:
        - Low energy: seek safety (cohesion, preservation)
        - High threat: defensive posture (separation, preservation)
        - High energy + low threat: opportunity for exploration
        - Medium states: balanced behavior
        """
        weights = self.base_weights.copy()

        # Clamp inputs
        energy = np.clip(energy, 0.0, 1.0)
        threat_level = np.clip(threat_level, 0.0, 1.0)

        # Low energy modulation
        # When tired, stay close to others and preserve state
        fatigue = 1.0 - energy
        weights[ForceType.COHESION] *= (1.0 + 0.5 * fatigue)
        weights[ForceType.PRESERVATION] *= (1.0 + 0.8 * fatigue)
        weights[ForceType.EXPLORATION] *= (1.0 - 0.7 * fatigue)
        weights[ForceType.DISRUPTION] *= (1.0 - 0.9 * fatigue)

        # Threat modulation
        # Under threat, separate and preserve but stay somewhat aligned
        weights[ForceType.SEPARATION] *= (1.0 + 1.0 * threat_level)
        weights[ForceType.PRESERVATION] *= (1.0 + 0.5 * threat_level)
        weights[ForceType.ALIGNMENT] *= (1.0 + 0.3 * threat_level)
        weights[ForceType.EXPLORATION] *= (1.0 - 0.8 * threat_level)
        weights[ForceType.DISRUPTION] *= (1.0 - 0.5 * threat_level)

        # Opportunity modulation (high energy, low threat)
        opportunity = energy * (1.0 - threat_level)
        weights[ForceType.EXPLORATION] *= (1.0 + 1.5 * opportunity)
        weights[ForceType.EXPLOITATION] *= (1.0 + 0.5 * opportunity)

        return weights

    def resolve(
        self,
        forces: List[Force],
        energy: float = 1.0,
        threat_level: float = 0.0
    ) -> np.ndarray:
        """
        Resolve forces with context-dependent weighting.

        Args:
            forces: List of forces acting on the agent
            energy: Agent's current energy level [0, 1]
            threat_level: Perceived threat in environment [0, 1]

        Returns:
            Resolved action vector (normalized weighted sum)

        Behavior:
        - Low energy: favor cohesion and preservation (stay safe)
        - High threat: favor separation and preservation (flee/defend)
        - High energy, low threat: favor exploration (opportunity)
        """
        if not forces:
            return np.zeros(2)

        # Get context-dependent weights
        weights = self._compute_dynamic_weights(energy, threat_level)

        result = np.zeros_like(forces[0].direction)
        total_weight = 0.0

        for force in forces:
            weight = weights.get(force.force_type, 1.0) * force.magnitude
            result += weight * force.direction
            total_weight += weight

        if total_weight > 0:
            result /= total_weight

        return result


class PaladinChaosBalance:
    """
    The fundamental tension: preservation vs disruption.

    Paladin agents preserve stability.
    Chaos agents test resilience.
    Neither is complete without the other.
    """

    def __init__(
        self,
        preservation_strength: float = 1.0,
        disruption_strength: float = 0.3
    ):
        self.preservation_strength = preservation_strength
        self.disruption_strength = disruption_strength

    def paladin_force(
        self,
        current_state: np.ndarray,
        target_state: np.ndarray
    ) -> Force:
        """
        Compute preserving force toward target state.

        Paladins pull the system toward stability.
        """
        direction = target_state - current_state
        magnitude = np.linalg.norm(direction)
        if magnitude > 0:
            direction = direction / magnitude

        return Force(
            force_type=ForceType.PRESERVATION,
            direction=direction,
            magnitude=self.preservation_strength * magnitude
        )

    def chaos_force(
        self,
        current_state: np.ndarray,
        perturbation_scale: float = 0.1
    ) -> Force:
        """
        Compute disrupting force (random perturbation).

        Chaos agents probe the system's resilience.
        """
        direction = np.random.randn(*current_state.shape)
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        return Force(
            force_type=ForceType.DISRUPTION,
            direction=direction,
            magnitude=self.disruption_strength * perturbation_scale
        )
