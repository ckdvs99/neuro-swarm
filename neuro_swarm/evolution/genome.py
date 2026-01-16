"""
neuro_swarm/evolution/genome.py

Genome representations for evolutionary optimization.

A genome encodes the parameters that define agent behavior.
The genome is the genotype; the resulting swarm behavior is the phenotype.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


class Genome(ABC):
    """
    Abstract base for genome representations.

    A genome must support:
    - Serialization (to_dict, from_dict) for distributed evaluation
    - Mutation for variation
    - Random initialization
    """

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize genome to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Genome":
        """Deserialize genome from dictionary."""
        pass

    @abstractmethod
    def mutate(self, rng: np.random.Generator) -> "Genome":
        """Return a mutated copy of this genome."""
        pass

    @classmethod
    @abstractmethod
    def random(cls, rng: np.random.Generator) -> "Genome":
        """Generate a random genome."""
        pass


@dataclass
class AgentGenome(Genome):
    """
    Genome for a single NeuroAgent's parameters.

    Maps directly to AgentConfig but with evolvable bounds.
    """

    # SSM-inspired parameters
    memory_persistence: float = 0.9    # A matrix analog (0.5-0.99)
    observation_weight: float = 0.1    # B matrix analog (0.01-0.3)

    # Energy dynamics
    energy_decay: float = 0.05         # Action cost (0.01-0.15)
    energy_recovery: float = 0.15      # Rest benefit (0.05-0.3)
    rest_threshold: float = 0.1        # Mandatory rest (0.05-0.2)

    # Behavior weights
    cohesion_weight: float = 0.3       # Pull toward neighbors (0-1)
    separation_weight: float = 0.5     # Push from neighbors (0-1)
    alignment_weight: float = 0.2      # Match neighbor velocity (0-1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "AgentGenome",
            "memory_persistence": self.memory_persistence,
            "observation_weight": self.observation_weight,
            "energy_decay": self.energy_decay,
            "energy_recovery": self.energy_recovery,
            "rest_threshold": self.rest_threshold,
            "cohesion_weight": self.cohesion_weight,
            "separation_weight": self.separation_weight,
            "alignment_weight": self.alignment_weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentGenome":
        return cls(
            memory_persistence=data.get("memory_persistence", 0.9),
            observation_weight=data.get("observation_weight", 0.1),
            energy_decay=data.get("energy_decay", 0.05),
            energy_recovery=data.get("energy_recovery", 0.15),
            rest_threshold=data.get("rest_threshold", 0.1),
            cohesion_weight=data.get("cohesion_weight", 0.3),
            separation_weight=data.get("separation_weight", 0.5),
            alignment_weight=data.get("alignment_weight", 0.2),
        )

    def mutate(self, rng: np.random.Generator, sigma: float = 0.1) -> "AgentGenome":
        """Gaussian mutation with bounds."""
        def mutate_param(val: float, low: float, high: float) -> float:
            new_val = val + rng.normal(0, sigma * (high - low))
            return float(np.clip(new_val, low, high))

        return AgentGenome(
            memory_persistence=mutate_param(self.memory_persistence, 0.5, 0.99),
            observation_weight=mutate_param(self.observation_weight, 0.01, 0.3),
            energy_decay=mutate_param(self.energy_decay, 0.01, 0.15),
            energy_recovery=mutate_param(self.energy_recovery, 0.05, 0.3),
            rest_threshold=mutate_param(self.rest_threshold, 0.05, 0.2),
            cohesion_weight=mutate_param(self.cohesion_weight, 0.0, 1.0),
            separation_weight=mutate_param(self.separation_weight, 0.0, 1.0),
            alignment_weight=mutate_param(self.alignment_weight, 0.0, 1.0),
        )

    @classmethod
    def random(cls, rng: np.random.Generator) -> "AgentGenome":
        return cls(
            memory_persistence=rng.uniform(0.5, 0.99),
            observation_weight=rng.uniform(0.01, 0.3),
            energy_decay=rng.uniform(0.01, 0.15),
            energy_recovery=rng.uniform(0.05, 0.3),
            rest_threshold=rng.uniform(0.05, 0.2),
            cohesion_weight=rng.uniform(0.0, 1.0),
            separation_weight=rng.uniform(0.0, 1.0),
            alignment_weight=rng.uniform(0.0, 1.0),
        )


@dataclass
class SwarmGenome(Genome):
    """
    Genome for an entire swarm configuration.

    Includes agent parameters plus swarm-level parameters.
    """

    agent_genome: AgentGenome = field(default_factory=AgentGenome)

    # Swarm-level parameters
    num_agents: int = 7                # Default to Miller's number
    substrate_decay: float = 0.95      # How fast traces fade (0.8-0.99)
    substrate_diffusion: float = 0.1   # How fast traces spread (0.01-0.3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "SwarmGenome",
            "agent_genome": self.agent_genome.to_dict(),
            "num_agents": self.num_agents,
            "substrate_decay": self.substrate_decay,
            "substrate_diffusion": self.substrate_diffusion,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SwarmGenome":
        agent_data = data.get("agent_genome", {})
        return cls(
            agent_genome=AgentGenome.from_dict(agent_data),
            num_agents=data.get("num_agents", 7),
            substrate_decay=data.get("substrate_decay", 0.95),
            substrate_diffusion=data.get("substrate_diffusion", 0.1),
        )

    def mutate(self, rng: np.random.Generator, sigma: float = 0.1) -> "SwarmGenome":
        return SwarmGenome(
            agent_genome=self.agent_genome.mutate(rng, sigma),
            num_agents=self.num_agents,  # Don't mutate agent count
            substrate_decay=float(np.clip(
                self.substrate_decay + rng.normal(0, sigma * 0.19),
                0.8, 0.99
            )),
            substrate_diffusion=float(np.clip(
                self.substrate_diffusion + rng.normal(0, sigma * 0.29),
                0.01, 0.3
            )),
        )

    @classmethod
    def random(cls, rng: np.random.Generator) -> "SwarmGenome":
        return cls(
            agent_genome=AgentGenome.random(rng),
            num_agents=7,
            substrate_decay=rng.uniform(0.8, 0.99),
            substrate_diffusion=rng.uniform(0.01, 0.3),
        )


def crossover(
    parent1: Genome,
    parent2: Genome,
    rng: np.random.Generator
) -> Genome:
    """
    Uniform crossover between two genomes.

    Each parameter is randomly selected from either parent.
    """
    if type(parent1) != type(parent2):
        raise ValueError("Cannot crossover different genome types")

    d1 = parent1.to_dict()
    d2 = parent2.to_dict()

    def crossover_dict(dict1: Dict, dict2: Dict) -> Dict:
        result = {}
        for key in dict1:
            if key == "type":
                result[key] = dict1[key]
            elif isinstance(dict1[key], dict):
                result[key] = crossover_dict(dict1[key], dict2[key])
            else:
                result[key] = dict1[key] if rng.random() < 0.5 else dict2[key]
        return result

    child_dict = crossover_dict(d1, d2)
    return type(parent1).from_dict(child_dict)
