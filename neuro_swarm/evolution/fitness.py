"""
neuro_swarm/evolution/fitness.py

Fitness functions and behavioral descriptors for swarm evaluation.

Fitness is what we optimize. Behavior is what we measure.
In quality-diversity algorithms, we optimize fitness within behavioral niches.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Callable, Optional
import numpy as np


@dataclass
class EvaluationResult:
    """
    Result of evaluating a genome.

    Contains both fitness (how good) and behavior (what kind).
    Optionally includes the genome that produced this result (for MAP-Elites).
    """

    fitness: float
    behavior: np.ndarray  # Behavioral descriptor vector
    metadata: Dict[str, Any] = field(default_factory=dict)
    genome: Optional[Any] = None  # Optional genome reference for archive storage

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "fitness": self.fitness,
            "behavior": self.behavior.tolist(),
            "metadata": self.metadata,
        }
        if self.genome is not None and hasattr(self.genome, "to_dict"):
            result["genome"] = self.genome.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        return cls(
            fitness=data["fitness"],
            behavior=np.array(data["behavior"]),
            metadata=data.get("metadata", {}),
            genome=data.get("genome"),  # Will be dict, needs reconstruction if used
        )


@dataclass
class BehavioralDescriptor:
    """
    Defines the behavioral space for quality-diversity.

    Each dimension captures a different aspect of swarm behavior.
    """

    names: List[str]  # Names of behavioral dimensions
    bounds: List[Tuple[float, float]]  # (min, max) for each dimension

    @property
    def dimensions(self) -> int:
        return len(self.names)

    def validate(self, behavior: np.ndarray) -> bool:
        """Check if behavior is within bounds."""
        if len(behavior) != self.dimensions:
            return False
        for i, (low, high) in enumerate(self.bounds):
            if not (low <= behavior[i] <= high):
                return False
        return True


# Default behavioral descriptor for swarms
SWARM_BEHAVIOR_DESCRIPTOR = BehavioralDescriptor(
    names=["cohesion", "dispersion"],
    bounds=[(0.0, 1.0), (0.0, 1.0)],
)


def discretize_behavior(
    behavior: np.ndarray,
    bounds: List[Tuple[float, float]],
    resolution: int,
) -> Tuple[int, ...]:
    """
    Map continuous behavior to discrete archive cell.

    Used by MAP-Elites to index the behavior archive.
    """
    indices = []
    for i, (low, high) in enumerate(bounds):
        # Normalize to [0, 1]
        normalized = (behavior[i] - low) / (high - low)
        # Clamp and discretize
        idx = int(np.clip(normalized * resolution, 0, resolution - 1))
        indices.append(idx)
    return tuple(indices)


class FitnessFunction(ABC):
    """
    Abstract base for fitness functions.

    A fitness function evaluates a swarm simulation and returns
    both a scalar fitness and a behavioral descriptor.
    """

    @abstractmethod
    def evaluate(
        self,
        trajectory: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> EvaluationResult:
        """
        Evaluate a swarm simulation.

        Args:
            trajectory: List of simulation states over time
            config: Genome configuration used

        Returns:
            EvaluationResult with fitness and behavior
        """
        pass

    @property
    @abstractmethod
    def behavior_descriptor(self) -> BehavioralDescriptor:
        """Return the behavioral descriptor for this fitness function."""
        pass


class SwarmCoherenceFitness(FitnessFunction):
    """
    Fitness based on swarm coherence and efficiency.

    Rewards:
    - Staying together (cohesion)
    - Moving as a group (alignment)
    - Energy efficiency

    Behavioral dimensions:
    - Average cohesion over time
    - Average dispersion over time
    """

    def __init__(
        self,
        cohesion_weight: float = 0.4,
        alignment_weight: float = 0.3,
        efficiency_weight: float = 0.3,
    ):
        self.cohesion_weight = cohesion_weight
        self.alignment_weight = alignment_weight
        self.efficiency_weight = efficiency_weight
        self._descriptor = BehavioralDescriptor(
            names=["cohesion", "dispersion"],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
        )

    @property
    def behavior_descriptor(self) -> BehavioralDescriptor:
        return self._descriptor

    def evaluate(
        self,
        trajectory: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> EvaluationResult:
        """Evaluate swarm coherence from trajectory."""
        if not trajectory:
            return EvaluationResult(
                fitness=0.0,
                behavior=np.array([0.0, 1.0]),
                metadata={"error": "empty trajectory"},
            )

        cohesions = []
        dispersions = []
        alignments = []
        energies = []

        for state in trajectory:
            positions = state.get("positions", [])
            velocities = state.get("velocities", [])
            agent_energies = state.get("energies", [])

            if len(positions) < 2:
                continue

            positions = np.array(positions)
            velocities = np.array(velocities)

            # Cohesion: inverse of average distance to centroid
            centroid = positions.mean(axis=0)
            distances = np.linalg.norm(positions - centroid, axis=1)
            avg_distance = distances.mean()
            cohesion = 1.0 / (1.0 + avg_distance)
            cohesions.append(cohesion)

            # Dispersion: standard deviation of positions
            dispersion = distances.std() / (avg_distance + 1e-6)
            dispersion = min(dispersion, 1.0)
            dispersions.append(dispersion)

            # Alignment: average velocity correlation
            if len(velocities) > 0 and np.any(velocities):
                norms = np.linalg.norm(velocities, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-6)
                normalized = velocities / norms
                avg_direction = normalized.mean(axis=0)
                alignment = np.linalg.norm(avg_direction)
                alignments.append(alignment)

            # Energy efficiency
            if agent_energies:
                energies.append(np.mean(agent_energies))

        # Compute final metrics
        avg_cohesion = np.mean(cohesions) if cohesions else 0.0
        avg_dispersion = np.mean(dispersions) if dispersions else 1.0
        avg_alignment = np.mean(alignments) if alignments else 0.0
        avg_energy = np.mean(energies) if energies else 0.5

        # Fitness is weighted combination
        fitness = (
            self.cohesion_weight * avg_cohesion
            + self.alignment_weight * avg_alignment
            + self.efficiency_weight * avg_energy
        )

        return EvaluationResult(
            fitness=fitness,
            behavior=np.array([avg_cohesion, avg_dispersion]),
            metadata={
                "cohesion": avg_cohesion,
                "dispersion": avg_dispersion,
                "alignment": avg_alignment,
                "energy": avg_energy,
                "steps": len(trajectory),
            },
        )


class TaskCompletionFitness(FitnessFunction):
    """
    Fitness based on task completion.

    For scenarios with explicit goals (e.g., reach target, patrol area).
    """

    def __init__(
        self,
        target_position: Optional[np.ndarray] = None,
        completion_radius: float = 5.0,
    ):
        self.target_position = target_position or np.array([50.0, 50.0])
        self.completion_radius = completion_radius
        self._descriptor = BehavioralDescriptor(
            names=["speed", "path_efficiency"],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
        )

    @property
    def behavior_descriptor(self) -> BehavioralDescriptor:
        return self._descriptor

    def evaluate(
        self,
        trajectory: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> EvaluationResult:
        """Evaluate task completion from trajectory."""
        if not trajectory:
            return EvaluationResult(
                fitness=0.0,
                behavior=np.array([0.0, 0.0]),
                metadata={"error": "empty trajectory"},
            )

        # Get start and end positions
        start_positions = np.array(trajectory[0].get("positions", [[0, 0]]))
        end_positions = np.array(trajectory[-1].get("positions", [[0, 0]]))

        start_centroid = start_positions.mean(axis=0)
        end_centroid = end_positions.mean(axis=0)

        # Distance to target
        final_distance = np.linalg.norm(end_centroid - self.target_position)
        initial_distance = np.linalg.norm(start_centroid - self.target_position)

        # Progress toward goal
        progress = max(0, initial_distance - final_distance) / (initial_distance + 1e-6)

        # Completion bonus
        completed = final_distance < self.completion_radius
        completion_bonus = 1.0 if completed else 0.0

        # Path efficiency (straight line vs actual path)
        total_movement = 0.0
        for i in range(1, len(trajectory)):
            prev = np.array(trajectory[i - 1].get("positions", []))
            curr = np.array(trajectory[i].get("positions", []))
            if len(prev) > 0 and len(curr) > 0:
                total_movement += np.linalg.norm(
                    curr.mean(axis=0) - prev.mean(axis=0)
                )

        direct_distance = np.linalg.norm(end_centroid - start_centroid)
        path_efficiency = direct_distance / (total_movement + 1e-6)
        path_efficiency = min(path_efficiency, 1.0)

        # Speed (normalized by steps)
        speed = direct_distance / (len(trajectory) + 1e-6)
        speed = min(speed / 2.0, 1.0)  # Normalize

        # Fitness
        fitness = 0.5 * progress + 0.3 * completion_bonus + 0.2 * path_efficiency

        return EvaluationResult(
            fitness=fitness,
            behavior=np.array([speed, path_efficiency]),
            metadata={
                "progress": progress,
                "completed": completed,
                "final_distance": final_distance,
                "path_efficiency": path_efficiency,
                "speed": speed,
            },
        )
