"""
core/rhythm.py

Biological systems have cycles: circadian, refractory, seasonal.
Constant vigilance is unsustainable and unnatural.
Our agents pulse. They breathe.

Inspired by:
- Circadian rhythms
- Neural oscillations (theta, gamma)
- Refractory periods
- Predator-prey cycles
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Callable
import numpy as np


@dataclass
class RhythmConfig:
    """Configuration for temporal dynamics."""
    base_period: float = 10.0          # Base oscillation period (time steps)
    phase_noise: float = 0.1           # Random phase variation
    coupling_strength: float = 0.05    # How strongly agents synchronize


class Rhythm:
    """
    Temporal dynamics manager for an agent or swarm.

    Manages:
    - Oscillatory activity levels
    - Phase relationships between agents
    - Refractory periods

    Principles embodied:
    - Rhythm, not reaction (Principle 5)
    - Balance over power (Principle 2)
    """

    def __init__(
        self,
        config: Optional[RhythmConfig] = None,
        initial_phase: Optional[float] = None
    ):
        self.config = config or RhythmConfig()

        # Phase in [0, 2*pi)
        if initial_phase is not None:
            self.phase = initial_phase
        else:
            self.phase = np.random.uniform(0, 2 * np.pi)

        self.time = 0

    def step(self, neighbor_phases: Optional[List[float]] = None) -> float:
        """
        Advance rhythm by one time step.

        Returns the current activity level [0, 1].

        If neighbor_phases provided, weakly couples to neighbors
        (Kuramoto-style synchronization).
        """
        self.time += 1

        # Base phase advance
        phase_increment = 2 * np.pi / self.config.base_period

        # Add noise
        phase_increment += np.random.normal(0, self.config.phase_noise)

        # Kuramoto coupling: tendency to synchronize with neighbors
        if neighbor_phases:
            coupling = 0.0
            for neighbor_phase in neighbor_phases:
                coupling += np.sin(neighbor_phase - self.phase)
            coupling *= self.config.coupling_strength / len(neighbor_phases)
            phase_increment += coupling

        # Update phase
        self.phase = (self.phase + phase_increment) % (2 * np.pi)

        return self.get_activity()

    def get_activity(self) -> float:
        """
        Current activity level based on phase.

        Returns value in [0, 1] where:
        - 1 = peak activity
        - 0 = minimum activity (rest)
        """
        # Cosine oscillation mapped to [0, 1]
        return (1 + np.cos(self.phase)) / 2

    def is_resting(self, threshold: float = 0.2) -> bool:
        """Check if currently in rest phase."""
        return self.get_activity() < threshold

    def is_active(self, threshold: float = 0.8) -> bool:
        """Check if currently in active phase."""
        return self.get_activity() > threshold

    def time_to_next_peak(self) -> float:
        """Estimate time steps until next activity peak."""
        # Peak is at phase = 0
        if self.phase == 0:
            return self.config.base_period

        remaining_phase = (2 * np.pi - self.phase) % (2 * np.pi)
        return remaining_phase / (2 * np.pi) * self.config.base_period

    def __repr__(self) -> str:
        return (
            f"Rhythm(phase={self.phase:.2f}, "
            f"activity={self.get_activity():.2f}, "
            f"time={self.time})"
        )


class SwarmRhythm:
    """
    Collective rhythm dynamics for a swarm.

    Manages phase relationships and synchronization
    across multiple agents.
    """

    def __init__(
        self,
        n_agents: int,
        config: Optional[RhythmConfig] = None,
        synchronize_initial: bool = False
    ):
        self.config = config or RhythmConfig()
        self.n_agents = n_agents

        if synchronize_initial:
            # All agents start in phase
            initial_phase = np.random.uniform(0, 2 * np.pi)
            self.rhythms = [
                Rhythm(config, initial_phase) for _ in range(n_agents)
            ]
        else:
            # Random initial phases
            self.rhythms = [Rhythm(config) for _ in range(n_agents)]

    def step(self, neighbor_graph: Optional[List[List[int]]] = None) -> List[float]:
        """
        Advance all rhythms by one step.

        neighbor_graph: List of neighbor indices for each agent

        Returns list of activity levels.
        """
        activities = []

        for i, rhythm in enumerate(self.rhythms):
            if neighbor_graph and i < len(neighbor_graph):
                neighbor_phases = [
                    self.rhythms[j].phase
                    for j in neighbor_graph[i]
                    if j < len(self.rhythms)
                ]
            else:
                neighbor_phases = None

            activity = rhythm.step(neighbor_phases)
            activities.append(activity)

        return activities

    def get_synchrony(self) -> float:
        """
        Measure of global phase synchrony.

        Returns value in [0, 1] where:
        - 1 = perfect synchronization
        - 0 = uniformly distributed phases

        Uses Kuramoto order parameter.
        """
        phases = np.array([r.phase for r in self.rhythms])
        complex_phases = np.exp(1j * phases)
        order_parameter = np.abs(complex_phases.mean())
        return float(order_parameter)

    def get_mean_activity(self) -> float:
        """Average activity across all agents."""
        return np.mean([r.get_activity() for r in self.rhythms])

    def __repr__(self) -> str:
        return (
            f"SwarmRhythm(n={self.n_agents}, "
            f"synchrony={self.get_synchrony():.2f}, "
            f"mean_activity={self.get_mean_activity():.2f})"
        )
