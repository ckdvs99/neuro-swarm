"""
core/substrate.py

The environment as memory. Stigmergy over direct communication.

The ant doesn't say "food is north."
The ant walks north, leaving a trace. Others follow the trace.
The message IS the behavior.

Inspired by:
- Ant pheromone trails
- Neural synaptic traces
- Environmental affordances
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class SubstrateConfig:
    """Configuration for the stigmergic substrate."""
    grid_size: Tuple[int, int] = (100, 100)     # Spatial resolution
    trace_dim: int = 4                           # Dimensionality of traces
    decay_rate: float = 0.95                     # How quickly traces fade
    diffusion_rate: float = 0.1                  # How traces spread spatially
    world_bounds: Tuple[float, float] = (-50, 50)  # Physical world coordinates


class Substrate:
    """
    The stigmergic communication layer.

    Agents read and write traces to this shared medium.
    Information persists, decays, and diffuses over time.

    Principles embodied:
    - The medium is the message (Principle 3)
    - Locality is honesty (Principle 4) - only local reads/writes
    """

    def __init__(self, config: Optional[SubstrateConfig] = None):
        self.config = config or SubstrateConfig()

        # Initialize trace field
        # Shape: (grid_x, grid_y, trace_dim)
        self.traces = np.zeros(
            (*self.config.grid_size, self.config.trace_dim),
            dtype=np.float64
        )

        # For coordinate mapping
        self._world_min = self.config.world_bounds[0]
        self._world_max = self.config.world_bounds[1]
        self._world_range = self._world_max - self._world_min

    def read(self, position: np.ndarray) -> np.ndarray:
        """
        Read the trace at a given position.

        Returns the trace vector at the nearest grid cell.
        Agents can only read locally - no global access.
        """
        grid_pos = self._world_to_grid(position)
        return self.traces[grid_pos[0], grid_pos[1]].copy()

    def write(self, position: np.ndarray, trace: np.ndarray) -> None:
        """
        Write a trace at a given position.

        Trace is added (accumulated) to existing trace.
        This allows multiple agents to contribute to the same location.
        """
        grid_pos = self._world_to_grid(position)
        trace_bounded = np.clip(trace[:self.config.trace_dim], -1, 1)

        # Pad if trace is too short
        if len(trace_bounded) < self.config.trace_dim:
            trace_bounded = np.pad(
                trace_bounded,
                (0, self.config.trace_dim - len(trace_bounded))
            )

        # Accumulate (saturating)
        self.traces[grid_pos[0], grid_pos[1]] = np.clip(
            self.traces[grid_pos[0], grid_pos[1]] + trace_bounded,
            -5, 5
        )

    def step(self) -> None:
        """
        Advance the substrate by one time step.

        - Traces decay (fade over time)
        - Traces diffuse (spread to neighbors)

        This creates temporal and spatial dynamics in the medium.
        """
        # Decay
        self.traces *= self.config.decay_rate

        # Simple diffusion (average with neighbors)
        if self.config.diffusion_rate > 0:
            self._diffuse()

    def _diffuse(self) -> None:
        """Apply diffusion to spread traces spatially."""
        kernel_weight = self.config.diffusion_rate / 4
        center_weight = 1 - self.config.diffusion_rate

        new_traces = center_weight * self.traces.copy()

        # Add contributions from 4-neighbors
        new_traces[1:, :, :] += kernel_weight * self.traces[:-1, :, :]
        new_traces[:-1, :, :] += kernel_weight * self.traces[1:, :, :]
        new_traces[:, 1:, :] += kernel_weight * self.traces[:, :-1, :]
        new_traces[:, :-1, :] += kernel_weight * self.traces[:, 1:, :]

        self.traces = new_traces

    def _world_to_grid(self, position: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        # Clamp to world bounds
        pos_clamped = np.clip(position[:2], self._world_min, self._world_max - 0.001)

        # Normalize to [0, 1]
        pos_normalized = (pos_clamped - self._world_min) / self._world_range

        # Scale to grid
        grid_x = int(pos_normalized[0] * self.config.grid_size[0])
        grid_y = int(pos_normalized[1] * self.config.grid_size[1])

        # Safety clamp
        grid_x = max(0, min(self.config.grid_size[0] - 1, grid_x))
        grid_y = max(0, min(self.config.grid_size[1] - 1, grid_y))

        return (grid_x, grid_y)

    def get_total_trace_magnitude(self) -> float:
        """Total magnitude of all traces (for monitoring)."""
        return float(np.abs(self.traces).sum())

    def __repr__(self) -> str:
        return (
            f"Substrate(grid={self.config.grid_size}, "
            f"total_trace={self.get_total_trace_magnitude():.2f})"
        )
