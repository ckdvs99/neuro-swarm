"""
observations/visualize.py

Watch. Learn. Adjust.

You cannot debug what you cannot see.
You cannot understand what you do not watch.
Patience is methodology.

Inspired by:
- Scientific visualization
- Debugger interfaces
- Nature documentaries
"""

from __future__ import annotations
from typing import Optional, List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from neuro_swarm.environments.simple_field import SimpleField


class SwarmVisualizer:
    """
    Visualization tools for swarm observation.

    Principles embodied:
    - Observation before intervention (Principle 6)
    - We log everything; storage is cheap; insight is precious
    """

    def __init__(
        self,
        field: SimpleField,
        figsize: tuple = (10, 10),
        trail_length: int = 50
    ):
        self.field = field
        self.figsize = figsize
        self.trail_length = trail_length

        # History for trails
        self.position_history: List[np.ndarray] = []

        # Lazy import matplotlib
        self._plt = None
        self._fig = None
        self._ax = None

    def _setup_plot(self):
        """Initialize matplotlib figure."""
        import matplotlib.pyplot as plt
        self._plt = plt

        self._fig, self._ax = plt.subplots(figsize=self.figsize)
        self._ax.set_xlim(self.field.config.bounds)
        self._ax.set_ylim(self.field.config.bounds)
        self._ax.set_aspect('equal')
        self._ax.set_facecolor('#1a1a2e')
        self._fig.patch.set_facecolor('#16213e')

    def record_frame(self) -> None:
        """Record current positions for trail rendering."""
        positions = self.field.get_positions()
        self.position_history.append(positions.copy())

        # Trim to trail length
        if len(self.position_history) > self.trail_length:
            self.position_history = self.position_history[-self.trail_length:]

    def render(
        self,
        show_trails: bool = True,
        show_velocities: bool = True,
        show_energy: bool = True,
        show_substrate: bool = True
    ) -> None:
        """
        Render current state of the swarm.

        Options:
        - show_trails: Show movement history
        - show_velocities: Show heading arrows
        - show_energy: Color by energy level
        - show_substrate: Show stigmergic traces
        """
        if self._plt is None:
            self._setup_plot()

        self._ax.clear()
        self._ax.set_xlim(self.field.config.bounds)
        self._ax.set_ylim(self.field.config.bounds)
        self._ax.set_facecolor('#1a1a2e')

        # Substrate visualization
        if show_substrate and self.field.substrate is not None:
            self._render_substrate()

        # Trails
        if show_trails and len(self.position_history) > 1:
            self._render_trails()

        # Agents
        positions = self.field.get_positions()
        velocities = self.field.get_velocities()
        energies = self.field.get_energies()

        if len(positions) == 0:
            return

        # Color by energy
        if show_energy:
            colors = self._plt.cm.viridis(energies)
        else:
            colors = '#4cc9f0'

        # Plot agents
        self._ax.scatter(
            positions[:, 0], positions[:, 1],
            c=colors if show_energy else colors,
            s=50, alpha=0.8, edgecolors='white', linewidths=0.5
        )

        # Velocity arrows
        if show_velocities:
            self._ax.quiver(
                positions[:, 0], positions[:, 1],
                velocities[:, 0], velocities[:, 1],
                color='#f72585', alpha=0.6, scale=20
            )

        # Info text
        self._ax.set_title(
            f"Time: {self.field.time} | Agents: {len(self.field.agents)} | "
            f"Mean Energy: {energies.mean():.2f}",
            color='white', fontsize=12
        )

        self._plt.pause(0.01)

    def _render_trails(self) -> None:
        """Render movement trails with fading."""
        n_frames = len(self.position_history)
        for i, positions in enumerate(self.position_history[:-1]):
            next_positions = self.position_history[i + 1]
            alpha = (i + 1) / n_frames * 0.3

            for j in range(len(positions)):
                if j < len(next_positions):
                    self._ax.plot(
                        [positions[j, 0], next_positions[j, 0]],
                        [positions[j, 1], next_positions[j, 1]],
                        color='#4361ee', alpha=alpha, linewidth=1
                    )

    def _render_substrate(self) -> None:
        """Render stigmergic substrate as heatmap."""
        traces = self.field.substrate.traces
        # Sum across trace dimensions for visualization
        trace_magnitude = np.abs(traces).sum(axis=2)

        if trace_magnitude.max() > 0:
            self._ax.imshow(
                trace_magnitude.T,
                extent=[*self.field.config.bounds, *self.field.config.bounds],
                origin='lower',
                cmap='inferno',
                alpha=0.4,
                vmin=0,
                vmax=trace_magnitude.max()
            )

    def save_frame(self, path: str) -> None:
        """Save current frame to file."""
        if self._fig is not None:
            self._fig.savefig(path, dpi=150, facecolor=self._fig.get_facecolor())

    def close(self) -> None:
        """Close the visualization."""
        if self._plt is not None:
            self._plt.close(self._fig)


def animate_study(
    field: SimpleField,
    steps: int = 100,
    interval: int = 50,
    save_path: Optional[str] = None
) -> None:
    """
    Run and animate a study.

    This is the primary observation tool.
    Watch before you hypothesize.
    """
    viz = SwarmVisualizer(field)

    try:
        for _ in range(steps):
            field.step()
            viz.record_frame()
            viz.render()

        if save_path:
            viz.save_frame(save_path)

    finally:
        viz.close()
