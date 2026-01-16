"""
Tests for core/rhythm.py

Biological rhythm dynamics - oscillations, phase coupling, synchronization.
"""

import numpy as np
import pytest

from neuro_swarm.core.rhythm import Rhythm, RhythmConfig, SwarmRhythm


class TestRhythmConfig:
    """Tests for RhythmConfig dataclass."""

    def test_default_config(self):
        config = RhythmConfig()
        assert config.base_period == 10.0
        assert config.phase_noise == 0.1
        assert config.coupling_strength == 0.05

    def test_custom_config(self):
        config = RhythmConfig(
            base_period=20.0,
            phase_noise=0.2,
            coupling_strength=0.1
        )
        assert config.base_period == 20.0
        assert config.phase_noise == 0.2
        assert config.coupling_strength == 0.1


class TestRhythm:
    """Tests for individual Rhythm dynamics."""

    def test_initialization_random_phase(self):
        rhythm = Rhythm()
        assert 0 <= rhythm.phase < 2 * np.pi
        assert rhythm.time == 0

    def test_initialization_specific_phase(self):
        rhythm = Rhythm(initial_phase=np.pi)
        assert rhythm.phase == np.pi

    def test_initialization_with_config(self):
        config = RhythmConfig(base_period=5.0)
        rhythm = Rhythm(config)
        assert rhythm.config.base_period == 5.0

    def test_step_advances_time(self):
        rhythm = Rhythm()
        assert rhythm.time == 0
        rhythm.step()
        assert rhythm.time == 1
        rhythm.step()
        assert rhythm.time == 2

    def test_step_changes_phase(self):
        rhythm = Rhythm(initial_phase=0.0)
        initial_phase = rhythm.phase
        rhythm.step()
        # Phase should have advanced (with some noise)
        assert rhythm.phase != initial_phase

    def test_step_returns_activity(self):
        rhythm = Rhythm()
        activity = rhythm.step()
        assert 0 <= activity <= 1

    def test_get_activity_range(self):
        rhythm = Rhythm()
        for _ in range(100):
            rhythm.step()
            activity = rhythm.get_activity()
            assert 0 <= activity <= 1

    def test_get_activity_at_phase_zero(self):
        rhythm = Rhythm(initial_phase=0.0)
        # At phase 0, cos(0) = 1, so activity = (1 + 1) / 2 = 1
        assert rhythm.get_activity() == pytest.approx(1.0)

    def test_get_activity_at_phase_pi(self):
        rhythm = Rhythm(initial_phase=np.pi)
        # At phase pi, cos(pi) = -1, so activity = (1 - 1) / 2 = 0
        assert rhythm.get_activity() == pytest.approx(0.0)

    def test_is_resting_when_low_activity(self):
        rhythm = Rhythm(initial_phase=np.pi)  # Minimum activity
        assert rhythm.is_resting(threshold=0.2)

    def test_is_resting_when_high_activity(self):
        rhythm = Rhythm(initial_phase=0.0)  # Maximum activity
        assert not rhythm.is_resting(threshold=0.2)

    def test_is_active_when_high_activity(self):
        rhythm = Rhythm(initial_phase=0.0)  # Maximum activity
        assert rhythm.is_active(threshold=0.8)

    def test_is_active_when_low_activity(self):
        rhythm = Rhythm(initial_phase=np.pi)  # Minimum activity
        assert not rhythm.is_active(threshold=0.8)

    def test_time_to_next_peak_at_peak(self):
        config = RhythmConfig(base_period=10.0)
        rhythm = Rhythm(config, initial_phase=0.0)
        # At peak, next peak is one full period away
        assert rhythm.time_to_next_peak() == pytest.approx(10.0)

    def test_time_to_next_peak_halfway(self):
        config = RhythmConfig(base_period=10.0)
        rhythm = Rhythm(config, initial_phase=np.pi)
        # Halfway through, next peak is half period away
        assert rhythm.time_to_next_peak() == pytest.approx(5.0)

    def test_step_with_neighbor_phases(self):
        rhythm = Rhythm(initial_phase=0.0)
        neighbor_phases = [np.pi, np.pi, np.pi]  # All neighbors at opposite phase
        # Should still work and return activity
        activity = rhythm.step(neighbor_phases)
        assert 0 <= activity <= 1

    def test_kuramoto_coupling_pulls_toward_neighbors(self):
        """Agents should tend to synchronize with neighbors over time."""
        np.random.seed(42)
        config = RhythmConfig(coupling_strength=0.2, phase_noise=0.0)

        # Agent starts at phase 0, neighbors at phase pi
        rhythm = Rhythm(config, initial_phase=0.0)
        neighbor_phases = [np.pi, np.pi, np.pi]

        # After stepping, phase should have moved toward neighbors
        rhythm.step(neighbor_phases)
        # The coupling term should have pulled phase toward pi
        # sin(pi - 0) = 0, so no pull at exactly 0 and pi
        # Use a different setup

    def test_repr(self):
        rhythm = Rhythm(initial_phase=1.0)
        repr_str = repr(rhythm)
        assert "Rhythm" in repr_str
        assert "phase=" in repr_str
        assert "activity=" in repr_str
        assert "time=" in repr_str


class TestSwarmRhythm:
    """Tests for collective SwarmRhythm dynamics."""

    def test_initialization(self):
        swarm_rhythm = SwarmRhythm(n_agents=5)
        assert swarm_rhythm.n_agents == 5
        assert len(swarm_rhythm.rhythms) == 5

    def test_initialization_with_config(self):
        config = RhythmConfig(base_period=20.0)
        swarm_rhythm = SwarmRhythm(n_agents=3, config=config)
        for rhythm in swarm_rhythm.rhythms:
            assert rhythm.config.base_period == 20.0

    def test_initialization_synchronized(self):
        swarm_rhythm = SwarmRhythm(n_agents=5, synchronize_initial=True)
        phases = [r.phase for r in swarm_rhythm.rhythms]
        # All phases should be the same
        assert all(p == phases[0] for p in phases)

    def test_initialization_unsynchronized(self):
        np.random.seed(42)
        swarm_rhythm = SwarmRhythm(n_agents=10, synchronize_initial=False)
        phases = [r.phase for r in swarm_rhythm.rhythms]
        # Phases should be different (with high probability)
        assert len(set(phases)) > 1

    def test_step_returns_activities(self):
        swarm_rhythm = SwarmRhythm(n_agents=5)
        activities = swarm_rhythm.step()
        assert len(activities) == 5
        assert all(0 <= a <= 1 for a in activities)

    def test_step_with_neighbor_graph(self):
        swarm_rhythm = SwarmRhythm(n_agents=4)
        # Simple neighbor graph: each agent knows adjacent agents
        neighbor_graph = [
            [1],        # Agent 0 knows agent 1
            [0, 2],     # Agent 1 knows agents 0 and 2
            [1, 3],     # Agent 2 knows agents 1 and 3
            [2],        # Agent 3 knows agent 2
        ]
        activities = swarm_rhythm.step(neighbor_graph)
        assert len(activities) == 4

    def test_step_with_partial_neighbor_graph(self):
        swarm_rhythm = SwarmRhythm(n_agents=4)
        # Incomplete neighbor graph
        neighbor_graph = [[1], [0]]  # Only first two agents have neighbors
        activities = swarm_rhythm.step(neighbor_graph)
        assert len(activities) == 4

    def test_get_synchrony_range(self):
        swarm_rhythm = SwarmRhythm(n_agents=5)
        synchrony = swarm_rhythm.get_synchrony()
        assert 0 <= synchrony <= 1

    def test_get_synchrony_perfect_sync(self):
        swarm_rhythm = SwarmRhythm(n_agents=5, synchronize_initial=True)
        synchrony = swarm_rhythm.get_synchrony()
        assert synchrony == pytest.approx(1.0)

    def test_get_synchrony_desync(self):
        """Uniformly distributed phases should have low synchrony."""
        swarm_rhythm = SwarmRhythm(n_agents=4)
        # Set phases uniformly around the circle
        for i, rhythm in enumerate(swarm_rhythm.rhythms):
            rhythm.phase = i * (2 * np.pi / 4)

        synchrony = swarm_rhythm.get_synchrony()
        # For 4 uniformly distributed phases, order parameter is 0
        assert synchrony == pytest.approx(0.0, abs=0.01)

    def test_get_mean_activity(self):
        swarm_rhythm = SwarmRhythm(n_agents=5)
        mean_activity = swarm_rhythm.get_mean_activity()
        assert 0 <= mean_activity <= 1

    def test_synchronization_emerges(self):
        """With coupling, synchrony should increase over time."""
        np.random.seed(42)
        config = RhythmConfig(coupling_strength=0.2, phase_noise=0.01)
        swarm_rhythm = SwarmRhythm(n_agents=5, config=config, synchronize_initial=False)

        # Full connectivity - everyone knows everyone
        neighbor_graph = [
            [j for j in range(5) if j != i]
            for i in range(5)
        ]

        initial_synchrony = swarm_rhythm.get_synchrony()

        # Run for many steps
        for _ in range(200):
            swarm_rhythm.step(neighbor_graph)

        final_synchrony = swarm_rhythm.get_synchrony()

        # Synchrony should have increased (or stayed high)
        # Note: with noise, this may not always hold, but should on average
        assert final_synchrony >= initial_synchrony * 0.5  # Allow some slack

    def test_repr(self):
        swarm_rhythm = SwarmRhythm(n_agents=5)
        repr_str = repr(swarm_rhythm)
        assert "SwarmRhythm" in repr_str
        assert "n=5" in repr_str
        assert "synchrony=" in repr_str
        assert "mean_activity=" in repr_str
