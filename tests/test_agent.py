"""
Tests for the NeuroAgent core component.

Every study is a test. Observation is methodology.
"""

import numpy as np
import pytest

from neuro_swarm.core.agent import NeuroAgent, AgentConfig, AgentState


class TestAgentState:
    """Tests for AgentState dataclass."""

    def test_state_initialization(self):
        """Agent state initializes with correct types."""
        state = AgentState(
            position=np.array([1.0, 2.0]),
            velocity=np.array([0.1, 0.2]),
            internal=np.zeros(8),
        )
        assert state.position.dtype == np.float64
        assert state.velocity.dtype == np.float64
        assert state.energy == 1.0
        assert state.age == 0

    def test_state_from_lists(self):
        """Agent state converts lists to numpy arrays."""
        state = AgentState(
            position=[1.0, 2.0],
            velocity=[0.0, 0.0],
            internal=[0] * 8,
        )
        assert isinstance(state.position, np.ndarray)
        assert isinstance(state.velocity, np.ndarray)


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_config(self):
        """Default config has expected values."""
        config = AgentConfig()
        assert config.state_dim == 8
        assert config.neighbor_limit == 7
        assert config.rest_threshold == 0.1

    def test_custom_config(self):
        """Custom config values are respected."""
        config = AgentConfig(
            state_dim=16,
            neighbor_limit=5,
            energy_decay=0.1,
        )
        assert config.state_dim == 16
        assert config.neighbor_limit == 5
        assert config.energy_decay == 0.1


class TestNeuroAgent:
    """Tests for NeuroAgent."""

    def test_agent_creation(self):
        """Agent creates with valid state."""
        agent = NeuroAgent("test_agent")
        assert agent.id == "test_agent"
        assert agent.state.energy == 1.0
        assert len(agent.state.internal) == 8

    def test_agent_with_position(self):
        """Agent respects initial position."""
        pos = np.array([10.0, -5.0])
        agent = NeuroAgent("positioned", position=pos)
        np.testing.assert_array_equal(agent.state.position, pos)

    def test_agent_with_custom_config(self):
        """Agent uses custom config."""
        config = AgentConfig(state_dim=16)
        agent = NeuroAgent("custom", config=config)
        assert len(agent.state.internal) == 16

    def test_perceive_no_neighbors(self):
        """Agent perceives empty neighborhood."""
        agent = NeuroAgent("lonely")
        observation = agent.perceive([], None)
        assert len(observation) == 10  # 6 neighbor + 4 substrate

    def test_perceive_with_neighbors(self):
        """Agent perceives neighbors correctly."""
        agent = NeuroAgent("observer", position=np.array([0.0, 0.0]))
        neighbor_states = [
            AgentState(
                position=np.array([1.0, 0.0]),
                velocity=np.array([0.1, 0.0]),
                internal=np.zeros(8),
            )
        ]
        observation = agent.perceive(neighbor_states, None)
        assert len(observation) == 10
        # Relative position should be [1, 0]
        assert observation[0] == pytest.approx(1.0)

    def test_update_decreases_energy(self):
        """Update step costs energy."""
        agent = NeuroAgent("active")
        initial_energy = agent.state.energy
        observation = np.zeros(10)
        agent.update(observation)
        assert agent.state.energy < initial_energy

    def test_rest_when_depleted(self):
        """Agent rests when energy is low."""
        config = AgentConfig(rest_threshold=0.5)
        agent = NeuroAgent("tired", config=config)
        agent.state.energy = 0.1  # Below threshold

        initial_energy = agent.state.energy
        agent.update(np.zeros(10))

        # Should have recovered energy, not depleted it
        assert agent.state.energy > initial_energy

    def test_act_produces_movement(self):
        """Action produces velocity change."""
        agent = NeuroAgent("mover")
        # Give it some internal state to produce motion
        agent.state.internal[:2] = [1.0, 0.5]
        velocity = agent.act()
        assert np.linalg.norm(velocity) > 0

    def test_act_limited_when_depleted(self):
        """Depleted agent doesn't move."""
        config = AgentConfig(rest_threshold=0.5)
        agent = NeuroAgent("exhausted", config=config)
        agent.state.energy = 0.1
        velocity = agent.act()
        np.testing.assert_array_equal(velocity, [0, 0])

    def test_history_recording(self):
        """Agent records history when enabled."""
        agent = NeuroAgent("historian")
        agent.record_history = True

        for _ in range(5):
            agent.update(np.zeros(10))

        assert len(agent.history) == 5

    def test_distance_to(self):
        """Distance calculation is correct."""
        agent1 = NeuroAgent("a", position=np.array([0.0, 0.0]))
        agent2 = NeuroAgent("b", position=np.array([3.0, 4.0]))
        assert agent1.distance_to(agent2) == pytest.approx(5.0)
