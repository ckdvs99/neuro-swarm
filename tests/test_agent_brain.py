"""
Tests for core/agent_brain.py

Brain-augmented agents with distilled model integration.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from neuro_swarm.core.agent_brain import (
    BrainConfig,
    BrainAugmentedAgent,
    TriumvirateAgent,
    PaladinAgent,
    ChaosAgent,
    SwarmUnitAgent,
    create_agent,
)
from neuro_swarm.core.agent import AgentState, AgentConfig


class TestBrainConfig:
    """Tests for BrainConfig dataclass."""

    def test_default_config(self):
        config = BrainConfig()
        assert config.brain_type is None
        assert config.model_path is None
        assert config.brain_influence == 0.7
        assert config.decision_interval == 10
        assert config.confidence_threshold == 0.3

    def test_custom_config(self):
        config = BrainConfig(
            brain_type="triumvirate",
            model_path="/path/to/model",
            brain_influence=0.9,
            decision_interval=5,
            confidence_threshold=0.5
        )
        assert config.brain_type == "triumvirate"
        assert config.model_path == "/path/to/model"
        assert config.brain_influence == 0.9
        assert config.decision_interval == 5
        assert config.confidence_threshold == 0.5


class TestBrainAugmentedAgent:
    """Tests for BrainAugmentedAgent without actual model loading."""

    def test_initialization_without_brain(self):
        agent = BrainAugmentedAgent("agent_0")

        assert agent.id == "agent_0"
        assert agent.brain is None
        assert agent.last_brain_decision is None
        assert agent.steps_since_brain_decision == 0

    def test_initialization_with_position(self):
        position = np.array([10.0, 20.0])
        agent = BrainAugmentedAgent("agent_0", position)

        assert np.allclose(agent.state.position, position)

    def test_initialization_with_config(self):
        agent_config = AgentConfig(energy_decay=0.1)
        brain_config = BrainConfig(brain_influence=0.5)

        agent = BrainAugmentedAgent(
            "agent_0",
            config=agent_config,
            brain_config=brain_config
        )

        assert agent.config.energy_decay == 0.1
        assert agent.brain_config.brain_influence == 0.5

    def test_should_consult_brain_no_brain(self):
        agent = BrainAugmentedAgent("agent_0")
        assert not agent._should_consult_brain()

    def test_should_consult_brain_interval(self):
        brain_config = BrainConfig(decision_interval=5)
        agent = BrainAugmentedAgent("agent_0", brain_config=brain_config)
        agent.brain = Mock()  # Fake brain

        # Initially should not consult (steps_since = 0)
        assert not agent._should_consult_brain()

        # After 5 steps, should consult
        agent.steps_since_brain_decision = 5
        assert agent._should_consult_brain()

    def test_get_observation_dict(self):
        agent = BrainAugmentedAgent("agent_0", np.array([1.0, 2.0]))
        agent.state.velocity = np.array([0.5, -0.5])
        agent.state.energy = 0.8

        obs_dict = agent._get_observation_dict([])

        assert obs_dict["position"] == [1.0, 2.0]
        assert obs_dict["velocity"] == [0.5, -0.5]
        assert obs_dict["energy"] == 0.8

    def test_get_context_dict(self):
        agent = BrainAugmentedAgent("agent_0", np.array([0.0, 0.0]))

        # Create mock neighbor states with required fields
        neighbor1 = AgentState(
            position=np.array([5.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            internal=np.zeros(8)
        )
        neighbor2 = AgentState(
            position=np.array([0.0, 5.0]),
            velocity=np.array([0.0, 0.0]),
            internal=np.zeros(8)
        )

        ctx_dict = agent._get_context_dict([neighbor1, neighbor2])

        assert ctx_dict["num_neighbors"] == 2
        assert len(ctx_dict["neighbors"]) == 2
        assert ctx_dict["neighbors"][0]["relative_position"] == [5.0, 0.0]

    def test_apply_brain_decision_direction(self):
        brain_config = BrainConfig(brain_influence=0.5)
        agent = BrainAugmentedAgent("agent_0", brain_config=brain_config)

        decision = {"direction": [1.0, 0.0], "intensity": 1.0}
        result = agent._apply_brain_decision(decision)

        # direction * intensity * brain_influence = [1, 0] * 1.0 * 0.5
        assert np.allclose(result, [0.5, 0.0])

    def test_apply_brain_decision_with_intensity(self):
        brain_config = BrainConfig(brain_influence=1.0)
        agent = BrainAugmentedAgent("agent_0", brain_config=brain_config)

        decision = {"direction": [1.0, 1.0], "intensity": 0.5}
        result = agent._apply_brain_decision(decision)

        assert np.allclose(result, [0.5, 0.5])

    def test_apply_brain_decision_defensive_action(self):
        agent = BrainAugmentedAgent("agent_0")

        decision = {"action": "defensive stance"}
        result = agent._apply_brain_decision(decision)

        # Defensive = hold position
        assert np.allclose(result, [0.0, 0.0])

    def test_apply_brain_decision_empty(self):
        agent = BrainAugmentedAgent("agent_0")

        decision = {}
        result = agent._apply_brain_decision(decision)

        assert np.allclose(result, [0.0, 0.0])

    def test_perceive_stores_neighbors(self):
        agent = BrainAugmentedAgent("agent_0")

        neighbor = AgentState(
            position=np.array([5.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            internal=np.zeros(8)
        )
        agent.perceive([neighbor], None)

        assert hasattr(agent, '_last_neighbors')
        assert len(agent._last_neighbors) == 1

    def test_update_increments_step_counter(self):
        agent = BrainAugmentedAgent("agent_0")
        initial_steps = agent.steps_since_brain_decision

        observation = np.zeros(agent.config.state_dim + 4)  # Match expected size
        agent.update(observation)

        assert agent.steps_since_brain_decision == initial_steps + 1

    def test_update_without_brain(self):
        agent = BrainAugmentedAgent("agent_0")

        # Should not raise even without brain
        observation = np.zeros(agent.config.state_dim + 4)
        agent.update(observation)

    def test_get_brain_stats_no_brain(self):
        agent = BrainAugmentedAgent("agent_0")

        stats = agent.get_brain_stats()

        assert stats["has_brain"] is False
        assert stats["brain_type"] is None
        assert stats["last_decision"] is None

    def test_get_brain_stats_with_config(self):
        brain_config = BrainConfig(brain_type="paladin", brain_influence=0.8)
        agent = BrainAugmentedAgent("agent_0", brain_config=brain_config)

        stats = agent.get_brain_stats()

        assert stats["brain_type"] == "paladin"
        assert stats["brain_influence"] == 0.8

    def test_repr_without_brain(self):
        agent = BrainAugmentedAgent("agent_0", np.array([1.0, 2.0]))
        repr_str = repr(agent)

        assert "BrainAugmentedAgent" in repr_str
        assert "agent_0" in repr_str
        assert "brain=" not in repr_str

    def test_repr_with_brain_type(self):
        brain_config = BrainConfig(brain_type="triumvirate")
        agent = BrainAugmentedAgent("agent_0", brain_config=brain_config)
        agent.brain = Mock()  # Fake brain

        repr_str = repr(agent)

        assert "brain=triumvirate" in repr_str


class TestTriumvirateAgent:
    """Tests for TriumvirateAgent specialized class."""

    def test_initialization(self):
        agent = TriumvirateAgent("coordinator_0")

        assert agent.id == "coordinator_0"
        assert agent.config.energy_decay == 0.03
        assert agent.config.memory_persistence == 0.95
        assert agent.brain_config.brain_type == "triumvirate"
        assert agent.brain_config.brain_influence == 0.8
        assert agent.brain_config.decision_interval == 5

    def test_initialization_with_position(self):
        position = np.array([0.0, 0.0])
        agent = TriumvirateAgent("coordinator_0", position)

        assert np.allclose(agent.state.position, position)


class TestPaladinAgent:
    """Tests for PaladinAgent specialized class."""

    def test_initialization(self):
        agent = PaladinAgent("defender_0")

        assert agent.id == "defender_0"
        assert agent.config.energy_decay == 0.04
        assert agent.config.observation_weight == 0.15
        assert agent.brain_config.brain_type == "paladin"
        assert agent.brain_config.decision_interval == 3  # Quick response

    def test_has_higher_observation_weight(self):
        paladin = PaladinAgent("defender_0")
        default_agent = BrainAugmentedAgent("default_0")

        assert paladin.config.observation_weight > default_agent.config.observation_weight


class TestChaosAgent:
    """Tests for ChaosAgent specialized class."""

    def test_initialization(self):
        agent = ChaosAgent("adversary_0")

        assert agent.id == "adversary_0"
        assert agent.config.energy_decay == 0.06  # Higher cost
        assert agent.config.observation_weight == 0.2  # Highly reactive
        assert agent.brain_config.brain_type == "chaos"

    def test_has_higher_energy_decay(self):
        chaos = ChaosAgent("adversary_0")
        default_agent = BrainAugmentedAgent("default_0")

        assert chaos.config.energy_decay > default_agent.config.energy_decay


class TestSwarmUnitAgent:
    """Tests for SwarmUnitAgent specialized class."""

    def test_initialization(self):
        agent = SwarmUnitAgent("unit_0")

        assert agent.id == "unit_0"
        assert agent.config.state_dim == 8
        assert agent.brain_config.brain_type == "swarm_unit"
        assert agent.brain_config.decision_interval == 1  # Every step

    def test_fast_decision_interval(self):
        unit = SwarmUnitAgent("unit_0")
        paladin = PaladinAgent("defender_0")

        # Swarm unit should decide every step
        assert unit.brain_config.decision_interval < paladin.brain_config.decision_interval


class TestCreateAgent:
    """Tests for create_agent factory function."""

    def test_create_triumvirate(self):
        agent = create_agent("agent_0", "triumvirate")

        assert isinstance(agent, TriumvirateAgent)
        assert agent.id == "agent_0"

    def test_create_paladin(self):
        agent = create_agent("agent_0", "paladin")

        assert isinstance(agent, PaladinAgent)

    def test_create_chaos(self):
        agent = create_agent("agent_0", "chaos")

        assert isinstance(agent, ChaosAgent)

    def test_create_swarm_unit(self):
        agent = create_agent("agent_0", "swarm_unit")

        assert isinstance(agent, SwarmUnitAgent)

    def test_create_with_position(self):
        position = np.array([10.0, 20.0])
        agent = create_agent("agent_0", "paladin", position)

        assert np.allclose(agent.state.position, position)

    def test_create_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_agent("agent_0", "unknown_type")


class TestBrainIntegration:
    """Integration tests for brain-augmented behavior."""

    def test_agent_hierarchy_energy_costs(self):
        """Higher tier agents should have lower energy costs (more efficient)."""
        triumvirate = TriumvirateAgent("t_0")
        paladin = PaladinAgent("p_0")
        chaos = ChaosAgent("c_0")

        # Triumvirate is most efficient
        assert triumvirate.config.energy_decay < paladin.config.energy_decay
        assert paladin.config.energy_decay < chaos.config.energy_decay

    def test_agent_hierarchy_brain_influence(self):
        """Higher tier agents should rely more on brain."""
        triumvirate = TriumvirateAgent("t_0")
        swarm_unit = SwarmUnitAgent("s_0")

        assert triumvirate.brain_config.brain_influence > swarm_unit.brain_config.brain_influence

    def test_multiple_agents_independent(self):
        """Multiple agents should have independent states."""
        agents = [create_agent(f"agent_{i}", "swarm_unit") for i in range(5)]

        # Set different positions
        for i, agent in enumerate(agents):
            agent.state.position = np.array([float(i), float(i)])

        # Verify independence
        for i, agent in enumerate(agents):
            assert np.allclose(agent.state.position, [float(i), float(i)])
