"""
Tests for environments/simple_field.py

2D continuous space simulation environment.
"""

import numpy as np
import pytest

from neuro_swarm.environments.simple_field import SimpleField, FieldConfig
from neuro_swarm.core.agent import AgentConfig
from neuro_swarm.core.substrate import SubstrateConfig


class TestFieldConfig:
    """Tests for FieldConfig dataclass."""

    def test_default_config(self):
        config = FieldConfig()
        assert config.bounds == (-50.0, 50.0)
        assert config.wrap_edges is True
        assert config.friction == 0.98
        assert config.substrate_enabled is True

    def test_custom_config(self):
        config = FieldConfig(
            bounds=(-100.0, 100.0),
            wrap_edges=False,
            friction=0.9,
            substrate_enabled=False
        )
        assert config.bounds == (-100.0, 100.0)
        assert config.wrap_edges is False
        assert config.friction == 0.9
        assert config.substrate_enabled is False


class TestSimpleField:
    """Tests for SimpleField environment."""

    def test_initialization_default(self):
        field = SimpleField()
        assert field.config.bounds == (-50.0, 50.0)
        assert field.time == 0
        assert len(field.agents) == 0
        assert field.substrate is not None

    def test_initialization_without_substrate(self):
        config = FieldConfig(substrate_enabled=False)
        field = SimpleField(config)
        assert field.substrate is None

    def test_initialization_with_substrate_config(self):
        field_config = FieldConfig()
        substrate_config = SubstrateConfig(world_bounds=(-25.0, 25.0))
        field = SimpleField(field_config, substrate_config)
        assert field.substrate is not None
        assert field.substrate.config.world_bounds == (-25.0, 25.0)

    def test_add_agent_with_position(self):
        field = SimpleField()
        position = np.array([10.0, 20.0])
        agent = field.add_agent("agent_0", position)

        assert "agent_0" in field.agents
        assert np.allclose(agent.state.position, position)

    def test_add_agent_random_position(self):
        field = SimpleField()
        agent = field.add_agent("agent_0")

        assert "agent_0" in field.agents
        # Position should be within bounds
        assert field.config.bounds[0] <= agent.state.position[0] <= field.config.bounds[1]
        assert field.config.bounds[0] <= agent.state.position[1] <= field.config.bounds[1]

    def test_add_agent_with_config(self):
        field = SimpleField()
        agent_config = AgentConfig(neighbor_limit=5)
        agent = field.add_agent("agent_0", agent_config=agent_config)

        assert agent.config.neighbor_limit == 5

    def test_add_multiple_agents(self):
        field = SimpleField()
        field.add_agent("agent_0")
        field.add_agent("agent_1")
        field.add_agent("agent_2")

        assert len(field.agents) == 3

    def test_remove_agent(self):
        field = SimpleField()
        field.add_agent("agent_0")
        field.add_agent("agent_1")

        removed = field.remove_agent("agent_0")

        assert removed is not None
        assert "agent_0" not in field.agents
        assert "agent_1" in field.agents

    def test_remove_nonexistent_agent(self):
        field = SimpleField()
        removed = field.remove_agent("nonexistent")
        assert removed is None

    def test_get_neighbors_empty(self):
        field = SimpleField()
        field.add_agent("agent_0")

        neighbors = field.get_neighbors("agent_0")
        assert len(neighbors) == 0

    def test_get_neighbors_with_agents(self):
        field = SimpleField()
        field.add_agent("agent_0", np.array([0.0, 0.0]))
        field.add_agent("agent_1", np.array([5.0, 0.0]))
        field.add_agent("agent_2", np.array([10.0, 0.0]))

        neighbors = field.get_neighbors("agent_0")
        assert len(neighbors) == 2

    def test_get_neighbors_sorted_by_distance(self):
        field = SimpleField()
        field.add_agent("agent_0", np.array([0.0, 0.0]))
        field.add_agent("agent_1", np.array([10.0, 0.0]))  # Distance 10
        field.add_agent("agent_2", np.array([5.0, 0.0]))   # Distance 5

        neighbors = field.get_neighbors("agent_0")
        # First neighbor should be closer
        dist1 = np.linalg.norm(neighbors[0].position)
        dist2 = np.linalg.norm(neighbors[1].position)
        assert dist1 <= dist2

    def test_get_neighbors_respects_limit(self):
        field = SimpleField()
        field.add_agent("agent_0", np.array([0.0, 0.0]))
        for i in range(10):
            field.add_agent(f"agent_{i+1}", np.array([float(i+1), 0.0]))

        neighbors = field.get_neighbors("agent_0", limit=5)
        assert len(neighbors) == 5

    def test_get_neighbors_respects_radius(self):
        field = SimpleField()
        field.add_agent("agent_0", np.array([0.0, 0.0]))
        field.add_agent("agent_1", np.array([5.0, 0.0]))   # Within radius
        field.add_agent("agent_2", np.array([15.0, 0.0]))  # Outside radius

        neighbors = field.get_neighbors("agent_0", radius=10.0)
        assert len(neighbors) == 1

    def test_get_neighbors_nonexistent_agent(self):
        field = SimpleField()
        neighbors = field.get_neighbors("nonexistent")
        assert len(neighbors) == 0

    def test_step_advances_time(self):
        field = SimpleField()
        field.add_agent("agent_0")

        assert field.time == 0
        field.step()
        assert field.time == 1

    def test_step_updates_agents(self):
        field = SimpleField()
        agent = field.add_agent("agent_0", np.array([0.0, 0.0]))
        agent.state.velocity = np.array([1.0, 0.0])

        initial_pos = agent.state.position.copy()
        field.step()

        # Position should have changed due to velocity
        assert not np.allclose(agent.state.position, initial_pos)

    def test_step_applies_friction(self):
        config = FieldConfig(friction=0.5)
        field = SimpleField(config)
        agent = field.add_agent("agent_0")
        agent.state.velocity = np.array([10.0, 10.0])

        initial_velocity_mag = np.linalg.norm(agent.state.velocity)
        field.step()
        final_velocity_mag = np.linalg.norm(agent.state.velocity)

        # Velocity should have decreased due to friction
        assert final_velocity_mag < initial_velocity_mag

    def test_step_wraps_edges_toroidal(self):
        config = FieldConfig(bounds=(-10.0, 10.0), wrap_edges=True)
        field = SimpleField(config)
        agent = field.add_agent("agent_0", np.array([9.0, 0.0]))
        agent.state.velocity = np.array([5.0, 0.0])

        field.step()

        # Position should wrap around
        assert config.bounds[0] <= agent.state.position[0] <= config.bounds[1]

    def test_step_bounces_at_boundary(self):
        config = FieldConfig(bounds=(-10.0, 10.0), wrap_edges=False, friction=1.0)
        field = SimpleField(config)
        # Start at boundary with velocity pushing out
        agent = field.add_agent("agent_0", np.array([10.0, 0.0]))
        agent.state.velocity = np.array([5.0, 0.0])

        field.step()

        # Position should be clamped at boundary
        assert agent.state.position[0] <= config.bounds[1]
        # Velocity should have reversed (bounce) and reduced
        assert agent.state.velocity[0] < 0

    def test_step_with_substrate(self):
        field = SimpleField()
        agent = field.add_agent("agent_0", np.array([0.0, 0.0]))

        # Step should work with substrate
        field.step()
        assert field.substrate is not None

    def test_get_state_snapshot(self):
        field = SimpleField()
        field.add_agent("agent_0", np.array([1.0, 2.0]))
        field.add_agent("agent_1", np.array([3.0, 4.0]))

        snapshot = field.get_state_snapshot()

        assert "agent_0" in snapshot
        assert "agent_1" in snapshot
        assert np.allclose(snapshot["agent_0"].position, [1.0, 2.0])

    def test_get_positions(self):
        field = SimpleField()
        field.add_agent("agent_0", np.array([1.0, 2.0]))
        field.add_agent("agent_1", np.array([3.0, 4.0]))

        positions = field.get_positions()

        assert positions.shape == (2, 2)

    def test_get_velocities(self):
        field = SimpleField()
        agent0 = field.add_agent("agent_0")
        agent0.state.velocity = np.array([1.0, 0.0])
        agent1 = field.add_agent("agent_1")
        agent1.state.velocity = np.array([0.0, 1.0])

        velocities = field.get_velocities()

        assert velocities.shape == (2, 2)

    def test_get_energies(self):
        field = SimpleField()
        agent0 = field.add_agent("agent_0")
        agent0.state.energy = 0.8
        agent1 = field.add_agent("agent_1")
        agent1.state.energy = 0.5

        energies = field.get_energies()

        assert len(energies) == 2
        assert 0.5 in energies
        assert 0.8 in energies

    def test_repr(self):
        field = SimpleField()
        field.add_agent("agent_0")
        field.add_agent("agent_1")
        field.step()

        repr_str = repr(field)

        assert "SimpleField" in repr_str
        assert "agents=2" in repr_str
        assert "time=1" in repr_str
        assert "substrate=" in repr_str

    def test_multiple_steps(self):
        field = SimpleField()
        for i in range(5):
            field.add_agent(f"agent_{i}")

        for _ in range(10):
            field.step()

        assert field.time == 10

    def test_agents_interact_through_perception(self):
        field = SimpleField()
        # Add agents close enough to be neighbors
        field.add_agent("agent_0", np.array([0.0, 0.0]))
        field.add_agent("agent_1", np.array([5.0, 0.0]))

        # Running steps should allow agents to perceive each other
        for _ in range(5):
            field.step()

        # Both agents should still exist and have valid states
        assert len(field.agents) == 2
        for agent in field.agents.values():
            assert 0 <= agent.state.energy <= 1
