"""
Tests for neuro_swarm/protocols/

Tests attention, consensus, and tension protocols.
"""

import pytest
import numpy as np

from neuro_swarm.core.agent import AgentState
from neuro_swarm.protocols.attention import LocalAttention
from neuro_swarm.protocols.consensus import (
    Vote,
    TriumvirateConsensus,
    LocalConsensus,
    RoleBasedConsensus,
)
from neuro_swarm.protocols.tension import (
    Force,
    ForceType,
    LinearTension,
    DynamicTension,
    PaladinChaosBalance,
)


# ==================== Attention Tests ====================

class TestLocalAttention:
    """Tests for LocalAttention protocol."""

    def test_select_neighbors_empty(self):
        """Empty neighbor list returns empty."""
        attention = LocalAttention(neighbor_limit=7)
        agent = AgentState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([1.0, 0.0]),
            internal=np.zeros(8),
        )
        result = attention.select_neighbors(agent, [])
        assert result == []

    def test_select_neighbors_respects_limit(self):
        """Neighbor selection respects limit."""
        attention = LocalAttention(neighbor_limit=3)
        agent = AgentState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([1.0, 0.0]),
            internal=np.zeros(8),
        )
        neighbors = [
            AgentState(
                position=np.array([float(i), 0.0]),
                velocity=np.array([1.0, 0.0]),
                internal=np.zeros(8),
            )
            for i in range(1, 10)
        ]
        result = attention.select_neighbors(agent, neighbors)
        assert len(result) == 3

    def test_select_neighbors_sorted_by_distance(self):
        """Nearest neighbors are selected first."""
        attention = LocalAttention(neighbor_limit=2)
        agent = AgentState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([1.0, 0.0]),
            internal=np.zeros(8),
        )
        neighbors = [
            AgentState(position=np.array([10.0, 0.0]), velocity=np.zeros(2), internal=np.zeros(8)),
            AgentState(position=np.array([1.0, 0.0]), velocity=np.zeros(2), internal=np.zeros(8)),
            AgentState(position=np.array([5.0, 0.0]), velocity=np.zeros(2), internal=np.zeros(8)),
        ]
        result = attention.select_neighbors(agent, neighbors)
        # First result should be the nearest (distance 1)
        assert np.allclose(result[0][0].position, [1.0, 0.0])

    def test_select_neighbors_radius_filter(self):
        """Radius filter excludes distant neighbors."""
        attention = LocalAttention(neighbor_limit=7, attention_radius=5.0)
        agent = AgentState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([1.0, 0.0]),
            internal=np.zeros(8),
        )
        neighbors = [
            AgentState(position=np.array([1.0, 0.0]), velocity=np.zeros(2), internal=np.zeros(8)),
            AgentState(position=np.array([10.0, 0.0]), velocity=np.zeros(2), internal=np.zeros(8)),
        ]
        result = attention.select_neighbors(agent, neighbors)
        assert len(result) == 1

    def test_compute_attention_weights_sum_to_one(self):
        """Attention weights should sum to 1."""
        attention = LocalAttention(neighbor_limit=7)
        agent = AgentState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([1.0, 0.0]),
            internal=np.zeros(8),
        )
        neighbors = [
            AgentState(position=np.array([1.0, 0.0]), velocity=np.array([1.0, 0.0]), internal=np.zeros(8)),
            AgentState(position=np.array([2.0, 0.0]), velocity=np.array([1.0, 0.0]), internal=np.zeros(8)),
        ]
        weights = attention.compute_attention_weights(agent, neighbors)
        assert np.isclose(weights.sum(), 1.0)

    def test_attend_returns_correct_shape(self):
        """Attend returns 6-dimensional vector."""
        attention = LocalAttention(neighbor_limit=7)
        agent = AgentState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([1.0, 0.0]),
            internal=np.zeros(8),
        )
        neighbors = [
            AgentState(position=np.array([1.0, 0.0]), velocity=np.array([1.0, 0.0]), internal=np.zeros(8)),
        ]
        weights = np.array([1.0])
        result = attention.attend(agent, neighbors, weights)
        assert result.shape == (6,)

    def test_attend_empty_neighbors(self):
        """Attend with empty neighbors returns zeros."""
        attention = LocalAttention(neighbor_limit=7)
        agent = AgentState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([1.0, 0.0]),
            internal=np.zeros(8),
        )
        result = attention.attend(agent, [], np.array([]))
        assert np.allclose(result, np.zeros(6))


# ==================== Consensus Tests ====================

class TestTriumvirateConsensus:
    """Tests for TriumvirateConsensus protocol."""

    def test_propose_creates_valid_vote(self):
        """Propose creates a valid Vote object."""
        consensus = TriumvirateConsensus()
        vote = consensus.propose("agent_1", np.array([1.0, 0.0]), 0.9)
        assert vote.agent_id == "agent_1"
        assert np.allclose(vote.value, [1.0, 0.0])
        assert vote.confidence == 0.9

    def test_propose_clips_confidence(self):
        """Confidence is clipped to [0, 1]."""
        consensus = TriumvirateConsensus()
        vote = consensus.propose("agent_1", np.array([1.0, 0.0]), 1.5)
        assert vote.confidence == 1.0

    def test_has_consensus_with_agreement(self):
        """Consensus detected when votes agree."""
        consensus = TriumvirateConsensus(agreement_threshold=0.8)
        votes = [
            Vote("a1", np.array([1.0, 0.0]), 0.9, 1),
            Vote("a2", np.array([1.0, 0.1]), 0.8, 2),
            Vote("a3", np.array([1.0, -0.1]), 0.7, 3),
        ]
        assert consensus.has_consensus(votes)

    def test_has_consensus_without_agreement(self):
        """No consensus when votes disagree."""
        consensus = TriumvirateConsensus(agreement_threshold=0.9)
        votes = [
            Vote("a1", np.array([1.0, 0.0]), 0.9, 1),
            Vote("a2", np.array([-1.0, 0.0]), 0.8, 2),
            Vote("a3", np.array([0.0, 1.0]), 0.7, 3),
        ]
        assert not consensus.has_consensus(votes)

    def test_aggregate_returns_value_on_agreement(self):
        """Aggregate returns consensus value when agreement exists."""
        consensus = TriumvirateConsensus(agreement_threshold=0.7)
        votes = [
            Vote("a1", np.array([1.0, 0.0]), 0.9, 1),
            Vote("a2", np.array([1.0, 0.0]), 0.8, 2),
            Vote("a3", np.array([0.0, 1.0]), 0.7, 3),  # Dissenter
        ]
        result = consensus.aggregate(votes)
        assert result is not None
        assert np.linalg.norm(result - np.array([1.0, 0.0])) < 0.5

    def test_aggregate_returns_none_on_disagreement(self):
        """Aggregate returns None when no agreement."""
        consensus = TriumvirateConsensus(agreement_threshold=0.95)
        votes = [
            Vote("a1", np.array([1.0, 0.0]), 0.9, 1),
            Vote("a2", np.array([-1.0, 0.0]), 0.8, 2),
            Vote("a3", np.array([0.0, 1.0]), 0.7, 3),
        ]
        result = consensus.aggregate(votes)
        assert result is None

    def test_aggregate_insufficient_votes(self):
        """Aggregate returns None with insufficient votes."""
        consensus = TriumvirateConsensus()
        votes = [Vote("a1", np.array([1.0, 0.0]), 0.9, 1)]
        result = consensus.aggregate(votes)
        assert result is None


class TestLocalConsensus:
    """Tests for LocalConsensus protocol."""

    def test_propose_creates_valid_vote(self):
        """Propose creates a valid Vote object."""
        consensus = LocalConsensus(neighbor_limit=7)
        vote = consensus.propose("agent_1", np.array([1.0, 0.0]), 0.9)
        assert vote.agent_id == "agent_1"

    def test_aggregate_respects_neighbor_limit(self):
        """Aggregate respects neighbor limit."""
        consensus = LocalConsensus(neighbor_limit=3)
        votes = [
            Vote(f"a{i}", np.array([1.0, 0.0]), 0.9, i)
            for i in range(10)
        ]
        result = consensus.aggregate(votes)
        assert result is not None

    def test_has_consensus_with_similar_votes(self):
        """Consensus detected when local votes are similar."""
        consensus = LocalConsensus(neighbor_limit=5, agreement_threshold=0.7)
        votes = [
            Vote("a1", np.array([1.0, 0.0]), 0.9, 1),
            Vote("a2", np.array([1.1, 0.0]), 0.8, 2),
            Vote("a3", np.array([0.9, 0.0]), 0.7, 3),
        ]
        assert consensus.has_consensus(votes)


class TestRoleBasedConsensus:
    """Tests for RoleBasedConsensus protocol."""

    def test_propose_applies_role_weight(self):
        """Role weight is applied to confidence."""
        consensus = RoleBasedConsensus(paladin_weight=2.0)
        vote = consensus.propose("agent_1", np.array([1.0, 0.0]), 0.5, role="paladin")
        assert vote.confidence == 1.0  # 0.5 * 2.0

    def test_aggregate_weights_by_role(self):
        """Aggregation weights by role."""
        consensus = RoleBasedConsensus(paladin_weight=2.0, explorer_weight=0.5)
        votes = [
            consensus.propose("paladin", np.array([1.0, 0.0]), 1.0, role="paladin"),
            consensus.propose("explorer", np.array([0.0, 1.0]), 1.0, role="explorer"),
        ]
        result = consensus.aggregate(votes)
        assert result is not None
        # Paladin has higher weight, result should be closer to [1, 0]
        assert result[0] > result[1]


# ==================== Tension Tests ====================

class TestLinearTension:
    """Tests for LinearTension resolver."""

    def test_resolve_empty_forces(self):
        """Empty force list returns zero vector."""
        resolver = LinearTension()
        result = resolver.resolve([])
        assert np.allclose(result, [0.0, 0.0])

    def test_resolve_single_force(self):
        """Single force returns scaled direction."""
        resolver = LinearTension(weights={ForceType.COHESION: 1.0})
        forces = [
            Force(ForceType.COHESION, np.array([1.0, 0.0]), 1.0)
        ]
        result = resolver.resolve(forces)
        assert np.allclose(result, [1.0, 0.0])

    def test_resolve_opposing_forces(self):
        """Opposing forces balance out."""
        resolver = LinearTension(weights={
            ForceType.COHESION: 1.0,
            ForceType.SEPARATION: 1.0
        })
        forces = [
            Force(ForceType.COHESION, np.array([1.0, 0.0]), 1.0),
            Force(ForceType.SEPARATION, np.array([-1.0, 0.0]), 1.0),
        ]
        result = resolver.resolve(forces)
        assert np.allclose(result, [0.0, 0.0])


class TestDynamicTension:
    """Tests for DynamicTension resolver."""

    def test_resolve_low_energy_favors_cohesion(self):
        """Low energy increases cohesion weight."""
        resolver = DynamicTension()
        forces = [
            Force(ForceType.COHESION, np.array([1.0, 0.0]), 1.0),
            Force(ForceType.EXPLORATION, np.array([0.0, 1.0]), 1.0),
        ]

        # High energy
        high_energy_result = resolver.resolve(forces, energy=1.0, threat_level=0.0)

        # Low energy
        low_energy_result = resolver.resolve(forces, energy=0.1, threat_level=0.0)

        # Low energy should favor cohesion (x component) more
        assert low_energy_result[0] > high_energy_result[0]

    def test_resolve_high_threat_favors_separation(self):
        """High threat increases separation weight."""
        resolver = DynamicTension()
        forces = [
            Force(ForceType.COHESION, np.array([1.0, 0.0]), 1.0),
            Force(ForceType.SEPARATION, np.array([-1.0, 0.0]), 1.0),
        ]

        # Low threat
        low_threat = resolver.resolve(forces, energy=1.0, threat_level=0.0)

        # High threat
        high_threat = resolver.resolve(forces, energy=1.0, threat_level=1.0)

        # High threat should shift result toward separation (negative x)
        assert high_threat[0] < low_threat[0]

    def test_resolve_clamps_energy(self):
        """Energy is clamped to valid range."""
        resolver = DynamicTension()
        forces = [Force(ForceType.COHESION, np.array([1.0, 0.0]), 1.0)]

        # Should not raise with out-of-range values
        result = resolver.resolve(forces, energy=2.0, threat_level=-0.5)
        assert result is not None


class TestPaladinChaosBalance:
    """Tests for PaladinChaosBalance."""

    def test_paladin_force_toward_target(self):
        """Paladin force points toward target."""
        balance = PaladinChaosBalance()
        current = np.array([0.0, 0.0])
        target = np.array([1.0, 0.0])

        force = balance.paladin_force(current, target)

        assert force.force_type == ForceType.PRESERVATION
        assert np.allclose(force.direction, [1.0, 0.0])

    def test_paladin_force_zero_at_target(self):
        """Paladin force is zero when at target."""
        balance = PaladinChaosBalance()
        position = np.array([1.0, 0.0])

        force = balance.paladin_force(position, position)

        assert force.magnitude == 0.0

    def test_chaos_force_random_direction(self):
        """Chaos force has random direction."""
        balance = PaladinChaosBalance()
        np.random.seed(42)

        force1 = balance.chaos_force(np.array([0.0, 0.0]))
        force2 = balance.chaos_force(np.array([0.0, 0.0]))

        # Different random directions
        assert force1.force_type == ForceType.DISRUPTION
        assert not np.allclose(force1.direction, force2.direction)

    def test_chaos_force_normalized(self):
        """Chaos force direction is normalized."""
        balance = PaladinChaosBalance()

        force = balance.chaos_force(np.array([0.0, 0.0]))

        assert np.isclose(np.linalg.norm(force.direction), 1.0)
