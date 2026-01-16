"""
protocols/consensus.py

Emergent agreement protocols.

Three is the smallest number that can achieve consensus without deadlock.
Three perspectives can triangulate truth.
Three forces can balance.

Inspired by:
- Byzantine fault tolerance
- Quorum sensing in bacteria
- Neural population codes
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Any
import numpy as np


@dataclass
class Vote:
    """A single agent's contribution to consensus."""
    agent_id: str
    value: np.ndarray
    confidence: float
    timestamp: int


class ConsensusProtocol(ABC):
    """
    Abstract base for consensus mechanisms.

    Principles embodied:
    - Hierarchical sovereignty (Principle 7)
    - Locality is honesty (Principle 4) - consensus emerges, not imposed
    """

    @abstractmethod
    def propose(self, agent_id: str, value: np.ndarray, confidence: float) -> Vote:
        """Submit a proposal to the consensus process."""
        pass

    @abstractmethod
    def aggregate(self, votes: List[Vote]) -> Optional[np.ndarray]:
        """Aggregate votes into a consensus value (or None if no consensus)."""
        pass

    @abstractmethod
    def has_consensus(self, votes: List[Vote]) -> bool:
        """Check if consensus has been reached."""
        pass


class TriumvirateConsensus(ConsensusProtocol):
    """
    Three-agent consensus mechanism.

    The Triumvirate:
    - The Preserver: maintains coherence, resists disruption
    - The Challenger: tests assumptions, probes weakness
    - The Integrator: synthesizes, finds the path between
    """

    def __init__(
        self,
        agreement_threshold: float = 0.8,
        min_confidence: float = 0.5
    ):
        self.agreement_threshold = agreement_threshold
        self.min_confidence = min_confidence
        self._current_timestamp = 0

    def propose(self, agent_id: str, value: np.ndarray, confidence: float) -> Vote:
        """
        Submit a proposal to the triumvirate.

        The proposal is weighted by the agent's confidence.
        Low confidence proposals can still contribute but carry less weight.
        """
        self._current_timestamp += 1
        return Vote(
            agent_id=agent_id,
            value=np.asarray(value, dtype=np.float64),
            confidence=np.clip(confidence, 0.0, 1.0),
            timestamp=self._current_timestamp
        )

    def _compute_agreement(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute agreement between two value vectors.

        Returns a value in [0, 1] where 1 = perfect agreement.
        Uses cosine similarity for direction and magnitude similarity.
        """
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-8 and norm2 < 1e-8:
            return 1.0  # Both zero vectors agree
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0  # One is zero, other isn't

        # Cosine similarity for direction
        cosine = np.dot(v1, v2) / (norm1 * norm2)
        direction_agreement = (cosine + 1.0) / 2.0  # Map [-1, 1] to [0, 1]

        # Magnitude similarity
        max_norm = max(norm1, norm2)
        magnitude_agreement = min(norm1, norm2) / max_norm

        # Combined agreement
        return 0.7 * direction_agreement + 0.3 * magnitude_agreement

    def aggregate(self, votes: List[Vote]) -> Optional[np.ndarray]:
        """
        Aggregate three votes into consensus.

        Requires agreement between at least 2 of 3.
        The triumvirate pattern: Preserver, Challenger, Integrator.

        Returns the consensus value or None if no agreement reached.
        """
        if len(votes) < 2:
            return None

        # Filter by minimum confidence
        valid_votes = [v for v in votes if v.confidence >= self.min_confidence]
        if len(valid_votes) < 2:
            return None

        # For triumvirate (3 votes), find the agreeing pair(s)
        if len(valid_votes) >= 3:
            # Check all pairs for agreement
            agreeing_pairs = []
            for i in range(len(valid_votes)):
                for j in range(i + 1, len(valid_votes)):
                    agreement = self._compute_agreement(
                        valid_votes[i].value,
                        valid_votes[j].value
                    )
                    if agreement >= self.agreement_threshold:
                        agreeing_pairs.append((i, j, agreement))

            if not agreeing_pairs:
                return None

            # Find the best agreeing group
            # If all three agree, use weighted mean of all
            all_agree = len(agreeing_pairs) >= 3
            if all_agree:
                # Unanimous: weighted mean by confidence
                total_weight = sum(v.confidence for v in valid_votes)
                consensus = sum(
                    v.confidence * v.value for v in valid_votes
                ) / total_weight
                return consensus

            # Otherwise, use the best agreeing pair
            best_pair = max(agreeing_pairs, key=lambda x: x[2])
            i, j, _ = best_pair
            v1, v2 = valid_votes[i], valid_votes[j]

            # Weighted mean of agreeing pair
            total_weight = v1.confidence + v2.confidence
            consensus = (v1.confidence * v1.value + v2.confidence * v2.value) / total_weight
            return consensus

        # For 2 votes, check direct agreement
        if len(valid_votes) == 2:
            agreement = self._compute_agreement(
                valid_votes[0].value,
                valid_votes[1].value
            )
            if agreement >= self.agreement_threshold:
                total_weight = valid_votes[0].confidence + valid_votes[1].confidence
                consensus = (
                    valid_votes[0].confidence * valid_votes[0].value +
                    valid_votes[1].confidence * valid_votes[1].value
                ) / total_weight
                return consensus

        return None

    def has_consensus(self, votes: List[Vote]) -> bool:
        """
        Check if 2/3 or 3/3 agreement exists.

        Returns True if at least 2 votes agree above the threshold.
        """
        valid_votes = [v for v in votes if v.confidence >= self.min_confidence]
        if len(valid_votes) < 2:
            return False

        # Check all pairs for agreement
        for i in range(len(valid_votes)):
            for j in range(i + 1, len(valid_votes)):
                agreement = self._compute_agreement(
                    valid_votes[i].value,
                    valid_votes[j].value
                )
                if agreement >= self.agreement_threshold:
                    return True

        return False


class LocalConsensus(ConsensusProtocol):
    """
    Neighborhood-based consensus for larger swarms.

    Each agent achieves local consensus with neighbors,
    which propagates to form global patterns.

    Implements Principle 4: Locality is honesty — consensus emerges
    from local interactions, never imposed from above.
    """

    def __init__(
        self,
        neighbor_limit: int = 7,
        agreement_threshold: float = 0.6,
        quorum_fraction: float = 0.5
    ):
        self.neighbor_limit = neighbor_limit
        self.agreement_threshold = agreement_threshold
        self.quorum_fraction = quorum_fraction
        self._current_timestamp = 0

    def propose(self, agent_id: str, value: np.ndarray, confidence: float) -> Vote:
        """
        Submit a local proposal to the neighborhood.

        Proposals are time-stamped for consistency checking.
        """
        self._current_timestamp += 1
        return Vote(
            agent_id=agent_id,
            value=np.asarray(value, dtype=np.float64),
            confidence=np.clip(confidence, 0.0, 1.0),
            timestamp=self._current_timestamp
        )

    def aggregate(self, votes: List[Vote]) -> Optional[np.ndarray]:
        """
        Aggregate local neighborhood votes.

        Uses a robust mean that:
        1. Weights by confidence
        2. Respects the neighbor limit
        3. Requires quorum for valid consensus

        This creates a gradient-based consensus that propagates
        through the swarm via overlapping neighborhoods.
        """
        if not votes:
            return None

        # Respect neighbor limit
        votes = votes[:self.neighbor_limit]

        # Check quorum
        min_votes = max(2, int(len(votes) * self.quorum_fraction))
        if len(votes) < min_votes:
            return None

        # Sort votes by confidence (highest first)
        sorted_votes = sorted(votes, key=lambda v: v.confidence, reverse=True)

        # Compute weighted mean
        total_weight = 0.0
        weighted_sum = np.zeros_like(sorted_votes[0].value)

        for vote in sorted_votes:
            weight = vote.confidence
            weighted_sum += weight * vote.value
            total_weight += weight

        if total_weight < 1e-8:
            return None

        consensus = weighted_sum / total_weight
        return consensus

    def has_consensus(self, votes: List[Vote]) -> bool:
        """
        Check if local neighborhood agrees.

        Requires:
        1. Quorum met
        2. Variance in proposals below threshold
        """
        if not votes:
            return False

        votes = votes[:self.neighbor_limit]
        min_votes = max(2, int(len(votes) * self.quorum_fraction))
        if len(votes) < min_votes:
            return False

        # Compute centroid
        values = np.array([v.value for v in votes])
        centroid = values.mean(axis=0)

        # Compute average distance from centroid
        distances = np.linalg.norm(values - centroid, axis=1)
        avg_distance = distances.mean()

        # Normalize by centroid magnitude
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 1e-8:
            normalized_distance = avg_distance / centroid_norm
        else:
            normalized_distance = avg_distance

        # Consensus if variance is low
        return normalized_distance < (1.0 - self.agreement_threshold)


class RoleBasedConsensus(ConsensusProtocol):
    """
    Consensus with explicit roles for cyber-physical defense.

    Roles:
    - Paladins: Preserve stability (high weight on consistency)
    - Explorers: Probe alternatives (weight on novelty)
    - Integrators: Balance the two

    Implements the Paladin/Chaos dynamic from the philosophy.
    """

    def __init__(
        self,
        paladin_weight: float = 1.5,
        explorer_weight: float = 0.7,
        integrator_weight: float = 1.0,
        agreement_threshold: float = 0.7
    ):
        self.role_weights = {
            "paladin": paladin_weight,
            "explorer": explorer_weight,
            "integrator": integrator_weight
        }
        self.agreement_threshold = agreement_threshold
        self._current_timestamp = 0

    def propose(
        self,
        agent_id: str,
        value: np.ndarray,
        confidence: float,
        role: str = "integrator"
    ) -> Vote:
        """
        Submit a role-weighted proposal.

        The role determines how the vote is weighted in aggregation.
        """
        self._current_timestamp += 1
        return Vote(
            agent_id=agent_id,
            value=np.asarray(value, dtype=np.float64),
            confidence=np.clip(confidence, 0.0, 1.0) * self.role_weights.get(role, 1.0),
            timestamp=self._current_timestamp
        )

    def aggregate(self, votes: List[Vote]) -> Optional[np.ndarray]:
        """Aggregate role-weighted votes."""
        if len(votes) < 2:
            return None

        total_weight = sum(v.confidence for v in votes)
        if total_weight < 1e-8:
            return None

        consensus = sum(v.confidence * v.value for v in votes) / total_weight
        return consensus

    def has_consensus(self, votes: List[Vote]) -> bool:
        """Check if roles have reached agreement."""
        if len(votes) < 2:
            return False

        values = np.array([v.value for v in votes])
        weights = np.array([v.confidence for v in votes])

        # Weighted centroid
        total_weight = weights.sum()
        if total_weight < 1e-8:
            return False

        centroid = (weights[:, np.newaxis] * values).sum(axis=0) / total_weight

        # Check weighted agreement
        for i, vote in enumerate(votes):
            distance = np.linalg.norm(vote.value - centroid)
            norm = np.linalg.norm(centroid)
            if norm > 1e-8:
                normalized = distance / norm
                if normalized > (1.0 - self.agreement_threshold):
                    return False

        return True
