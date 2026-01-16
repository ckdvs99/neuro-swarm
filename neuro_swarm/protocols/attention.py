"""
protocols/attention.py

Local neighborhood attention mechanisms.

No agent has global information. Any architecture that assumes global state is lying.
Seven neighbors. That's enough. That's real.

Inspired by:
- Starling flocking (7 nearest neighbors)
- Cortical local connectivity
- Transformer attention (but localized)
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

from neuro_swarm.core.agent import AgentState


class LocalAttention:
    """
    Attention mechanism constrained to local neighborhood.

    Principles embodied:
    - Locality is honesty (Principle 4)
    - State is memory; gating is attention (Principle 8)
    """

    def __init__(
        self,
        neighbor_limit: int = 7,
        attention_radius: Optional[float] = None
    ):
        self.neighbor_limit = neighbor_limit
        self.attention_radius = attention_radius

    def select_neighbors(
        self,
        agent_state: AgentState,
        all_states: List[AgentState],
        exclude_self: bool = True
    ) -> List[Tuple[AgentState, float]]:
        """
        Select the most relevant neighbors for attention.

        Returns list of (neighbor_state, attention_weight) tuples,
        limited to neighbor_limit entries.

        Implements Principle 4: Locality is honesty — 7±2 neighbors max.
        """
        if not all_states:
            return []

        candidates = []
        for state in all_states:
            # Skip self if requested (compare by position identity)
            if exclude_self and np.allclose(state.position, agent_state.position):
                continue

            # Compute distance
            distance = np.linalg.norm(state.position - agent_state.position)

            # Apply radius filter if specified
            if self.attention_radius is not None and distance > self.attention_radius:
                continue

            candidates.append((state, distance))

        # Sort by distance (nearest first)
        candidates.sort(key=lambda x: x[1])

        # Limit to neighbor_limit
        candidates = candidates[:self.neighbor_limit]

        if not candidates:
            return []

        # Compute attention weights for selected neighbors
        neighbor_states = [c[0] for c in candidates]
        weights = self.compute_attention_weights(agent_state, neighbor_states)

        return list(zip(neighbor_states, weights))

    def compute_attention_weights(
        self,
        agent_state: AgentState,
        neighbor_states: List[AgentState]
    ) -> np.ndarray:
        """
        Compute attention weights for given neighbors.

        Returns array of weights that sum to 1.

        Attention is computed based on:
        - Distance (closer = more attention)
        - Heading alignment (aligned neighbors = more attention)
        - Energy state (active neighbors = more attention)

        Implements Principle 8: State is memory; gating is attention.
        """
        if not neighbor_states:
            return np.array([])

        n = len(neighbor_states)
        scores = np.zeros(n)

        for i, neighbor in enumerate(neighbor_states):
            # Distance component: inverse distance weighting
            distance = np.linalg.norm(neighbor.position - agent_state.position)
            distance_score = 1.0 / (1.0 + distance)

            # Heading alignment: cosine similarity of velocities
            agent_vel_norm = np.linalg.norm(agent_state.velocity)
            neighbor_vel_norm = np.linalg.norm(neighbor.velocity)

            if agent_vel_norm > 1e-6 and neighbor_vel_norm > 1e-6:
                alignment = np.dot(agent_state.velocity, neighbor.velocity)
                alignment = alignment / (agent_vel_norm * neighbor_vel_norm)
                alignment_score = (alignment + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
            else:
                alignment_score = 0.5  # Neutral if not moving

            # Energy component: active neighbors are more informative
            energy_score = neighbor.energy

            # Combined score (weighted average)
            scores[i] = 0.5 * distance_score + 0.3 * alignment_score + 0.2 * energy_score

        # Softmax normalization for attention weights
        scores = scores - scores.max()  # Numerical stability
        exp_scores = np.exp(scores)
        weights = exp_scores / (exp_scores.sum() + 1e-8)

        return weights

    def attend(
        self,
        agent_state: AgentState,
        neighbor_states: List[AgentState],
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Compute attended representation of neighbors.

        Weighted combination of neighbor information in egocentric frame.

        Returns:
            Attended vector containing:
            - Weighted relative position (2D)
            - Weighted relative velocity (2D)
            - Weighted average energy (1D)
            - Attention entropy (1D) - measure of attention distribution
        """
        if not neighbor_states or len(weights) == 0:
            return np.zeros(6)

        # Ensure weights sum to 1
        weights = weights / (weights.sum() + 1e-8)

        # Compute weighted features
        weighted_rel_pos = np.zeros(2)
        weighted_rel_vel = np.zeros(2)
        weighted_energy = 0.0

        for i, neighbor in enumerate(neighbor_states):
            w = weights[i]
            # Relative position (egocentric)
            rel_pos = neighbor.position - agent_state.position
            weighted_rel_pos += w * rel_pos

            # Relative velocity
            rel_vel = neighbor.velocity - agent_state.velocity
            weighted_rel_vel += w * rel_vel

            # Energy
            weighted_energy += w * neighbor.energy

        # Attention entropy: how spread out is attention?
        # Low entropy = focused attention, high entropy = diffuse attention
        entropy = -np.sum(weights * np.log(weights + 1e-8))
        max_entropy = np.log(len(weights) + 1e-8)
        normalized_entropy = entropy / (max_entropy + 1e-8)

        return np.concatenate([
            weighted_rel_pos,
            weighted_rel_vel,
            [weighted_energy],
            [normalized_entropy]
        ])
