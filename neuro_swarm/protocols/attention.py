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
        """
        # TODO: Implement neighbor selection
        # - Compute distances
        # - Apply radius filter if specified
        # - Select top-k by relevance (distance, heading alignment, etc.)
        # - Compute attention weights
        raise NotImplementedError("LocalAttention.select_neighbors")

    def compute_attention_weights(
        self,
        agent_state: AgentState,
        neighbor_states: List[AgentState]
    ) -> np.ndarray:
        """
        Compute attention weights for given neighbors.

        Returns array of weights that sum to 1.
        """
        # TODO: Implement attention weight computation
        # - Consider distance (closer = more attention)
        # - Consider heading alignment (aligned = more attention)
        # - Consider energy states
        raise NotImplementedError("LocalAttention.compute_attention_weights")

    def attend(
        self,
        agent_state: AgentState,
        neighbor_states: List[AgentState],
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Compute attended representation of neighbors.

        Weighted combination of neighbor information.
        """
        # TODO: Implement attended representation
        raise NotImplementedError("LocalAttention.attend")
