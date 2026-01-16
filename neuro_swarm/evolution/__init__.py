"""
neuro_swarm/evolution/

Evolutionary algorithms for distributed swarm parameter optimization.

Key insight: Evolution is embarrassingly parallel at the evaluation step.
- Generate candidates (fast, centralized)
- Evaluate candidates (slow, distributed)
- Select and breed (fast, centralized)

Algorithms:
- EvolutionaryStrategy (ES): OpenAI-style gradient-free optimization
- MAPElites: Quality-diversity via behavioral niches
- NoveltySearch: Reward novelty, escape deceptive landscapes
"""

from .algorithms import (
    EvolutionConfig,
    EvolutionaryAlgorithm,
    EvolutionaryStrategy,
    MAPElites,
    NoveltySearch,
)
from .genome import Genome, SwarmGenome, crossover
from .fitness import EvaluationResult, FitnessFunction, BehavioralDescriptor

__all__ = [
    "EvolutionConfig",
    "EvolutionaryAlgorithm",
    "EvolutionaryStrategy",
    "MAPElites",
    "NoveltySearch",
    "Genome",
    "SwarmGenome",
    "crossover",
    "EvaluationResult",
    "FitnessFunction",
    "BehavioralDescriptor",
]
