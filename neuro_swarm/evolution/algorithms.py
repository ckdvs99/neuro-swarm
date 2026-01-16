"""
evolution/algorithms.py

Evolutionary Algorithms designed for distributed execution.

Key insight: Evolution is embarrassingly parallel at the evaluation step.
- Generate candidates (fast, centralized)
- Evaluate candidates (slow, distributed)
- Select and breed (fast, centralized)

These algorithms are designed to work with a distributed worker pool.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict
import time
import json

from .genome import Genome, SwarmGenome, crossover
from .fitness import (
    EvaluationResult, 
    FitnessFunction, 
    BehavioralDescriptor,
    discretize_behavior
)


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary algorithms."""
    population_size: int = 100
    generations: int = 500
    mutation_rate: float = 0.1
    crossover_rate: float = 0.3
    elite_fraction: float = 0.1
    tournament_size: int = 3
    seed: int = 42


class EvolutionaryAlgorithm(ABC):
    """
    Abstract base for evolutionary algorithms.
    
    Designed for distributed evaluation via task queues.
    """
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.generation = 0
        self.evaluations = 0
        self.history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def ask(self) -> List[Genome]:
        """
        Generate candidates for evaluation.
        
        Returns list of genomes to be evaluated by workers.
        """
        pass
    
    @abstractmethod
    def tell(self, results: List[EvaluationResult]) -> None:
        """
        Receive evaluation results and update state.
        
        Called after workers have evaluated the candidates from ask().
        """
        pass
    
    @abstractmethod
    def get_best(self) -> Tuple[Genome, float]:
        """Return best genome and its fitness."""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current algorithm statistics."""
        return {
            'generation': self.generation,
            'evaluations': self.evaluations,
            'algorithm': self.__class__.__name__,
        }


class EvolutionaryStrategy(EvolutionaryAlgorithm):
    """
    OpenAI-style Evolution Strategy.
    
    Key features:
    - Generates perturbations around a mean genome
    - Uses fitness-weighted averaging to update mean
    - Highly parallelizable (all perturbations independent)
    - No gradient required
    
    Reference: "Evolution Strategies as a Scalable Alternative to Reinforcement Learning"
    """
    
    def __init__(
        self, 
        config: EvolutionConfig,
        base_genome: Optional[Genome] = None,
        sigma: float = 0.1,
        learning_rate: float = 0.01,
    ):
        super().__init__(config)
        self.sigma = sigma
        self.learning_rate = learning_rate
        
        # Initialize mean genome
        self.mean_genome = base_genome or SwarmGenome.random(self.rng)
        self.mean_params = self._genome_to_vector(self.mean_genome)
        
        # Track perturbations for tell()
        self.current_perturbations: List[np.ndarray] = []
        self.current_genomes: List[Genome] = []
        
        # Best seen so far
        self.best_genome = self.mean_genome
        self.best_fitness = float('-inf')
    
    def _genome_to_vector(self, genome: Genome) -> np.ndarray:
        """Flatten genome to parameter vector (continuous params only)."""
        params = []
        d = genome.to_dict()
        
        # Extract nested if SwarmGenome
        if 'agent_genome' in d:
            agent_d = d['agent_genome']
            for key, val in agent_d.items():
                if isinstance(val, (int, float)) and key != 'type':
                    params.append(float(val))
        
        for key, val in d.items():
            if isinstance(val, (int, float)) and key != 'type':
                params.append(float(val))
        
        return np.array(params)
    
    def _vector_to_genome(self, params: np.ndarray, template: Genome) -> Genome:
        """Reconstruct genome from parameter vector."""
        d = template.to_dict()
        idx = 0
        
        # Handle nested agent_genome
        if 'agent_genome' in d:
            agent_d = d['agent_genome'].copy()
            for key in list(agent_d.keys()):
                if isinstance(agent_d[key], (int, float)) and key != 'type':
                    agent_d[key] = type(agent_d[key])(params[idx])
                    idx += 1
            d['agent_genome'] = agent_d
        
        for key in list(d.keys()):
            if isinstance(d[key], (int, float)) and key != 'type':
                d[key] = type(d[key])(params[idx])
                idx += 1
        
        return type(template).from_dict(d)
    
    def ask(self) -> List[Genome]:
        """Generate perturbed candidates."""
        self.current_perturbations = []
        self.current_genomes = []
        
        for _ in range(self.config.population_size):
            # Sample perturbation
            eps = self.rng.standard_normal(len(self.mean_params))
            self.current_perturbations.append(eps)
            
            # Create perturbed genome
            perturbed_params = self.mean_params + self.sigma * eps
            genome = self._vector_to_genome(perturbed_params, self.mean_genome)
            self.current_genomes.append(genome)
        
        return self.current_genomes
    
    def tell(self, results: List[EvaluationResult]) -> None:
        """Update mean using fitness-weighted perturbations."""
        # Extract fitnesses
        fitnesses = np.array([r.fitness for r in results])
        
        # Update best
        best_idx = fitnesses.argmax()
        if fitnesses[best_idx] > self.best_fitness:
            self.best_fitness = fitnesses[best_idx]
            self.best_genome = self.current_genomes[best_idx]
        
        # Normalize fitnesses (fitness shaping)
        ranks = np.argsort(np.argsort(fitnesses))
        shaped = ranks / (len(ranks) - 1) - 0.5  # Center around 0
        
        # Compute gradient estimate
        gradient = np.zeros_like(self.mean_params)
        for eps, f in zip(self.current_perturbations, shaped):
            gradient += f * eps
        gradient /= (len(results) * self.sigma)
        
        # Update mean
        self.mean_params += self.learning_rate * gradient
        self.mean_genome = self._vector_to_genome(self.mean_params, self.mean_genome)
        
        # Update statistics
        self.generation += 1
        self.evaluations += len(results)
        self.history.append({
            'generation': self.generation,
            'mean_fitness': fitnesses.mean(),
            'max_fitness': fitnesses.max(),
            'min_fitness': fitnesses.min(),
            'best_overall': self.best_fitness,
        })
    
    def get_best(self) -> Tuple[Genome, float]:
        return self.best_genome, self.best_fitness
    
    def get_statistics(self) -> Dict[str, Any]:
        stats = super().get_statistics()
        stats.update({
            'sigma': self.sigma,
            'learning_rate': self.learning_rate,
            'best_fitness': self.best_fitness,
        })
        return stats


class MAPElites(EvolutionaryAlgorithm):
    """
    MAP-Elites: Quality-Diversity Algorithm
    
    Maintains an archive of best solutions for each behavioral niche.
    Instead of converging to one optimum, finds diverse high-quality solutions.
    
    Reference: "Illuminating search spaces by mapping elites"
    """
    
    def __init__(
        self,
        config: EvolutionConfig,
        behavior_descriptor: BehavioralDescriptor,
        archive_resolution: int = 20,
        initial_population: Optional[List[Genome]] = None,
    ):
        super().__init__(config)
        self.behavior_descriptor = behavior_descriptor
        self.archive_resolution = archive_resolution
        
        # Archive: mapping from cell index to (genome, fitness, behavior)
        self.archive: Dict[Tuple[int, ...], Tuple[Genome, float, np.ndarray]] = {}

        # Initialize with random genomes if provided
        self.pending_genomes: List[Genome] = initial_population or []
        if not self.pending_genomes:
            self.pending_genomes = [
                SwarmGenome.random(self.rng)
                for _ in range(config.population_size)
            ]

        # Track current batch of genomes for result association
        self._current_genomes: List[Genome] = []
    
    def _get_cell(self, behavior: np.ndarray) -> Tuple[int, ...]:
        """Map behavior to archive cell."""
        return discretize_behavior(
            behavior, 
            self.behavior_descriptor.bounds,
            self.archive_resolution
        )
    
    def ask(self) -> List[Genome]:
        """Generate candidates via mutation of archive elites."""
        if self.pending_genomes:
            # Return pending (initial) genomes
            genomes = self.pending_genomes
            self.pending_genomes = []
            self._current_genomes = genomes
            return genomes

        # Generate new candidates by mutating archive members
        candidates = []
        archive_list = list(self.archive.values())

        for _ in range(self.config.population_size):
            if archive_list and self.rng.random() > 0.1:
                # Select parent from archive (uniform random)
                parent, _, _ = archive_list[self.rng.integers(len(archive_list))]
                if parent is not None:
                    child = parent.mutate(self.rng)

                    # Optional crossover
                    if self.rng.random() < self.config.crossover_rate and len(archive_list) > 1:
                        other, _, _ = archive_list[self.rng.integers(len(archive_list))]
                        if other is not None:
                            child = crossover(child, other, self.rng)

                    candidates.append(child)
                else:
                    # Fallback if genome is None
                    candidates.append(SwarmGenome.random(self.rng))
            else:
                # Random genome for exploration
                candidates.append(SwarmGenome.random(self.rng))

        self._current_genomes = candidates
        return candidates
    
    def tell(self, results: List[EvaluationResult]) -> None:
        """Update archive with new results."""
        additions = 0
        improvements = 0

        for i, result in enumerate(results):
            cell = self._get_cell(result.behavior)

            # Get genome: prefer from result, then metadata, then tracked genomes
            genome = result.genome
            if genome is None and hasattr(result, 'metadata'):
                # Check if genome was stored in metadata (from controller)
                genome = result.metadata.get('_genome')
            if genome is None and i < len(self._current_genomes):
                # Fall back to tracked genomes from ask()
                genome = self._current_genomes[i]

            if cell not in self.archive:
                # New cell discovered
                self.archive[cell] = (
                    genome,
                    result.fitness,
                    result.behavior
                )
                additions += 1
            elif result.fitness > self.archive[cell][1]:
                # Better solution for this cell
                self.archive[cell] = (
                    genome,
                    result.fitness,
                    result.behavior
                )
                improvements += 1
        
        self.generation += 1
        self.evaluations += len(results)
        
        fitnesses = [r.fitness for r in results]
        self.history.append({
            'generation': self.generation,
            'archive_size': len(self.archive),
            'coverage': len(self.archive) / (self.archive_resolution ** self.behavior_descriptor.dimensions),
            'additions': additions,
            'improvements': improvements,
            'mean_fitness': np.mean(fitnesses),
            'max_fitness': np.max(fitnesses),
        })
    
    def get_best(self) -> Tuple[Optional[Genome], float]:
        """Return globally best genome in archive."""
        if not self.archive:
            return None, float('-inf')
        
        best_cell = max(self.archive.keys(), key=lambda c: self.archive[c][1])
        genome, fitness, _ = self.archive[best_cell]
        return genome, fitness
    
    def get_archive_stats(self) -> Dict[str, Any]:
        """Get detailed archive statistics."""
        if not self.archive:
            return {'empty': True}
        
        fitnesses = [v[1] for v in self.archive.values()]
        behaviors = np.array([v[2] for v in self.archive.values()])
        
        return {
            'size': len(self.archive),
            'coverage': len(self.archive) / (self.archive_resolution ** self.behavior_descriptor.dimensions),
            'fitness_mean': np.mean(fitnesses),
            'fitness_std': np.std(fitnesses),
            'fitness_max': np.max(fitnesses),
            'fitness_min': np.min(fitnesses),
            'behavior_coverage': {
                name: (behaviors[:, i].min(), behaviors[:, i].max())
                for i, name in enumerate(self.behavior_descriptor.names)
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        stats = super().get_statistics()
        stats.update(self.get_archive_stats())
        return stats


class NoveltySearch(EvolutionaryAlgorithm):
    """
    Novelty Search: Reward behavioral novelty, not fitness.
    
    Useful for deceptive fitness landscapes where greedy optimization fails.
    
    Reference: "Abandoning Objectives: Evolution Through the Search for Novelty Alone"
    """
    
    def __init__(
        self,
        config: EvolutionConfig,
        behavior_descriptor: BehavioralDescriptor,
        k_neighbors: int = 15,
        archive_threshold: float = 0.1,
        fitness_weight: float = 0.0,  # 0 = pure novelty, 1 = pure fitness
    ):
        super().__init__(config)
        self.behavior_descriptor = behavior_descriptor
        self.k_neighbors = k_neighbors
        self.archive_threshold = archive_threshold
        self.fitness_weight = fitness_weight
        
        # Behavior archive (for novelty computation)
        self.behavior_archive: List[np.ndarray] = []
        
        # Population
        self.population: List[Genome] = [
            SwarmGenome.random(self.rng) 
            for _ in range(config.population_size)
        ]
        self.population_behaviors: List[np.ndarray] = []
        self.population_fitnesses: List[float] = []
        
        # Best seen
        self.best_genome: Optional[Genome] = None
        self.best_fitness = float('-inf')
    
    def _compute_novelty(self, behavior: np.ndarray) -> float:
        """Compute novelty as average distance to k nearest neighbors."""
        if not self.behavior_archive and not self.population_behaviors:
            return 1.0  # Maximum novelty if no reference points
        
        # Combine archive and current population
        all_behaviors = self.behavior_archive + self.population_behaviors
        if not all_behaviors:
            return 1.0
        
        # Compute distances
        distances = [
            np.linalg.norm(behavior - b) 
            for b in all_behaviors
        ]
        distances.sort()
        
        # Average of k nearest (excluding self if present)
        k = min(self.k_neighbors, len(distances))
        return np.mean(distances[:k])
    
    def ask(self) -> List[Genome]:
        """Return current population for evaluation."""
        return self.population
    
    def tell(self, results: List[EvaluationResult]) -> None:
        """Update population based on novelty (and optionally fitness)."""
        # Store behaviors and fitnesses
        self.population_behaviors = [r.behavior for r in results]
        self.population_fitnesses = [r.fitness for r in results]
        
        # Compute novelty scores
        novelties = [self._compute_novelty(b) for b in self.population_behaviors]
        
        # Combined score
        scores = []
        for nov, fit in zip(novelties, self.population_fitnesses):
            score = (1 - self.fitness_weight) * nov + self.fitness_weight * fit
            scores.append(score)
        
        # Update archive with novel behaviors
        for behavior, novelty in zip(self.population_behaviors, novelties):
            if novelty > self.archive_threshold:
                self.behavior_archive.append(behavior)
        
        # Update best (by fitness)
        best_idx = np.argmax(self.population_fitnesses)
        if self.population_fitnesses[best_idx] > self.best_fitness:
            self.best_fitness = self.population_fitnesses[best_idx]
            self.best_genome = self.population[best_idx]
        
        # Selection and reproduction
        new_population = []
        
        # Elitism
        elite_count = int(self.config.elite_fraction * self.config.population_size)
        elite_indices = np.argsort(scores)[-elite_count:]
        for idx in elite_indices:
            new_population.append(self.population[idx])
        
        # Tournament selection and mutation
        while len(new_population) < self.config.population_size:
            # Tournament selection based on combined score
            tournament = self.rng.choice(
                len(self.population), 
                size=self.config.tournament_size, 
                replace=False
            )
            winner_idx = tournament[np.argmax([scores[i] for i in tournament])]
            parent = self.population[winner_idx]
            
            # Mutation
            child = parent.mutate(self.rng)
            
            # Optional crossover
            if self.rng.random() < self.config.crossover_rate:
                other_idx = self.rng.integers(len(self.population))
                child = crossover(child, self.population[other_idx], self.rng)
            
            new_population.append(child)
        
        self.population = new_population
        
        # Update statistics
        self.generation += 1
        self.evaluations += len(results)
        self.history.append({
            'generation': self.generation,
            'archive_size': len(self.behavior_archive),
            'mean_novelty': np.mean(novelties),
            'max_novelty': np.max(novelties),
            'mean_fitness': np.mean(self.population_fitnesses),
            'max_fitness': np.max(self.population_fitnesses),
            'best_overall': self.best_fitness,
        })
    
    def get_best(self) -> Tuple[Optional[Genome], float]:
        return self.best_genome, self.best_fitness
    
    def get_statistics(self) -> Dict[str, Any]:
        stats = super().get_statistics()
        stats.update({
            'archive_size': len(self.behavior_archive),
            'k_neighbors': self.k_neighbors,
            'fitness_weight': self.fitness_weight,
            'best_fitness': self.best_fitness,
        })
        return stats
