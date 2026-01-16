"""
Tests for neuro_swarm/evolution/

Tests algorithms, genome, and fitness functions.
"""

import pytest
import numpy as np

from neuro_swarm.evolution.algorithms import (
    EvolutionConfig,
    EvolutionaryStrategy,
    MAPElites,
    NoveltySearch,
)
from neuro_swarm.evolution.genome import AgentGenome, SwarmGenome, crossover
from neuro_swarm.evolution.fitness import (
    EvaluationResult,
    BehavioralDescriptor,
    SwarmCoherenceFitness,
    TaskCompletionFitness,
    discretize_behavior,
)


# ==================== Genome Tests ====================

class TestAgentGenome:
    """Tests for AgentGenome."""

    def test_random_creates_valid_genome(self):
        """Random creates genome with valid parameters."""
        rng = np.random.default_rng(42)
        genome = AgentGenome.random(rng)

        assert 0.5 <= genome.memory_persistence <= 0.99
        assert 0.01 <= genome.observation_weight <= 0.3
        assert 0.0 <= genome.cohesion_weight <= 1.0

    def test_serialization(self):
        """Genome serializes and deserializes correctly."""
        rng = np.random.default_rng(42)
        genome = AgentGenome.random(rng)

        data = genome.to_dict()
        restored = AgentGenome.from_dict(data)

        assert restored.memory_persistence == genome.memory_persistence
        assert restored.observation_weight == genome.observation_weight

    def test_mutation_produces_different_genome(self):
        """Mutation produces different genome."""
        rng = np.random.default_rng(42)
        genome = AgentGenome.random(rng)
        mutated = genome.mutate(rng)

        # Should be different in at least one parameter
        assert genome.to_dict() != mutated.to_dict()

    def test_mutation_respects_bounds(self):
        """Mutation respects parameter bounds."""
        rng = np.random.default_rng(42)
        genome = AgentGenome.random(rng)

        for _ in range(10):
            mutated = genome.mutate(rng)
            assert 0.5 <= mutated.memory_persistence <= 0.99
            assert 0.01 <= mutated.observation_weight <= 0.3


class TestSwarmGenome:
    """Tests for SwarmGenome."""

    def test_random_creates_valid_genome(self):
        """Random creates genome with valid swarm parameters."""
        rng = np.random.default_rng(42)
        genome = SwarmGenome.random(rng)

        assert genome.num_agents == 7
        assert 0.8 <= genome.substrate_decay <= 0.99
        assert isinstance(genome.agent_genome, AgentGenome)

    def test_serialization_preserves_nested(self):
        """Serialization preserves nested agent genome."""
        rng = np.random.default_rng(42)
        genome = SwarmGenome.random(rng)

        data = genome.to_dict()
        restored = SwarmGenome.from_dict(data)

        assert restored.num_agents == genome.num_agents
        assert restored.agent_genome.memory_persistence == genome.agent_genome.memory_persistence


class TestCrossover:
    """Tests for crossover function."""

    def test_crossover_produces_valid_genome(self):
        """Crossover produces valid genome."""
        rng = np.random.default_rng(42)
        parent1 = SwarmGenome.random(rng)
        parent2 = SwarmGenome.random(rng)

        child = crossover(parent1, parent2, rng)

        assert isinstance(child, SwarmGenome)
        assert 0.8 <= child.substrate_decay <= 0.99

    def test_crossover_different_types_raises(self):
        """Crossover with different types raises error."""
        rng = np.random.default_rng(42)
        genome1 = AgentGenome.random(rng)
        genome2 = SwarmGenome.random(rng)

        with pytest.raises(ValueError):
            crossover(genome1, genome2, rng)


# ==================== Fitness Tests ====================

class TestEvaluationResult:
    """Tests for EvaluationResult."""

    def test_basic_creation(self):
        """Basic result creation works."""
        result = EvaluationResult(
            fitness=0.5,
            behavior=np.array([0.3, 0.7]),
        )
        assert result.fitness == 0.5
        assert len(result.behavior) == 2

    def test_genome_field(self):
        """Genome field is preserved."""
        rng = np.random.default_rng(42)
        genome = SwarmGenome.random(rng)

        result = EvaluationResult(
            fitness=0.5,
            behavior=np.array([0.3, 0.7]),
            genome=genome,
        )

        assert result.genome is not None
        assert isinstance(result.genome, SwarmGenome)

    def test_serialization_with_genome(self):
        """Serialization includes genome."""
        rng = np.random.default_rng(42)
        genome = SwarmGenome.random(rng)

        result = EvaluationResult(
            fitness=0.5,
            behavior=np.array([0.3, 0.7]),
            genome=genome,
        )

        data = result.to_dict()
        assert "genome" in data
        assert data["genome"]["type"] == "SwarmGenome"


class TestDiscretizeBehavior:
    """Tests for discretize_behavior function."""

    def test_maps_to_correct_cell(self):
        """Behavior maps to correct archive cell."""
        behavior = np.array([0.5, 0.5])
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        resolution = 10

        cell = discretize_behavior(behavior, bounds, resolution)

        assert cell == (5, 5)

    def test_clamps_out_of_bounds(self):
        """Out of bounds values are clamped."""
        behavior = np.array([1.5, -0.5])
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        resolution = 10

        cell = discretize_behavior(behavior, bounds, resolution)

        assert cell == (9, 0)  # Clamped to valid range


class TestSwarmCoherenceFitness:
    """Tests for SwarmCoherenceFitness."""

    def test_evaluate_empty_trajectory(self):
        """Empty trajectory returns zero fitness."""
        fitness = SwarmCoherenceFitness()
        result = fitness.evaluate([], {})

        assert result.fitness == 0.0

    def test_evaluate_valid_trajectory(self):
        """Valid trajectory returns positive fitness."""
        fitness = SwarmCoherenceFitness()
        trajectory = [
            {
                "positions": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                "velocities": [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                "energies": [0.8, 0.9, 0.7],
            }
            for _ in range(5)
        ]

        result = fitness.evaluate(trajectory, {})

        assert result.fitness > 0.0
        assert len(result.behavior) == 2


# ==================== Algorithm Tests ====================

class TestEvolutionaryStrategy:
    """Tests for EvolutionaryStrategy."""

    def test_ask_returns_population_size_genomes(self):
        """Ask returns correct number of genomes."""
        config = EvolutionConfig(population_size=10, seed=42)
        es = EvolutionaryStrategy(config)

        genomes = es.ask()

        assert len(genomes) == 10

    def test_tell_updates_best(self):
        """Tell updates best genome."""
        config = EvolutionConfig(population_size=5, seed=42)
        es = EvolutionaryStrategy(config)

        genomes = es.ask()
        results = [
            EvaluationResult(
                fitness=float(i),
                behavior=np.array([0.5, 0.5]),
            )
            for i in range(5)
        ]

        es.tell(results)

        best_genome, best_fitness = es.get_best()
        assert best_fitness == 4.0

    def test_generation_increments(self):
        """Generation counter increments."""
        config = EvolutionConfig(population_size=3, seed=42)
        es = EvolutionaryStrategy(config)

        assert es.generation == 0

        genomes = es.ask()
        results = [
            EvaluationResult(fitness=0.5, behavior=np.array([0.5, 0.5]))
            for _ in genomes
        ]
        es.tell(results)

        assert es.generation == 1


class TestMAPElites:
    """Tests for MAPElites."""

    def test_ask_returns_genomes(self):
        """Ask returns genomes."""
        config = EvolutionConfig(population_size=10, seed=42)
        descriptor = BehavioralDescriptor(
            names=["x", "y"],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
        )
        me = MAPElites(config, descriptor)

        genomes = me.ask()

        assert len(genomes) == 10

    def test_tell_populates_archive(self):
        """Tell populates the archive."""
        config = EvolutionConfig(population_size=5, seed=42)
        descriptor = BehavioralDescriptor(
            names=["x", "y"],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
        )
        me = MAPElites(config, descriptor, archive_resolution=5)

        genomes = me.ask()
        results = [
            EvaluationResult(
                fitness=0.5,
                behavior=np.array([i * 0.2, i * 0.2]),
                genome=genomes[i],
            )
            for i in range(5)
        ]

        me.tell(results)

        assert len(me.archive) > 0

    def test_archive_stores_genomes_correctly(self):
        """Archive stores genomes correctly (bug fix test)."""
        config = EvolutionConfig(population_size=3, seed=42)
        descriptor = BehavioralDescriptor(
            names=["x", "y"],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
        )
        me = MAPElites(config, descriptor, archive_resolution=10)

        genomes = me.ask()
        results = [
            EvaluationResult(
                fitness=float(i),
                behavior=np.array([i * 0.3, i * 0.3]),
                genome=genomes[i],
            )
            for i in range(3)
        ]

        me.tell(results)

        # Check that genomes are stored correctly (not function references)
        for cell, (genome, fitness, behavior) in me.archive.items():
            assert genome is not None
            assert isinstance(genome, SwarmGenome)
            assert hasattr(genome, "to_dict")

    def test_get_best_returns_genome(self):
        """get_best returns a valid genome."""
        config = EvolutionConfig(population_size=5, seed=42)
        descriptor = BehavioralDescriptor(
            names=["x", "y"],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
        )
        me = MAPElites(config, descriptor)

        genomes = me.ask()
        results = [
            EvaluationResult(
                fitness=float(i),
                behavior=np.array([i * 0.2, 0.5]),
                genome=genomes[i],
            )
            for i in range(5)
        ]

        me.tell(results)

        best_genome, best_fitness = me.get_best()
        assert best_fitness == 4.0
        assert best_genome is not None
        assert isinstance(best_genome, SwarmGenome)


class TestNoveltySearch:
    """Tests for NoveltySearch."""

    def test_ask_returns_population(self):
        """Ask returns population."""
        config = EvolutionConfig(population_size=10, seed=42)
        descriptor = BehavioralDescriptor(
            names=["x", "y"],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
        )
        ns = NoveltySearch(config, descriptor)

        genomes = ns.ask()

        assert len(genomes) == 10

    def test_tell_populates_archive(self):
        """Tell populates behavior archive."""
        config = EvolutionConfig(population_size=5, seed=42)
        descriptor = BehavioralDescriptor(
            names=["x", "y"],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
        )
        ns = NoveltySearch(config, descriptor, archive_threshold=0.0)

        genomes = ns.ask()
        results = [
            EvaluationResult(
                fitness=0.5,
                behavior=np.array([i * 0.2, i * 0.2]),
            )
            for i in range(5)
        ]

        ns.tell(results)

        assert len(ns.behavior_archive) > 0

    def test_novelty_rewards_diversity(self):
        """Novelty search rewards diverse behaviors."""
        config = EvolutionConfig(population_size=3, seed=42)
        descriptor = BehavioralDescriptor(
            names=["x", "y"],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
        )
        ns = NoveltySearch(config, descriptor)

        # First batch
        ns.ask()
        results1 = [
            EvaluationResult(
                fitness=0.5,
                behavior=np.array([0.1, 0.1]),
            )
            for _ in range(3)
        ]
        ns.tell(results1)

        # Second batch with different behaviors
        ns.ask()
        results2 = [
            EvaluationResult(
                fitness=0.5,
                behavior=np.array([0.9, 0.9]),
            )
            for _ in range(3)
        ]

        # Compute novelty manually
        novelties = [ns._compute_novelty(r.behavior) for r in results2]

        # Should have high novelty (different from archive)
        assert all(n > 0.5 for n in novelties)
