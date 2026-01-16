"""
neuro_swarm/services/controller.py

Evolution controller service.

The controller manages the evolution process:
1. Generates candidate genomes (ask)
2. Distributes evaluation tasks to workers
3. Collects results from workers
4. Updates evolution state (tell)
5. Tracks progress and statistics

Designed for distributed execution with graceful degradation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Type
import time
import uuid
import logging
import json

import numpy as np

from neuro_swarm.evolution.algorithms import (
    EvolutionaryAlgorithm,
    EvolutionaryStrategy,
    MAPElites,
    NoveltySearch,
    EvolutionConfig,
)
from neuro_swarm.evolution.genome import Genome, SwarmGenome
from neuro_swarm.evolution.fitness import (
    EvaluationResult,
    BehavioralDescriptor,
    SWARM_BEHAVIOR_DESCRIPTOR,
)
from .queue import (
    TaskQueue,
    EvaluationTask,
    EvaluationResultMessage,
    create_task_queue,
)

logger = logging.getLogger(__name__)


@dataclass
class ControllerConfig:
    """Configuration for the evolution controller."""
    # Algorithm selection
    algorithm: str = "es"  # "es", "map_elites", or "novelty"

    # Evolution parameters
    population_size: int = 100
    generations: int = 500

    # Task distribution
    batch_size: int = 50  # Tasks per batch
    result_timeout: float = 60.0  # Seconds to wait for results
    task_timeout: float = 300.0  # Individual task timeout

    # Simulation parameters
    simulation_steps: int = 200
    num_agents: int = 7

    # Queue configuration
    queue_backend: str = "memory"  # "memory" or "redis"
    redis_url: str = "redis://localhost:6379"

    # Checkpointing
    checkpoint_interval: int = 10  # Generations between checkpoints
    checkpoint_path: Optional[str] = None

    # Random seed
    seed: int = 42


class EvolutionController:
    """
    Controller for distributed evolutionary optimization.

    Manages the ask-evaluate-tell loop across distributed workers.
    """

    def __init__(
        self,
        config: ControllerConfig,
        behavior_descriptor: Optional[BehavioralDescriptor] = None,
    ):
        self.config = config
        self.behavior_descriptor = behavior_descriptor or SWARM_BEHAVIOR_DESCRIPTOR

        # Initialize RNG
        self.rng = np.random.default_rng(config.seed)

        # Create task queue
        self.queue = create_task_queue(
            backend=config.queue_backend,
            redis_url=config.redis_url,
        )

        # Create algorithm
        self.algorithm = self._create_algorithm()

        # State tracking
        self.generation = 0
        self.total_evaluations = 0
        self.pending_tasks: Dict[str, EvaluationTask] = {}
        self.pending_genomes: Dict[str, Genome] = {}
        self.history: List[Dict[str, Any]] = []

        # Status
        self.running = False
        self.start_time: Optional[float] = None

        logger.info(f"Controller initialized with {config.algorithm} algorithm")

    def _create_algorithm(self) -> EvolutionaryAlgorithm:
        """Create the selected evolutionary algorithm."""
        evo_config = EvolutionConfig(
            population_size=self.config.population_size,
            generations=self.config.generations,
            seed=self.config.seed,
        )

        if self.config.algorithm == "es":
            return EvolutionaryStrategy(evo_config)
        elif self.config.algorithm == "map_elites":
            return MAPElites(
                evo_config,
                behavior_descriptor=self.behavior_descriptor,
            )
        elif self.config.algorithm == "novelty":
            return NoveltySearch(
                evo_config,
                behavior_descriptor=self.behavior_descriptor,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

    def _create_simulation_config(self) -> Dict[str, Any]:
        """Create simulation configuration for workers."""
        return {
            "steps": self.config.simulation_steps,
            "num_agents": self.config.num_agents,
            "world_size": 100.0,
            "record_interval": 10,
        }

    def generate_tasks(self) -> int:
        """
        Generate evaluation tasks for current generation.

        Returns number of tasks generated.
        """
        # Ask algorithm for genomes to evaluate
        genomes = self.algorithm.ask()

        sim_config = self._create_simulation_config()
        tasks_created = 0

        for genome in genomes:
            task_id = str(uuid.uuid4())

            task = EvaluationTask(
                task_id=task_id,
                genome_data=genome.to_dict(),
                simulation_config=sim_config,
                generation=self.generation,
                timeout_seconds=self.config.task_timeout,
            )

            if self.queue.push_task(task):
                self.pending_tasks[task_id] = task
                self.pending_genomes[task_id] = genome
                tasks_created += 1

        logger.info(f"Generation {self.generation}: Created {tasks_created} tasks")
        return tasks_created

    def collect_results(self, timeout: float = None) -> List[EvaluationResult]:
        """
        Collect results from workers.

        Args:
            timeout: Maximum time to wait for results

        Returns:
            List of evaluation results
        """
        timeout = timeout or self.config.result_timeout
        start_time = time.time()
        results = []
        expected = len(self.pending_tasks)

        while len(results) < expected:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(
                    f"Timeout waiting for results: got {len(results)}/{expected}"
                )
                break

            # Pop result from queue
            result_msg = self.queue.pop_result(timeout=1.0)
            if result_msg is None:
                continue

            # Convert to EvaluationResult
            if result_msg.error is not None:
                logger.warning(f"Task {result_msg.task_id} failed: {result_msg.error}")
                continue

            eval_result = EvaluationResult(
                fitness=result_msg.fitness,
                behavior=np.array(result_msg.behavior),
                metadata=result_msg.metadata,
            )

            # Store genome reference for MAP-Elites bug fix
            if result_msg.task_id in self.pending_genomes:
                eval_result.metadata["_genome"] = self.pending_genomes[result_msg.task_id]

            results.append(eval_result)

            # Clean up tracking
            self.pending_tasks.pop(result_msg.task_id, None)
            self.pending_genomes.pop(result_msg.task_id, None)
            self.total_evaluations += 1

        return results

    def step(self) -> Dict[str, Any]:
        """
        Execute one generation of evolution.

        Returns generation statistics.
        """
        gen_start = time.time()

        # Generate tasks
        num_tasks = self.generate_tasks()

        # Collect results
        results = self.collect_results()

        # Update algorithm
        if results:
            self.algorithm.tell(results)

        gen_time = time.time() - gen_start

        # Record statistics
        stats = {
            "generation": self.generation,
            "tasks_created": num_tasks,
            "results_received": len(results),
            "total_evaluations": self.total_evaluations,
            "generation_time": gen_time,
            **self.algorithm.get_statistics(),
        }

        self.history.append(stats)
        self.generation += 1

        best_fit = stats.get('best_fitness')
        best_fit_str = f"{best_fit:.4f}" if isinstance(best_fit, (int, float)) else "N/A"
        logger.info(
            f"Generation {self.generation - 1}: "
            f"{len(results)} results, "
            f"best fitness: {best_fit_str}"
        )

        return stats

    def run(self, generations: Optional[int] = None) -> Dict[str, Any]:
        """
        Run evolution for specified number of generations.

        Args:
            generations: Number of generations (default: from config)

        Returns:
            Final statistics
        """
        generations = generations or self.config.generations
        self.running = True
        self.start_time = time.time()

        logger.info(f"Starting evolution for {generations} generations")

        try:
            for _ in range(generations):
                if not self.running:
                    break

                stats = self.step()

                # Checkpoint
                if (
                    self.config.checkpoint_path and
                    self.generation % self.config.checkpoint_interval == 0
                ):
                    self.save_checkpoint()

        except KeyboardInterrupt:
            logger.info("Evolution interrupted by user")

        finally:
            self.running = False

        total_time = time.time() - self.start_time

        # Final results
        best_genome, best_fitness = self.algorithm.get_best()

        final_stats = {
            "total_generations": self.generation,
            "total_evaluations": self.total_evaluations,
            "total_time": total_time,
            "best_fitness": best_fitness,
            "best_genome": best_genome.to_dict() if best_genome else None,
            "algorithm": self.config.algorithm,
        }

        logger.info(
            f"Evolution complete: {self.generation} generations, "
            f"best fitness: {best_fitness:.4f}"
        )

        return final_stats

    def stop(self) -> None:
        """Stop evolution gracefully."""
        self.running = False
        logger.info("Stopping evolution...")

    def save_checkpoint(self, path: Optional[str] = None) -> None:
        """Save current state to checkpoint file."""
        path = path or self.config.checkpoint_path
        if path is None:
            return

        checkpoint = {
            "generation": self.generation,
            "total_evaluations": self.total_evaluations,
            "algorithm_stats": self.algorithm.get_statistics(),
            "history": self.history,
            "config": {
                "algorithm": self.config.algorithm,
                "population_size": self.config.population_size,
                "seed": self.config.seed,
            },
        }

        # Add best genome
        best_genome, best_fitness = self.algorithm.get_best()
        if best_genome:
            checkpoint["best_genome"] = best_genome.to_dict()
            checkpoint["best_fitness"] = best_fitness

        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load state from checkpoint file."""
        with open(path, "r") as f:
            checkpoint = json.load(f)

        self.generation = checkpoint.get("generation", 0)
        self.total_evaluations = checkpoint.get("total_evaluations", 0)
        self.history = checkpoint.get("history", [])

        logger.info(f"Checkpoint loaded from {path}: generation {self.generation}")

    def get_status(self) -> Dict[str, Any]:
        """Get current controller status."""
        elapsed = 0.0
        if self.start_time:
            elapsed = time.time() - self.start_time

        return {
            "running": self.running,
            "generation": self.generation,
            "total_evaluations": self.total_evaluations,
            "pending_tasks": len(self.pending_tasks),
            "queue_length": self.queue.get_queue_length(),
            "result_count": self.queue.get_result_count(),
            "elapsed_time": elapsed,
            "algorithm": self.config.algorithm,
        }


def run_controller(config: Optional[ControllerConfig] = None) -> None:
    """
    Run the controller as a standalone service.

    This is the entry point for the controller container.
    """
    import argparse
    import signal
    import sys

    parser = argparse.ArgumentParser(description="Evolution Controller")
    parser.add_argument("--algorithm", default="es", choices=["es", "map_elites", "novelty"])
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--redis-url", default="redis://localhost:6379")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if config is None:
        config = ControllerConfig(
            algorithm=args.algorithm,
            population_size=args.population_size,
            generations=args.generations,
            queue_backend="redis",
            redis_url=args.redis_url,
            checkpoint_path=args.checkpoint_path,
            seed=args.seed,
        )

    controller = EvolutionController(config)

    # Handle signals for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        controller.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run evolution
    controller.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_controller()
