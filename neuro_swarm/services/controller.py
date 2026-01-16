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

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from neuro_swarm.evolution.algorithms import (
    EvolutionaryAlgorithm,
    EvolutionaryStrategy,
    EvolutionConfig,
    MAPElites,
    NoveltySearch,
)
from neuro_swarm.evolution.fitness import (
    SWARM_BEHAVIOR_DESCRIPTOR,
    BehavioralDescriptor,
    EvaluationResult,
)
from neuro_swarm.evolution.genome import Genome

from .dashboard import (
    DASHBOARD_ARCHIVE_KEY,
    DASHBOARD_HISTORY_KEY,
    DASHBOARD_STATUS_KEY,
)
from .persistence import EvolutionPersistence, PersistenceConfig
from .queue import (
    EvaluationTask,
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
    checkpoint_path: str | None = None

    # Random seed
    seed: int = 42

    # Dashboard publishing
    publish_to_dashboard: bool = True
    dashboard_history_limit: int = 1000  # Max history entries in Redis

    # Database persistence
    db_persistence_enabled: bool = True
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "neuro_swarm"
    db_user: str = "postgres"
    db_password: str = ""
    db_store_genomes: bool = False
    db_archive_snapshot_interval: int = 10


class EvolutionController:
    """
    Controller for distributed evolutionary optimization.

    Manages the ask-evaluate-tell loop across distributed workers.
    """

    def __init__(
        self,
        config: ControllerConfig,
        behavior_descriptor: BehavioralDescriptor | None = None,
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
        self.pending_tasks: dict[str, EvaluationTask] = {}
        self.pending_genomes: dict[str, Genome] = {}
        self.history: list[dict[str, Any]] = []

        # Status
        self.running = False
        self.start_time: float | None = None

        # Dashboard Redis connection (lazy)
        self._dashboard_redis = None

        # Database persistence
        self.persistence = self._create_persistence()

        logger.info(f"Controller initialized with {config.algorithm} algorithm")

    def _create_persistence(self) -> EvolutionPersistence:
        """Create database persistence layer."""
        persistence_config = PersistenceConfig(
            host=self.config.db_host,
            port=self.config.db_port,
            database=self.config.db_name,
            user=self.config.db_user,
            password=self.config.db_password,
            enabled=self.config.db_persistence_enabled,
            store_genomes=self.config.db_store_genomes,
            archive_snapshot_interval=self.config.db_archive_snapshot_interval,
        )
        return EvolutionPersistence(persistence_config)

    def _get_dashboard_redis(self):
        """Get Redis connection for dashboard publishing."""
        if self._dashboard_redis is None and self.config.queue_backend == "redis":
            try:
                import redis
                self._dashboard_redis = redis.from_url(
                    self.config.redis_url, decode_responses=True
                )
                self._dashboard_redis.ping()
                logger.info("Dashboard Redis connection established")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis for dashboard: {e}")
                self._dashboard_redis = None
        return self._dashboard_redis

    def _publish_dashboard_status(self, stats: dict[str, Any]) -> None:
        """Publish current status to Redis for dashboard."""
        if not self.config.publish_to_dashboard:
            return

        r = self._get_dashboard_redis()
        if r is None:
            return

        try:
            # Get best genome
            best_genome, best_fitness = self.algorithm.get_best()

            # Build status payload
            elapsed = 0.0
            if self.start_time:
                elapsed = time.time() - self.start_time

            status = {
                "running": self.running,
                "generation": self.generation,
                "total_evaluations": self.total_evaluations,
                "elapsed_time": elapsed,
                "algorithm": self.config.algorithm,
                "best_fitness": best_fitness,
                "best_genome": best_genome.to_dict() if best_genome else None,
                "statistics": self.algorithm.get_statistics(),
            }

            # Publish status
            r.set(DASHBOARD_STATUS_KEY, json.dumps(status))

            # Publish to history list (prepend for most recent first)
            r.lpush(DASHBOARD_HISTORY_KEY, json.dumps(stats))
            r.ltrim(DASHBOARD_HISTORY_KEY, 0, self.config.dashboard_history_limit - 1)

            # Publish archive for MAP-Elites
            if self.config.algorithm == "map_elites":
                self._publish_archive()

        except Exception as e:
            logger.warning(f"Failed to publish dashboard status: {e}")

    def _publish_archive(self) -> None:
        """Publish MAP-Elites archive to Redis."""
        r = self._get_dashboard_redis()
        if r is None:
            return

        try:
            if hasattr(self.algorithm, 'archive') and hasattr(self.algorithm, 'grid_resolution'):
                archive = self.algorithm.archive
                grid_res = self.algorithm.grid_resolution

                # Convert archive to heatmap data
                grid_data = []
                for coords, (_genome, fitness) in archive.items():
                    grid_data.append({
                        "coords": list(coords),
                        "fitness": fitness,
                    })

                archive_payload = {
                    "grid": grid_data,
                    "grid_resolution": list(grid_res),
                    "coverage": len(archive) / (grid_res[0] * grid_res[1]),
                    "behavior_bounds": [[0, 1], [0, 1]],
                }

                r.set(DASHBOARD_ARCHIVE_KEY, json.dumps(archive_payload))

        except Exception as e:
            logger.warning(f"Failed to publish archive: {e}")

    def _persist_archive_snapshot(self) -> None:
        """Persist MAP-Elites archive snapshot to database."""
        try:
            if hasattr(self.algorithm, 'archive') and hasattr(self.algorithm, 'grid_resolution'):
                archive = self.algorithm.archive
                grid_res = self.algorithm.grid_resolution

                # Convert archive to serializable format
                grid_data = []
                for coords, (genome, fitness) in archive.items():
                    grid_data.append({
                        "coords": list(coords),
                        "fitness": fitness,
                        "genome": genome.to_dict() if hasattr(genome, 'to_dict') else None,
                    })

                archive_data = {
                    "grid": grid_data,
                    "grid_resolution": list(grid_res),
                    "coverage": len(archive) / (grid_res[0] * grid_res[1]),
                    "total_cells": grid_res[0] * grid_res[1],
                    "filled_cells": len(archive),
                }

                self.persistence.record_archive_snapshot(
                    generation=self.generation,
                    archive_data=archive_data,
                )

        except Exception as e:
            logger.warning(f"Failed to persist archive snapshot: {e}")

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

    def _create_simulation_config(self) -> dict[str, Any]:
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

    def collect_results(self, timeout: float = None) -> list[EvaluationResult]:
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

    def step(self) -> dict[str, Any]:
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

        # Publish to dashboard
        self._publish_dashboard_status(stats)

        # Persist to database
        best_genome, _ = self.algorithm.get_best()
        self.persistence.record_generation(
            generation=self.generation - 1,
            stats=stats,
            best_genome=best_genome.to_dict() if best_genome else None,
        )

        # Record archive snapshot periodically for MAP-Elites
        if (
            self.config.algorithm == "map_elites"
            and self.generation % self.config.db_archive_snapshot_interval == 0
        ):
            self._persist_archive_snapshot()

        best_fit = stats.get('best_fitness')
        best_fit_str = f"{best_fit:.4f}" if isinstance(best_fit, (int, float)) else "N/A"
        logger.info(
            f"Generation {self.generation - 1}: "
            f"{len(results)} results, "
            f"best fitness: {best_fit_str}"
        )

        return stats

    def run(self, generations: int | None = None) -> dict[str, Any]:
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

        # Start database run tracking
        run_config = {
            "population_size": self.config.population_size,
            "generations": generations,
            "batch_size": self.config.batch_size,
            "simulation_steps": self.config.simulation_steps,
            "num_agents": self.config.num_agents,
            "seed": self.config.seed,
        }
        self.persistence.start_run(self.config.algorithm, run_config)

        logger.info(f"Starting evolution for {generations} generations")

        try:
            for _ in range(generations):
                if not self.running:
                    break

                self.step()

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

        # End database run tracking
        self.persistence.end_run(
            best_fitness=best_fitness or 0.0,
            best_genome=best_genome.to_dict() if best_genome else None,
            total_generations=self.generation,
            total_evaluations=self.total_evaluations,
        )
        self.persistence.close()

        logger.info(
            f"Evolution complete: {self.generation} generations, "
            f"best fitness: {best_fitness:.4f}"
        )

        return final_stats

    def stop(self) -> None:
        """Stop evolution gracefully."""
        self.running = False
        logger.info("Stopping evolution...")

    def save_checkpoint(self, path: str | None = None) -> None:
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
        with open(path) as f:
            checkpoint = json.load(f)

        self.generation = checkpoint.get("generation", 0)
        self.total_evaluations = checkpoint.get("total_evaluations", 0)
        self.history = checkpoint.get("history", [])

        logger.info(f"Checkpoint loaded from {path}: generation {self.generation}")

    def get_status(self) -> dict[str, Any]:
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


def run_controller(config: ControllerConfig | None = None) -> None:
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

    # Database arguments
    parser.add_argument("--db-host", default=os.environ.get("POSTGRES_HOST", "localhost"))
    parser.add_argument("--db-port", type=int, default=int(os.environ.get("POSTGRES_PORT", "5432")))
    parser.add_argument("--db-name", default=os.environ.get("POSTGRES_DB", "neuro_swarm"))
    parser.add_argument("--db-user", default=os.environ.get("POSTGRES_USER", "postgres"))
    parser.add_argument("--db-password", default=os.environ.get("POSTGRES_PASSWORD", ""))
    parser.add_argument("--db-persistence-enabled", type=bool, default=True)

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
            db_host=args.db_host,
            db_port=args.db_port,
            db_name=args.db_name,
            db_user=args.db_user,
            db_password=args.db_password,
            db_persistence_enabled=args.db_persistence_enabled,
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
