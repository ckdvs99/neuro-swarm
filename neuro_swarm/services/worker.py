"""
neuro_swarm/services/worker.py

Evaluation worker service.

Workers perform the computationally expensive part of evolution:
1. Pull evaluation tasks from queue
2. Run swarm simulations
3. Compute fitness and behavioral descriptors
4. Push results back to queue

Designed for horizontal scaling - add more workers for faster evolution.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neuro_swarm.core.agent import AgentConfig, NeuroAgent
from neuro_swarm.core.substrate import SubstrateConfig
from neuro_swarm.environments.simple_field import FieldConfig, SimpleField
from neuro_swarm.evolution.fitness import (
    FitnessFunction,
    SwarmCoherenceFitness,
    TaskCompletionFitness,
)
from neuro_swarm.evolution.genome import SwarmGenome

from .dashboard import WORKER_PREFIX
from .queue import (
    EvaluationResultMessage,
    EvaluationTask,
    create_task_queue,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Configuration for evaluation workers."""
    # Worker identity
    worker_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Queue configuration
    queue_backend: str = "memory"
    redis_url: str = "redis://localhost:6379"

    # Simulation settings
    default_steps: int = 200
    default_world_size: float = 100.0

    # Fitness function
    fitness_type: str = "coherence"  # "coherence" or "task"

    # Worker behavior
    poll_interval: float = 0.1  # Seconds between task polls
    max_consecutive_errors: int = 5  # Errors before worker stops
    heartbeat_interval: float = 30.0  # Seconds between heartbeats

    # Random seed (None = random each simulation)
    seed: int | None = None

    # Dashboard heartbeat
    publish_heartbeat: bool = True
    heartbeat_ttl: int = 60  # Redis TTL for worker heartbeat


class SwarmSimulator:
    """
    Runs swarm simulations for genome evaluation.

    Encapsulates the simulation loop and trajectory recording.
    """

    def __init__(
        self,
        world_size: float = 100.0,
        seed: int | None = None,
    ):
        self.world_size = world_size
        self.rng = np.random.default_rng(seed)

    def run(
        self,
        genome: SwarmGenome,
        steps: int = 200,
        record_interval: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Run a swarm simulation.

        Args:
            genome: SwarmGenome defining the swarm
            steps: Number of simulation steps
            record_interval: Steps between trajectory recordings

        Returns:
            Trajectory as list of state dictionaries
        """
        # Create environment configuration
        field_config = FieldConfig(
            bounds=(-self.world_size, self.world_size),
            wrap_edges=True,
            friction=0.98,
            substrate_enabled=True,
        )

        substrate_config = SubstrateConfig(
            grid_size=(50, 50),
            trace_dim=4,
            decay_rate=genome.substrate_decay,
            diffusion_rate=genome.substrate_diffusion,
            world_bounds=(-self.world_size, self.world_size),
        )

        field = SimpleField(
            config=field_config,
            substrate_config=substrate_config,
        )

        # Create agent configuration from genome
        agent_config = AgentConfig(
            memory_persistence=genome.agent_genome.memory_persistence,
            observation_weight=genome.agent_genome.observation_weight,
            energy_decay=genome.agent_genome.energy_decay,
            energy_recovery=genome.agent_genome.energy_recovery,
            rest_threshold=genome.agent_genome.rest_threshold,
        )

        # Add agents to field
        for i in range(genome.num_agents):
            position = self.rng.uniform(
                -self.world_size / 4,
                self.world_size / 4,
                size=2
            )
            field.add_agent(f"agent_{i}", position=position, agent_config=agent_config)

        # Run simulation
        trajectory = []

        for step in range(steps):
            # Record state
            if step % record_interval == 0:
                state = self._record_state(list(field.agents.values()))
                trajectory.append(state)

            # Step simulation
            field.step()

        # Record final state
        trajectory.append(self._record_state(list(field.agents.values())))

        return trajectory

    def _record_state(self, agents: list[NeuroAgent]) -> dict[str, Any]:
        """Record current state for trajectory."""
        return {
            "positions": [a.state.position.tolist() for a in agents],
            "velocities": [a.state.velocity.tolist() for a in agents],
            "energies": [a.state.energy for a in agents],
            "ages": [a.state.age for a in agents],
        }


class EvaluationWorker:
    """
    Worker that evaluates genomes from the task queue.

    Runs continuously, pulling tasks and pushing results.
    """

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.worker_id = config.worker_id

        # Create queue connection
        self.queue = create_task_queue(
            backend=config.queue_backend,
            redis_url=config.redis_url,
        )

        # Create simulator
        self.simulator = SwarmSimulator(
            world_size=config.default_world_size,
            seed=config.seed,
        )

        # Create fitness function
        self.fitness_fn = self._create_fitness_function()

        # Status
        self.running = False
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.consecutive_errors = 0
        self.last_heartbeat = time.time()

        # Dashboard Redis connection (lazy)
        self._dashboard_redis = None

        logger.info(f"Worker {self.worker_id} initialized")

    def _create_fitness_function(self) -> FitnessFunction:
        """Create the fitness function based on config."""
        if self.config.fitness_type == "coherence":
            return SwarmCoherenceFitness()
        elif self.config.fitness_type == "task":
            return TaskCompletionFitness()
        else:
            raise ValueError(f"Unknown fitness type: {self.config.fitness_type}")

    def _get_dashboard_redis(self):
        """Get Redis connection for dashboard heartbeat."""
        if self._dashboard_redis is None and self.config.queue_backend == "redis":
            try:
                import redis
                self._dashboard_redis = redis.from_url(
                    self.config.redis_url, decode_responses=True
                )
                self._dashboard_redis.ping()
            except Exception as e:
                logger.warning(f"Failed to connect to Redis for heartbeat: {e}")
                self._dashboard_redis = None
        return self._dashboard_redis

    def _publish_heartbeat(self) -> None:
        """Publish worker heartbeat to Redis."""
        if not self.config.publish_heartbeat:
            return

        r = self._get_dashboard_redis()
        if r is None:
            return

        try:
            import json
            worker_key = f"{WORKER_PREFIX}{self.worker_id}"
            worker_data = {
                "worker_id": self.worker_id,
                "tasks_completed": self.tasks_completed,
                "tasks_failed": self.tasks_failed,
                "consecutive_errors": self.consecutive_errors,
                "running": self.running,
                "last_heartbeat": time.time(),
            }
            r.setex(worker_key, self.config.heartbeat_ttl, json.dumps(worker_data))
        except Exception as e:
            logger.warning(f"Failed to publish heartbeat: {e}")

    def evaluate_task(self, task: EvaluationTask) -> EvaluationResultMessage:
        """
        Evaluate a single task.

        Args:
            task: Evaluation task from queue

        Returns:
            Result message to push to result queue
        """
        start_time = time.time()

        try:
            # Parse genome
            genome = SwarmGenome.from_dict(task.genome_data)

            # Get simulation config
            sim_config = task.simulation_config
            steps = sim_config.get("steps", self.config.default_steps)
            record_interval = sim_config.get("record_interval", 10)

            # Run simulation
            trajectory = self.simulator.run(
                genome=genome,
                steps=steps,
                record_interval=record_interval,
            )

            # Compute fitness
            result = self.fitness_fn.evaluate(trajectory, task.genome_data)

            eval_time = time.time() - start_time

            return EvaluationResultMessage(
                task_id=task.task_id,
                fitness=result.fitness,
                behavior=result.behavior.tolist(),
                genome_data=task.genome_data,
                metadata={
                    **result.metadata,
                    "worker_id": self.worker_id,
                    "eval_time": eval_time,
                    "steps": steps,
                },
                worker_id=self.worker_id,
            )

        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            return EvaluationResultMessage(
                task_id=task.task_id,
                fitness=0.0,
                behavior=[0.0, 0.0],
                genome_data=task.genome_data,
                metadata={"error": str(e)},
                worker_id=self.worker_id,
                error=str(e),
            )

    def process_one(self) -> bool:
        """
        Process a single task if available.

        Returns True if a task was processed, False if queue was empty.
        """
        task = self.queue.pop_task(timeout=self.config.poll_interval)
        if task is None:
            return False

        logger.debug(f"Processing task {task.task_id}")

        result = self.evaluate_task(task)

        if result.error is None:
            self.tasks_completed += 1
            self.consecutive_errors = 0
        else:
            self.tasks_failed += 1
            self.consecutive_errors += 1

        self.queue.push_result(result)
        return True

    def run(self) -> None:
        """
        Run worker continuously until stopped.

        Polls for tasks and processes them.
        """
        self.running = True
        logger.info(f"Worker {self.worker_id} starting")

        # Publish initial heartbeat
        self._publish_heartbeat()

        try:
            while self.running:
                # Check for too many consecutive errors
                if self.consecutive_errors >= self.config.max_consecutive_errors:
                    logger.error(
                        f"Too many consecutive errors ({self.consecutive_errors}), stopping"
                    )
                    break

                # Process task
                processed = self.process_one()

                # Heartbeat logging and Redis publish
                now = time.time()
                if now - self.last_heartbeat >= self.config.heartbeat_interval:
                    logger.info(
                        f"Worker {self.worker_id} heartbeat: "
                        f"{self.tasks_completed} completed, {self.tasks_failed} failed"
                    )
                    self._publish_heartbeat()
                    self.last_heartbeat = now

                # Small sleep if no task to prevent busy-waiting
                if not processed:
                    time.sleep(self.config.poll_interval)

        except KeyboardInterrupt:
            logger.info("Worker interrupted by user")

        finally:
            self.running = False
            logger.info(
                f"Worker {self.worker_id} stopped: "
                f"{self.tasks_completed} completed, {self.tasks_failed} failed"
            )

    def stop(self) -> None:
        """Stop worker gracefully."""
        self.running = False

    def get_status(self) -> dict[str, Any]:
        """Get current worker status."""
        return {
            "worker_id": self.worker_id,
            "running": self.running,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "consecutive_errors": self.consecutive_errors,
        }


def run_worker(config: WorkerConfig | None = None) -> None:
    """
    Run the worker as a standalone service.

    This is the entry point for the worker container.
    """
    import argparse
    import signal
    import sys

    parser = argparse.ArgumentParser(description="Evaluation Worker")
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--redis-url", default="redis://localhost:6379")
    parser.add_argument("--fitness-type", default="coherence", choices=["coherence", "task"])
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if config is None:
        config = WorkerConfig(
            worker_id=args.worker_id or str(uuid.uuid4())[:8],
            queue_backend="redis",
            redis_url=args.redis_url,
            fitness_type=args.fitness_type,
            seed=args.seed,
        )

    worker = EvaluationWorker(config)

    # Handle signals for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run worker
    worker.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_worker()
