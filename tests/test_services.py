"""
Tests for neuro_swarm/services/

Tests queue, controller, and worker components.
"""

import pytest
import numpy as np
import time
import json

from neuro_swarm.services.queue import (
    TaskStatus,
    EvaluationTask,
    EvaluationResultMessage,
    InMemoryTaskQueue,
    create_task_queue,
)
from neuro_swarm.services.controller import EvolutionController, ControllerConfig
from neuro_swarm.services.worker import EvaluationWorker, WorkerConfig, SwarmSimulator
from neuro_swarm.evolution.genome import SwarmGenome


# ==================== Queue Tests ====================

class TestEvaluationTask:
    """Tests for EvaluationTask."""

    def test_serialization(self):
        """Task serializes and deserializes correctly."""
        task = EvaluationTask(
            task_id="test-123",
            genome_data={"type": "SwarmGenome", "num_agents": 7},
            simulation_config={"steps": 100},
            generation=5,
        )

        json_str = task.to_json()
        restored = EvaluationTask.from_json(json_str)

        assert restored.task_id == task.task_id
        assert restored.genome_data == task.genome_data
        assert restored.generation == task.generation

    def test_default_timeout(self):
        """Task has default timeout."""
        task = EvaluationTask(
            task_id="test",
            genome_data={},
            simulation_config={},
        )
        assert task.timeout_seconds == 300.0


class TestEvaluationResultMessage:
    """Tests for EvaluationResultMessage."""

    def test_serialization(self):
        """Result serializes and deserializes correctly."""
        result = EvaluationResultMessage(
            task_id="test-123",
            fitness=0.85,
            behavior=[0.5, 0.5],
            genome_data={"type": "SwarmGenome"},
            metadata={"steps": 100},
            worker_id="worker-1",
        )

        json_str = result.to_json()
        restored = EvaluationResultMessage.from_json(json_str)

        assert restored.task_id == result.task_id
        assert restored.fitness == result.fitness
        assert restored.behavior == result.behavior
        assert restored.worker_id == result.worker_id

    def test_error_field(self):
        """Error field is preserved."""
        result = EvaluationResultMessage(
            task_id="test",
            fitness=0.0,
            behavior=[0.0, 0.0],
            genome_data={},
            error="Simulation failed",
        )

        json_str = result.to_json()
        restored = EvaluationResultMessage.from_json(json_str)

        assert restored.error == "Simulation failed"


class TestInMemoryTaskQueue:
    """Tests for InMemoryTaskQueue."""

    def test_push_and_pop_task(self):
        """Tasks can be pushed and popped."""
        queue = InMemoryTaskQueue()
        task = EvaluationTask(
            task_id="test-1",
            genome_data={"test": True},
            simulation_config={},
        )

        queue.push_task(task)
        assert queue.get_queue_length() == 1

        popped = queue.pop_task(timeout=0.1)
        assert popped is not None
        assert popped.task_id == task.task_id
        assert queue.get_queue_length() == 0

    def test_push_and_pop_result(self):
        """Results can be pushed and popped."""
        queue = InMemoryTaskQueue()
        result = EvaluationResultMessage(
            task_id="test-1",
            fitness=0.5,
            behavior=[0.5, 0.5],
            genome_data={},
        )

        queue.push_result(result)
        assert queue.get_result_count() == 1

        popped = queue.pop_result(timeout=0.1)
        assert popped is not None
        assert popped.task_id == result.task_id

    def test_pop_empty_queue_returns_none(self):
        """Popping empty queue returns None."""
        queue = InMemoryTaskQueue()

        task = queue.pop_task(timeout=0.01)
        assert task is None

        result = queue.pop_result(timeout=0.01)
        assert result is None

    def test_fifo_ordering(self):
        """Tasks are processed in FIFO order."""
        queue = InMemoryTaskQueue()

        for i in range(3):
            task = EvaluationTask(
                task_id=f"task-{i}",
                genome_data={},
                simulation_config={},
            )
            queue.push_task(task)

        for i in range(3):
            task = queue.pop_task(timeout=0.1)
            assert task.task_id == f"task-{i}"

    def test_clear(self):
        """Clear empties all queues."""
        queue = InMemoryTaskQueue()

        queue.push_task(EvaluationTask("t1", {}, {}))
        queue.push_result(EvaluationResultMessage("r1", 0.5, [0.5], {}))

        queue.clear()

        assert queue.get_queue_length() == 0
        assert queue.get_result_count() == 0

    def test_task_status_tracking(self):
        """Task status is tracked correctly."""
        queue = InMemoryTaskQueue()
        task = EvaluationTask("test-1", {}, {})

        queue.push_task(task)
        assert queue.get_task_status("test-1") == TaskStatus.PENDING

        queue.pop_task(timeout=0.1)
        assert queue.get_task_status("test-1") == TaskStatus.IN_PROGRESS


class TestCreateTaskQueue:
    """Tests for queue factory function."""

    def test_create_memory_queue(self):
        """Factory creates memory queue."""
        queue = create_task_queue(backend="memory")
        assert isinstance(queue, InMemoryTaskQueue)

    def test_create_unknown_backend_raises(self):
        """Unknown backend raises ValueError."""
        with pytest.raises(ValueError):
            create_task_queue(backend="unknown")


# ==================== Simulator Tests ====================

class TestSwarmSimulator:
    """Tests for SwarmSimulator."""

    def test_run_returns_trajectory(self):
        """Simulator returns trajectory list."""
        rng = np.random.default_rng(42)
        genome = SwarmGenome.random(rng)
        simulator = SwarmSimulator(world_size=50.0, seed=42)

        trajectory = simulator.run(genome, steps=50, record_interval=10)

        assert isinstance(trajectory, list)
        assert len(trajectory) > 0

    def test_trajectory_contains_required_fields(self):
        """Trajectory states contain required fields."""
        rng = np.random.default_rng(42)
        genome = SwarmGenome.random(rng)
        simulator = SwarmSimulator(seed=42)

        trajectory = simulator.run(genome, steps=20, record_interval=10)

        for state in trajectory:
            assert "positions" in state
            assert "velocities" in state
            assert "energies" in state
            assert "ages" in state

    def test_trajectory_positions_are_valid(self):
        """Trajectory positions are valid arrays."""
        rng = np.random.default_rng(42)
        genome = SwarmGenome.random(rng)
        simulator = SwarmSimulator(seed=42)

        trajectory = simulator.run(genome, steps=20, record_interval=10)

        for state in trajectory:
            positions = state["positions"]
            assert len(positions) == genome.num_agents
            for pos in positions:
                assert len(pos) == 2
                assert all(np.isfinite(pos))


# ==================== Worker Tests ====================

class TestEvaluationWorker:
    """Tests for EvaluationWorker."""

    def test_evaluate_task_returns_result(self):
        """Worker evaluates task and returns result."""
        config = WorkerConfig(queue_backend="memory", seed=42)
        worker = EvaluationWorker(config)

        rng = np.random.default_rng(42)
        genome = SwarmGenome.random(rng)

        task = EvaluationTask(
            task_id="test-1",
            genome_data=genome.to_dict(),
            simulation_config={"steps": 20, "record_interval": 10},
        )

        result = worker.evaluate_task(task)

        assert result.task_id == task.task_id
        assert result.error is None
        assert np.isfinite(result.fitness)

    def test_evaluate_task_handles_error(self):
        """Worker handles evaluation errors gracefully."""
        config = WorkerConfig(queue_backend="memory")
        worker = EvaluationWorker(config)

        # Use a genome with invalid nested structure that will cause parsing error
        task = EvaluationTask(
            task_id="test-1",
            genome_data={"agent_genome": "not_a_dict"},  # Invalid nested structure
            simulation_config={},
        )

        result = worker.evaluate_task(task)

        assert result.error is not None
        assert result.fitness == 0.0

    def test_process_one_with_queue(self):
        """Worker processes task from queue."""
        config = WorkerConfig(queue_backend="memory", seed=42)
        worker = EvaluationWorker(config)

        rng = np.random.default_rng(42)
        genome = SwarmGenome.random(rng)

        task = EvaluationTask(
            task_id="test-1",
            genome_data=genome.to_dict(),
            simulation_config={"steps": 20},
        )

        worker.queue.push_task(task)
        processed = worker.process_one()

        assert processed is True
        assert worker.tasks_completed == 1

        result = worker.queue.pop_result(timeout=0.1)
        assert result is not None
        assert result.task_id == task.task_id

    def test_get_status(self):
        """Worker status is correctly reported."""
        config = WorkerConfig(queue_backend="memory")
        worker = EvaluationWorker(config)

        status = worker.get_status()

        assert "worker_id" in status
        assert "running" in status
        assert "tasks_completed" in status


# ==================== Controller Tests ====================

class TestEvolutionController:
    """Tests for EvolutionController."""

    def test_generate_tasks(self):
        """Controller generates evaluation tasks."""
        config = ControllerConfig(
            algorithm="es",
            population_size=10,
            queue_backend="memory",
        )
        controller = EvolutionController(config)

        num_tasks = controller.generate_tasks()

        assert num_tasks == 10
        assert controller.queue.get_queue_length() == 10
        assert len(controller.pending_tasks) == 10

    def test_step_with_results(self):
        """Controller step processes results."""
        config = ControllerConfig(
            algorithm="es",
            population_size=5,
            queue_backend="memory",
            result_timeout=1.0,
        )
        controller = EvolutionController(config)

        # Generate tasks
        controller.generate_tasks()

        # Simulate worker processing
        while controller.queue.get_queue_length() > 0:
            task = controller.queue.pop_task(timeout=0.1)
            if task:
                result = EvaluationResultMessage(
                    task_id=task.task_id,
                    fitness=np.random.random(),
                    behavior=[np.random.random(), np.random.random()],
                    genome_data=task.genome_data,
                )
                controller.queue.push_result(result)

        # Collect results
        results = controller.collect_results(timeout=1.0)

        assert len(results) == 5
        assert controller.total_evaluations == 5

    def test_get_status(self):
        """Controller status is correctly reported."""
        config = ControllerConfig(queue_backend="memory")
        controller = EvolutionController(config)

        status = controller.get_status()

        assert "running" in status
        assert "generation" in status
        assert "algorithm" in status


# ==================== Integration Tests ====================

class TestControllerWorkerIntegration:
    """Integration tests for controller and worker."""

    def test_single_generation(self):
        """Controller and worker complete a generation."""
        # Setup controller
        controller_config = ControllerConfig(
            algorithm="es",
            population_size=5,
            queue_backend="memory",
            result_timeout=10.0,
        )
        controller = EvolutionController(controller_config)

        # Setup worker
        worker_config = WorkerConfig(queue_backend="memory", seed=42)
        # Share the same queue
        worker = EvaluationWorker(worker_config)
        worker.queue = controller.queue

        # Generate tasks
        controller.generate_tasks()

        # Process all tasks
        while controller.queue.get_queue_length() > 0:
            worker.process_one()

        # Collect results
        results = controller.collect_results(timeout=5.0)

        assert len(results) == 5
        assert all(r.fitness >= 0 for r in results)

    def test_multiple_generations(self):
        """Multiple generations complete successfully."""
        controller_config = ControllerConfig(
            algorithm="es",
            population_size=3,
            queue_backend="memory",
            result_timeout=10.0,
        )
        controller = EvolutionController(controller_config)

        worker_config = WorkerConfig(queue_backend="memory", seed=42)
        worker = EvaluationWorker(worker_config)
        worker.queue = controller.queue

        for gen in range(3):
            # Generate
            controller.generate_tasks()

            # Process
            while controller.queue.get_queue_length() > 0:
                worker.process_one()

            # Collect
            results = controller.collect_results(timeout=5.0)
            controller.algorithm.tell(results)
            controller.generation += 1

        assert controller.generation == 3
        assert controller.total_evaluations == 9
