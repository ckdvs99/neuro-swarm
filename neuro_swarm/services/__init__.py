"""
neuro_swarm/services/

Distributed services for evolutionary optimization.

Architecture:
- Controller: Manages evolution state, generates candidates, aggregates results
- Worker: Evaluates candidate genomes in parallel
- Queue: Redis-based task distribution

The controller generates genomes and pushes them to a task queue.
Workers pull tasks, run simulations, and push results to a result queue.
The controller aggregates results and advances evolution.

This is embarrassingly parallel at the evaluation step.
"""

from .queue import TaskQueue, TaskStatus, EvaluationTask, EvaluationResultMessage
from .controller import EvolutionController, ControllerConfig
from .worker import EvaluationWorker, WorkerConfig

__all__ = [
    "TaskQueue",
    "TaskStatus",
    "EvaluationTask",
    "EvaluationResultMessage",
    "EvolutionController",
    "ControllerConfig",
    "EvaluationWorker",
    "WorkerConfig",
]
