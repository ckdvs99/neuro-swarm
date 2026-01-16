"""
neuro_swarm/services/queue.py

Redis-based task queue for distributed evolution.

The queue provides:
- Task distribution to workers
- Result collection from workers
- Task status tracking
- Timeout handling for failed workers
- Graceful degradation when Redis is unavailable

Inspired by simple, robust message passing systems.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
import json
import time
import uuid
import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of an evaluation task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class EvaluationTask:
    """
    A single evaluation task for a worker.

    Contains everything needed to evaluate a genome:
    - The genome configuration
    - Simulation parameters
    - Task metadata
    """
    task_id: str
    genome_data: Dict[str, Any]
    simulation_config: Dict[str, Any]
    generation: int = 0
    created_at: float = field(default_factory=time.time)
    timeout_seconds: float = 300.0  # 5 minute default

    def to_json(self) -> str:
        """Serialize task to JSON."""
        return json.dumps({
            "task_id": self.task_id,
            "genome_data": self.genome_data,
            "simulation_config": self.simulation_config,
            "generation": self.generation,
            "created_at": self.created_at,
            "timeout_seconds": self.timeout_seconds,
        })

    @classmethod
    def from_json(cls, data: str) -> "EvaluationTask":
        """Deserialize task from JSON."""
        d = json.loads(data)
        return cls(
            task_id=d["task_id"],
            genome_data=d["genome_data"],
            simulation_config=d["simulation_config"],
            generation=d.get("generation", 0),
            created_at=d.get("created_at", time.time()),
            timeout_seconds=d.get("timeout_seconds", 300.0),
        )


@dataclass
class EvaluationResultMessage:
    """
    Result of evaluating a genome.

    Sent from worker back to controller.
    """
    task_id: str
    fitness: float
    behavior: List[float]
    genome_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    worker_id: str = ""
    completed_at: float = field(default_factory=time.time)
    error: Optional[str] = None

    def to_json(self) -> str:
        """Serialize result to JSON."""
        return json.dumps({
            "task_id": self.task_id,
            "fitness": self.fitness,
            "behavior": self.behavior,
            "genome_data": self.genome_data,
            "metadata": self.metadata,
            "worker_id": self.worker_id,
            "completed_at": self.completed_at,
            "error": self.error,
        })

    @classmethod
    def from_json(cls, data: str) -> "EvaluationResultMessage":
        """Deserialize result from JSON."""
        d = json.loads(data)
        return cls(
            task_id=d["task_id"],
            fitness=d["fitness"],
            behavior=d["behavior"],
            genome_data=d["genome_data"],
            metadata=d.get("metadata", {}),
            worker_id=d.get("worker_id", ""),
            completed_at=d.get("completed_at", time.time()),
            error=d.get("error"),
        )


class TaskQueue(ABC):
    """
    Abstract base for task queue implementations.

    Provides interface for distributed task management.
    """

    @abstractmethod
    def push_task(self, task: EvaluationTask) -> bool:
        """Push a task to the queue. Returns True if successful."""
        pass

    @abstractmethod
    def pop_task(self, timeout: float = 1.0) -> Optional[EvaluationTask]:
        """Pop a task from the queue. Returns None if queue is empty."""
        pass

    @abstractmethod
    def push_result(self, result: EvaluationResultMessage) -> bool:
        """Push a result to the result queue. Returns True if successful."""
        pass

    @abstractmethod
    def pop_result(self, timeout: float = 1.0) -> Optional[EvaluationResultMessage]:
        """Pop a result from the result queue. Returns None if empty."""
        pass

    @abstractmethod
    def get_queue_length(self) -> int:
        """Get number of pending tasks."""
        pass

    @abstractmethod
    def get_result_count(self) -> int:
        """Get number of pending results."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all queues."""
        pass


class RedisTaskQueue(TaskQueue):
    """
    Redis-backed task queue for production use.

    Uses Redis lists for FIFO queuing with blocking operations.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        task_queue_key: str = "neuro_swarm:tasks",
        result_queue_key: str = "neuro_swarm:results",
        status_key_prefix: str = "neuro_swarm:status:",
    ):
        self.redis_url = redis_url
        self.task_queue_key = task_queue_key
        self.result_queue_key = result_queue_key
        self.status_key_prefix = status_key_prefix
        self._redis = None

    def _get_redis(self):
        """Lazy connection to Redis."""
        if self._redis is None:
            try:
                import redis
                self._redis = redis.from_url(self.redis_url)
                self._redis.ping()  # Test connection
                logger.info(f"Connected to Redis at {self.redis_url}")
            except ImportError:
                raise ImportError(
                    "redis package required for RedisTaskQueue. "
                    "Install with: pip install redis"
                )
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        return self._redis

    def push_task(self, task: EvaluationTask) -> bool:
        """Push task to Redis list."""
        try:
            r = self._get_redis()
            r.rpush(self.task_queue_key, task.to_json())
            r.set(
                f"{self.status_key_prefix}{task.task_id}",
                TaskStatus.PENDING.value,
                ex=int(task.timeout_seconds * 2)
            )
            return True
        except Exception as e:
            logger.error(f"Failed to push task: {e}")
            return False

    def pop_task(self, timeout: float = 1.0) -> Optional[EvaluationTask]:
        """Pop task from Redis list with blocking."""
        try:
            r = self._get_redis()
            result = r.blpop(self.task_queue_key, timeout=timeout)
            if result is None:
                return None
            _, data = result
            task = EvaluationTask.from_json(data.decode("utf-8"))
            r.set(
                f"{self.status_key_prefix}{task.task_id}",
                TaskStatus.IN_PROGRESS.value,
                ex=int(task.timeout_seconds * 2)
            )
            return task
        except Exception as e:
            logger.error(f"Failed to pop task: {e}")
            return None

    def push_result(self, result: EvaluationResultMessage) -> bool:
        """Push result to Redis list."""
        try:
            r = self._get_redis()
            r.rpush(self.result_queue_key, result.to_json())
            status = TaskStatus.COMPLETED if result.error is None else TaskStatus.FAILED
            r.set(
                f"{self.status_key_prefix}{result.task_id}",
                status.value,
                ex=3600  # Keep status for 1 hour
            )
            return True
        except Exception as e:
            logger.error(f"Failed to push result: {e}")
            return False

    def pop_result(self, timeout: float = 1.0) -> Optional[EvaluationResultMessage]:
        """Pop result from Redis list with blocking."""
        try:
            r = self._get_redis()
            result = r.blpop(self.result_queue_key, timeout=timeout)
            if result is None:
                return None
            _, data = result
            return EvaluationResultMessage.from_json(data.decode("utf-8"))
        except Exception as e:
            logger.error(f"Failed to pop result: {e}")
            return None

    def get_queue_length(self) -> int:
        """Get number of pending tasks."""
        try:
            r = self._get_redis()
            return r.llen(self.task_queue_key)
        except Exception:
            return 0

    def get_result_count(self) -> int:
        """Get number of pending results."""
        try:
            r = self._get_redis()
            return r.llen(self.result_queue_key)
        except Exception:
            return 0

    def clear(self) -> None:
        """Clear all queues."""
        try:
            r = self._get_redis()
            r.delete(self.task_queue_key)
            r.delete(self.result_queue_key)
            # Clear status keys (scan pattern)
            cursor = 0
            while True:
                cursor, keys = r.scan(cursor, match=f"{self.status_key_prefix}*")
                if keys:
                    r.delete(*keys)
                if cursor == 0:
                    break
            logger.info("Cleared all queues")
        except Exception as e:
            logger.error(f"Failed to clear queues: {e}")

    def get_task_status(self, task_id: str) -> TaskStatus:
        """Get status of a specific task."""
        try:
            r = self._get_redis()
            status = r.get(f"{self.status_key_prefix}{task_id}")
            if status is None:
                return TaskStatus.PENDING
            return TaskStatus(status.decode("utf-8"))
        except Exception:
            return TaskStatus.PENDING


class InMemoryTaskQueue(TaskQueue):
    """
    In-memory task queue for testing and single-machine use.

    Thread-safe implementation using queues.
    """

    def __init__(self):
        import queue
        import threading
        self._task_queue: queue.Queue = queue.Queue()
        self._result_queue: queue.Queue = queue.Queue()
        self._status: Dict[str, TaskStatus] = {}
        self._lock = threading.Lock()

    def push_task(self, task: EvaluationTask) -> bool:
        """Push task to in-memory queue."""
        try:
            self._task_queue.put(task)
            with self._lock:
                self._status[task.task_id] = TaskStatus.PENDING
            return True
        except Exception:
            return False

    def pop_task(self, timeout: float = 1.0) -> Optional[EvaluationTask]:
        """Pop task from in-memory queue."""
        try:
            import queue
            task = self._task_queue.get(timeout=timeout)
            with self._lock:
                self._status[task.task_id] = TaskStatus.IN_PROGRESS
            return task
        except Exception:
            return None

    def push_result(self, result: EvaluationResultMessage) -> bool:
        """Push result to in-memory queue."""
        try:
            self._result_queue.put(result)
            with self._lock:
                status = TaskStatus.COMPLETED if result.error is None else TaskStatus.FAILED
                self._status[result.task_id] = status
            return True
        except Exception:
            return False

    def pop_result(self, timeout: float = 1.0) -> Optional[EvaluationResultMessage]:
        """Pop result from in-memory queue."""
        try:
            import queue
            return self._result_queue.get(timeout=timeout)
        except Exception:
            return None

    def get_queue_length(self) -> int:
        """Get number of pending tasks."""
        return self._task_queue.qsize()

    def get_result_count(self) -> int:
        """Get number of pending results."""
        return self._result_queue.qsize()

    def clear(self) -> None:
        """Clear all queues."""
        import queue
        while True:
            try:
                self._task_queue.get_nowait()
            except Exception:
                break
        while True:
            try:
                self._result_queue.get_nowait()
            except Exception:
                break
        with self._lock:
            self._status.clear()

    def get_task_status(self, task_id: str) -> TaskStatus:
        """Get status of a specific task."""
        with self._lock:
            return self._status.get(task_id, TaskStatus.PENDING)


def create_task_queue(
    backend: str = "memory",
    redis_url: str = "redis://localhost:6379",
    **kwargs
) -> TaskQueue:
    """
    Factory function to create a task queue.

    Args:
        backend: "memory" or "redis"
        redis_url: Redis connection URL (for redis backend)
        **kwargs: Additional backend-specific options

    Returns:
        TaskQueue instance
    """
    if backend == "memory":
        return InMemoryTaskQueue()
    elif backend == "redis":
        return RedisTaskQueue(redis_url=redis_url, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
