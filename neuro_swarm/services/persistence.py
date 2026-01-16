"""
neuro_swarm/services/persistence.py

Database persistence layer for evolution data.

Stores evolution runs, generation statistics, and archive snapshots
to TimescaleDB for long-term analysis and reproducibility.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PersistenceConfig:
    """Configuration for database persistence."""

    # Connection settings
    host: str = "localhost"
    port: int = 5432
    database: str = "neuro_swarm"
    user: str = "postgres"
    password: str = ""

    # Behavior settings
    enabled: bool = True
    archive_snapshot_interval: int = 10  # Generations between archive snapshots
    store_genomes: bool = False  # Store individual evaluated genomes (large)

    @classmethod
    def from_env(cls) -> PersistenceConfig:
        """Create config from environment variables."""
        return cls(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            database=os.environ.get("POSTGRES_DB", "neuro_swarm"),
            user=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", ""),
            enabled=os.environ.get("DB_PERSISTENCE_ENABLED", "true").lower() == "true",
            store_genomes=os.environ.get("DB_STORE_GENOMES", "false").lower()
            == "true",
        )

    @property
    def connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        return (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )


class EvolutionPersistence:
    """
    Handles persistence of evolution data to TimescaleDB.

    Thread-safe connection management with lazy initialization.
    """

    def __init__(self, config: PersistenceConfig):
        self.config = config
        self._conn = None
        self._run_id: uuid.UUID | None = None

    def _get_connection(self) -> Any:
        """Get or create database connection."""
        if not self.config.enabled:
            return None

        if self._conn is None:
            try:
                import psycopg2

                self._conn = psycopg2.connect(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.user,
                    password=self.config.password,
                )
                self._conn.autocommit = True
                logger.info(
                    f"Database connection established to {self.config.host}:{self.config.port}"
                )
            except ImportError:
                logger.warning(
                    "psycopg2 not installed. Install with: pip install psycopg2-binary"
                )
                self.config.enabled = False
                return None
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                self.config.enabled = False
                return None

        return self._conn

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("Database connection closed")

    def start_run(
        self,
        algorithm: str,
        config: dict[str, Any],
    ) -> uuid.UUID | None:
        """
        Start a new evolution run.

        Args:
            algorithm: Name of the algorithm (es, map_elites, novelty)
            config: Evolution configuration dictionary

        Returns:
            Run ID if successful, None otherwise
        """
        conn = self._get_connection()
        if conn is None:
            return None

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO evolution_runs (algorithm, config, status)
                    VALUES (%s, %s, 'running')
                    RETURNING run_id
                    """,
                    (algorithm, json.dumps(config)),
                )
                result = cur.fetchone()
                if result:
                    self._run_id = result[0]
                    logger.info(f"Started evolution run: {self._run_id}")
                    return self._run_id
        except Exception as e:
            logger.error(f"Failed to start run: {e}")

        return None

    def end_run(
        self,
        best_fitness: float,
        best_genome: dict[str, Any] | None,
        total_generations: int,
        total_evaluations: int,
    ) -> None:
        """
        Mark an evolution run as completed.

        Args:
            best_fitness: Final best fitness achieved
            best_genome: Best genome found (as dict)
            total_generations: Total generations run
            total_evaluations: Total evaluations performed
        """
        if self._run_id is None:
            return

        conn = self._get_connection()
        if conn is None:
            return

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE evolution_runs
                    SET ended_at = NOW(),
                        status = 'completed',
                        final_best_fitness = %s,
                        final_best_genome = %s,
                        total_generations = %s,
                        total_evaluations = %s
                    WHERE run_id = %s
                    """,
                    (
                        best_fitness,
                        json.dumps(best_genome) if best_genome else None,
                        total_generations,
                        total_evaluations,
                        str(self._run_id),
                    ),
                )
                logger.info(f"Ended evolution run: {self._run_id}")
        except Exception as e:
            logger.error(f"Failed to end run: {e}")

    def record_generation(
        self,
        generation: int,
        stats: dict[str, Any],
        best_genome: dict[str, Any] | None = None,
    ) -> None:
        """
        Record statistics for a generation.

        Args:
            generation: Generation number
            stats: Statistics dictionary from algorithm
            best_genome: Best genome for this generation
        """
        if self._run_id is None:
            return

        conn = self._get_connection()
        if conn is None:
            return

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO generation_stats (
                        run_id, generation, best_fitness, mean_fitness,
                        min_fitness, std_fitness, archive_size, archive_coverage,
                        generation_time_ms, evaluations, best_genome, behavior_coverage
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(self._run_id),
                        generation,
                        stats.get("best_fitness"),
                        stats.get("mean_fitness"),
                        stats.get("min_fitness"),
                        stats.get("std_fitness"),
                        stats.get("archive_size"),
                        stats.get("archive_coverage"),
                        int(stats.get("generation_time", 0) * 1000),
                        stats.get("results_received", 0),
                        json.dumps(best_genome) if best_genome else None,
                        json.dumps(stats.get("behavior_coverage"))
                        if stats.get("behavior_coverage")
                        else None,
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to record generation {generation}: {e}")

    def record_archive_snapshot(
        self,
        generation: int,
        archive_data: dict[str, Any],
    ) -> None:
        """
        Record a snapshot of the MAP-Elites archive.

        Args:
            generation: Generation number
            archive_data: Archive data (grid, coverage, etc.)
        """
        if self._run_id is None:
            return

        conn = self._get_connection()
        if conn is None:
            return

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO archive_snapshots (run_id, generation, archive_data)
                    VALUES (%s, %s, %s)
                    """,
                    (
                        str(self._run_id),
                        generation,
                        json.dumps(archive_data),
                    ),
                )
                logger.debug(f"Recorded archive snapshot at generation {generation}")
        except Exception as e:
            logger.error(f"Failed to record archive snapshot: {e}")

    def record_evaluated_genome(
        self,
        generation: int,
        genome: dict[str, Any],
        fitness: float,
        behavior: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record an evaluated genome (optional, can generate large data).

        Args:
            generation: Generation number
            genome: Genome data as dict
            fitness: Fitness value
            behavior: Behavioral descriptor
            metadata: Additional metadata
        """
        if not self.config.store_genomes or self._run_id is None:
            return

        conn = self._get_connection()
        if conn is None:
            return

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO evaluated_genomes (
                        run_id, generation, genome, fitness, behavior, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(self._run_id),
                        generation,
                        json.dumps(genome),
                        fitness,
                        behavior,
                        json.dumps(metadata) if metadata else None,
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to record evaluated genome: {e}")

    def record_worker_stats(
        self,
        worker_id: str,
        tasks_completed: int,
        tasks_failed: int,
        avg_task_time_ms: float,
    ) -> None:
        """
        Record worker statistics.

        Args:
            worker_id: Worker identifier
            tasks_completed: Total tasks completed
            tasks_failed: Total tasks failed
            avg_task_time_ms: Average task time in milliseconds
        """
        conn = self._get_connection()
        if conn is None:
            return

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO worker_stats (
                        worker_id, run_id, tasks_completed, tasks_failed, avg_task_time_ms
                    ) VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        worker_id,
                        str(self._run_id) if self._run_id else None,
                        tasks_completed,
                        tasks_failed,
                        avg_task_time_ms,
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to record worker stats: {e}")

    @property
    def run_id(self) -> uuid.UUID | None:
        """Get current run ID."""
        return self._run_id

    def get_run_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get recent evolution runs.

        Args:
            limit: Maximum number of runs to return

        Returns:
            List of run summaries
        """
        conn = self._get_connection()
        if conn is None:
            return []

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT run_id, started_at, ended_at, algorithm, status,
                           final_best_fitness, total_generations, total_evaluations
                    FROM evolution_runs
                    ORDER BY started_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
                return [
                    {
                        "run_id": str(row[0]),
                        "started_at": row[1].isoformat() if row[1] else None,
                        "ended_at": row[2].isoformat() if row[2] else None,
                        "algorithm": row[3],
                        "status": row[4],
                        "best_fitness": row[5],
                        "generations": row[6],
                        "evaluations": row[7],
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to get run history: {e}")
            return []

    def get_generation_history(
        self, run_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """
        Get generation statistics for a run.

        Args:
            run_id: Run ID (defaults to current run)
            limit: Maximum generations to return

        Returns:
            List of generation statistics
        """
        run_id = run_id or (str(self._run_id) if self._run_id else None)
        if run_id is None:
            return []

        conn = self._get_connection()
        if conn is None:
            return []

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT generation, best_fitness, mean_fitness, min_fitness,
                           std_fitness, archive_size, archive_coverage,
                           generation_time_ms, evaluations
                    FROM generation_stats
                    WHERE run_id = %s
                    ORDER BY generation DESC
                    LIMIT %s
                    """,
                    (run_id, limit),
                )
                rows = cur.fetchall()
                return [
                    {
                        "generation": row[0],
                        "best_fitness": row[1],
                        "mean_fitness": row[2],
                        "min_fitness": row[3],
                        "std_fitness": row[4],
                        "archive_size": row[5],
                        "archive_coverage": row[6],
                        "generation_time_ms": row[7],
                        "evaluations": row[8],
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to get generation history: {e}")
            return []
