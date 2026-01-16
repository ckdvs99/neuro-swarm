"""
neuro_swarm/services/dashboard.py

Web-based monitoring dashboard for the distributed evolution system.

Provides real-time visibility into:
- Evolution progress (generations, fitness curves)
- MAP-Elites archive heatmap
- Worker pool status
- Queue health metrics

The dashboard is read-only and does not control evolution.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for the dashboard service."""

    redis_url: str = "redis://localhost:6379"
    host: str = "0.0.0.0"
    port: int = 8082
    update_interval: float = 1.0  # WebSocket push interval
    history_limit: int = 1000  # Max generations to keep in history
    worker_ttl: int = 60  # Worker heartbeat TTL in seconds


# Redis key constants
REDIS_KEY_PREFIX = "neuro_swarm:"
DASHBOARD_STATUS_KEY = f"{REDIS_KEY_PREFIX}dashboard:status"
DASHBOARD_HISTORY_KEY = f"{REDIS_KEY_PREFIX}dashboard:history"
DASHBOARD_ARCHIVE_KEY = f"{REDIS_KEY_PREFIX}dashboard:archive"
WORKER_PREFIX = f"{REDIS_KEY_PREFIX}workers:"
TASK_QUEUE_KEY = f"{REDIS_KEY_PREFIX}tasks"
RESULT_QUEUE_KEY = f"{REDIS_KEY_PREFIX}results"


class DashboardState:
    """
    Manages dashboard state from Redis.

    Provides methods to read shared state without interfering with evolution.
    """

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._redis = None

    def _get_redis(self) -> Any:
        """Lazy connection to Redis."""
        if self._redis is None:
            try:
                import redis

                self._redis = redis.from_url(self.redis_url, decode_responses=True)
                self._redis.ping()
                logger.info(f"Dashboard connected to Redis at {self.redis_url}")
            except ImportError as err:
                raise ImportError(
                    "redis package required for dashboard. "
                    "Install with: pip install redis"
                ) from err
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        return self._redis

    def get_status(self) -> dict[str, Any]:
        """Get current controller status."""
        try:
            r = self._get_redis()
            status_json = r.get(DASHBOARD_STATUS_KEY)
            if status_json:
                return json.loads(status_json)
            return {
                "running": False,
                "generation": 0,
                "total_evaluations": 0,
                "elapsed_time": 0.0,
                "algorithm": "unknown",
            }
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return {"error": str(e)}

    def get_statistics(self) -> dict[str, Any]:
        """Get algorithm statistics."""
        try:
            r = self._get_redis()
            status_json = r.get(DASHBOARD_STATUS_KEY)
            if status_json:
                status = json.loads(status_json)
                return status.get("statistics", {})
            return {}
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}

    def get_history(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Get generation history with pagination."""
        try:
            r = self._get_redis()
            # History stored as a list in Redis
            history_json = r.lrange(DASHBOARD_HISTORY_KEY, offset, offset + limit - 1)
            return [json.loads(h) for h in history_json]
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return []

    def get_best(self) -> dict[str, Any]:
        """Get best genome found."""
        try:
            r = self._get_redis()
            status_json = r.get(DASHBOARD_STATUS_KEY)
            if status_json:
                status = json.loads(status_json)
                return {
                    "fitness": status.get("best_fitness"),
                    "genome": status.get("best_genome"),
                }
            return {"fitness": None, "genome": None}
        except Exception as e:
            logger.error(f"Failed to get best: {e}")
            return {"error": str(e)}

    def get_archive(self) -> dict[str, Any]:
        """Get MAP-Elites archive as heatmap data."""
        try:
            r = self._get_redis()
            archive_json = r.get(DASHBOARD_ARCHIVE_KEY)
            if archive_json:
                return json.loads(archive_json)
            return {
                "grid": [],
                "grid_resolution": [20, 20],
                "coverage": 0.0,
                "behavior_bounds": [[0, 1], [0, 1]],
            }
        except Exception as e:
            logger.error(f"Failed to get archive: {e}")
            return {"error": str(e)}

    def get_workers(self) -> list[dict[str, Any]]:
        """Get worker status list."""
        try:
            r = self._get_redis()
            workers = []

            # Scan for worker keys
            cursor = 0
            while True:
                cursor, keys = r.scan(cursor, match=f"{WORKER_PREFIX}*")
                for key in keys:
                    worker_json = r.get(key)
                    if worker_json:
                        worker_data = json.loads(worker_json)
                        # Calculate time since last heartbeat
                        last_seen = worker_data.get("last_heartbeat", 0)
                        worker_data["seconds_since_heartbeat"] = time.time() - last_seen
                        workers.append(worker_data)
                if cursor == 0:
                    break

            return workers
        except Exception as e:
            logger.error(f"Failed to get workers: {e}")
            return []

    def get_queue_stats(self) -> dict[str, Any]:
        """Get queue lengths and health metrics."""
        try:
            r = self._get_redis()
            task_length = r.llen(TASK_QUEUE_KEY)
            result_length = r.llen(RESULT_QUEUE_KEY)

            # Get throughput from status if available
            status_json = r.get(DASHBOARD_STATUS_KEY)
            throughput = 0.0
            if status_json:
                status = json.loads(status_json)
                elapsed = status.get("elapsed_time", 1.0) or 1.0
                total_evals = status.get("total_evaluations", 0)
                throughput = total_evals / elapsed

            return {
                "task_queue_length": task_length,
                "result_queue_length": result_length,
                "throughput": throughput,
                "healthy": task_length < 1000,  # Alert if queue backing up
            }
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {"error": str(e)}


def create_app(config: DashboardConfig) -> Any:
    """Create the FastAPI application."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(
        title="Neuro-Swarm Dashboard",
        description="Monitoring dashboard for distributed evolution",
        version="0.1.0",
    )

    # Dashboard state
    state = DashboardState(config.redis_url)

    # WebSocket connections
    active_connections: set[WebSocket] = set()

    # Static files directory
    static_dir = Path(__file__).parent / "static"

    # Mount static files if directory exists
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def root() -> Any:
        """Serve the dashboard HTML."""
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return HTMLResponse(
            content="<h1>Dashboard static files not found</h1>",
            status_code=404,
        )

    @app.get("/api/status")
    async def get_status() -> dict[str, Any]:
        """Get current controller status."""
        return state.get_status()

    @app.get("/api/statistics")
    async def get_statistics() -> dict[str, Any]:
        """Get algorithm statistics."""
        return state.get_statistics()

    @app.get("/api/history")
    async def get_history(limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Get generation history with pagination."""
        return state.get_history(limit=limit, offset=offset)

    @app.get("/api/best")
    async def get_best() -> dict[str, Any]:
        """Get best genome found."""
        return state.get_best()

    @app.get("/api/archive")
    async def get_archive() -> dict[str, Any]:
        """Get MAP-Elites archive as heatmap data."""
        return state.get_archive()

    @app.get("/api/workers")
    async def get_workers() -> list[dict[str, Any]]:
        """Get worker status list."""
        return state.get_workers()

    @app.get("/api/queue")
    async def get_queue() -> dict[str, Any]:
        """Get queue lengths and health metrics."""
        return state.get_queue_stats()

    @app.websocket("/ws/updates")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """WebSocket endpoint for real-time updates."""
        await websocket.accept()
        active_connections.add(websocket)
        logger.info(
            f"WebSocket connected. Total connections: {len(active_connections)}"
        )

        try:
            while True:
                # Send updates periodically
                update = {
                    "type": "update",
                    "timestamp": time.time(),
                    "status": state.get_status(),
                    "queue": state.get_queue_stats(),
                }
                await websocket.send_json(update)
                await asyncio.sleep(config.update_interval)
        except WebSocketDisconnect:
            active_connections.discard(websocket)
            logger.info(
                f"WebSocket disconnected. Total connections: {len(active_connections)}"
            )
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            active_connections.discard(websocket)

    @app.on_event("startup")
    async def startup() -> None:
        """Log startup."""
        logger.info(f"Dashboard starting on {config.host}:{config.port}")

    @app.on_event("shutdown")
    async def shutdown() -> None:
        """Clean up on shutdown."""
        logger.info("Dashboard shutting down")
        # Close all WebSocket connections
        for conn in active_connections:
            try:
                await conn.close()
            except Exception:
                pass

    return app


def run_dashboard(config: DashboardConfig | None = None) -> None:
    """
    Run the dashboard as a standalone service.

    This is the entry point for the dashboard container.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Evolution Dashboard")
    parser.add_argument("--redis-url", default="redis://localhost:6379")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)

    args = parser.parse_args()

    if config is None:
        config = DashboardConfig(
            redis_url=args.redis_url,
            host=args.host,
            port=args.port,
        )

    app = create_app(config)

    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_dashboard()
