"""
Persistence layer.

Responsibility: durably store and retrieve `JobRecord` rows. This layer
knows nothing about job *lifecycle rules* (which transitions are valid,
locking semantics for concurrent updates, etc.) — that belongs to
`src/state`. It also knows nothing about video files — those belong to
`src/artifacts`. This module is a dumb, swappable storage backend: today
it's one JSON file per job on local disk; tomorrow it could be a Postgres
table or a Redis hash without any other layer changing.
"""
from __future__ import annotations

import abc
import asyncio
import json
from pathlib import Path
from typing import Optional

from src.models import JobRecord


class JobNotFoundError(KeyError):
    pass


class JobRepository(abc.ABC):
    """Abstract persistence contract for job metadata."""

    @abc.abstractmethod
    async def save(self, job: JobRecord) -> None: ...

    @abc.abstractmethod
    async def get(self, job_id: str) -> JobRecord: ...

    @abc.abstractmethod
    async def list_all(self) -> list[JobRecord]: ...

    @abc.abstractmethod
    async def delete(self, job_id: str) -> None: ...


class JsonFileJobRepository(JobRepository):
    """
    Durable job storage: one JSON file per job under `directory`.

    An in-memory cache mirrors what's on disk for fast reads; every write
    goes through to disk immediately (write-through), so a process restart
    can recover full job history by reloading the directory at startup via
    `load_from_disk()`.
    """

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()

    def _path_for(self, job_id: str) -> Path:
        return self._dir / f"{job_id}.json"

    async def load_from_disk(self) -> None:
        """Populate the in-memory cache from whatever is already on disk.
        Call once at startup to recover job history across restarts."""

        def _read_all() -> dict[str, JobRecord]:
            jobs: dict[str, JobRecord] = {}
            for path in self._dir.glob("*.json"):
                try:
                    data = json.loads(path.read_text())
                    job = JobRecord.model_validate(data)
                    jobs[job.job_id] = job
                except (json.JSONDecodeError, ValueError):
                    continue  # skip corrupt/partial files rather than crash startup
            return jobs

        loaded = await asyncio.to_thread(_read_all)
        async with self._lock:
            self._cache.update(loaded)

    async def save(self, job: JobRecord) -> None:
        def _write() -> None:
            path = self._path_for(job.job_id)
            path.write_text(job.model_dump_json(indent=2))

        async with self._lock:
            self._cache[job.job_id] = job.model_copy()
        await asyncio.to_thread(_write)

    async def get(self, job_id: str) -> JobRecord:
        async with self._lock:
            job = self._cache.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            return job.model_copy()

    async def list_all(self) -> list[JobRecord]:
        async with self._lock:
            return [job.model_copy() for job in self._cache.values()]

    async def delete(self, job_id: str) -> None:
        def _delete() -> None:
            path = self._path_for(job_id)
            path.unlink(missing_ok=True)

        async with self._lock:
            self._cache.pop(job_id, None)
        await asyncio.to_thread(_delete)
