"""
Artifact storage layer.

Responsibility: persist and retrieve the actual generated *video files*.
This is deliberately separate from `src/persistence` (which stores small
JSON job metadata records): artifacts are large binary blobs with
different lifecycle, storage, and retrieval needs (e.g. streaming a file
response, moving to object storage / a CDN later). Swapping this for an S3-
backed implementation should require no changes to state, generation, or
API layers beyond constructing a different `ArtifactStore`.
"""
from __future__ import annotations

import abc
import asyncio
from pathlib import Path

from src.models import ArtifactInfo


class ArtifactNotFoundError(KeyError):
    pass


class ArtifactStore(abc.ABC):
    @abc.abstractmethod
    async def save(self, job_id: str, source_path: Path, duration_seconds: float) -> ArtifactInfo: ...

    @abc.abstractmethod
    async def path_for(self, job_id: str) -> Path:
        """Return the filesystem path of a stored artifact, for streaming/serving."""

    @abc.abstractmethod
    async def exists(self, job_id: str) -> bool: ...

    @abc.abstractmethod
    async def delete(self, job_id: str) -> None: ...


class LocalArtifactStore(ArtifactStore):
    """Stores finished video files as plain files on local disk."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, job_id: str) -> Path:
        return self._dir / f"{job_id}.mp4"

    async def save(self, job_id: str, source_path: Path, duration_seconds: float) -> ArtifactInfo:
        dest = self._path_for(job_id)

        def _move() -> int:
            data = source_path.read_bytes()
            dest.write_bytes(data)
            return len(data)

        size_bytes = await asyncio.to_thread(_move)
        return ArtifactInfo(
            job_id=job_id,
            content_type="video/mp4",
            size_bytes=size_bytes,
            duration_seconds=duration_seconds,
        )

    async def path_for(self, job_id: str) -> Path:
        path = self._path_for(job_id)
        if not await asyncio.to_thread(path.exists):
            raise ArtifactNotFoundError(job_id)
        return path

    async def exists(self, job_id: str) -> bool:
        return await asyncio.to_thread(self._path_for(job_id).exists)

    async def delete(self, job_id: str) -> None:
        await asyncio.to_thread(lambda: self._path_for(job_id).unlink(missing_ok=True))
