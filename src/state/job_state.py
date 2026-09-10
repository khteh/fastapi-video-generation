"""
Job state layer.

Responsibility: enforce job *lifecycle rules* — valid status transitions,
progress bookkeeping, and atomic read-modify-write updates — on top of a
`JobRepository`. This layer knows the business rules (e.g. "you can't
cancel a completed job", "an interrupted job should be marked failed on
restart") but delegates all actual durability to persistence, and knows
nothing about how videos are generated or stored.
"""
from __future__ import annotations

import asyncio
import uuid

from src.models import ArtifactInfo, JobRecord, JobStatus, VideoRequest
from src.persistence.repository import JobNotFoundError, JobRepository

__all__ = ["JobNotFoundError", "InvalidTransitionError", "JobStateManager"]


class InvalidTransitionError(Exception):
    """Raised when an operation would move a job through an illegal state transition."""


class JobStateManager:
    """
    The single place business rules about job lifecycle live. All mutations
    to a job go through here (never write to the repository directly from
    the API or worker layers) so the rules are enforced in one spot.
    """

    def __init__(self, repository: JobRepository) -> None:
        self._repo = repository
        self._lock = asyncio.Lock()

    async def create(self, request: VideoRequest, provider: str) -> JobRecord:
        job = JobRecord(
            job_id=str(uuid.uuid4()),
            status=JobStatus.PENDING,
            topic=request.topic,
            difficulty=request.difficulty,
            duration_seconds=request.duration_seconds,
            provider=provider,
            stage="queued",
        )
        async with self._lock:
            await self._repo.save(job)
        return job

    async def get(self, job_id: str) -> JobRecord:
        return await self._repo.get(job_id)

    async def list_all(self) -> list[JobRecord]:
        jobs = await self._repo.list_all()
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    async def _mutate(self, job_id: str, mutator) -> JobRecord:
        """Atomic read-modify-write: fetch, apply `mutator`, persist, return."""
        async with self._lock:
            job = await self._repo.get(job_id)
            mutator(job)
            job.touch()
            await self._repo.save(job)
            return job.model_copy()

    async def mark_processing(self, job_id: str, stage: str, progress: int = 0) -> JobRecord:
        def _mutate(job: JobRecord) -> None:
            if job.status.is_terminal:
                raise InvalidTransitionError(
                    f"cannot move job {job_id} to processing from terminal state {job.status}"
                )
            job.status = JobStatus.PROCESSING
            job.stage = stage
            job.progress = progress

        return await self._mutate(job_id, _mutate)

    async def update_progress(self, job_id: str, *, stage: str, progress: int) -> JobRecord:
        def _mutate(job: JobRecord) -> None:
            if job.status.is_terminal:
                # Job was cancelled/failed/completed concurrently; ignore
                # a late progress update from an in-flight worker instead
                # of resurrecting it.
                return
            job.stage = stage
            job.progress = progress

        return await self._mutate(job_id, _mutate)

    async def mark_completed(self, job_id: str, artifact: ArtifactInfo) -> JobRecord:
        def _mutate(job: JobRecord) -> None:
            job.status = JobStatus.COMPLETED
            job.stage = "done"
            job.progress = 100
            job.artifact = artifact
            job.error = None

        return await self._mutate(job_id, _mutate)

    async def mark_failed(self, job_id: str, error: str) -> JobRecord:
        def _mutate(job: JobRecord) -> None:
            if job.status == JobStatus.CANCELLED:
                return  # don't overwrite an explicit cancellation
            job.status = JobStatus.FAILED
            job.stage = "failed"
            job.error = error

        return await self._mutate(job_id, _mutate)

    async def mark_cancelled(self, job_id: str) -> JobRecord:
        def _mutate(job: JobRecord) -> None:
            if job.status.is_terminal:
                raise InvalidTransitionError(
                    f"job {job_id} is already {job.status.value} and cannot be cancelled"
                )
            job.status = JobStatus.CANCELLED
            job.stage = "cancelled"

        return await self._mutate(job_id, _mutate)

    async def recover_interrupted_jobs(self) -> list[JobRecord]:
        """
        Call once at startup. Any job still PENDING/PROCESSING in the
        persisted store must have been interrupted by a previous process
        crash or restart — nothing resumes an in-flight generation, so mark
        those jobs FAILED rather than leaving them stuck forever.
        """
        recovered: list[JobRecord] = []
        for job in await self._repo.list_all():
            if not job.status.is_terminal:
                updated = await self.mark_failed(
                    job.job_id, "interrupted: server restarted while this job was running"
                )
                recovered.append(updated)
        return recovered
