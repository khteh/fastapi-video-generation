"""
Unit tests for src.state.job_state — job lifecycle business rules.

Uses a minimal in-memory fake JobRepository (not JsonFileJobRepository) so
these tests exercise *only* the state layer's rules, independent of the
persistence backend.
"""
from __future__ import annotations

import pytest

from src.models import ArtifactInfo, DifficultyLevel, JobStatus, VideoRequest
from src.persistence.repository import JobNotFoundError, JobRepository
from src.state.job_state import InvalidTransitionError, JobStateManager


class FakeRepository(JobRepository):
    def __init__(self):
        self._jobs = {}

    async def save(self, job):
        self._jobs[job.job_id] = job.model_copy()

    async def get(self, job_id):
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return job.model_copy()

    async def list_all(self):
        return [j.model_copy() for j in self._jobs.values()]

    async def delete(self, job_id):
        self._jobs.pop(job_id, None)


def _request() -> VideoRequest:
    return VideoRequest(topic="What is entropy?", difficulty=DifficultyLevel.beginner, duration_seconds=30)


@pytest.mark.asyncio
async def test_create_starts_pending():
    state = JobStateManager(FakeRepository())
    job = await state.create(_request(), provider="simulated")
    assert job.status == JobStatus.PENDING
    assert job.progress == 0


@pytest.mark.asyncio
async def test_mark_processing_updates_status_and_stage():
    state = JobStateManager(FakeRepository())
    job = await state.create(_request(), provider="simulated")
    updated = await state.mark_processing(job.job_id, stage="writing_script", progress=5)
    assert updated.status == JobStatus.PROCESSING
    assert updated.stage == "writing_script"
    assert updated.progress == 5


@pytest.mark.asyncio
async def test_update_progress_ignored_after_terminal():
    state = JobStateManager(FakeRepository())
    job = await state.create(_request(), provider="simulated")
    await state.mark_cancelled(job.job_id)
    # Late progress update from a still-running task must not resurrect it.
    updated = await state.update_progress(job.job_id, stage="rendering", progress=50)
    assert updated.status == JobStatus.CANCELLED


@pytest.mark.asyncio
async def test_mark_completed_sets_artifact_and_progress_100():
    state = JobStateManager(FakeRepository())
    job = await state.create(_request(), provider="simulated")
    artifact = ArtifactInfo(job_id=job.job_id, size_bytes=1234, duration_seconds=42.0)
    updated = await state.mark_completed(job.job_id, artifact)
    assert updated.status == JobStatus.COMPLETED
    assert updated.progress == 100
    assert updated.artifact.size_bytes == 1234


@pytest.mark.asyncio
async def test_mark_failed_sets_error():
    state = JobStateManager(FakeRepository())
    job = await state.create(_request(), provider="simulated")
    updated = await state.mark_failed(job.job_id, "boom")
    assert updated.status == JobStatus.FAILED
    assert updated.error == "boom"


@pytest.mark.asyncio
async def test_mark_failed_does_not_overwrite_cancelled():
    state = JobStateManager(FakeRepository())
    job = await state.create(_request(), provider="simulated")
    await state.mark_cancelled(job.job_id)
    updated = await state.mark_failed(job.job_id, "late failure")
    assert updated.status == JobStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_twice_raises_invalid_transition():
    state = JobStateManager(FakeRepository())
    job = await state.create(_request(), provider="simulated")
    await state.mark_cancelled(job.job_id)
    with pytest.raises(InvalidTransitionError):
        await state.mark_cancelled(job.job_id)


@pytest.mark.asyncio
async def test_cancel_completed_job_raises_invalid_transition():
    state = JobStateManager(FakeRepository())
    job = await state.create(_request(), provider="simulated")
    artifact = ArtifactInfo(job_id=job.job_id, size_bytes=1, duration_seconds=1.0)
    await state.mark_completed(job.job_id, artifact)
    with pytest.raises(InvalidTransitionError):
        await state.mark_cancelled(job.job_id)


@pytest.mark.asyncio
async def test_list_all_sorted_newest_first():
    state = JobStateManager(FakeRepository())
    first = await state.create(_request(), provider="simulated")
    second = await state.create(_request(), provider="simulated")
    jobs = await state.list_all()
    assert jobs[0].job_id == second.job_id
    assert jobs[1].job_id == first.job_id


@pytest.mark.asyncio
async def test_recover_interrupted_jobs_marks_non_terminal_as_failed():
    repo = FakeRepository()
    state = JobStateManager(repo)
    pending_job = await state.create(_request(), provider="simulated")
    processing_job = await state.create(_request(), provider="simulated")
    await state.mark_processing(processing_job.job_id, stage="rendering", progress=40)
    completed_job = await state.create(_request(), provider="simulated")
    artifact = ArtifactInfo(job_id=completed_job.job_id, size_bytes=1, duration_seconds=1.0)
    await state.mark_completed(completed_job.job_id, artifact)

    recovered = await state.recover_interrupted_jobs()
    recovered_ids = {j.job_id for j in recovered}
    assert recovered_ids == {pending_job.job_id, processing_job.job_id}

    final_completed = await state.get(completed_job.job_id)
    assert final_completed.status == JobStatus.COMPLETED  # untouched
