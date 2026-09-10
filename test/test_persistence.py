"""Unit tests for src.persistence — durable job metadata storage, isolated
from state/lifecycle rules and from generation/artifacts."""
from __future__ import annotations

import pytest

from src.models import DifficultyLevel, JobRecord, JobStatus
from src.persistence.repository import JobNotFoundError, JsonFileJobRepository


def _job(job_id: str, status: JobStatus = JobStatus.PENDING) -> JobRecord:
    return JobRecord(
        job_id=job_id,
        status=status,
        topic="What is entropy?",
        difficulty=DifficultyLevel.beginner,
        duration_seconds=30,
        provider="simulated",
    )


@pytest.mark.asyncio
async def test_save_and_get(tmp_path):
    repo = JsonFileJobRepository(tmp_path)
    await repo.save(_job("a"))
    fetched = await repo.get("a")
    assert fetched.job_id == "a"
    assert fetched.status == JobStatus.PENDING


@pytest.mark.asyncio
async def test_get_missing_raises(tmp_path):
    repo = JsonFileJobRepository(tmp_path)
    with pytest.raises(JobNotFoundError):
        await repo.get("missing")


@pytest.mark.asyncio
async def test_save_writes_a_json_file_to_disk(tmp_path):
    repo = JsonFileJobRepository(tmp_path)
    await repo.save(_job("a"))
    assert (tmp_path / "a.json").exists()


@pytest.mark.asyncio
async def test_list_all_returns_saved_jobs(tmp_path):
    repo = JsonFileJobRepository(tmp_path)
    await repo.save(_job("a"))
    await repo.save(_job("b"))
    jobs = await repo.list_all()
    assert {j.job_id for j in jobs} == {"a", "b"}


@pytest.mark.asyncio
async def test_delete_removes_from_cache_and_disk(tmp_path):
    repo = JsonFileJobRepository(tmp_path)
    await repo.save(_job("a"))
    await repo.delete("a")
    assert not (tmp_path / "a.json").exists()
    with pytest.raises(JobNotFoundError):
        await repo.get("a")


@pytest.mark.asyncio
async def test_load_from_disk_recovers_jobs_across_restart(tmp_path):
    repo1 = JsonFileJobRepository(tmp_path)
    await repo1.save(_job("persisted-job", JobStatus.PROCESSING))

    # Simulate a fresh process: new repository instance, same directory.
    repo2 = JsonFileJobRepository(tmp_path)
    # Before loading, the new instance's cache is empty.
    with pytest.raises(JobNotFoundError):
        await repo2.get("persisted-job")

    await repo2.load_from_disk()
    recovered = await repo2.get("persisted-job")
    assert recovered.job_id == "persisted-job"
    assert recovered.status == JobStatus.PROCESSING


@pytest.mark.asyncio
async def test_load_from_disk_skips_corrupt_files(tmp_path):
    (tmp_path / "corrupt.json").write_text("{not valid json")
    repo = JsonFileJobRepository(tmp_path)
    await repo.load_from_disk()  # should not raise
    jobs = await repo.list_all()
    assert jobs == []
