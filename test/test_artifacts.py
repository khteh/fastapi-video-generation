"""Unit tests for src.artifacts — video file storage, isolated from job metadata."""
from __future__ import annotations

import pytest

from src.artifacts.store import ArtifactNotFoundError, LocalArtifactStore


@pytest.mark.asyncio
async def test_save_and_retrieve(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake mp4 bytes")

    store = LocalArtifactStore(tmp_path / "artifacts")
    info = await store.save("job-1", source, duration_seconds=12.5)

    assert info.job_id == "job-1"
    assert info.size_bytes == len(b"fake mp4 bytes")
    assert info.duration_seconds == 12.5
    assert info.content_type == "video/mp4"

    path = await store.path_for("job-1")
    assert path.read_bytes() == b"fake mp4 bytes"


@pytest.mark.asyncio
async def test_exists_reflects_saved_state(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    assert await store.exists("missing") is False

    source = tmp_path / "source.mp4"
    source.write_bytes(b"data")
    await store.save("job-1", source, duration_seconds=1.0)
    assert await store.exists("job-1") is True


@pytest.mark.asyncio
async def test_path_for_missing_raises(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    with pytest.raises(ArtifactNotFoundError):
        await store.path_for("nope")


@pytest.mark.asyncio
async def test_delete_removes_artifact(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"data")
    store = LocalArtifactStore(tmp_path / "artifacts")
    await store.save("job-1", source, duration_seconds=1.0)

    await store.delete("job-1")
    assert await store.exists("job-1") is False
