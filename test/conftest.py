"""
Shared pytest fixtures.

Sets environment variables *before* `src.config` (and therefore anything
that imports it) is first imported, so:
  - job metadata and artifacts are written to an isolated temp directory
    per test session, never the real `data/` directory.
  - video dimensions/fps are small, keeping real ffmpeg encoding fast.

Tests exercise the REAL generation pipeline (genuine ffmpeg + flite speech
synthesis, genuine PIL slide rendering, genuine video encoding) rather than
mocks — this repo's ffmpeg build supports the `flite` speech filter, so
there's no need to fake it out. A handful of full-generation tests take a
few seconds each as a result; that's an intentional trade-off for
end-to-end confidence over raw test speed.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

_TEST_DATA_ROOT = Path(tempfile.mkdtemp(prefix="stemvideo_test_"))
os.environ.setdefault("VIDEO_JOBS_DIR", str(_TEST_DATA_ROOT / "jobs"))
os.environ.setdefault("VIDEO_ARTIFACTS_DIR", str(_TEST_DATA_ROOT / "artifacts"))
os.environ.setdefault("VIDEO_WIDTH", "480")
os.environ.setdefault("VIDEO_HEIGHT", "270")
os.environ.setdefault("VIDEO_FPS", "10")
os.environ.setdefault("VIDEO_NUM_WORKERS", "2")
# Tests always use the offline "simulated" provider — fast, deterministic,
# no API keys/network required. The "ai" provider has its own dedicated,
# availability-gated tests in test_providers.py.
os.environ.setdefault("GENERATION_PROVIDER", "simulated")

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.main import app


@pytest_asyncio.fixture
async def client():
    """An httpx AsyncClient bound directly to the ASGI app (no real socket),
    with the app's lifespan (worker pool startup/shutdown) triggered."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        async with _LifespanContext():
            yield ac


class _LifespanContext:
    """Drives the FastAPI lifespan context manager for the test client."""

    async def __aenter__(self):
        self._ctx = app.router.lifespan_context(app)
        await self._ctx.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._ctx.__aexit__(exc_type, exc, tb)


async def poll_until(condition, *, timeout: float = 30.0, interval: float = 0.2):
    """Poll an async `condition()` callable until it returns truthy or timeout.

    Real video generation (ffmpeg + speech synthesis) takes a few seconds
    even at the small test resolution, hence the generous default timeout.
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        result = await condition()
        if result:
            return result
        await asyncio.sleep(interval)
    raise AssertionError("condition not met before timeout")


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TEST_DATA_ROOT, ignore_errors=True)
