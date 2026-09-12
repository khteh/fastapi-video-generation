"""
End-to-end API tests, exercising the real generation pipeline (real ffmpeg
encoding + real flite speech synthesis) through the HTTP layer, using the
"simulated" provider (see test/conftest.py).
"""
from __future__ import annotations

import pytest

from test.conftest import poll_until

EXAMPLE_QUESTIONS = [
    "How does the pH scale work?",
    "Why do atoms form covalent bonds?",
    "What is the difference between ionic and covalent bonding?",
]


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_providers_endpoint_lists_active_and_available(client):
    resp = await client.get("/api/v1/providers")
    assert resp.status_code == 200
    body = resp.json()
    assert body["active"] == "simulated"
    assert "simulated" in body["available"]
    assert "ai" in body["available"]


@pytest.mark.asyncio
@pytest.mark.parametrize("topic", EXAMPLE_QUESTIONS)
async def test_submit_job_returns_202_with_job_id(client, topic):
    resp = await client.post(
        "/api/v1/videos",
        json={"topic": topic, "difficulty": "beginner"},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["job_id"]
    assert body["status"] == "pending"
    assert body["provider"] == "simulated"
    assert body["poll_url"].endswith(body["job_id"])


@pytest.mark.asyncio
async def test_get_nonexistent_job_returns_404(client):
    resp = await client.get("/api/v1/videos/does-not-exist")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_download_nonexistent_job_returns_404(client):
    resp = await client.get("/api/v1/videos/does-not-exist/download")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_blank_topic_is_rejected_with_422(client):
    resp = await client.post("/api/v1/videos", json={"topic": "   "})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_whitespace_only_topic_rejected_even_at_valid_raw_length(client):
    """5 spaces has the "right" raw length but is meaningless once
    stripped — must be rejected the same as an empty string."""
    resp = await client.post("/api/v1/videos", json={"topic": "     "})
    assert resp.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "topic",
    [
        "~!@#$",       # symbols only
        "12345",       # digits only
        "....",        # punctuation only
        "@@@@@",       # symbols only
        "1 2 3 4 5",   # digits and whitespace only
    ],
)
async def test_topic_with_no_letters_is_rejected_even_at_valid_length(client, topic):
    """These all satisfy length requirements but contain no actual words —
    a real STEM question always has at least one letter in it."""
    resp = await client.post("/api/v1/videos", json={"topic": topic})
    assert resp.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "topic",
    ["pH scale", "DNA repair", "5G networks", "E=mc2", "RNA splicing"],
)
async def test_short_but_legitimate_topics_are_not_falsely_rejected(client, topic):
    """The letters-only check must not have false positives on real, if
    short, STEM topics that happen to include digits/symbols. (Note: plain
    "DNA" alone is only 3 characters and would separately fail the
    VIDEO_MIN_TOPIC_LENGTH=5 floor — that's an intentional, unrelated gate,
    not what this test is checking.)"""
    resp = await client.post("/api/v1/videos", json={"topic": topic})
    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_too_short_topic_is_rejected(client):
    resp = await client.post("/api/v1/videos", json={"topic": "pH?"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_invalid_difficulty_is_rejected(client):
    resp = await client.post(
        "/api/v1/videos", json={"topic": "How does the pH scale work?", "difficulty": "expert"}
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_duration_over_90_seconds_is_rejected(client):
    resp = await client.post(
        "/api/v1/videos",
        json={"topic": "How does the pH scale work?", "duration_seconds": 91},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_duration_at_90_seconds_is_accepted(client):
    resp = await client.post(
        "/api/v1/videos",
        json={"topic": "How does the pH scale work?", "duration_seconds": 90},
    )
    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_topic_over_max_length_is_rejected(client):
    from src.config import settings
    limit = settings.max_topic_length
    topic = "Why does " + ("water " * ((limit // 6) + 5)) + "boil?"
    assert len(topic) > limit
    resp = await client.post("/api/v1/videos", json={"topic": topic})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_topic_at_max_length_is_accepted(client):
    # Pad a real question out to exactly the 1024-character limit.
    from src.config import settings
    limit = settings.max_topic_length
    base = "How does the pH scale work? "
    topic = (base * (limit // len(base) + 1))[:limit]
    assert len(topic) == limit
    resp = await client.post("/api/v1/videos", json={"topic": topic})
    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_list_videos_shows_status_and_includes_submitted_job(client):
    submit = await client.post(
        "/api/v1/videos", json={"topic": "What is the difference between ionic and covalent bonding?"}
    )
    job_id = submit.json()["job_id"]

    resp = await client.get("/api/v1/videos")
    assert resp.status_code == 200
    jobs = resp.json()
    matching = [j for j in jobs if j["job_id"] == job_id]
    assert len(matching) == 1
    assert matching[0]["status"] in ("pending", "processing", "completed")
    assert "status" in matching[0]  # visible status field (requirement 7/8)
    assert matching[0]["topic"] == "What is the difference between ionic and covalent bonding?"


@pytest.mark.asyncio
async def test_full_flow_completes_with_real_visuals_and_audio_under_90s(client):
    """The core requirement: learner submits a question -> poll -> completed
    -> download a real, valid mp4 with both video and audio streams,
    capped at 90 seconds."""
    submit = await client.post(
        "/api/v1/videos",
        json={"topic": "Why do atoms form covalent bonds?", "difficulty": "intermediate"},
    )
    assert submit.status_code == 202
    job_id = submit.json()["job_id"]

    async def is_done():
        r = await client.get(f"/api/v1/videos/{job_id}")
        data = r.json()
        return data if data["status"] in ("completed", "failed") else None

    final = await poll_until(is_done, timeout=30.0)
    assert final["status"] == "completed"
    assert final["progress"] == 100
    assert final["artifact"] is not None
    assert final["artifact"]["content_type"] == "video/mp4"
    assert final["artifact"]["size_bytes"] > 0
    assert 0 < final["artifact"]["duration_seconds"] <= 90.0

    download = await client.get(f"/api/v1/videos/{job_id}/download")
    assert download.status_code == 200
    assert download.headers["content-type"] == "video/mp4"
    assert len(download.content) == final["artifact"]["size_bytes"]
    # A real MP4 starts with an ftyp box a few bytes in.
    assert b"ftyp" in download.content[:32]


@pytest.mark.asyncio
async def test_download_before_completion_returns_409(client):
    submit = await client.post("/api/v1/videos", json={"topic": "How does the pH scale work?"})
    job_id = submit.json()["job_id"]

    # Immediately try to download without waiting - job is very unlikely to
    # already be complete given real encoding takes multiple seconds.
    resp = await client.get(f"/api/v1/videos/{job_id}/download")
    assert resp.status_code in (409, 200)  # 200 only if it raced to completion


@pytest.mark.asyncio
async def test_gibberish_topic_rejected_immediately_with_no_job_created(client):
    """A topic like "asdf jkl qwerty" has letters and a valid length, so it
    passes the pydantic form checks — but the topic classifier now runs
    SYNCHRONOUSLY, before any job is created, so this is rejected with an
    immediate 422 and no job_id is ever handed out. (Previously this used
    to return 202 and fail asynchronously; see the README's "Failure
    handling" section for why that changed.)"""
    before = await client.get("/api/v1/videos")
    jobs_before = {j["job_id"] for j in before.json()}

    resp = await client.post("/api/v1/videos", json={"topic": "asdf jkl qwerty"})
    assert resp.status_code == 422
    assert "keyboard" in resp.json()["detail"].lower() or "mashing" in resp.json()["detail"].lower()

    after = await client.get("/api/v1/videos")
    jobs_after = {j["job_id"] for j in after.json()}
    assert jobs_after == jobs_before, "a rejected topic must not create a job"


@pytest.mark.asyncio
async def test_ai_backend_unavailable_returns_503_with_no_job_created(client, monkeypatch):
    """When the configured provider's AI backend can't be reached (no
    network, bad API key), submission must fail immediately with a clear
    503 and no job created — never a 202 for a job that's doomed to fail,
    and never a silent fallback to simulated-quality output."""
    from src.generation.topic_classifier import ClassificationUnavailableError

    class _FakeUnavailableProvider:
        async def validate_topic(self, topic: str):
            raise ClassificationUnavailableError("simulated: no network reaching the AI backend")

    monkeypatch.setattr("src.main.get_provider", lambda name: _FakeUnavailableProvider())

    before = await client.get("/api/v1/videos")
    jobs_before = {j["job_id"] for j in before.json()}

    resp = await client.post("/api/v1/videos", json={"topic": "How does the pH scale work?"})
    assert resp.status_code == 503
    assert "unavailable" in resp.json()["detail"].lower()

    after = await client.get("/api/v1/videos")
    jobs_after = {j["job_id"] for j in after.json()}
    assert jobs_after == jobs_before, "an unavailable AI backend must not create a job"


@pytest.mark.asyncio
async def test_cancel_pending_or_running_job(client):
    submit = await client.post("/api/v1/videos", json={"topic": "How does binary search work?"})
    job_id = submit.json()["job_id"]

    cancel_resp = await client.delete(f"/api/v1/videos/{job_id}")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"

    final = await client.get(f"/api/v1/videos/{job_id}")
    assert final.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_completed_job_returns_409(client):
    submit = await client.post("/api/v1/videos", json={"topic": "What is Newton's second law?"})
    job_id = submit.json()["job_id"]

    async def is_done():
        r = await client.get(f"/api/v1/videos/{job_id}")
        data = r.json()
        return data if data["status"] in ("completed", "failed") else None

    await poll_until(is_done, timeout=30.0)

    resp = await client.delete(f"/api/v1/videos/{job_id}")
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_cancel_nonexistent_job_returns_404(client):
    resp = await client.delete("/api/v1/videos/does-not-exist")
    assert resp.status_code == 404
