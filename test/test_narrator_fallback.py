"""
Tests for src.generation.narrator's fallback chain — specifically the
scenario where a network-backed narrator (EdgeTTSNarrator) is *nominally*
available (package installed) but fails at actual synthesis time (no
network, network drops mid-session, then recovers).
"""
from __future__ import annotations

import wave, pytest
from pathlib import Path
from src.generation.narrator import (
    EdgeTTSNarrator,
    FallbackNarrator,
    FliteNarrator,
    NarrationUnavailableError,
    PiperNarrator,
    ToneNarrator,
    select_narrator,
)

class _AlwaysFailsNarrator:
    """Simulates a narrator whose is_available() check passes (e.g. package
    installed) but whose synthesize() always fails (e.g. no network) —
    exactly the EdgeTTSNarrator-with-no-network scenario."""

    name = "fake-network-backend"

    def __init__(self) -> None:
        self.call_count = 0

    async def is_available(self) -> bool:
        return True

    async def synthesize(self, text: str, out_path: Path) -> float:
        self.call_count += 1
        raise NarrationUnavailableError("simulated: no network")


class _FlakyNarrator:
    """Fails its first N calls, then succeeds — simulates network coming
    back mid-session."""

    name = "fake-flaky-backend"

    def __init__(self, fail_count: int) -> None:
        self._fail_count = fail_count
        self.call_count = 0

    async def is_available(self) -> bool:
        return True

    async def synthesize(self, text: str, out_path: Path) -> float:
        self.call_count += 1
        if self.call_count <= self._fail_count:
            raise NarrationUnavailableError("simulated: still down")
        out_path.write_bytes(b"used-primary")
        return 1.0


class _AlwaysSucceedsNarrator:
    name = "fake-fallback-backend"

    async def is_available(self) -> bool:
        return True

    async def synthesize(self, text: str, out_path: Path) -> float:
        out_path.write_bytes(b"used-fallback")
        return 1.0


@pytest.mark.asyncio
async def test_fallback_cascades_on_every_call_not_just_once(tmp_path):
    """The core bug being fixed: a narrator that's nominally 'available'
    but always fails at synthesis time must cascade to the fallback on
    EVERY call, not just fail permanently after the first attempt."""
    primary = _AlwaysFailsNarrator()
    fallback = _AlwaysSucceedsNarrator()
    chain = FallbackNarrator(primary=primary, fallback=fallback)

    for i in range(3):
        out = tmp_path / f"slide_{i}.wav"
        await chain.synthesize(f"slide {i}", out)
        assert out.read_bytes() == b"used-fallback"

    assert primary.call_count == 3, "primary must be retried on every call, not cached as permanently dead"


@pytest.mark.asyncio
async def test_fallback_recovers_automatically_once_primary_starts_working(tmp_path):
    """If the primary starts failing and then recovers (e.g. network comes
    back), later calls must use it again without any restart/re-selection."""
    primary = _FlakyNarrator(fail_count=2)
    fallback = _AlwaysSucceedsNarrator()
    chain = FallbackNarrator(primary=primary, fallback=fallback)

    used = []
    for i in range(4):
        out = tmp_path / f"slide_{i}.wav"
        await chain.synthesize(f"slide {i}", out)
        used.append(out.read_bytes())

    assert used == [b"used-fallback", b"used-fallback", b"used-primary", b"used-primary"]


@pytest.mark.asyncio
async def test_select_narrator_realistic_chain_ends_in_tone(monkeypatch):
    """Whatever backends are actually available on this machine, the
    'realistic' chain must always include tone as the guaranteed final
    fallback (it has no dependencies and can't itself fail)."""
    from src.generation import narrator as narrator_module

    narrator_module._cache.clear()
    chain = await select_narrator(prefer_realistic=True)
    assert "tone" in chain.name


@pytest.mark.asyncio
async def test_select_narrator_produces_working_chain_on_this_machine(tmp_path):
    """Whatever the real chain resolves to here, it must actually produce
    valid, non-empty audio when asked."""
    from src.generation import narrator as narrator_module

    narrator_module._cache.clear()
    chain = await select_narrator(prefer_realistic=True)
    out = tmp_path / "test.wav"
    duration = await chain.synthesize("Testing the real narrator chain on this machine.", out)
    assert out.exists()
    assert duration > 0
    with wave.open(str(out), "rb") as wav:
        assert wav.getnframes() > 0


@pytest.mark.asyncio
async def test_flite_synthesis_failure_is_normalized_to_narration_unavailable(tmp_path, monkeypatch):
    """A subprocess-level failure inside FliteNarrator must surface as
    NarrationUnavailableError (so a wrapping FallbackNarrator can catch
    it), not a raw SubprocessError."""
    from src.generation.subprocess_utils import SubprocessError

    narrator = FliteNarrator(voice="kal", sample_rate=22050)

    async def _always_fails(*args, **kwargs):
        raise SubprocessError("simulated ffmpeg crash")

    monkeypatch.setattr("src.generation.narrator.run_subprocess", _always_fails)

    with pytest.raises(NarrationUnavailableError):
        await narrator.synthesize("test", tmp_path / "out.wav")


@pytest.mark.asyncio
async def test_select_ai_narrator_never_includes_flite_or_tone():
    """Product requirement: 'ai' mode must never silently produce
    simulated-quality (flite/tone) audio. Whatever select_ai_narrator()
    resolves to on this machine, its name must not mention either."""
    from src.generation import narrator as narrator_module

    narrator_module._cache.clear()
    narrator = await narrator_module.select_ai_narrator()
    assert "flite" not in narrator.name
    assert "tone" not in narrator.name


# after
@pytest.mark.asyncio
async def test_select_ai_narrator_fails_clearly_with_no_real_backend_available(monkeypatch):
    """select_ai_narrator() must resolve to the dedicated sentinel and
    fail clearly when NEITHER edge-tts nor Piper is actually usable.
    Forced deterministically via monkeypatch rather than assumed from the
    test environment's real package state — if `uv sync --extra ai` was
    run, `edge_tts`/`piper` may genuinely be importable here, which would
    make EdgeTTSNarrator/PiperNarrator.is_available() return True (that
    check only confirms the package is installed, not that a real
    backend is reachable) and resolve to a real narrator instead of the
    sentinel, breaking a version of this test that assumed otherwise."""
    from src.generation import narrator as narrator_module

    async def _always_unavailable(self) -> bool:
        return False

    monkeypatch.setattr(EdgeTTSNarrator, "is_available", _always_unavailable)
    monkeypatch.setattr(PiperNarrator, "is_available", _always_unavailable)

    narrator_module._cache.clear()
    narrator = await narrator_module.select_ai_narrator()

    assert narrator.name == "no-ai-voice-available"

    with pytest.raises(NarrationUnavailableError) as exc_info:
        await narrator.synthesize("test", Path("/tmp/unused.wav"))
    assert "edge-tts" in str(exc_info.value)
    assert "Piper" in str(exc_info.value)

@pytest.mark.asyncio
async def test_select_ai_narrator_is_cached_across_calls():
    from src.generation import narrator as narrator_module

    narrator_module._cache.clear()
    first = await narrator_module.select_ai_narrator()
    second = await narrator_module.select_ai_narrator()
    assert first is second
