"""
Tests for src.generation.providers — the plug-and-play registry between
"simulated" and "ai" generation providers.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.generation.providers import (
    AIProvider,
    SimulatedProvider,
    UnknownProviderError,
    available_providers,
    get_provider,
)


def test_available_providers_lists_both_builtin_providers():
    names = available_providers()
    assert "simulated" in names
    assert "ai" in names


def test_get_provider_returns_correct_types():
    assert isinstance(get_provider("simulated"), SimulatedProvider)
    assert isinstance(get_provider("ai"), AIProvider)


def test_get_provider_is_cached_singleton_per_name():
    a = get_provider("simulated")
    b = get_provider("simulated")
    assert a is b


def test_get_provider_unknown_name_raises():
    with pytest.raises(UnknownProviderError):
        get_provider("not-a-real-provider")


@pytest.mark.asyncio
async def test_simulated_provider_generates_real_video():
    provider = SimulatedProvider()
    await provider.initialize()
    assert provider.name == "simulated"

    with tempfile.TemporaryDirectory() as tmp:
        result = await provider.generate(
            topic="How does the pH scale work?",
            difficulty="beginner",
            work_dir=Path(tmp),
        )
        assert result.video_path.exists()
        assert result.video_path.stat().st_size > 0
        assert 0 < result.duration_seconds <= 90.0


@pytest.mark.asyncio
async def test_ai_provider_fails_clearly_without_silently_degrading(monkeypatch):
    """Deliberate product requirement: unlike topic classification (see
    validate_topic tests below), script writing and voice directly
    determine generated video quality, so the 'ai' provider must NOT
    silently substitute simulated-quality output (the template script
    writer, flite/tone voice) when the Anthropic API or a real voice
    backend is unavailable — it must fail with a clear, specific error
    instead, so a caller who asked for "ai" never unknowingly receives
    "simulated" output.

    This environment has no ANTHROPIC_API_KEY and no edge-tts/Piper
    configured, so script generation fails first."""
    from src.generation.script_providers import ProviderUnavailableError

    provider = AIProvider()
    await provider.initialize()
    assert provider.name == "ai"

    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ProviderUnavailableError):
            await provider.generate(
                topic="Why do atoms form covalent bonds?",
                difficulty="beginner",
                work_dir=Path(tmp),
            )
        # Confirm nothing was silently written despite the failure.
        assert list(Path(tmp).glob("*.mp4")) == []


@pytest.mark.asyncio
async def test_ai_provider_strict_narrator_never_resolves_to_flite_or_tone():
    """The 'ai' provider's narrator must never be (or fall back to) flite
    or tone — those are what 'simulated' mode uses. If no real AI voice
    backend is available, it should be the dedicated
    'no-ai-voice-available' sentinel that fails clearly, not a silent
    downgrade."""
    provider = AIProvider()
    await provider.initialize()
    name = provider._narrator.name
    assert "flite" not in name
    assert "tone" not in name
    assert name == "no-ai-voice-available" or "edge-tts" in name or "piper" in name


@pytest.mark.asyncio
async def test_simulated_validate_topic_rejects_gibberish_no_exception():
    """SimulatedProvider.validate_topic() must never raise — the heuristic
    classifier has no external dependencies to fail."""
    provider = SimulatedProvider()
    result = await provider.validate_topic("asdf jkl qwerty")
    assert result.is_valid is False
    assert result.reason
    assert result.classifier == "heuristic"


@pytest.mark.asyncio
async def test_simulated_validate_topic_accepts_real_topic():
    provider = SimulatedProvider()
    result = await provider.validate_topic("How does the pH scale work?")
    assert result.is_valid is True


@pytest.mark.asyncio
async def test_ai_validate_topic_raises_when_backend_unavailable():
    """Core requirement: in 'ai' mode, with no network/API key,
    validate_topic() must raise (not silently fall back to the
    heuristic), so main.py can turn this into an immediate 503 before any
    job is created."""
    from src.generation.topic_classifier import ClassificationUnavailableError

    provider = AIProvider()
    with pytest.raises(ClassificationUnavailableError):
        await provider.validate_topic("How does the pH scale work?")


@pytest.mark.asyncio
async def test_ai_validate_topic_uses_real_classifier_when_available():
    """With a working (faked) Anthropic client, validate_topic() must
    return the real classifier's verdict, not the heuristic's."""
    provider = AIProvider()

    class _FakeTextBlock:
        def __init__(self, text):
            self.type = "text"
            self.text = text

    class _FakeResponse:
        def __init__(self, text):
            self.content = [_FakeTextBlock(text)]

    class _FakeMessages:
        def __init__(self, text):
            self._text = text

        async def create(self, **kwargs):
            return _FakeResponse(self._text)

    class _FakeClient:
        def __init__(self, text):
            self.messages = _FakeMessages(text)

    provider._topic_classifier._api_key = "fake-key-for-test"
    provider._topic_classifier._client = _FakeClient("VALID")
    result = await provider.validate_topic("How does the pH scale work?")
    assert result.is_valid is True
    assert result.classifier == "anthropic"

    provider._topic_classifier._client = _FakeClient("INVALID")
    result2 = await provider.validate_topic("asdf jkl qwerty")
    assert result2.is_valid is False
    assert result2.reason


@pytest.mark.asyncio
async def test_provider_generate_lazily_initializes_if_not_called_explicitly():
    """A provider constructed but never explicitly initialize()'d should
    still work (defensive lazy-init in _PipelineBackedProvider.generate)."""
    provider = SimulatedProvider()
    with tempfile.TemporaryDirectory() as tmp:
        result = await provider.generate(
            topic="What is Newton's second law?",
            difficulty="beginner",
            work_dir=Path(tmp),
        )
        assert result.video_path.exists()
