"""
Tests for src.generation.topic_classifier.

The heuristic classifier is tested exhaustively here since it's the one
every submission goes through by default (directly for "simulated", as
the fallback for "ai"). The Anthropic classifier is tested via a fake
client (no network/API key needed) covering its three response-parsing
outcomes, plus the fallback wrapper's degradation behavior.
"""
from __future__ import annotations

import pytest

from src.generation.topic_classifier import (
    AnthropicTopicClassifier,
    ClassificationUnavailableError,
    FallbackTopicClassifier,
    HeuristicTopicClassifier,
)

GIBBERISH_TOPICS = [
    "asdf jkl qwerty",
    "qwerty asdf zxcvbn",
    "jkl jkl jkl",
    "aaaaaa bbbbbb cccccc",
    "ababab cdcdcd efefef",
    "zzzzz xxxxx",
]

REAL_STEM_TOPICS = [
    "How does the pH scale work?",
    "Why do atoms form covalent bonds?",
    "What is the difference between ionic and covalent bonding?",
    "How does binary search work?",
    "What is Newton's second law?",
    "DNA repair",
    "CRISPR-Cas9 gene editing",
    "Kirchhoff's voltage law",
    "quantum chromodynamics",
    "pH scale",
    "5G networks",
    "E=mc2",
    "RNA splicing",
    "How do black holes form?",
    "What is the Pythagorean theorem?",
    "Explain photosynthesis",
    "machine learning gradient descent",
    "Heisenberg uncertainty principle",
    "thermodynamics entropy",
    "what is quicksort algorithm complexity",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("topic", GIBBERISH_TOPICS)
async def test_heuristic_rejects_keyboard_mashing_and_low_entropy(topic):
    clf = HeuristicTopicClassifier()
    valid, reason = await clf.is_valid(topic)
    assert valid is False
    assert reason


@pytest.mark.asyncio
@pytest.mark.parametrize("topic", REAL_STEM_TOPICS)
async def test_heuristic_accepts_real_stem_topics_no_false_positives(topic):
    clf = HeuristicTopicClassifier()
    valid, reason = await clf.is_valid(topic)
    assert valid is True, f"false positive on legitimate topic: {topic!r} (reason: {reason})"


@pytest.mark.asyncio
async def test_heuristic_accepts_bare_short_acronym():
    """Short tokens (<=4 chars) are exempt from the vowel-less check, since
    real acronyms/units are commonly vowel-less (DNA, RNA, kg, mm)."""
    clf = HeuristicTopicClassifier()
    valid, _ = await clf.is_valid("DNA repair")
    assert valid is True


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    async def create(self, **kwargs):
        return _FakeResponse(self._response_text)


class _FakeClient:
    def __init__(self, response_text: str) -> None:
        self.messages = _FakeMessages(response_text)


@pytest.mark.asyncio
async def test_anthropic_classifier_parses_valid_response():
    clf = AnthropicTopicClassifier(api_key="fake-key", model="claude-haiku-4-5-20251001")
    clf._client = _FakeClient("VALID")
    valid, reason = await clf.is_valid("How does the pH scale work?")
    assert valid is True
    assert reason == ""


@pytest.mark.asyncio
async def test_anthropic_classifier_parses_invalid_response():
    clf = AnthropicTopicClassifier(api_key="fake-key", model="claude-haiku-4-5-20251001")
    clf._client = _FakeClient("INVALID")
    valid, reason = await clf.is_valid("asdf jkl qwerty")
    assert valid is False
    assert reason


@pytest.mark.asyncio
async def test_anthropic_classifier_raises_on_malformed_response():
    """A response that doesn't cleanly parse as VALID/INVALID must raise
    (so the fallback wrapper degrades to the heuristic) rather than
    silently guessing either way."""
    clf = AnthropicTopicClassifier(api_key="fake-key", model="claude-haiku-4-5-20251001")
    clf._client = _FakeClient("I think this is probably fine actually")
    with pytest.raises(ClassificationUnavailableError):
        await clf.is_valid("something")


@pytest.mark.asyncio
async def test_anthropic_classifier_unavailable_without_api_key():
    clf = AnthropicTopicClassifier(api_key="", model="claude-haiku-4-5-20251001")
    assert await clf.is_available() is False


@pytest.mark.asyncio
async def test_fallback_classifier_degrades_to_heuristic_without_api_key():
    """Core requirement: 'ai' mode must fall back to the offline heuristic
    if the AI classifier doesn't work (invalid API key, no network)."""
    primary = AnthropicTopicClassifier(api_key="", model="claude-haiku-4-5-20251001")
    fallback = HeuristicTopicClassifier()
    combo = FallbackTopicClassifier(primary=primary, fallback=fallback)

    valid, reason = await combo.is_valid("asdf jkl qwerty")
    assert valid is False
    assert reason

    valid, reason = await combo.is_valid("How does the pH scale work?")
    assert valid is True


@pytest.mark.asyncio
async def test_fallback_classifier_degrades_on_api_call_failure():
    """Simulates a configured-but-broken API key (e.g. call raises) rather
    than simply an absent one — should still degrade cleanly."""

    class _AlwaysFailsClassifier:
        name = "broken"

        async def is_available(self) -> bool:
            return True

        async def is_valid(self, topic: str):
            raise ClassificationUnavailableError("simulated network failure")

    combo = FallbackTopicClassifier(primary=_AlwaysFailsClassifier(), fallback=HeuristicTopicClassifier())
    valid, reason = await combo.is_valid("asdf jkl qwerty")
    assert valid is False
    assert reason


@pytest.mark.asyncio
async def test_fallback_classifier_uses_primary_when_it_works():
    class _AlwaysValidClassifier:
        name = "stub"

        async def is_available(self) -> bool:
            return True

        async def is_valid(self, topic: str):
            return True, ""

    combo = FallbackTopicClassifier(primary=_AlwaysValidClassifier(), fallback=HeuristicTopicClassifier())
    # Even gibberish should be accepted here, proving the primary (not the
    # fallback) is what actually ran.
    valid, _ = await combo.is_valid("asdf jkl qwerty")
    assert valid is True
