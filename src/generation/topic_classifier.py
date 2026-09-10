"""
Topic classification: catches nonsensical input that slips past the basic
`VideoRequest.topic` validation in `src/models.py` (blank, no-letters,
length bounds) but is still meaningless — e.g. keyboard mashing like
"asdf jkl qwerty". That basic validation only checks *form*; this checks
(approximately) *content*.

Classification is called SYNCHRONOUSLY from `POST /api/v1/videos`, before
any job is created (see `providers.py`'s `validate_topic()` and its
module docstring for the full policy) — so a semantically-invalid topic,
or an unavailable "ai"-mode backend, is rejected immediately with no job
ever created, rather than discovered later via polling.

Two implementations, mirroring the same plug-and-play pattern as
`script_providers.py`:

  - `HeuristicTopicClassifier`: fully offline, no dependencies, no network.
    Flags tokens that are QWERTY keyboard walks ("asdf", "qwerty", "jkl"),
    low-character-variety ("aaaaaa", "ababab"), or long vowel-less runs
    that aren't short acronyms. This is a *pattern* detector, not a real
    language understanding — it reliably catches accidental/test-style
    garbage but will not catch nonsense that happens to consist of real
    words strung together meaninglessly, and never needs the network.
    Used by the "simulated" generation provider. Can never raise
    `ClassificationUnavailableError` — it has no external dependencies.

  - `AnthropicTopicClassifier`: a real, cheap/fast LLM classification call
    (reuses the same Anthropic client pattern as `AnthropicScriptProvider`)
    that actually understands meaning — correctly distinguishes "asdf jkl
    qwerty" from obscure-but-real jargon like "Kirchhoff's voltage law".
    Requires `ANTHROPIC_API_KEY` and network. Used directly (unwrapped, no
    fallback) by the "ai" generation provider: a failure here (no network,
    bad key, timeout) raises `ClassificationUnavailableError`, which
    `main.py` turns into an immediate `503` with no job created — this
    call doubles as a live health check for the same backend script
    generation will need a moment later.

  - `FallbackTopicClassifier`: wraps a primary + fallback classifier — if
    the primary is unavailable or a call fails, it transparently falls
    back rather than raising. Still available as a general-purpose
    utility and exercised directly in tests, but note it is *not*
    currently used to build the "ai" provider (see providers.py for why:
    once "ai" mode is a hard gate on every network-dependent component,
    letting only the classifier silently degrade would be inconsistent).
"""
from __future__ import annotations

import abc
import re
from dataclasses import dataclass

_QWERTY_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
_VOWELS = set("aeiouy")
_WORD_RE = re.compile(r"[a-zA-Z]+")


@dataclass(frozen=True)
class ClassificationResult:
    """Typed result returned by `GenerationProvider.validate_topic()` in
    providers.py — a small convenience wrapper around the
    `(is_valid, reason)` tuple every `TopicClassifier.is_valid()` returns,
    plus which classifier actually produced the verdict."""

    is_valid: bool
    reason: str = ""
    classifier: str = ""


class ClassificationUnavailableError(Exception):
    """Raised when a classifier can't run at all (missing package/API key/network)."""


class TopicClassifier(abc.ABC):
    name: str

    @abc.abstractmethod
    async def is_valid(self, topic: str) -> tuple[bool, str]:
        """Returns (is_valid, reason). `reason` is a human-readable
        explanation when invalid, and may be empty when valid."""

    async def is_available(self) -> bool:
        return True


def _is_keyboard_walk(token: str) -> bool:
    """Catches runs of horizontally-adjacent QWERTY keys, e.g. 'asdf',
    'qwerty', 'jkl', 'zxcvbn' — classic accidental/test-input mashing."""
    if len(token) < 3:
        return False
    for row in _QWERTY_ROWS:
        if token in row or token in row[::-1]:
            return True
    return False


def _is_low_variety(token: str) -> bool:
    """Catches very low character diversity, e.g. 'aaaaaa', 'ababab'."""
    if len(token) < 3:
        return False
    return len(set(token)) <= 2


def _lacks_vowels(token: str) -> bool:
    """Catches long vowel-less runs. Short tokens are exempt since real
    short acronyms/units are commonly vowel-less (DNA, RNA, kg, mm)."""
    if len(token) <= 4:
        return False
    return not any(c in _VOWELS for c in token)


def _looks_suspicious(token: str) -> bool:
    lower = token.lower()
    return _is_keyboard_walk(lower) or _is_low_variety(lower) or _lacks_vowels(lower)


class HeuristicTopicClassifier(TopicClassifier):
    name = "heuristic"

    # Reject only when a clear majority of meaningful tokens look
    # suspicious — a single odd token in an otherwise sensible sentence
    # shouldn't sink the whole topic.
    _SUSPICIOUS_FRACTION_THRESHOLD = 0.6

    async def is_valid(self, topic: str) -> tuple[bool, str]:
        tokens = [w for w in _WORD_RE.findall(topic) if len(w) >= 3]
        if not tokens:
            # Nothing substantial to judge (e.g. "pH" alone) — the basic
            # length/letter checks in models.py already gate this; don't
            # double-reject here.
            return True, ""

        suspicious = [t for t in tokens if _looks_suspicious(t)]
        fraction = len(suspicious) / len(tokens)
        if fraction >= self._SUSPICIOUS_FRACTION_THRESHOLD:
            return False, (
                "topic looks like keyboard mashing or random characters rather "
                "than a real question or subject"
            )
        return True, ""


_CLASSIFIER_SYSTEM_PROMPT = """You classify whether input text is a coherent \
question or subject (in any field, especially STEM) versus meaningless \
gibberish, keyboard mashing, or random characters.

Be lenient: if there is ANY reasonable interpretation of the input as a \
real topic or question — including obscure jargon, abbreviations, or \
informally phrased questions — classify it as VALID. Only classify as \
INVALID if it is clearly meaningless (e.g. keyboard mashing like "asdf jkl \
qwerty", random character strings, or repeated nonsense).

Respond with exactly one word: VALID or INVALID. No other text."""


class AnthropicTopicClassifier(TopicClassifier):
    """Real LLM-based classification via the Anthropic API. Deliberately
    uses a small, fast/cheap model — this is a binary classification call,
    not content generation."""

    name = "anthropic"

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ClassificationUnavailableError(
                    "the 'anthropic' package is not installed; install it with "
                    "`uv sync --extra ai` to use the AI topic classifier"
                ) from exc
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def is_available(self) -> bool:
        if not self._api_key:
            return False
        try:
            self._get_client()
        except ClassificationUnavailableError:
            return False
        return True

    async def is_valid(self, topic: str) -> tuple[bool, str]:
        if not self._api_key:
            raise ClassificationUnavailableError("ANTHROPIC_API_KEY is not set")
        client = self._get_client()

        try:
            response = await client.messages.create(
                model=self._model,
                max_tokens=10,
                system=_CLASSIFIER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": topic}],
            )
        except Exception as exc:  # auth errors, rate limits, network errors, etc.
            raise ClassificationUnavailableError(f"Anthropic API call failed: {exc}") from exc

        text_parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        verdict = "".join(text_parts).strip().upper()

        if verdict.startswith("VALID"):
            return True, ""
        if verdict.startswith("INVALID"):
            return False, "the AI classifier flagged this topic as not a coherent question or subject"

        # Response didn't match the expected format — treat as a failure
        # (caller falls back) rather than silently guessing.
        raise ClassificationUnavailableError(f"unexpected classifier response: {verdict!r}")


class FallbackTopicClassifier(TopicClassifier):
    """Wraps a primary classifier with a fallback, exactly like
    `FallbackScriptProvider`: if the primary is unavailable or a call
    fails (bad API key, no network, malformed response), transparently
    fall back instead of blocking submission."""

    def __init__(self, primary: TopicClassifier, fallback: TopicClassifier) -> None:
        self._primary = primary
        self._fallback = fallback
        self.name = f"{primary.name}(fallback={fallback.name})"

    async def is_available(self) -> bool:
        return True  # the fallback is always available, so this wrapper always is

    async def is_valid(self, topic: str) -> tuple[bool, str]:
        if await self._primary.is_available():
            try:
                return await self._primary.is_valid(topic)
            except ClassificationUnavailableError:
                pass  # fall through to the fallback below
        return await self._fallback.is_valid(topic)
