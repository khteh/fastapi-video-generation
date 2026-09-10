"""
Script generation providers: turn a STEM topic/question into a short
sequence of narrated slides.

This is the concrete example of requirement "plug-and-play of both
simulated generation and real AI/video-generation providers": two
implementations of the same `ScriptProvider` interface —

  - `SimulatedScriptProvider`: a deterministic, fully offline template
    writer. No API keys, no network, runs in milliseconds. This is what
    the "simulated" generation provider (see providers.py) and the test
    suite use by default.
  - `AnthropicScriptProvider`: calls a real LLM (the Anthropic API) to
    write a genuinely topic-specific script — actually answering "How
    does the pH scale work?" rather than filling in a generic template.
    Used directly (unwrapped) by the "ai" generation provider: if it's
    unavailable (no `ANTHROPIC_API_KEY`, package missing) or a call fails
    (no network, API error), it raises `ProviderUnavailableError` and the
    job fails with that clear message — see providers.py's module
    docstring for why "ai" mode deliberately does NOT fall back to
    `SimulatedScriptProvider` here (script content directly determines
    output quality; a caller who asked for "ai" should never silently
    receive template-quality output instead).

`FallbackScriptProvider` (below) still exists as a general-purpose
composable wrapper — degrade primary-to-fallback on
`ProviderUnavailableError` — and is exercised directly in
test_generation.py, but note it is *not* currently used to build the
"ai" provider for the reason above.

Both script providers are swappable behind the same interface with zero
changes to slide rendering, narration, or video assembly. Adding a third
provider (e.g. a different LLM, or a retrieval-augmented one with a
curated STEM knowledge base) means implementing `ScriptProvider` and
registering it in `providers.py` — nothing else in the pipeline needs to
change.
"""
from __future__ import annotations

import abc
import json
import string
from dataclasses import dataclass

_DIFFICULTY_FRAMING = {
    "beginner": "a clear, simple explanation using everyday language and analogies",
    "intermediate": "a solid explanation with the key terminology defined",
    "advanced": "a precise, technical explanation assuming prior background in the subject",
}

# Max length for the topic text used in on-screen headings/body lines
# (which have limited visual width on a 1280px-wide slide) — narration
# (spoken audio) always uses the full topic regardless of length, since
# audio has no such constraint. See `_short_title()` below.
_MAX_VISUAL_TITLE_CHARS = 70


def _short_title(topic: str, max_chars: int = _MAX_VISUAL_TITLE_CHARS) -> str:
    """
    A concise version of `topic` for slide headings/body text. Long
    queries (the pydantic layer allows up to 1024 characters) simply
    don't fit on a slide at a readable font size — this truncates at a
    word boundary and adds an ellipsis rather than letting the renderer
    either overflow off-frame or shrink the font to the point of being
    unreadable. `slide_renderer.py` still wraps/fits whatever heading
    text it's given as a second, independent safety net (e.g. for
    AI-generated headings that don't come through this function at all).
    """
    stripped = topic.strip()
    if len(stripped) <= max_chars:
        return stripped
    truncated = stripped[:max_chars].rsplit(" ", 1)[0].rstrip(string.punctuation + " ")
    if not truncated:
        # A single "word" longer than max_chars with no space to break on.
        truncated = stripped[:max_chars].rstrip(string.punctuation + " ")
    return truncated + "…"

# Slide count scales with difficulty: the 90-second video length cap has
# real headroom at 5 slides (~53s of narration at a 145wpm pace, ~40%
# unused), and 5-6 slides isn't enough room for genuine depth on a complex
# topic — barely space for title/what/why/takeaway/summary, with no room
# for a mechanism walkthrough, a common misconception, or how the topic
# connects to related ideas. So beginner stays quick or an "elevator
# pitch" (5 slides), while intermediate and advanced use more of the
# available budget for actual depth, not padding.
_DIFFICULTY_SLIDE_COUNT = {
    "beginner": 5,
    "intermediate": 7,
    "advanced": 10,
}

# Rough word-count target per difficulty so the *total* narration stays
# comfortably under the 90s hard cap even at 10 slides (see pipeline.py's
# trim safety net for what happens if a script runs long despite this).
_DIFFICULTY_WORD_BUDGET = {
    "beginner": "about 110-140 words total, roughly 45-58 seconds",
    "intermediate": "about 150-175 words total, roughly 62-72 seconds",
    "advanced": "about 190-215 words total, roughly 78-89 seconds",
}


@dataclass
class Slide:
    heading: str
    body_lines: list[str]
    narration: str
    kind: str = "content"  # "title" | "content" | "diagram" | "summary"


class ProviderUnavailableError(Exception):
    """Raised when a provider can't run at all (missing package/API key/model)."""


class ScriptProvider(abc.ABC):
    name: str

    @abc.abstractmethod
    async def generate_script(self, topic: str, difficulty: str) -> list[Slide]:
        """Produce a narrated script for `topic`. Slide count scales with
        `difficulty` — see `_DIFFICULTY_SLIDE_COUNT` (5/7/10 for
        beginner/intermediate/advanced)."""

    async def is_available(self) -> bool:
        """Cheap check for whether this provider can actually run right now
        (package installed, API key set, etc). Default: always available."""
        return True


class SimulatedScriptProvider(ScriptProvider):
    """
    Deterministic, offline, template-based script writer.

    It does not "know" chemistry, physics, or math — it produces a
    consistently-structured explainer shell with the topic woven in,
    which is exactly what you want for fast, free, reproducible testing
    and demos. For genuinely correct, topic-specific answers to "How does
    the pH scale work?", use `AnthropicScriptProvider` instead.

    Slide count scales with difficulty (see `_DIFFICULTY_SLIDE_COUNT`):
    beginner stays a quick 5-slide overview; intermediate and advanced
    add genuinely new sections (a mechanism walkthrough, a common
    misconception, real-world applications, connections to related
    ideas) rather than just repeating the same five slides with more
    words stuffed in.
    """

    name = "simulated"

    async def generate_script(self, topic: str, difficulty: str) -> list[Slide]:
        framing = _DIFFICULTY_FRAMING.get(difficulty, _DIFFICULTY_FRAMING["beginner"])
        topic_text = topic.strip()
        short_topic = _short_title(topic_text)
        target_count = _DIFFICULTY_SLIDE_COUNT.get(difficulty, _DIFFICULTY_SLIDE_COUNT["beginner"])

        title_slide = Slide(
            heading=f"Exploring: {short_topic}",
            body_lines=[f"A {difficulty}-level STEM explainer"],
            narration=(
                f"Welcome. Today we're exploring: {topic_text} Let's dive in with {framing}."
            ),
            kind="title",
        )

        # Ordered pool of middle slides. Every prefix of this list must
        # read as a coherent script on its own, since beginner takes the
        # first 2, intermediate the first 4, and advanced all 7 — so
        # "how it works" is split into two consecutive parts (kept
        # together for intermediate) rather than interleaved with the
        # misconception/application slides that only advanced reaches.
        # Narration is kept tight (~20 words/slide average) so all 10
        # advanced-tier slides fit comfortably under the 90s video cap.
        middle_pool = [
            Slide(
                heading="What's going on here?",
                body_lines=[
                    "This topic describes a specific pattern",
                    "that scientists and engineers rely on.",
                ],
                narration=(
                    f"Let's dig in. {topic_text} This touches on a well-studied pattern "
                    f"experts use to predict how a system behaves."
                ),
                kind="content",
            ),
            Slide(
                heading="Why it matters",
                body_lines=[
                    "Understanding this helps predict outcomes",
                    "and connects to real-world applications.",
                ],
                narration=(
                    "This matters because it shapes how real systems behave, how "
                    "professionals design around it, and how related ideas build on "
                    "top of it."
                ),
                kind="diagram",
            ),
            Slide(
                heading="How it works: getting started",
                body_lines=[
                    "Start with the initial setup",
                    "every explanation of this builds from.",
                ],
                narration=(
                    "Let's break it down. Start with the initial conditions — the "
                    "state every explanation of this topic builds from."
                ),
                kind="diagram",
            ),
            Slide(
                heading="How it works: what follows",
                body_lines=[
                    "From the starting point,",
                    "a predictable sequence takes over.",
                ],
                narration=(
                    "From there, a predictable sequence of changes carries the process "
                    "through to its outcome."
                ),
                kind="diagram",
            ),
            Slide(
                heading="A common misconception",
                body_lines=[
                    "It's tempting to oversimplify this,",
                    "but the details matter.",
                ],
                narration=(
                    f"A common mistake: oversimplifying {topic_text} Skipping the details "
                    f"is exactly where misunderstandings creep in."
                ),
                kind="content",
            ),
            Slide(
                heading="Where you'll see this",
                body_lines=[
                    "This shows up in real,",
                    "practical settings — not just theory.",
                ],
                narration=(
                    f"This isn't just theoretical — {topic_text} shows up in real, "
                    f"practical settings worth understanding deeply."
                ),
                kind="content",
            ),
            Slide(
                heading="How this connects",
                body_lines=[
                    "This doesn't exist in isolation —",
                    "it links to other ideas in the field.",
                ],
                narration=(
                    "This doesn't exist in isolation — it connects to related ideas, "
                    "and understanding one often makes the others easier to grasp."
                ),
                kind="diagram",
            ),
        ]

        takeaway_slide = Slide(
            heading="Key takeaway",
            body_lines=[
                "Keep the core idea in mind:",
                "cause and effect, step by step.",
            ],
            narration=(
                "The key takeaway: this comes down to cause and effect — one step "
                "leading logically to the next, once you see the pattern."
            ),
            kind="content",
        )

        summary_slide = Slide(
            heading="Thanks for watching",
            body_lines=[f"You've now got the basics of: {short_topic}"],
            narration=f"That covers the basics of: {topic_text} Thanks for watching!",
            kind="summary",
        )

        # Total = title + middle + takeaway + summary, so the middle pool
        # needs (target_count - 3) entries.
        middle_needed = max(0, target_count - 3)
        middle_slides = middle_pool[:middle_needed]

        return [title_slide, *middle_slides, takeaway_slide, summary_slide]


_ANTHROPIC_SYSTEM_PROMPT = """You write short scripts for STEM explainer videos aimed at learners.

Given a topic/question, a difficulty level, and a target slide count and
word budget, respond with ONLY a JSON array (no markdown fences, no
commentary before or after) of exactly the requested number of slide
objects. Each object must have exactly these keys:
  "heading": a short slide title (max 6 words)
  "body_lines": a list of 1-2 short strings shown as on-screen bullet text
  "narration": 2-4 natural, spoken-style sentences a text-to-speech engine
               will read aloud for this slide. No markdown, no bullet
               symbols, no LaTeX — plain spoken English only.
  "kind": one of "title", "content", "diagram", "summary" (use "title" for
          the first slide and "summary" for the last)

Be scientifically accurate and directly answer the learner's question.
When the target slide count is higher (intermediate/advanced requests),
use the extra slides for genuine depth — a step-by-step mechanism, a
common misconception, a real-world application, or how the topic connects
to related ideas — not padding or repeating earlier slides in other
words. Stay close to the requested word budget so the narration fits
within the video's time limit."""


class AnthropicScriptProvider(ScriptProvider):
    """
    Real AI script writer: calls the Anthropic API to produce a genuinely
    topic-specific, accurate script instead of a generic template.

    Requires the `anthropic` package (`uv sync --extra ai`) and an
    `ANTHROPIC_API_KEY` environment variable. If either is missing, it
    raises `ProviderUnavailableError` — used directly/unwrapped by the
    "ai" provider (see providers.py for why "ai" mode has no fallback to
    `SimulatedScriptProvider`).

    Slide count and word budget scale with difficulty exactly like
    `SimulatedScriptProvider` (see `_DIFFICULTY_SLIDE_COUNT` /
    `_DIFFICULTY_WORD_BUDGET` above) — the target is passed to the model
    explicitly rather than left to its own judgment, so "advanced" 
    requests reliably get noticeably more depth than "beginner" ones.
    """

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
                raise ProviderUnavailableError(
                    "the 'anthropic' package is not installed; install it with "
                    "`uv sync --extra ai` to use the AI script provider"
                ) from exc
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def is_available(self) -> bool:
        if not self._api_key:
            return False
        try:
            self._get_client()
        except ProviderUnavailableError:
            return False
        return True

    async def generate_script(self, topic: str, difficulty: str) -> list[Slide]:
        if not self._api_key:
            raise ProviderUnavailableError("ANTHROPIC_API_KEY is not set")
        client = self._get_client()

        slide_count = _DIFFICULTY_SLIDE_COUNT.get(difficulty, _DIFFICULTY_SLIDE_COUNT["beginner"])
        word_budget = _DIFFICULTY_WORD_BUDGET.get(difficulty, _DIFFICULTY_WORD_BUDGET["beginner"])

        try:
            response = await client.messages.create(
                model=self._model,
                max_tokens=2500,
                system=_ANTHROPIC_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Topic/question: {topic}\n"
                            f"Difficulty: {difficulty}\n"
                            f"Target: exactly {slide_count} slides, {word_budget}."
                        ),
                    }
                ],
            )
        except Exception as exc:  # covers auth errors, rate limits, network errors, etc.
            raise ProviderUnavailableError(f"Anthropic API call failed: {exc}") from exc

        text_parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        raw_text = "".join(text_parts).strip()

        # Be tolerant of the model wrapping the JSON in a code fence despite
        # instructions not to.
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ProviderUnavailableError(
                f"Anthropic response was not valid JSON: {exc}"
            ) from exc

        slides: list[Slide] = []
        for item in data:
            try:
                slides.append(
                    Slide(
                        heading=str(item["heading"]),
                        body_lines=[str(line) for line in item.get("body_lines", [])],
                        narration=str(item["narration"]),
                        kind=str(item.get("kind", "content")),
                    )
                )
            except (KeyError, TypeError) as exc:
                raise ProviderUnavailableError(f"malformed slide in Anthropic response: {exc}") from exc

        if not slides:
            raise ProviderUnavailableError("Anthropic response contained no slides")
        return slides


class FallbackScriptProvider(ScriptProvider):
    """
    Wraps a primary `ScriptProvider` with a fallback: if the primary is
    unavailable (missing package/API key) or fails at generation time
    (network error, bad response, etc), transparently use the fallback
    instead of failing the whole job. Used by the "ai" generation provider
    to degrade from `AnthropicScriptProvider` to `SimulatedScriptProvider`
    rather than failing every job when the API key isn't configured.
    """

    def __init__(self, primary: ScriptProvider, fallback: ScriptProvider) -> None:
        self._primary = primary
        self._fallback = fallback
        self.name = f"{primary.name}(fallback={fallback.name})"

    async def is_available(self) -> bool:
        return True  # the fallback is always available, so this wrapper always is

    async def generate_script(self, topic: str, difficulty: str) -> list[Slide]:
        if await self._primary.is_available():
            try:
                return await self._primary.generate_script(topic, difficulty)
            except ProviderUnavailableError:
                pass  # fall through to the fallback below
        return await self._fallback.generate_script(topic, difficulty)
