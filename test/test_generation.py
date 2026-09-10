"""
Tests for src.generation — fully decoupled from state/persistence/artifacts
(no pydantic dependency at all). Uses the real ffmpeg/flite backend since
it's genuinely available; these are the most valuable tests in the suite
because they prove actual audio+video artifacts are produced correctly.
"""
from __future__ import annotations

import wave

import pytest

from src.generation.narrator import FliteNarrator, ToneNarrator, select_narrator
from src.generation.pipeline import generate
from src.generation.script_providers import (
    AnthropicScriptProvider,
    FallbackScriptProvider,
    SimulatedScriptProvider,
    Slide,
    _short_title,
)
from src.generation.slide_renderer import PillowSlideRenderer
from src.generation.subprocess_utils import probe_duration
from src.generation.video_assembler import FfmpegVideoAssembler

EXAMPLE_QUESTIONS = [
    "How does the pH scale work?",
    "Why do atoms form covalent bonds?",
    "What is the difference between ionic and covalent bonding?",
]


@pytest.mark.asyncio
async def test_simulated_script_provider_handles_example_questions():
    provider = SimulatedScriptProvider()
    for topic in EXAMPLE_QUESTIONS:
        slides = await provider.generate_script(topic, "beginner")
        assert len(slides) >= 3
        assert all(slide.narration.strip() for slide in slides)
        assert all(slide.heading.strip() for slide in slides)
        # The learner's actual question should appear somewhere in the script.
        assert any(topic in s.narration or topic in s.heading for s in slides)


@pytest.mark.asyncio
async def test_simulated_script_provider_unknown_difficulty_falls_back_gracefully():
    provider = SimulatedScriptProvider()
    slides = await provider.generate_script("What is entropy?", "not-a-real-difficulty")
    assert len(slides) >= 1  # doesn't crash; falls back to beginner framing


@pytest.mark.asyncio
async def test_slide_count_scales_with_difficulty():
    """Requirement: beginner ~5, intermediate ~7, advanced ~9-10 slides,
    using more of the 90s budget for genuine depth as difficulty rises
    rather than staying flat at a fixed count."""
    provider = SimulatedScriptProvider()
    topic = "How does the pH scale work?"

    beginner = await provider.generate_script(topic, "beginner")
    intermediate = await provider.generate_script(topic, "intermediate")
    advanced = await provider.generate_script(topic, "advanced")

    assert len(beginner) == 5
    assert len(intermediate) == 7
    assert len(advanced) == 10
    # Strictly increasing, not just "different".
    assert len(beginner) < len(intermediate) < len(advanced)


@pytest.mark.asyncio
async def test_higher_difficulty_slides_are_not_just_beginner_slides_repeated():
    """The extra slides at higher difficulty must be genuinely new
    content (mechanism, misconception, application, connections) — not
    the same beginner slides padded with more words."""
    provider = SimulatedScriptProvider()
    topic = "How does the pH scale work?"

    beginner = await provider.generate_script(topic, "beginner")
    advanced = await provider.generate_script(topic, "advanced")

    beginner_headings = {s.heading for s in beginner}
    advanced_headings = {s.heading for s in advanced}

    # Every beginner slide's heading should still appear in advanced...
    assert beginner_headings <= advanced_headings
    # ...but advanced must have genuinely new headings too, not just the
    # same five with longer narration.
    assert len(advanced_headings - beginner_headings) >= 4


@pytest.mark.asyncio
async def test_advanced_script_narration_fits_within_90_second_cap():
    """The whole point of scaling slide count with difficulty: it must
    stay within the video length requirement, not just get silently
    trimmed off the end by pipeline.py's safety net."""
    provider = SimulatedScriptProvider()
    slides = await provider.generate_script("How does the pH scale work?", "advanced")
    total_words = sum(len(s.narration.split()) for s in slides)
    estimated_seconds = total_words / 145 * 60  # matches narrator.py's pacing constant
    assert estimated_seconds < 90.0
    # Leave a reasonable safety margin below the hard cap for narrator
    # timing variance (real TTS pacing isn't exactly 145 wpm).
    assert estimated_seconds < 85.0


def test_short_title_leaves_short_topics_unchanged():
    topic = "How does the pH scale work?"
    assert _short_title(topic) == topic


def test_short_title_truncates_long_topic_at_word_boundary_with_ellipsis():
    long_topic = (
        "What is the difference between the various types of intermolecular "
        "forces including hydrogen bonding, van der Waals forces, and "
        "dipole-dipole interactions?"
    )
    short = _short_title(long_topic, max_chars=70)
    assert len(short) <= 71  # 70 + the ellipsis character
    assert short.endswith("…")
    # Must not cut mid-word: strip the ellipsis and confirm what's left
    # is a prefix of the original topic ending at a word boundary.
    without_ellipsis = short[:-1].rstrip()
    assert long_topic.startswith(without_ellipsis)
    assert long_topic[len(without_ellipsis)] == " "


def test_short_title_handles_single_unbreakable_word():
    """No spaces at all to break on — must still terminate and produce
    something reasonably short, not crash or return the whole string."""
    no_spaces = "Supercalifragilisticexpialidocious" * 5
    short = _short_title(no_spaces, max_chars=70)
    assert len(short) <= 71
    assert short.endswith("…")


@pytest.mark.asyncio
async def test_long_topic_narration_keeps_full_text_only_visual_text_shortened():
    """Core requirement: the FULL topic is always spoken in narration
    (audio has no width constraint) — only the on-screen heading/body
    text gets shortened, since that's what doesn't fit on a slide."""
    long_topic = (
        "What is the difference between the various types of intermolecular "
        "forces including hydrogen bonding, van der Waals forces, and "
        "dipole-dipole interactions?"
    )
    provider = SimulatedScriptProvider()
    slides = await provider.generate_script(long_topic, "beginner")

    title_slide = slides[0]
    assert long_topic in title_slide.narration
    assert long_topic not in title_slide.heading  # heading is shortened
    assert len(title_slide.heading) < len(long_topic)

    summary_slide = slides[-1]
    assert long_topic in summary_slide.narration
    assert long_topic not in summary_slide.body_lines[0]


@pytest.mark.asyncio
async def test_slide_renderer_wraps_long_heading_without_overflow():
    """The renderer-level safety net: given ANY long heading (regardless
    of source), it must wrap to at most 2 lines and fit within the
    frame — verified by measuring actual rendered text width, not just
    checking it doesn't crash."""
    from PIL import ImageDraw

    long_heading_slide = Slide(
        heading=(
            "Exploring: What is the difference between the various types of "
            "intermolecular forces including hydrogen bonding?"
        ),
        body_lines=["test"],
        narration="test",
        kind="title",
    )
    renderer = PillowSlideRenderer(1280, 720)
    img = renderer.render(long_heading_slide, 0, 1)
    assert img.size == (1280, 720)

    # Re-derive what the renderer computed and confirm every line
    # actually fits within the available width.
    draw = ImageDraw.Draw(img)
    from src.generation.slide_renderer import _font

    heading_font = _font(44)
    max_width = 1280 - 2 * 50
    lines = renderer._fit_lines(draw, long_heading_slide.heading, heading_font, max_width, max_lines=2)
    assert len(lines) <= 2
    for line in lines:
        assert draw.textlength(line, font=heading_font) <= max_width


def test_slide_renderer_single_unbreakable_word_does_not_overflow():
    from PIL import ImageDraw

    from src.generation.slide_renderer import _font

    pathological_slide = Slide(
        heading="Exploring: " + "Supercalifragilisticexpialidocious" * 5,
        body_lines=[],
        narration="test",
        kind="title",
    )
    renderer = PillowSlideRenderer(1280, 720)
    img = renderer.render(pathological_slide, 0, 1)
    draw = ImageDraw.Draw(img)
    heading_font = _font(44)
    max_width = 1280 - 2 * 50
    lines = renderer._fit_lines(draw, pathological_slide.heading, heading_font, max_width, max_lines=2)
    for line in lines:
        assert draw.textlength(line, font=heading_font) <= max_width


@pytest.mark.asyncio
async def test_simulated_script_provider_always_available():
    provider = SimulatedScriptProvider()
    assert await provider.is_available() is True


@pytest.mark.asyncio
async def test_anthropic_provider_unavailable_without_api_key():
    provider = AnthropicScriptProvider(api_key="", model="claude-sonnet-5")
    assert await provider.is_available() is False


@pytest.mark.asyncio
async def test_fallback_script_provider_degrades_to_simulated_without_api_key():
    """Core plug-and-play/graceful-degradation guarantee: the 'ai' provider
    must still produce a script even when Anthropic isn't configured."""
    primary = AnthropicScriptProvider(api_key="", model="claude-sonnet-5")
    fallback = SimulatedScriptProvider()
    combo = FallbackScriptProvider(primary=primary, fallback=fallback)

    slides = await combo.generate_script("How does the pH scale work?", "beginner")
    assert len(slides) >= 3
    assert "simulated" in combo.name


def test_slide_renderer_produces_correctly_sized_image():
    from src.generation.script_providers import Slide

    slide = Slide(heading="Test", body_lines=["line one"], narration="Testing.", kind="title")
    renderer = PillowSlideRenderer(400, 300)
    image = renderer.render(slide, 0, 1)
    assert image.size == (400, 300)


def test_slide_renderer_scales_proportionally_for_4k():
    """4K output must be a properly scaled-up layout, not a 720p-sized
    layout stretched onto a bigger canvas. Downscaling a 4K render should
    closely match the equivalent 720p render, pixel-for-pixel."""
    from src.generation.script_providers import Slide

    slide = Slide(
        heading="Understanding: How does the pH scale work?",
        body_lines=["A beginner-level STEM explainer"],
        narration="test",
        kind="title",
    )

    render_720p = PillowSlideRenderer(1280, 720).render(slide, 0, 5)
    render_4k = PillowSlideRenderer(3840, 2160).render(slide, 0, 5)

    assert render_4k.size == (3840, 2160)
    downscaled = render_4k.resize((1280, 720))

    # Compare mean pixel difference rather than exact equality — font
    # rasterization at different sizes won't be bit-identical, but the
    # overall layout (header band position, diagram placement, footer
    # dots) should match closely.
    import numpy as np

    diff = np.abs(
        np.asarray(downscaled, dtype=np.int16) - np.asarray(render_720p, dtype=np.int16)
    )
    assert diff.mean() < 5.0  # near-identical on average across all pixels


@pytest.mark.asyncio
async def test_tone_narrator_produces_valid_timed_wav(tmp_path):
    narrator = ToneNarrator(sample_rate=8000)
    out_path = tmp_path / "narration.wav"
    duration = await narrator.synthesize("This is a short test sentence.", out_path)

    assert out_path.exists()
    assert duration > 0
    with wave.open(str(out_path), "rb") as wav:
        measured = wav.getnframes() / float(wav.getframerate())
    assert measured == pytest.approx(duration, abs=0.01)


@pytest.mark.asyncio
async def test_select_narrator_simulated_mode_never_picks_network_backends():
    narrator = await select_narrator(prefer_realistic=False)
    assert narrator.name in ("flite", "tone")


@pytest.mark.asyncio
async def test_select_narrator_realistic_mode_returns_working_backend(tmp_path):
    narrator = await select_narrator(prefer_realistic=True)
    out_path = tmp_path / "narration.wav"
    duration = await narrator.synthesize("Testing narrator selection.", out_path)
    assert out_path.exists()
    assert duration > 0


@pytest.mark.asyncio
async def test_flite_narrator_produces_real_speech_audio(tmp_path):
    """This environment's ffmpeg build includes libflite support; verify it
    actually produces speech audio, not just any WAV file."""
    narrator = FliteNarrator(voice="kal", sample_rate=22050)
    out_path = tmp_path / "narration.wav"
    duration = await narrator.synthesize(
        "The pH scale measures how acidic or basic a solution is.", out_path
    )
    assert duration > 1.0  # a full sentence should take more than a second to speak
    with wave.open(str(out_path), "rb") as wav:
        assert wav.getnframes() > 0
        assert wav.getframerate() == 22050


@pytest.mark.asyncio
async def test_full_pipeline_produces_valid_mp4_for_a_real_stem_question(tmp_path):
    script_provider = SimulatedScriptProvider()
    narrator = await select_narrator(prefer_realistic=False)
    renderer = PillowSlideRenderer(480, 270)
    assembler = FfmpegVideoAssembler(480, 270, 10)

    progress_events: list[tuple[int, str]] = []

    async def on_progress(pct, stage):
        progress_events.append((pct, stage))

    result = await generate(
        topic="Why do atoms form covalent bonds?",
        difficulty="beginner",
        script_provider=script_provider,
        narrator=narrator,
        slide_renderer=renderer,
        assembler=assembler,
        work_dir=tmp_path,
        on_progress=on_progress,
    )

    assert result.video_path.exists()
    assert result.video_path.stat().st_size > 0
    assert result.duration_seconds > 0

    assert len(progress_events) >= 3
    percentages = [p for p, _ in progress_events]
    assert percentages == sorted(percentages)

    probed_duration = await probe_duration(result.video_path)
    assert probed_duration == pytest.approx(result.duration_seconds, abs=0.5)


@pytest.mark.asyncio
async def test_pipeline_enforces_hard_duration_cap(tmp_path):
    script_provider = SimulatedScriptProvider()
    narrator = await select_narrator(prefer_realistic=False)
    renderer = PillowSlideRenderer(480, 270)
    assembler = FfmpegVideoAssembler(480, 270, 10)

    result = await generate(
        topic="What is the difference between ionic and covalent bonding?",
        difficulty="advanced",
        script_provider=script_provider,
        narrator=narrator,
        slide_renderer=renderer,
        assembler=assembler,
        work_dir=tmp_path,
        max_duration_seconds=6.0,  # deliberately tiny to force trimming
    )

    assert result.duration_seconds <= 6.0 + 0.25
    probed = await probe_duration(result.video_path)
    assert probed <= 6.5


@pytest.mark.asyncio
async def test_pipeline_respects_default_90_second_cap(tmp_path):
    """Without an explicit override, the pipeline should fall back to the
    configured 90-second ceiling (requirement: limit video length to 90s)."""
    from src.config import settings

    assert settings.max_duration_seconds == 90.0

    script_provider = SimulatedScriptProvider()
    narrator = await select_narrator(prefer_realistic=False)
    renderer = PillowSlideRenderer(480, 270)
    assembler = FfmpegVideoAssembler(480, 270, 10)

    result = await generate(
        topic="What is Newton's second law?",
        difficulty="beginner",
        script_provider=script_provider,
        narrator=narrator,
        slide_renderer=renderer,
        assembler=assembler,
        work_dir=tmp_path,
    )
    assert result.duration_seconds <= 90.0
