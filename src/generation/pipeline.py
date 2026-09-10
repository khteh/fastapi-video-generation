"""
Orchestrates the async video-generation pipeline.

Deliberately decoupled from job state, persistence, and artifact storage:
`generate()` is a pure function of (topic, difficulty, injected providers,
progress callback) that returns a finished video file's path and duration.
The caller (see `src/worker.py`) is responsible for turning that into a
stored artifact and recorded job state. This keeps the generation logic
independently testable and swappable without touching how jobs are tracked
or how artifacts are stored.

Note: topic classification is NOT part of this pipeline. It's checked
synchronously by `GenerationProvider.validate_topic()` (see providers.py)
BEFORE a job — and therefore before any call to this function — is
created, so a semantically-invalid topic is rejected immediately with no
job ever created, rather than failing partway through generation.

Every step is provided by an injected interface (`ScriptProvider`,
`Narrator`, `PillowSlideRenderer`, `VideoAssembler`) — this function itself
has no opinion on whether the script came from a template or a real LLM,
or whether the voice is a tone, flite, Piper, or a cloud neural TTS. See
`src/generation/providers.py` for how "simulated" vs "ai" wire different
combinations of these together.
"""
from __future__ import annotations

from pathlib import Path
from typing import Awaitable, Callable, Optional

from src.config import settings
from src.generation.hw_accel import select_video_codec
from src.generation.narrator import Narrator
from src.generation.script_providers import ScriptProvider
from src.generation.slide_renderer import PillowSlideRenderer
from src.generation.subprocess_utils import probe_duration, run_subprocess
from src.generation.video_assembler import VideoAssembler

ProgressCallback = Callable[[int, str], Awaitable[None]]


class GenerationError(Exception):
    """Raised when the pipeline cannot produce a video for the given input."""


class GenerationResult:
    def __init__(self, video_path: Path, duration_seconds: float) -> None:
        self.video_path = video_path
        self.duration_seconds = duration_seconds


async def _noop_progress(progress: int, stage: str) -> None:
    return None


async def generate(
    topic: str,
    difficulty: str,
    script_provider: ScriptProvider,
    narrator: Narrator,
    slide_renderer: PillowSlideRenderer,
    assembler: VideoAssembler,
    work_dir: Path,
    on_progress: Optional[ProgressCallback] = None,
    max_duration_seconds: Optional[float] = None,
) -> GenerationResult:
    """
    Run the full pipeline inside `work_dir` (caller owns its lifecycle —
    typically a TemporaryDirectory that gets cleaned up after the artifact
    is copied out). Reports progress after each major phase via
    `on_progress(progress_percent, stage_name)`.

    `max_duration_seconds` is a hard ceiling: since actual video length is
    emergent from how long the narration takes to speak (not something we
    can dial precisely ahead of time), the assembled video is trimmed down
    to this length if it runs over, as a safety net. Defaults to
    `settings.max_duration_seconds` (90s) when not given explicitly.
    """
    progress_cb = on_progress or _noop_progress
    cap = max_duration_seconds if max_duration_seconds is not None else settings.max_duration_seconds
    work_dir.mkdir(parents=True, exist_ok=True)

    await progress_cb(5, "writing_script")
    slides = await script_provider.generate_script(topic, difficulty)
    if not slides:
        raise GenerationError(f"could not produce a script for topic '{topic}'")

    slide_audio_pairs: list[tuple[Path, Path]] = []
    total_slides = len(slides)

    for i, slide in enumerate(slides):
        image_path = work_dir / f"slide_{i:03d}.png"
        audio_path = work_dir / f"slide_{i:03d}.wav"

        image = slide_renderer.render(slide, i, total_slides)
        image.save(image_path)

        await narrator.synthesize(slide.narration, audio_path)

        slide_audio_pairs.append((image_path, audio_path))

        # Slide production spans progress 10% -> 70%.
        progress = 10 + int(60 * (i + 1) / total_slides)
        await progress_cb(progress, f"rendering_slide_{i + 1}_of_{total_slides}")

    await progress_cb(75, "assembling_video")
    output_path = await assembler.assemble(slide_audio_pairs, work_dir, "final.mp4")
    duration = await probe_duration(output_path)

    if cap is not None and duration > cap:
        await progress_cb(90, "trimming_to_length_limit")
        trimmed_path = work_dir / "final_trimmed.mp4"
        codec = await select_video_codec()
        await run_subprocess(
            [
                settings.ffmpeg_binary,
                "-hide_banner",
                "-y",
                "-i",
                output_path.name,
                "-t",
                str(cap),
                "-c:v",
                codec.encoder_name,
                *codec.extra_args,
                "-c:a",
                "aac",
                "-pix_fmt",
                "yuv420p",
                trimmed_path.name,
            ],
            cwd=work_dir,
        )
        output_path = trimmed_path
        duration = await probe_duration(output_path)

    await progress_cb(95, "finalizing")
    return GenerationResult(video_path=output_path, duration_seconds=duration)
