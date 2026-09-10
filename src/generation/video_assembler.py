"""
Combines rendered slide images and their narration audio into a single MP4
using ffmpeg: one still-image-plus-audio segment per slide, concatenated
into the final video. No moviepy dependency — just the ffmpeg binary that
`narrator.py` already relies on for speech synthesis.

Encoding uses NVIDIA GPU acceleration (NVENC) automatically when a working
GPU is detected, falling back to CPU (libx264) otherwise — see
`hw_accel.py` for the (verified, not just feature-flag) detection logic.
"""
from __future__ import annotations

import abc
from pathlib import Path

from src.config import settings
from src.generation.hw_accel import VideoCodec, select_video_codec
from src.generation.subprocess_utils import run_subprocess


class VideoAssembler(abc.ABC):
    @abc.abstractmethod
    async def assemble(
        self,
        slide_audio_pairs: list[tuple[Path, Path]],
        work_dir: Path,
        output_filename: str,
    ) -> Path:
        """
        Combine (image_path, audio_path) pairs, in order, into a single MP4.

        Precondition: every image_path and audio_path must already live
        directly inside `work_dir` (the pipeline is responsible for writing
        slide images and narration audio there) — ffmpeg is invoked with
        `cwd=work_dir` and referenced by bare filename, which keeps this
        code simple and avoids filtergraph/path-escaping pitfalls entirely.

        Returns the path to the assembled file (inside `work_dir`).
        """


class FfmpegVideoAssembler(VideoAssembler):
    def __init__(self, width: int, height: int, fps: int) -> None:
        self._width = width
        self._height = height
        self._fps = fps
        self._codec: VideoCodec | None = None

    async def _get_codec(self) -> VideoCodec:
        if self._codec is None:
            self._codec = await select_video_codec()
        return self._codec

    async def assemble(
        self,
        slide_audio_pairs: list[tuple[Path, Path]],
        work_dir: Path,
        output_filename: str,
    ) -> Path:
        codec = await self._get_codec()
        segment_names: list[str] = []
        for i, (image_path, audio_path) in enumerate(slide_audio_pairs):
            if image_path.parent.resolve() != work_dir.resolve() or audio_path.parent.resolve() != work_dir.resolve():
                raise ValueError(
                    f"slide assets must live directly in work_dir ({work_dir}); "
                    f"got {image_path} and {audio_path}"
                )
            segment_name = f"segment_{i:03d}.mp4"
            await self._encode_segment(image_path.name, audio_path.name, work_dir, segment_name, codec)
            segment_names.append(segment_name)

        concat_list = work_dir / "concat.txt"
        concat_list.write_text("".join(f"file '{name}'\n" for name in segment_names))

        await run_subprocess(
            [
                settings.ffmpeg_binary,
                "-hide_banner",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_list.name,
                "-c",
                "copy",
                "-fflags",
                "+genpts",
                output_filename,
            ],
            cwd=work_dir,
        )
        return work_dir / output_filename

    async def _encode_segment(
        self, image_name: str, audio_name: str, work_dir: Path, segment_name: str, codec: VideoCodec
    ) -> None:
        await run_subprocess(
            [
                settings.ffmpeg_binary,
                "-hide_banner",
                "-y",
                "-loop",
                "1",
                "-i",
                image_name,
                "-i",
                audio_name,
                "-c:v",
                codec.encoder_name,
                *codec.extra_args,
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-pix_fmt",
                "yuv420p",
                "-r",
                str(self._fps),
                "-vf",
                f"scale={self._width}:{self._height}",
                "-shortest",
                segment_name,
            ],
            cwd=work_dir,
        )
