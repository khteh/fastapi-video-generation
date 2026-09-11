"""
Video encoder selection: NVIDIA GPU acceleration (NVENC) with a verified,
automatic fallback to CPU encoding (libx264).

Critically, availability is checked by actually attempting a tiny real
encode with `h264_nvenc`, not just by checking whether ffmpeg lists the
encoder as compiled in. An ffmpeg build can have NVENC support compiled
in while the machine it's running on has no NVIDIA GPU or driver
installed at all — in that case `h264_nvenc` is listed but fails at
encode time (`Cannot load libcuda.so.1`). Only a real encode attempt
tells you which situation you're actually in, the same principle used
for flite speech-backend detection in narrator.py.
"""
from __future__ import annotations

import logging, tempfile
from dataclasses import dataclass
from pathlib import Path

from src.config import settings
from src.generation.subprocess_utils import SubprocessError, run_subprocess


@dataclass(frozen=True)
class VideoCodec:
    name: str
    # ffmpeg args for this codec, inserted as `-c:v <codec_args...>` — i.e.
    # everything after `-c:v <encoder_name>` and before the shared
    # `-pix_fmt yuv420p` / output-file args every codec gets.
    encoder_name: str
    extra_args: list[str]


LIBX264 = VideoCodec(
    name="libx264 (CPU)",
    encoder_name="libx264",
    extra_args=["-tune", "stillimage"],
)

H264_NVENC = VideoCodec(
    name="h264_nvenc (NVIDIA GPU)",
    encoder_name="h264_nvenc",
    extra_args=["-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", "23"],
)


async def _nvenc_actually_works() -> bool:
    """Attempt a real, tiny h264_nvenc encode. Returns False on any failure
    (missing driver, no GPU, encoder not compiled in, etc) rather than
    raising — this is a capability probe, not something that should crash
    startup."""
    try:
        with tempfile.TemporaryDirectory(prefix="nvenc_probe_") as tmp:
            work_dir = Path(tmp)
            await run_subprocess(
                [
                    settings.ffmpeg_binary,
                    "-hide_banner",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    f"color=c=black:s={settings.video_width}x{settings.video_height}:d=0.1",
                    "-c:v",
                    "h264_nvenc",
                    "-pix_fmt",
                    "yuv420p",
                    "-frames:v",
                    "1",
                    "probe.mp4",
                ],
                cwd=work_dir,
                timeout=15.0,
            )
        return True
    except SubprocessError as e:
        logging.exception(f"NVENC probe failed; falling back to CPU libx264: {e}")
        return False


_cached_codec: VideoCodec | None = None


async def select_video_codec() -> VideoCodec:
    """
    Return the video codec to use, honoring `VIDEO_HW_ACCEL`:
      - "cpu":  always libx264, no GPU probing at all.
      - "auto" (default): probe for a genuinely working NVENC GPU encoder;
        use it if available, otherwise fall back to libx264 automatically.
    Result is cached — the probe only runs once per process.
    """
    global _cached_codec
    if _cached_codec is not None:
        return _cached_codec

    if settings.video_hw_accel == "cpu":
        _cached_codec = LIBX264
    elif await _nvenc_actually_works():
        _cached_codec = H264_NVENC
    else:
        _cached_codec = LIBX264
    logging.info(f"Video codec selected: {_cached_codec.name}")
    return _cached_codec
