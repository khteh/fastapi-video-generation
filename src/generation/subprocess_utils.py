"""Small helper for running external commands (ffmpeg) asynchronously."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from src.config import settings


class SubprocessError(RuntimeError):
    """Raised when an external command exits non-zero."""


async def run_subprocess(cmd: list[str], cwd: Optional[Path] = None, timeout: float = 60.0) -> str:
    """
    Run `cmd` asynchronously, returning combined stderr+stdout text on
    success. Raises SubprocessError (including the last part of the
    process's output for debugging) if it exits non-zero or times out.

    Using asyncio.create_subprocess_exec (not `subprocess.run`) keeps this
    non-blocking so multiple video jobs can encode concurrently without
    stalling the event loop.
    """
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise SubprocessError(f"command timed out after {timeout}s: {' '.join(cmd)}")

    output = stdout.decode(errors="replace") if stdout else ""
    if process.returncode != 0:
        tail = "\n".join(output.strip().splitlines()[-25:])
        raise SubprocessError(
            f"command exited with code {process.returncode}: {' '.join(cmd)}\n--- output tail ---\n{tail}"
        )
    return output


async def probe_duration(path: Path) -> float:
    """Return the duration (seconds) of a media file via ffprobe."""
    output = await run_subprocess(
        [
            settings.ffprobe_binary,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path.name,
        ],
        cwd=path.parent,
        timeout=15.0,
    )
    return float(output.strip())
