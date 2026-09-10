"""Tests for src.generation.hw_accel — GPU encoder detection and fallback."""
from __future__ import annotations

import pytest

from src.generation import hw_accel


@pytest.fixture(autouse=True)
def _reset_codec_cache():
    """The selected codec is cached at module scope (the probe should only
    run once per process in production); reset it around each test so
    tests don't leak state into each other."""
    hw_accel._cached_codec = None
    yield
    hw_accel._cached_codec = None


class _FakeSettings:
    def __init__(self, hw_accel_mode: str) -> None:
        self.video_hw_accel = hw_accel_mode
        self.ffmpeg_binary = "ffmpeg"


@pytest.mark.asyncio
async def test_cpu_mode_always_selects_libx264_without_probing(monkeypatch):
    monkeypatch.setattr(hw_accel, "settings", _FakeSettings("cpu"))

    probed = False

    async def _should_not_be_called():
        nonlocal probed
        probed = True
        return True

    monkeypatch.setattr(hw_accel, "_nvenc_actually_works", _should_not_be_called)

    codec = await hw_accel.select_video_codec()
    assert codec.encoder_name == "libx264"
    assert probed is False  # "cpu" mode must skip GPU probing entirely


@pytest.mark.asyncio
async def test_auto_mode_uses_nvenc_when_probe_succeeds(monkeypatch):
    monkeypatch.setattr(hw_accel, "settings", _FakeSettings("auto"))
    monkeypatch.setattr(hw_accel, "_nvenc_actually_works", lambda: _true())

    codec = await hw_accel.select_video_codec()
    assert codec.encoder_name == "h264_nvenc"


@pytest.mark.asyncio
async def test_auto_mode_falls_back_to_cpu_when_probe_fails(monkeypatch):
    monkeypatch.setattr(hw_accel, "settings", _FakeSettings("auto"))
    monkeypatch.setattr(hw_accel, "_nvenc_actually_works", lambda: _false())

    codec = await hw_accel.select_video_codec()
    assert codec.encoder_name == "libx264"


@pytest.mark.asyncio
async def test_selection_is_cached_across_calls(monkeypatch):
    monkeypatch.setattr(hw_accel, "settings", _FakeSettings("cpu"))
    first = await hw_accel.select_video_codec()
    second = await hw_accel.select_video_codec()
    assert first is second


@pytest.mark.asyncio
async def test_real_probe_on_this_machine_does_not_raise():
    """Whatever the real answer is on the machine running the tests, the
    probe must complete cleanly (no exception, no hang) and auto mode must
    land on a valid codec either way — this is the actual environment
    integration check, as opposed to the mocked unit tests above."""
    codec = await hw_accel.select_video_codec()
    assert codec.encoder_name in ("libx264", "h264_nvenc")


async def _true():
    return True


async def _false():
    return False
