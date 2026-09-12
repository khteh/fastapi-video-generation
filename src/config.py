"""
Application configuration.

Every tunable (worker count, output directories, video dimensions, TTS
backend, provider selection, etc.) is read from environment variables with
sane defaults, so deployment and tests can override them without touching
code.

A local `.env` file (see `.env.example`) is loaded automatically if
present, via python-dotenv — this is the recommended way to set secrets
like `ANTHROPIC_API_KEY` locally without exporting them in your shell or
committing them anywhere. Real environment variables (e.g. ones set by
your deployment platform) always take precedence over `.env` values.
"""
from __future__ import annotations

import json, os, logging, sys
from dataclasses import dataclass, field
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler
from dotenv import load_dotenv

# Load variables from a `.env` file in the current working directory (or
# the nearest parent) into os.environ, without overriding anything already
# set there — e.g. a real ANTHROPIC_API_KEY exported by a deploy platform
# wins over whatever happens to be in `.env`.
load_dotenv(override=False)

def _int_env(settings: dict, name: str, default: int) -> int:
    try:
        return int(settings[name]) if name in settings else int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

def _float_env(settings: dict, name: str, default: float) -> float:
    try:
        return float(settings[name]) if name in settings else float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

def _path_env(settings: dict, name: str, default: str) -> Path:
    return Path(settings[name]) if name in settings else Path(os.environ.get(name, default))

class ConfigSingleton(type): # Inherit from "type" in order to gain access to method __call__
    __registry = {}
    def __call__(cls, *args, **kwargs):
        registry = type(cls).__registry
        if cls not in registry:
              registry[cls] = (super().__call__(*args, **kwargs), args, kwargs)
        elif registry[cls][1] != args or registry(cls)[2] != kwargs:
              raise TypeError(f"Class already initialized with different arguments!")
        return registry[cls][0]

@dataclass()
class Settings:
    with open('/etc/fastapi-video-service_config.json', 'r') as f:
        config = json.load(f)
    environment = config["ENVIRONMENT"]
    LOGLEVEL = config['LOGLEVEL']
    SECRET_KEY = config["SECRET_KEY"] or "you-will-never-guess"
    # --- Concurrency -------------------------------------------------------
    num_workers: int = _int_env(config, "VIDEO_NUM_WORKERS", 3)

    # --- Output (requirement: everything the app produces lives under
    # output/) ----------------------------------------------------------
    # Durable storage for JobRecord JSON files (job statuses) — persistence
    # layer. Independent of where generated video artifacts live.
    jobs_dir: Path = field(default_factory=lambda cfg=config: _path_env(cfg, "VIDEO_JOBS_DIR", "output/jobs"))
    # Generated video files — artifact layer.
    artifacts_dir: Path = field(
        default_factory=lambda cfg=config: _path_env(cfg, "VIDEO_ARTIFACTS_DIR", "output/videos")
    )

    # --- Validation ----------------------------------------------------------
    min_topic_length: int = _int_env(config, "VIDEO_MIN_TOPIC_LENGTH", 5)
    max_topic_length: int = _int_env(config, "VIDEO_MAX_TOPIC_LENGTH", 1024)
    max_jobs_in_store: int = _int_env(config, "VIDEO_MAX_JOBS_IN_STORE", 10_000)

    # Hard ceiling on generated video length. Enforced in two places:
    # the API's VideoRequest.duration_seconds field (le=90) rejects
    # requests that ask for more up front, and the generation pipeline
    # independently trims the assembled video to this length as a safety
    # net, since actual length is emergent from narration timing rather
    # than precisely controllable ahead of time.
    max_duration_seconds: float = _float_env(config, "VIDEO_MAX_DURATION_SECONDS", 90.0)

    # --- Generation provider (plug-and-play) --------------------------------
    # "simulated": fully offline/local pipeline — template script writer +
    #   local TTS (flite/tone). Deterministic-ish, fast, no API keys, no
    #   network. This is what the test suite uses.
    # "ai": real AI script writing (Anthropic API) + the most natural voice
    #   backend available (edge-tts / Piper neural TTS, falling back to
    #   flite/tone if unconfigured). See src/generation/providers.py.
    generation_provider: str = config["GENERATION_PROVIDER"] if "GENERATION_PROVIDER" in config else os.environ.get("GENERATION_PROVIDER", "simulated")

    # --- Video rendering -----------------------------------------------------
    video_width: int = _int_env(config, "VIDEO_WIDTH", 1280)
    video_height: int = _int_env(config, "VIDEO_HEIGHT", 720)
    video_fps: int = _int_env(config, "VIDEO_FPS", 60)
    tts_sample_rate: int = _int_env(config, "VIDEO_TTS_SAMPLE_RATE", 22050)

    # "auto" (default): use NVIDIA GPU encoding (NVENC) if a working GPU is
    #   actually detected at startup (a real tiny encode is attempted, not
    #   just an ffmpeg feature check), otherwise fall back to CPU libx264.
    # "cpu": always use libx264, skip GPU probing entirely.
    video_hw_accel: str = config["VIDEO_HW_ACCEL"] if "VIDEO_HW_ACCEL" in config else os.environ.get("VIDEO_HW_ACCEL", "auto")

    # --- Narrator backend tuning ---------------------------------------------
    flite_voice: str = config["VIDEO_FLITE_VOICE"] if "VIDEO_FLITE_VOICE" in config else os.environ.get("VIDEO_FLITE_VOICE", "kal")
    edge_tts_voice: str = config["VIDEO_EDGE_TTS_VOICE"] if "VIDEO_EDGE_TTS_VOICE" in config else os.environ.get("VIDEO_EDGE_TTS_VOICE", "en-US-AndrewNeural")
    piper_model_path: str = config["VIDEO_PIPER_MODEL_PATH"] if "VIDEO_PIPER_MODEL_PATH" in config else os.environ.get("VIDEO_PIPER_MODEL_PATH", "")

    # --- Anthropic (real AI script provider) --------------------------------
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    anthropic_model: str = config["VIDEO_ANTHROPIC_MODEL"] if "VIDEO_ANTHROPIC_MODEL" in config else os.environ.get("VIDEO_ANTHROPIC_MODEL", "claude-sonnet-5")
    # Deliberately a smaller/faster/cheaper model than the script writer
    # above — topic classification is a binary yes/no call, not content
    # generation, so it doesn't need a flagship model.
    anthropic_classifier_model: str = config["VIDEO_ANTHROPIC_CLASSIFIER_MODEL"] if "VIDEO_ANTHROPIC_CLASSIFIER_MODEL" in config else os.environ.get(
        "VIDEO_ANTHROPIC_CLASSIFIER_MODEL", "claude-haiku-4-5-20251001"
    )

    # Path to the ffmpeg/ffprobe binaries used for speech synthesis (via
    # ffmpeg's `flite` lavfi source), video/audio encoding, and duration
    # probing.
    ffmpeg_binary: str = config["VIDEO_FFMPEG_BINARY"] if "VIDEO_FFMPEG_BINARY" in config else os.environ.get("VIDEO_FFMPEG_BINARY", "ffmpeg")
    ffprobe_binary: str = config["VIDEO_FFPROBE_BINARY"] if "VIDEO_FFPROBE_BINARY" in config else os.environ.get("VIDEO_FFPROBE_BINARY", "ffprobe")
    """
    https://docs.python.org/3/library/logging.html
    The level parameter now accepts a string representation of the level such as ‘INFO’ as an alternative to the integer constants such as INFO.
    """
    logging.getLogger("httpx").setLevel(logging.WARNING)
    """
    https://realpython.com/python-modulo-string-formatting/#fine-tune-your-output-with-conversion-flags
    -	Justification of values that are shorter than the specified field width
    The Hyphen-Minus Flag (-)
    When a formatted value is shorter than the specified field width, it’s usually right-justified in the field. The hyphen-minus (-) flag causes the value to be left-justified in the specified field instead.
    """
    if environment == "development":
        logging.basicConfig(filename='/var/log/fastapi-video-service/log', filemode='w', format='%(asctime)s %(levelname)-8s %(message)s', level=LOGLEVEL, datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(handlers=[
            TimedRotatingFileHandler(filename='/var/log/fastapi-video-service/log', when='d', interval=1, backupCount=3),
            logging.StreamHandler(sys.stdout)
        ], format='%(asctime)s %(levelname)-8s %(message)s', level=LOGLEVEL, datefmt='%Y-%m-%d %H:%M:%S')

settings = Settings()
