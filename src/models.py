"""
Domain models shared by every layer of the service.

These are pure data contracts: no layer-specific logic lives here. The
state, persistence, artifact, and generation layers all speak these types,
but don't depend on each other directly (see each package's module
docstring for its specific responsibility).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from src.config import settings
from pydantic import BaseModel, Field, field_validator

_HAS_LETTER = re.compile(r"[a-zA-Z]")


class DifficultyLevel(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


class VideoRequest(BaseModel):
    """Payload a learner sends to request a STEM explainer video.

    `topic` is meant to be a natural question or concept, e.g.:
      - "How does the pH scale work?"
      - "Why do atoms form covalent bonds?"
      - "What is the difference between ionic and covalent bonding?"
      - "How does binary search work?"
      - "What is Newton's second law?"
    """

    topic: str = Field(
        ...,
        description="The STEM topic or question to explain, e.g. 'How does the pH scale work?'",
        min_length=settings.min_topic_length,
        max_length=settings.max_topic_length,
    )
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.beginner,
        description="Target audience level for the explanation.",
    )
    duration_seconds: int = Field(
        default=60,
        ge=15,
        le=90,
        description="Approximate desired video length in seconds (advisory; actual "
        "length is driven by how long the narration takes to speak, and is "
        "hard-capped at 90 seconds regardless of this value).",
    )

    @field_validator("topic")
    @classmethod
    def topic_must_have_content(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("topic must not be blank")
        if not _HAS_LETTER.search(stripped):
            raise ValueError(
                "topic must contain at least one letter — it looks like it's "
                "made up entirely of digits, symbols, or whitespace"
            )
        return stripped


class ArtifactInfo(BaseModel):
    """Metadata about a stored, retrievable video artifact."""

    job_id: str
    content_type: str = "video/mp4"
    size_bytes: int
    duration_seconds: float


class JobRecord(BaseModel):
    """Internal + external representation of a video generation job."""

    job_id: str
    status: JobStatus
    topic: str
    difficulty: DifficultyLevel
    duration_seconds: int
    provider: str
    progress: int = Field(default=0, ge=0, le=100)
    stage: Optional[str] = None
    error: Optional[str] = None
    artifact: Optional[ArtifactInfo] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)


class JobSubmittedResponse(BaseModel):
    job_id: str
    status: JobStatus
    provider: str
    poll_url: str
