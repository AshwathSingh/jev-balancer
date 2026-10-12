"""Utilities for turning real prompt records into labeled workload traces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from math import ceil, isfinite

from jev_balancer.models import DurationClass, TraceJob


@dataclass(frozen=True, slots=True)
class PromptRecord:
    """One externally sourced prompt before runtime measurement.

    Attributes:
        request_id: Stable source identifier.
        payload: Human-written text supplied to the measured model.
    """

    request_id: str
    payload: str

    def __post_init__(self) -> None:
        """Reject empty identifiers and prompt text."""

        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")
        if not self.payload.strip():
            raise ValueError("payload must not be empty")


@dataclass(frozen=True, slots=True)
class DurationClassThresholds:
    """Training-fitted boundaries for short, medium, and long jobs.

    Attributes:
        short_max_ms: Largest service time labeled short.
        medium_max_ms: Largest service time labeled medium.
    """

    short_max_ms: float
    medium_max_ms: float

    def __post_init__(self) -> None:
        """Validate finite, positive, ordered boundaries."""

        if not isfinite(self.short_max_ms) or self.short_max_ms <= 0:
            raise ValueError("short_max_ms must be finite and positive")
        if not isfinite(self.medium_max_ms) or self.medium_max_ms <= 0:
            raise ValueError("medium_max_ms must be finite and positive")
        if self.medium_max_ms < self.short_max_ms:
            raise ValueError("medium_max_ms must be at least short_max_ms")

    @classmethod
    def fit(cls, jobs: Iterable[TraceJob]) -> DurationClassThresholds:
        """Fit approximate service-time tertiles from training jobs only."""

        service_times = tuple(sorted(job.service_ms for job in jobs))
        if not service_times:
            raise ValueError("at least one training job is required")
        short_index = ceil(len(service_times) / 3) - 1
        medium_index = ceil(len(service_times) * 2 / 3) - 1
        return cls(
            short_max_ms=service_times[short_index],
            medium_max_ms=service_times[medium_index],
        )

    def classify(self, service_ms: float) -> DurationClass:
        """Assign a duration class using the fitted boundaries."""

        if not isfinite(service_ms) or service_ms <= 0:
            raise ValueError("service_ms must be finite and positive")
        if service_ms <= self.short_max_ms:
            return DurationClass.SHORT
        if service_ms <= self.medium_max_ms:
            return DurationClass.MEDIUM
        return DurationClass.LONG

    def label(self, jobs: Iterable[TraceJob]) -> tuple[TraceJob, ...]:
        """Return jobs labeled without changing any scheduler-visible field."""

        return tuple(
            replace(job, duration_class=self.classify(job.service_ms)) for job in jobs
        )

    def to_dict(self) -> dict[str, float]:
        """Return JSON-ready millisecond boundaries."""

        return {
            "short_max_ms": self.short_max_ms,
            "medium_max_ms": self.medium_max_ms,
        }


def extract_oasst_prompts(
    rows: Iterable[Mapping[str, object]],
    *,
    language: str = "en",
    limit: int | None = None,
) -> tuple[PromptRecord, ...]:
    """Extract reviewed root prompts from OpenAssistant OASST1 rows.

    Args:
        rows: Dataset rows using the published OASST1 schema.
        language: Exact language tag to retain.
        limit: Optional positive maximum number of returned prompts.

    Returns:
        Prompts in source order, excluding replies, deleted rows, and rows that
        failed review.
    """

    if not language.strip():
        raise ValueError("language must not be empty")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")

    prompts: list[PromptRecord] = []
    for row in rows:
        if (
            row.get("role") != "prompter"
            or row.get("lang") != language
            or row.get("parent_id") is not None
            or row.get("deleted") is not False
            or row.get("review_result") is not True
        ):
            continue
        request_id = row.get("message_id")
        payload = row.get("text")
        if not isinstance(request_id, str) or not isinstance(payload, str):
            raise ValueError("OASST1 rows must contain string message_id and text")
        prompts.append(PromptRecord(request_id=request_id, payload=payload))
        if limit is not None and len(prompts) == limit:
            break
    return tuple(prompts)
