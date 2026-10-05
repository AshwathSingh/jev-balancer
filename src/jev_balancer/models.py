"""Provider-neutral domain models used by schedulers and simulations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from math import isclose, isfinite
from types import MappingProxyType


def _require_name(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _require_finite_nonnegative(value: float, field_name: str) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")


def _require_finite_positive(value: float, field_name: str) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be finite and positive")


class DurationClass(str, Enum):
    """Coarse service-time classes produced by a semantic predictor."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


@dataclass(frozen=True, slots=True)
class WorkItem:
    """A scheduler-visible request.

    True service time intentionally does not live on this type. Only the
    simulator and oracle policy may access that value through :class:`TraceJob`.
    """

    request_id: str
    arrival_ms: float
    payload: str
    input_units: int
    sla_ms: float | None = None

    def __post_init__(self) -> None:
        _require_name(self.request_id, "request_id")
        _require_finite_nonnegative(self.arrival_ms, "arrival_ms")
        if self.input_units < 0:
            raise ValueError("input_units must be non-negative")
        if self.sla_ms is not None:
            _require_finite_positive(self.sla_ms, "sla_ms")


@dataclass(frozen=True, slots=True)
class TraceJob:
    """A work item plus simulation truth hidden from normal policies."""

    item: WorkItem
    service_ms: float
    duration_class: DurationClass | None = None

    def __post_init__(self) -> None:
        _require_finite_positive(self.service_ms, "service_ms")


@dataclass(frozen=True, slots=True)
class WorkerSnapshot:
    """Observable worker load at a scheduling decision point."""

    worker_id: str
    job_count: int
    estimated_backlog_ms: float

    def __post_init__(self) -> None:
        _require_name(self.worker_id, "worker_id")
        if self.job_count < 0:
            raise ValueError("job_count must be non-negative")
        _require_finite_nonnegative(
            self.estimated_backlog_ms,
            "estimated_backlog_ms",
        )


@dataclass(frozen=True, slots=True)
class DurationPrediction:
    """A provider-neutral probabilistic service-time estimate."""

    probabilities: Mapping[DurationClass, float]
    expected_service_ms: float
    confidence: float
    predictor: str
    overhead_ms: float = 0.0

    def __post_init__(self) -> None:
        probabilities = dict(self.probabilities)
        expected_classes = set(DurationClass)
        if set(probabilities) != expected_classes:
            raise ValueError("probabilities must contain every duration class")

        for probability in probabilities.values():
            if not isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("probabilities must be finite values from 0 to 1")

        if not isclose(sum(probabilities.values()), 1.0, abs_tol=1e-9):
            raise ValueError("probabilities must sum to 1")

        _require_finite_positive(self.expected_service_ms, "expected_service_ms")
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be a finite value from 0 to 1")
        _require_name(self.predictor, "predictor")
        _require_finite_nonnegative(self.overhead_ms, "overhead_ms")

        object.__setattr__(self, "probabilities", MappingProxyType(probabilities))


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """The auditable result of assigning one request to one worker."""

    request_id: str
    worker_id: str
    policy: str
    decided_at_ms: float
    estimated_service_ms: float
    estimated_backlog_ms: float
    prediction: DurationPrediction | None = None

    def __post_init__(self) -> None:
        _require_name(self.request_id, "request_id")
        _require_name(self.worker_id, "worker_id")
        _require_name(self.policy, "policy")
        _require_finite_nonnegative(self.decided_at_ms, "decided_at_ms")
        _require_finite_positive(self.estimated_service_ms, "estimated_service_ms")
        _require_finite_nonnegative(
            self.estimated_backlog_ms,
            "estimated_backlog_ms",
        )

    @property
    def estimated_finish_ms(self) -> float:
        """Return the predicted finish time relative to the simulation clock."""

        backlog_done_at = self.decided_at_ms + self.estimated_backlog_ms
        return backlog_done_at + self.estimated_service_ms
