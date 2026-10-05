"""Provider-neutral domain models used by schedulers and simulations."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from math import isclose, isfinite


def _require_name(value: str, field_name: str) -> None:
    """Reject identifiers that are empty or contain only whitespace."""

    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _require_finite_nonnegative(value: float, field_name: str) -> None:
    """Require a finite numeric value greater than or equal to zero."""

    if not isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")


def _require_finite_positive(value: float, field_name: str) -> None:
    """Require a finite numeric value strictly greater than zero."""

    if not isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be finite and positive")


def _require_nonnegative_integer(value: int, field_name: str) -> None:
    """Require an integer count while explicitly rejecting booleans."""

    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


class DurationClass(str, Enum):
    """Coarse service-time classes produced by a duration predictor."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class _FrozenProbabilities(Mapping[DurationClass, float]):
    """Small immutable mapping that remains compatible with ``asdict``."""

    __slots__ = ("_items",)
    _items: tuple[tuple[DurationClass, float], ...]

    def __init__(self, probabilities: Mapping[DurationClass, float]) -> None:
        self._items = tuple(
            (duration_class, probabilities[duration_class])
            for duration_class in DurationClass
        )

    def __getitem__(self, key: DurationClass) -> float:
        """Return the probability for ``key``."""

        for duration_class, probability in self._items:
            if duration_class == key:
                return probability
        raise KeyError(key)

    def __iter__(self) -> Iterator[DurationClass]:
        """Iterate over duration classes in their declared order."""

        return (duration_class for duration_class, _ in self._items)

    def __len__(self) -> int:
        """Return the number of represented duration classes."""

        return len(self._items)

    def __deepcopy__(self, memo: dict[int, object]) -> _FrozenProbabilities:
        """Return this immutable value unchanged when copied."""

        return self


@dataclass(frozen=True, slots=True)
class WorkItem:
    """A scheduler-visible request.

    True service time intentionally does not live on this type. Only the
    simulator and oracle policy may access that value through :class:`TraceJob`.

    Attributes:
        request_id: Non-empty identifier unique within a simulation run.
        arrival_ms: Non-negative arrival time on the simulation clock.
        payload: Text available to semantic predictors.
        input_units: Non-negative integer used by non-semantic predictors.
        sla_ms: Optional positive end-to-end latency objective.
    """

    request_id: str
    arrival_ms: float
    payload: str
    input_units: int
    sla_ms: float | None = None

    def __post_init__(self) -> None:
        _require_name(self.request_id, "request_id")
        _require_finite_nonnegative(self.arrival_ms, "arrival_ms")
        _require_nonnegative_integer(self.input_units, "input_units")
        if self.sla_ms is not None:
            _require_finite_positive(self.sla_ms, "sla_ms")


@dataclass(frozen=True, slots=True)
class TraceJob:
    """A work item plus simulation truth hidden from normal policies.

    Attributes:
        item: Scheduler-visible request data.
        service_ms: Positive observed execution time used by the simulator.
        duration_class: Optional ground-truth class used for evaluation.
    """

    item: WorkItem
    service_ms: float
    duration_class: DurationClass | None = None

    def __post_init__(self) -> None:
        _require_finite_positive(self.service_ms, "service_ms")


@dataclass(frozen=True, slots=True)
class WorkerSnapshot:
    """Observable worker load at a scheduling decision point.

    Attributes:
        worker_id: Non-empty stable worker identifier.
        job_count: Number of running and queued jobs.
        estimated_backlog_ms: Non-negative estimated work remaining.
    """

    worker_id: str
    job_count: int
    estimated_backlog_ms: float

    def __post_init__(self) -> None:
        _require_name(self.worker_id, "worker_id")
        _require_nonnegative_integer(self.job_count, "job_count")
        _require_finite_nonnegative(
            self.estimated_backlog_ms,
            "estimated_backlog_ms",
        )


@dataclass(frozen=True, slots=True)
class DurationPrediction:
    """A provider-neutral probabilistic service-time estimate.

    ``expected_service_ms`` is calculated by the predictor from this
    distribution and its calibrated class means. This model validates the
    resulting value but intentionally does not own that calibration table.

    Attributes:
        probabilities: Probability for every :class:`DurationClass`, summing
            to one.
        expected_service_ms: Positive predicted execution time.
        confidence: Predictor confidence from zero through one.
        predictor: Non-empty name of the producing predictor.
        overhead_ms: Non-negative time spent producing the prediction.
    """

    probabilities: Mapping[DurationClass, float]
    expected_service_ms: float
    confidence: float
    predictor: str
    overhead_ms: float = 0.0

    def __post_init__(self) -> None:
        if any(not isinstance(key, DurationClass) for key in self.probabilities):
            raise ValueError("probability keys must be DurationClass values")

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

        object.__setattr__(self, "probabilities", _FrozenProbabilities(probabilities))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation for traces and benchmark output."""

        return {
            "probabilities": {
                duration_class.value: probability
                for duration_class, probability in self.probabilities.items()
            },
            "expected_service_ms": self.expected_service_ms,
            "confidence": self.confidence,
            "predictor": self.predictor,
            "overhead_ms": self.overhead_ms,
        }


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """The auditable result of assigning one request to one worker.

    Attributes:
        request_id: Identifier of the assigned request.
        worker_id: Identifier of the selected worker.
        policy: Non-empty name of the scheduling policy.
        decided_at_ms: Non-negative decision time on the simulation clock.
        estimated_service_ms: Positive service-time estimate for the request.
        estimated_backlog_ms: Worker backlog observed when routing began.
        prediction: Optional semantic prediction supporting the decision.
    """

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

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation for an auditable decision trace."""

        return {
            "request_id": self.request_id,
            "worker_id": self.worker_id,
            "policy": self.policy,
            "decided_at_ms": self.decided_at_ms,
            "estimated_service_ms": self.estimated_service_ms,
            "estimated_backlog_ms": self.estimated_backlog_ms,
            "estimated_finish_ms": self.estimated_finish_ms,
            "prediction": self.prediction.to_dict() if self.prediction else None,
        }
