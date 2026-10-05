"""Conventional scheduling policies and an oracle comparison bound."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType

from jev_balancer.models import TraceJob, WorkerSnapshot, WorkItem
from jev_balancer.scheduler import SchedulingChoice


def _require_positive(value: float, field_name: str) -> None:
    """Require a finite value strictly greater than zero."""

    if not isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be finite and positive")


def _require_nonnegative(value: float, field_name: str) -> None:
    """Require a finite value greater than or equal to zero."""

    if not isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")


def _require_workers(
    workers: tuple[WorkerSnapshot, ...],
) -> tuple[WorkerSnapshot, ...]:
    """Reject direct policy calls that omit worker snapshots."""

    if not workers:
        raise ValueError("at least one worker snapshot is required")
    return workers


def _least_backlog(workers: tuple[WorkerSnapshot, ...]) -> WorkerSnapshot:
    """Select estimated work first, then queue length and identifier."""

    return min(
        _require_workers(workers),
        key=lambda worker: (
            worker.estimated_backlog_ms,
            worker.job_count,
            worker.worker_id,
        ),
    )


@dataclass(slots=True)
class RoundRobinPolicy:
    """Cycle through workers without considering their current load.

    Attributes:
        service_estimate_ms: Constant estimate recorded for every assigned job.
        decision_latency_ms: Simulated time required to make each decision.
    """

    service_estimate_ms: float
    decision_latency_ms: float = 0.0
    name: str = field(default="round-robin", init=False)
    _next_index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the configured estimate and scheduling overhead."""

        _require_positive(self.service_estimate_ms, "service_estimate_ms")
        _require_nonnegative(self.decision_latency_ms, "decision_latency_ms")

    def reset(self) -> None:
        """Restart the worker cycle before a simulation run."""

        self._next_index = 0

    def choose(
        self,
        item: WorkItem,
        workers: tuple[WorkerSnapshot, ...],
        now_ms: float,
    ) -> SchedulingChoice:
        """Choose the next worker in the simulator's stable ordering."""

        available_workers = _require_workers(workers)
        worker = available_workers[self._next_index % len(available_workers)]
        self._next_index += 1
        return SchedulingChoice(
            worker_id=worker.worker_id,
            estimated_service_ms=self.service_estimate_ms,
            decision_latency_ms=self.decision_latency_ms,
        )


@dataclass(frozen=True, slots=True)
class LeastJobsPolicy:
    """Choose the worker with the fewest running and queued jobs."""

    service_estimate_ms: float
    decision_latency_ms: float = 0.0
    name: str = field(default="least-jobs", init=False)

    def __post_init__(self) -> None:
        """Validate the configured estimate and scheduling overhead."""

        _require_positive(self.service_estimate_ms, "service_estimate_ms")
        _require_nonnegative(self.decision_latency_ms, "decision_latency_ms")

    def reset(self) -> None:
        """Reset state; this policy is stateless."""

    def choose(
        self,
        item: WorkItem,
        workers: tuple[WorkerSnapshot, ...],
        now_ms: float,
    ) -> SchedulingChoice:
        """Choose job count first and worker identifier second."""

        worker = min(
            _require_workers(workers),
            key=lambda snapshot: (snapshot.job_count, snapshot.worker_id),
        )
        return SchedulingChoice(
            worker_id=worker.worker_id,
            estimated_service_ms=self.service_estimate_ms,
            decision_latency_ms=self.decision_latency_ms,
        )


@dataclass(frozen=True, slots=True)
class MeanWorkPolicy:
    """Choose the smallest backlog using one historical mean service time."""

    mean_service_ms: float
    decision_latency_ms: float = 0.0
    name: str = field(default="mean-work", init=False)

    def __post_init__(self) -> None:
        """Validate the historical mean and scheduling overhead."""

        _require_positive(self.mean_service_ms, "mean_service_ms")
        _require_nonnegative(self.decision_latency_ms, "decision_latency_ms")

    def reset(self) -> None:
        """Reset state; this policy is stateless."""

    def choose(
        self,
        item: WorkItem,
        workers: tuple[WorkerSnapshot, ...],
        now_ms: float,
    ) -> SchedulingChoice:
        """Choose the worker with the least estimated remaining work."""

        worker = _least_backlog(workers)
        return SchedulingChoice(
            worker_id=worker.worker_id,
            estimated_service_ms=self.mean_service_ms,
            decision_latency_ms=self.decision_latency_ms,
        )


@dataclass(frozen=True, slots=True)
class InputRegressionPolicy:
    """Estimate service time from input size without semantic information.

    The estimate is ``max(minimum_service_ms, intercept_ms +
    slope_ms_per_unit * item.input_units)``. Coefficients should be learned on
    training data rather than the evaluated trace.
    """

    intercept_ms: float
    slope_ms_per_unit: float
    minimum_service_ms: float = 1.0
    decision_latency_ms: float = 0.0
    name: str = field(default="input-regression", init=False)

    def __post_init__(self) -> None:
        """Validate regression coefficients and scheduling overhead."""

        _require_nonnegative(self.intercept_ms, "intercept_ms")
        _require_nonnegative(self.slope_ms_per_unit, "slope_ms_per_unit")
        _require_positive(self.minimum_service_ms, "minimum_service_ms")
        _require_nonnegative(self.decision_latency_ms, "decision_latency_ms")

    def reset(self) -> None:
        """Reset state; this policy is stateless."""

    def estimate_service_ms(self, item: WorkItem) -> float:
        """Return the clipped linear runtime estimate for one request."""

        estimate = self.intercept_ms + self.slope_ms_per_unit * item.input_units
        return max(self.minimum_service_ms, estimate)

    def choose(
        self,
        item: WorkItem,
        workers: tuple[WorkerSnapshot, ...],
        now_ms: float,
    ) -> SchedulingChoice:
        """Choose the smallest estimated backlog and record the size estimate."""

        worker = _least_backlog(workers)
        return SchedulingChoice(
            worker_id=worker.worker_id,
            estimated_service_ms=self.estimate_service_ms(item),
            decision_latency_ms=self.decision_latency_ms,
        )


@dataclass(frozen=True, slots=True)
class OracleWorkPolicy:
    """Upper bound that schedules with true service times.

    This policy is intentionally ineligible as a real scheduler. It exists only
    to show the best result achievable with perfect duration information.
    """

    service_times_ms: Mapping[str, float]
    decision_latency_ms: float = 0.0
    name: str = field(default="oracle-work", init=False)

    def __post_init__(self) -> None:
        """Copy and validate the request-to-service-time lookup."""

        service_times_ms = dict(self.service_times_ms)
        if not service_times_ms:
            raise ValueError("service_times_ms must not be empty")
        if any(not request_id.strip() for request_id in service_times_ms):
            raise ValueError("service time request identifiers must not be empty")
        for service_ms in service_times_ms.values():
            _require_positive(service_ms, "service time")
        _require_nonnegative(self.decision_latency_ms, "decision_latency_ms")
        object.__setattr__(
            self,
            "service_times_ms",
            MappingProxyType(service_times_ms),
        )

    @classmethod
    def from_jobs(
        cls,
        jobs: Iterable[TraceJob],
        *,
        decision_latency_ms: float = 0.0,
    ) -> OracleWorkPolicy:
        """Build the oracle lookup from a trace, rejecting duplicate IDs."""

        trace = tuple(jobs)
        service_times_ms = {job.item.request_id: job.service_ms for job in trace}
        if len(service_times_ms) != len(trace):
            raise ValueError("request identifiers must be unique")
        return cls(
            service_times_ms=service_times_ms,
            decision_latency_ms=decision_latency_ms,
        )

    def reset(self) -> None:
        """Reset state; this policy is stateless."""

    def estimate_service_ms(self, item: WorkItem) -> float:
        """Return true runtime for the oracle comparison request."""

        try:
            return self.service_times_ms[item.request_id]
        except KeyError as error:
            raise ValueError(
                f"oracle has no service time for request: {item.request_id}"
            ) from error

    def choose(
        self,
        item: WorkItem,
        workers: tuple[WorkerSnapshot, ...],
        now_ms: float,
    ) -> SchedulingChoice:
        """Choose the smallest exact backlog using otherwise normal inputs."""

        worker = _least_backlog(workers)
        return SchedulingChoice(
            worker_id=worker.worker_id,
            estimated_service_ms=self.estimate_service_ms(item),
            decision_latency_ms=self.decision_latency_ms,
        )
