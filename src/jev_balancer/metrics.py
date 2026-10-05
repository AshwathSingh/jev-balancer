"""Deterministic metrics calculated from completed simulation results."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil, floor
from statistics import fmean

from jev_balancer.simulation import SimulationResult


def _percentile(sorted_values: tuple[float, ...], quantile: float) -> float:
    """Calculate a linearly interpolated percentile from sorted values."""

    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * quantile
    lower_index = floor(position)
    upper_index = ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return lower + (upper - lower) * weight


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Count and distribution summary for one latency dimension, in ms."""

    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    @classmethod
    def from_values(cls, values: Iterable[float]) -> LatencySummary:
        """Summarize values using deterministic linear interpolation."""

        ordered = tuple(sorted(values))
        if not ordered:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return cls(
            count=len(ordered),
            mean_ms=fmean(ordered),
            p50_ms=_percentile(ordered, 0.50),
            p95_ms=_percentile(ordered, 0.95),
            p99_ms=_percentile(ordered, 0.99),
            max_ms=ordered[-1],
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready latency summary."""

        return {
            "count": self.count,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "max_ms": self.max_ms,
        }


@dataclass(frozen=True, slots=True)
class WorkerUtilization:
    """Observed busy time and utilization for one worker."""

    worker_id: str
    busy_ms: float
    utilization: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready worker utilization record."""

        return {
            "worker_id": self.worker_id,
            "busy_ms": self.busy_ms,
            "utilization": self.utilization,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    """Metrics used to compare one policy with another."""

    policy_name: str
    completed_jobs: int
    observation_ms: float
    makespan_ms: float
    throughput_per_second: float
    response_time: LatencySummary
    queue_wait: LatencySummary
    routing_wait: LatencySummary
    decision_latency: LatencySummary
    sla_evaluated_jobs: int
    sla_violations: int
    sla_violation_rate: float | None
    worker_utilization: tuple[WorkerUtilization, ...]
    utilization_range: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready metrics report."""

        return {
            "policy": self.policy_name,
            "completed_jobs": self.completed_jobs,
            "observation_ms": self.observation_ms,
            "makespan_ms": self.makespan_ms,
            "throughput_per_second": self.throughput_per_second,
            "response_time": self.response_time.to_dict(),
            "queue_wait": self.queue_wait.to_dict(),
            "routing_wait": self.routing_wait.to_dict(),
            "decision_latency": self.decision_latency.to_dict(),
            "sla_evaluated_jobs": self.sla_evaluated_jobs,
            "sla_violations": self.sla_violations,
            "sla_violation_rate": self.sla_violation_rate,
            "worker_utilization": [
                utilization.to_dict() for utilization in self.worker_utilization
            ],
            "utilization_range": self.utilization_range,
        }


def calculate_metrics(result: SimulationResult) -> BenchmarkMetrics:
    """Calculate comparable performance metrics for one simulation result.

    Throughput and utilization use the interval from the earliest request
    arrival through the final completion. SLA rate includes only requests that
    define ``sla_ms``.

    Args:
        result: Completed deterministic simulation output.

    Returns:
        Immutable latency, throughput, SLA, and utilization metrics.
    """

    executions = result.executions
    if executions:
        observation_start_ms = min(
            execution.job.item.arrival_ms for execution in executions
        )
        observation_ms = result.makespan_ms - observation_start_ms
    else:
        observation_ms = 0.0

    throughput_per_second = (
        result.completed_jobs * 1_000 / observation_ms if observation_ms > 0 else 0.0
    )

    sla_results = [
        execution.sla_violated
        for execution in executions
        if execution.sla_violated is not None
    ]
    sla_violations = sum(sla_results)
    sla_violation_rate = sla_violations / len(sla_results) if sla_results else None

    busy_by_worker = {worker_id: 0.0 for worker_id in result.worker_ids}
    for execution in executions:
        busy_by_worker[execution.decision.worker_id] += execution.job.service_ms
    worker_utilization = tuple(
        WorkerUtilization(
            worker_id=worker_id,
            busy_ms=busy_ms,
            utilization=busy_ms / observation_ms if observation_ms > 0 else 0.0,
        )
        for worker_id, busy_ms in busy_by_worker.items()
    )
    utilization_values = [worker.utilization for worker in worker_utilization]
    utilization_range = (
        max(utilization_values) - min(utilization_values) if utilization_values else 0.0
    )

    return BenchmarkMetrics(
        policy_name=result.policy_name,
        completed_jobs=result.completed_jobs,
        observation_ms=observation_ms,
        makespan_ms=result.makespan_ms,
        throughput_per_second=throughput_per_second,
        response_time=LatencySummary.from_values(
            execution.response_time_ms for execution in executions
        ),
        queue_wait=LatencySummary.from_values(
            execution.queue_wait_ms for execution in executions
        ),
        routing_wait=LatencySummary.from_values(
            execution.routing_wait_ms for execution in executions
        ),
        decision_latency=LatencySummary.from_values(
            execution.decision_latency_ms for execution in executions
        ),
        sla_evaluated_jobs=len(sla_results),
        sla_violations=sla_violations,
        sla_violation_rate=sla_violation_rate,
        worker_utilization=worker_utilization,
        utilization_range=utilization_range,
    )
