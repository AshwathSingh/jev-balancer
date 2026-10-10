"""Deterministic event simulation for FIFO workers and a single dispatcher."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isclose, isfinite

from jev_balancer.models import RoutingDecision, TraceJob, WorkerSnapshot
from jev_balancer.scheduler import SchedulingPolicy


@dataclass(frozen=True, slots=True)
class JobExecution:
    """Observed lifecycle of one completed simulated job.

    Attributes:
        job: Input trace record, including hidden true service time.
        decision: Auditable routing decision produced from the policy choice.
        routing_started_ms: Time the dispatcher began routing the request.
        start_ms: Time the selected worker began executing the request.
        finish_ms: Time execution completed.
    """

    job: TraceJob
    decision: RoutingDecision
    routing_started_ms: float
    start_ms: float
    finish_ms: float

    def __post_init__(self) -> None:
        """Validate lifecycle ordering and decision identity."""

        item = self.job.item
        if item.request_id != self.decision.request_id:
            raise ValueError("job and decision request identifiers must match")
        if not (
            item.arrival_ms
            <= self.routing_started_ms
            <= self.decision.decided_at_ms
            <= self.start_ms
            <= self.finish_ms
        ):
            raise ValueError("job lifecycle timestamps are out of order")
        observed_service_ms = self.finish_ms - self.start_ms
        if not isclose(
            observed_service_ms,
            self.job.service_ms,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("execution duration must match trace service_ms")

    @property
    def routing_wait_ms(self) -> float:
        """Return time waiting for the dispatcher after arrival."""

        return self.routing_started_ms - self.job.item.arrival_ms

    @property
    def decision_latency_ms(self) -> float:
        """Return time spent executing the scheduling policy."""

        return self.decision.decided_at_ms - self.routing_started_ms

    @property
    def queue_wait_ms(self) -> float:
        """Return time waiting for the selected worker after routing."""

        return self.start_ms - self.decision.decided_at_ms

    @property
    def response_time_ms(self) -> float:
        """Return total time from request arrival through completion."""

        return self.finish_ms - self.job.item.arrival_ms

    @property
    def sla_violated(self) -> bool | None:
        """Return SLA status, or ``None`` when the request has no SLA."""

        sla_ms = self.job.item.sla_ms
        return None if sla_ms is None else self.response_time_ms > sla_ms

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready record for benchmark traces."""

        return {
            "request_id": self.job.item.request_id,
            "worker_id": self.decision.worker_id,
            "arrival_ms": self.job.item.arrival_ms,
            "routing_started_ms": self.routing_started_ms,
            "decided_at_ms": self.decision.decided_at_ms,
            "start_ms": self.start_ms,
            "finish_ms": self.finish_ms,
            "service_ms": self.job.service_ms,
            "routing_wait_ms": self.routing_wait_ms,
            "decision_latency_ms": self.decision_latency_ms,
            "queue_wait_ms": self.queue_wait_ms,
            "response_time_ms": self.response_time_ms,
            "sla_ms": self.job.item.sla_ms,
            "sla_violated": self.sla_violated,
            "decision": self.decision.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Immutable output of one deterministic simulation run."""

    policy_name: str
    worker_ids: tuple[str, ...]
    executions: tuple[JobExecution, ...]

    @property
    def completed_jobs(self) -> int:
        """Return the number of completed jobs."""

        return len(self.executions)

    @property
    def makespan_ms(self) -> float:
        """Return the latest completion time, or zero for an empty trace."""

        return max((execution.finish_ms for execution in self.executions), default=0.0)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation of the simulation run."""

        return {
            "policy": self.policy_name,
            "worker_ids": list(self.worker_ids),
            "completed_jobs": self.completed_jobs,
            "makespan_ms": self.makespan_ms,
            "executions": [execution.to_dict() for execution in self.executions],
        }


def _snapshot_worker(
    worker_id: str,
    executions: list[JobExecution],
    now_ms: float,
) -> WorkerSnapshot:
    """Build observable load without exposing true remaining service time."""

    active = [execution for execution in executions if execution.finish_ms > now_ms]
    estimated_backlog_ms = 0.0
    for execution in active:
        estimated_service_ms = execution.decision.estimated_service_ms
        if execution.start_ms <= now_ms:
            elapsed_ms = now_ms - execution.start_ms
            estimated_backlog_ms += max(0.0, estimated_service_ms - elapsed_ms)
        else:
            estimated_backlog_ms += estimated_service_ms

    return WorkerSnapshot(
        worker_id=worker_id,
        job_count=len(active),
        estimated_backlog_ms=estimated_backlog_ms,
    )


def _validate_inputs(
    jobs: tuple[TraceJob, ...],
    worker_ids: tuple[str, ...],
    policy_name: str,
) -> None:
    """Validate identifiers before invoking mutable policy code."""

    if not worker_ids:
        raise ValueError("at least one worker_id is required")
    if any(not worker_id.strip() for worker_id in worker_ids):
        raise ValueError("worker_ids must not be empty")
    if len(set(worker_ids)) != len(worker_ids):
        raise ValueError("worker_ids must be unique")
    if not policy_name.strip():
        raise ValueError("policy name must not be empty")

    request_ids = [job.item.request_id for job in jobs]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("request identifiers must be unique")


def simulate(
    jobs: Iterable[TraceJob],
    worker_ids: Iterable[str],
    policy: SchedulingPolicy,
) -> SimulationResult:
    """Replay a workload through a scheduling policy deterministically.

    Jobs are ordered by ``(arrival_ms, request_id)``. A single dispatcher makes
    one decision at a time, so a policy's ``decision_latency_ms`` delays later
    routing decisions. Each worker executes its assigned jobs non-preemptively
    in FIFO order.

    The policy receives only :class:`~jev_balancer.models.WorkItem` and
    :class:`~jev_balancer.models.WorkerSnapshot` values. Actual ``service_ms``
    remains confined to the simulation engine.

    Args:
        jobs: Trace jobs to replay. Input order does not affect the result.
        worker_ids: Unique worker identifiers. They are sorted for stable ties.
        policy: Policy implementing reset and worker selection.

    Returns:
        Immutable execution records ordered by arrival time and request ID.

    Raises:
        ValueError: If identifiers are invalid or the policy selects an unknown
            worker.
    """

    trace = tuple(jobs)
    stable_worker_ids = tuple(sorted(worker_ids))
    policy_name = policy.name
    _validate_inputs(trace, stable_worker_ids, policy_name)
    policy.reset()

    worker_assignments: dict[str, list[JobExecution]] = {
        worker_id: [] for worker_id in stable_worker_ids
    }
    executions: list[JobExecution] = []
    dispatcher_available_ms = 0.0

    ordered_trace = sorted(
        trace,
        key=lambda trace_job: (
            trace_job.item.arrival_ms,
            trace_job.item.request_id,
        ),
    )
    for job in ordered_trace:
        routing_started_ms = max(job.item.arrival_ms, dispatcher_available_ms)
        snapshots = tuple(
            _snapshot_worker(
                worker_id,
                worker_assignments[worker_id],
                routing_started_ms,
            )
            for worker_id in stable_worker_ids
        )
        choice = policy.choose(job.item, snapshots, routing_started_ms)
        if choice.worker_id not in worker_assignments:
            raise ValueError(f"policy selected unknown worker: {choice.worker_id}")

        decided_at_ms = routing_started_ms + choice.decision_latency_ms
        dispatcher_available_ms = decided_at_ms
        selected_snapshot = next(
            snapshot for snapshot in snapshots if snapshot.worker_id == choice.worker_id
        )
        decision = RoutingDecision(
            request_id=job.item.request_id,
            worker_id=choice.worker_id,
            policy=policy_name,
            decided_at_ms=decided_at_ms,
            estimated_service_ms=choice.estimated_service_ms,
            estimated_backlog_ms=selected_snapshot.estimated_backlog_ms,
            estimate_method=choice.estimate_method,
            risk_quantile=choice.risk_quantile,
            blend_weight=choice.blend_weight,
            fallback_service_ms=choice.fallback_service_ms,
            prediction=choice.prediction,
        )

        assignments = worker_assignments[choice.worker_id]
        worker_available_ms = assignments[-1].finish_ms if assignments else 0.0
        start_ms = max(decided_at_ms, worker_available_ms)
        finish_ms = start_ms + job.service_ms
        if not isfinite(finish_ms):
            raise ValueError("simulated finish time must be finite")

        execution = JobExecution(
            job=job,
            decision=decision,
            routing_started_ms=routing_started_ms,
            start_ms=start_ms,
            finish_ms=finish_ms,
        )
        assignments.append(execution)
        executions.append(execution)

    return SimulationResult(
        policy_name=policy_name,
        worker_ids=stable_worker_ids,
        executions=tuple(executions),
    )
