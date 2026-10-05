from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from jev_balancer import (
    SchedulingChoice,
    SimulationResult,
    TraceJob,
    WorkerSnapshot,
    WorkItem,
    simulate,
)


def make_job(
    request_id: str,
    arrival_ms: float,
    service_ms: float,
    *,
    input_units: int = 10,
    sla_ms: float | None = None,
) -> TraceJob:
    """Create a compact trace job for simulator tests."""

    return TraceJob(
        item=WorkItem(
            request_id=request_id,
            arrival_ms=arrival_ms,
            payload=f"payload for {request_id}",
            input_units=input_units,
            sla_ms=sla_ms,
        ),
        service_ms=service_ms,
    )


@dataclass
class FirstWorkerPolicy:
    """Test policy that routes every job to the first worker."""

    estimated_service_ms: float = 10.0
    decision_latency_ms: float = 0.0
    name: str = "first-worker"

    def reset(self) -> None:
        """Reset state; this policy is stateless."""

    def choose(
        self,
        item: WorkItem,
        workers: tuple[WorkerSnapshot, ...],
        now_ms: float,
    ) -> SchedulingChoice:
        """Choose the first worker without inspecting simulation truth."""

        assert not hasattr(item, "service_ms")
        return SchedulingChoice(
            worker_id=workers[0].worker_id,
            estimated_service_ms=self.estimated_service_ms,
            decision_latency_ms=self.decision_latency_ms,
        )


@dataclass
class RoundRobinPolicy:
    """Stateful test policy used to verify reset and stable input ordering."""

    name: str = "round-robin-test"
    index: int = 0

    def reset(self) -> None:
        """Restart the worker cycle before every simulation."""

        self.index = 0

    def choose(
        self,
        item: WorkItem,
        workers: tuple[WorkerSnapshot, ...],
        now_ms: float,
    ) -> SchedulingChoice:
        """Choose the next worker in stable snapshot order."""

        worker = workers[self.index % len(workers)]
        self.index += 1
        return SchedulingChoice(
            worker_id=worker.worker_id,
            estimated_service_ms=max(float(item.input_units), 1.0),
        )


@dataclass
class LeastBacklogPolicy:
    """Test policy that records snapshots and chooses minimum estimated work."""

    name: str = "least-backlog-test"
    seen: list[tuple[WorkerSnapshot, ...]] = field(default_factory=list)

    def reset(self) -> None:
        """Clear observations so repeated simulations are independent."""

        self.seen.clear()

    def choose(
        self,
        item: WorkItem,
        workers: tuple[WorkerSnapshot, ...],
        now_ms: float,
    ) -> SchedulingChoice:
        """Choose estimated backlog first and worker identifier second."""

        self.seen.append(workers)
        worker = min(
            workers,
            key=lambda snapshot: (
                snapshot.estimated_backlog_ms,
                snapshot.worker_id,
            ),
        )
        return SchedulingChoice(
            worker_id=worker.worker_id,
            estimated_service_ms=max(float(item.input_units), 1.0),
        )


def test_simulate_runs_fifo_without_overlapping_jobs() -> None:
    result = simulate(
        [make_job("a", 0, 5), make_job("b", 1, 2)],
        ["worker-a"],
        FirstWorkerPolicy(estimated_service_ms=5),
    )

    first, second = result.executions
    assert (first.start_ms, first.finish_ms) == (0, 5)
    assert (second.start_ms, second.finish_ms) == (5, 7)
    assert second.start_ms >= first.finish_ms
    assert second.queue_wait_ms == 4
    assert second.response_time_ms == 6


def test_simulate_is_deterministic_and_resets_stateful_policy() -> None:
    jobs = [make_job("b", 0, 3), make_job("a", 0, 5), make_job("c", 1, 2)]
    policy = RoundRobinPolicy()

    first = simulate(jobs, ["worker-b", "worker-a"], policy)
    second = simulate(reversed(jobs), ["worker-a", "worker-b"], policy)

    assert first.to_dict() == second.to_dict()
    assert [execution.job.item.request_id for execution in first.executions] == [
        "a",
        "b",
        "c",
    ]
    assert [execution.decision.worker_id for execution in first.executions] == [
        "worker-a",
        "worker-b",
        "worker-a",
    ]


def test_estimated_backlog_uses_predictions_not_true_service_time() -> None:
    policy = LeastBacklogPolicy()
    result = simulate(
        [
            make_job("long", 0, 100, input_units=10),
            make_job("next", 1, 1, input_units=1),
        ],
        ["worker-a", "worker-b"],
        policy,
    )

    second_snapshots = policy.seen[1]
    assert second_snapshots[0] == WorkerSnapshot(
        worker_id="worker-a",
        job_count=1,
        estimated_backlog_ms=9,
    )
    assert result.executions[1].decision.worker_id == "worker-b"


def test_decision_latency_blocks_dispatcher_and_counts_toward_response() -> None:
    result = simulate(
        [make_job("a", 0, 1), make_job("b", 1, 1, sla_ms=15)],
        ["worker-a"],
        FirstWorkerPolicy(estimated_service_ms=1, decision_latency_ms=10),
    )

    first, second = result.executions
    assert (first.routing_started_ms, first.start_ms, first.finish_ms) == (0, 10, 11)
    assert (second.routing_started_ms, second.start_ms, second.finish_ms) == (
        10,
        20,
        21,
    )
    assert second.routing_wait_ms == 9
    assert second.decision_latency_ms == 10
    assert second.response_time_ms == 20
    assert second.sla_violated is True


def test_simulation_result_is_json_ready() -> None:
    result = simulate(
        [make_job("a", 0, 5)],
        ["worker-a"],
        FirstWorkerPolicy(),
    )

    assert isinstance(result, SimulationResult)
    assert result.completed_jobs == 1
    assert result.makespan_ms == 5
    json.dumps(result.to_dict())


def test_empty_trace_returns_an_empty_result() -> None:
    result = simulate([], ["worker-a"], FirstWorkerPolicy())

    assert result.completed_jobs == 0
    assert result.makespan_ms == 0
    assert result.executions == ()


@pytest.mark.parametrize(
    "worker_ids, message",
    [
        ([], "at least one"),
        (["worker-a", "worker-a"], "unique"),
        ([""], "must not be empty"),
    ],
)
def test_simulate_rejects_invalid_worker_identifiers(
    worker_ids: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        simulate([], worker_ids, FirstWorkerPolicy())


def test_simulate_rejects_duplicate_request_identifiers() -> None:
    with pytest.raises(ValueError, match="request identifiers"):
        simulate(
            [make_job("duplicate", 0, 1), make_job("duplicate", 1, 1)],
            ["worker-a"],
            FirstWorkerPolicy(),
        )


@dataclass
class UnknownWorkerPolicy(FirstWorkerPolicy):
    """Test policy that violates the worker-selection contract."""

    def choose(
        self,
        item: WorkItem,
        workers: tuple[WorkerSnapshot, ...],
        now_ms: float,
    ) -> SchedulingChoice:
        """Return a worker absent from the supplied snapshots."""

        return SchedulingChoice(worker_id="missing", estimated_service_ms=1)


def test_simulate_rejects_unknown_policy_worker() -> None:
    with pytest.raises(ValueError, match="unknown worker"):
        simulate(
            [make_job("a", 0, 1)],
            ["worker-a"],
            UnknownWorkerPolicy(),
        )
