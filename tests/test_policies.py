from __future__ import annotations

from math import inf, nan

import pytest

from jev_balancer import (
    InputRegressionPolicy,
    LeastJobsPolicy,
    MeanWorkPolicy,
    OracleWorkPolicy,
    RoundRobinPolicy,
    TraceJob,
    WorkerSnapshot,
    WorkItem,
)


def make_item(request_id: str = "request-1", input_units: int = 5) -> WorkItem:
    """Create a scheduler-visible request for policy tests."""

    return WorkItem(request_id, 0, "test payload", input_units)


def make_workers() -> tuple[WorkerSnapshot, ...]:
    """Create stable worker snapshots with different count and work signals."""

    return (
        WorkerSnapshot("worker-a", job_count=2, estimated_backlog_ms=5),
        WorkerSnapshot("worker-b", job_count=1, estimated_backlog_ms=10),
        WorkerSnapshot("worker-c", job_count=1, estimated_backlog_ms=0),
    )


def test_round_robin_cycles_and_resets() -> None:
    policy = RoundRobinPolicy(service_estimate_ms=5)
    workers = make_workers()

    first_pass = [policy.choose(make_item(), workers, 0).worker_id for _ in range(4)]
    policy.reset()

    assert first_pass == ["worker-a", "worker-b", "worker-c", "worker-a"]
    assert policy.choose(make_item(), workers, 0).worker_id == "worker-a"


def test_least_jobs_uses_identifier_as_stable_tie_breaker() -> None:
    policy = LeastJobsPolicy(service_estimate_ms=5)

    choice = policy.choose(make_item(), make_workers(), 0)

    assert choice.worker_id == "worker-b"
    assert choice.estimated_service_ms == 5


def test_mean_work_uses_estimated_backlog() -> None:
    policy = MeanWorkPolicy(mean_service_ms=25)

    choice = policy.choose(make_item(), make_workers(), 0)

    assert choice.worker_id == "worker-c"
    assert choice.estimated_service_ms == 25


def test_input_regression_estimates_from_request_size() -> None:
    policy = InputRegressionPolicy(
        intercept_ms=2,
        slope_ms_per_unit=3,
        minimum_service_ms=1,
    )

    choice = policy.choose(make_item(input_units=5), make_workers(), 0)

    assert choice.worker_id == "worker-c"
    assert choice.estimated_service_ms == 17


def test_input_regression_applies_minimum_estimate() -> None:
    policy = InputRegressionPolicy(
        intercept_ms=0,
        slope_ms_per_unit=0,
        minimum_service_ms=2,
    )

    assert policy.estimate_service_ms(make_item(input_units=0)) == 2


def test_oracle_uses_true_trace_service_time() -> None:
    jobs = (
        TraceJob(make_item("a"), service_ms=10),
        TraceJob(make_item("b"), service_ms=25),
    )
    policy = OracleWorkPolicy.from_jobs(jobs)

    assert policy.estimate_service_ms(jobs[1].item) == 25
    assert policy.choose(jobs[0].item, make_workers(), 0).worker_id == "worker-c"


def test_oracle_rejects_unknown_request() -> None:
    policy = OracleWorkPolicy({"known": 10})

    with pytest.raises(ValueError, match="no service time"):
        policy.estimate_service_ms(make_item("unknown"))


def test_oracle_from_jobs_rejects_duplicate_request_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        OracleWorkPolicy.from_jobs(
            [
                TraceJob(make_item("duplicate"), service_ms=10),
                TraceJob(make_item("duplicate"), service_ms=20),
            ]
        )


@pytest.mark.parametrize("service_estimate_ms", [0.0, -1.0, inf, nan])
def test_round_robin_requires_positive_estimate(service_estimate_ms: float) -> None:
    with pytest.raises(ValueError, match="service_estimate_ms"):
        RoundRobinPolicy(service_estimate_ms=service_estimate_ms)


def test_policies_reject_empty_worker_snapshots() -> None:
    with pytest.raises(ValueError, match="worker snapshot"):
        MeanWorkPolicy(mean_service_ms=10).choose(make_item(), (), 0)
