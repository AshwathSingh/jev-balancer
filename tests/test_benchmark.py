from __future__ import annotations

import json

import pytest

from jev_balancer import (
    LeastJobsPolicy,
    MeanWorkPolicy,
    RoundRobinPolicy,
    TraceJob,
    WorkItem,
    benchmark_policies,
)


def make_jobs() -> tuple[TraceJob, ...]:
    """Create a small shared trace for benchmark tests."""

    return (
        TraceJob(WorkItem("b", 1, "second", 2), service_ms=2),
        TraceJob(WorkItem("a", 0, "first", 5), service_ms=5),
    )


def test_benchmark_replays_same_trace_through_each_policy() -> None:
    report = benchmark_policies(
        make_jobs(),
        ["worker-b", "worker-a"],
        [
            RoundRobinPolicy(service_estimate_ms=3.5),
            LeastJobsPolicy(service_estimate_ms=3.5),
            MeanWorkPolicy(mean_service_ms=3.5),
        ],
    )

    assert report.request_ids == ("a", "b")
    assert report.worker_ids == ("worker-a", "worker-b")
    assert [run.metrics.policy_name for run in report.policies] == [
        "round-robin",
        "least-jobs",
        "mean-work",
    ]
    assert all(run.metrics.completed_jobs == 2 for run in report.policies)
    json.dumps(report.to_dict())
    compact = report.to_dict(include_executions=False)
    assert "simulation" not in compact["policies"]["round-robin"]  # type: ignore[index]


def test_benchmark_requires_at_least_one_policy() -> None:
    with pytest.raises(ValueError, match="at least one policy"):
        benchmark_policies(make_jobs(), ["worker-a"], [])


def test_benchmark_requires_unique_policy_names() -> None:
    with pytest.raises(ValueError, match="policy names"):
        benchmark_policies(
            make_jobs(),
            ["worker-a"],
            [
                RoundRobinPolicy(service_estimate_ms=1),
                RoundRobinPolicy(service_estimate_ms=2),
            ],
        )
