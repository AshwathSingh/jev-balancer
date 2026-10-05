from __future__ import annotations

import json

import pytest

from jev_balancer import (
    RoundRobinPolicy,
    TraceJob,
    WorkItem,
    calculate_metrics,
    simulate,
)


def test_calculate_metrics_reports_latency_sla_and_utilization() -> None:
    jobs = (
        TraceJob(WorkItem("a", 0, "first", 4, sla_ms=3), service_ms=4),
        TraceJob(WorkItem("b", 0, "second", 2, sla_ms=2), service_ms=2),
    )
    result = simulate(
        jobs,
        ["worker-a", "worker-b"],
        RoundRobinPolicy(service_estimate_ms=3),
    )

    metrics = calculate_metrics(result)

    assert metrics.completed_jobs == 2
    assert metrics.observation_ms == 4
    assert metrics.throughput_per_second == 500
    assert metrics.response_time.mean_ms == 3
    assert metrics.response_time.p50_ms == 3
    assert metrics.response_time.p95_ms == pytest.approx(3.9)
    assert metrics.response_time.p99_ms == pytest.approx(3.98)
    assert metrics.sla_evaluated_jobs == 2
    assert metrics.sla_violations == 1
    assert metrics.sla_violation_rate == 0.5
    assert [worker.utilization for worker in metrics.worker_utilization] == [1, 0.5]
    assert metrics.utilization_range == 0.5
    json.dumps(metrics.to_dict())


def test_calculate_metrics_handles_empty_trace() -> None:
    result = simulate(
        [],
        ["worker-a", "worker-b"],
        RoundRobinPolicy(service_estimate_ms=1),
    )

    metrics = calculate_metrics(result)

    assert metrics.completed_jobs == 0
    assert metrics.observation_ms == 0
    assert metrics.throughput_per_second == 0
    assert metrics.response_time.count == 0
    assert metrics.sla_violation_rate is None
    assert [worker.utilization for worker in metrics.worker_utilization] == [0, 0]
