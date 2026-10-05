"""Run all v0.1 baseline policies against a small deterministic trace."""

from __future__ import annotations

import json

from jev_balancer import (
    InputRegressionPolicy,
    LeastJobsPolicy,
    MeanWorkPolicy,
    OracleWorkPolicy,
    RoundRobinPolicy,
    TraceJob,
    WorkItem,
    benchmark_policies,
)


def build_trace() -> tuple[TraceJob, ...]:
    """Return a trace with deliberately varied request sizes and runtimes."""

    return (
        TraceJob(WorkItem("long-a", 0, "long request", 10, 120), service_ms=100),
        TraceJob(WorkItem("short-a", 1, "short request", 1, 10), service_ms=2),
        TraceJob(WorkItem("short-b", 2, "short request", 1, 10), service_ms=2),
        TraceJob(WorkItem("long-b", 3, "long request", 8, 100), service_ms=80),
    )


def main() -> None:
    """Compare conventional schedulers and print JSON metrics."""

    jobs = build_trace()
    historical_mean_ms = 46.0
    policies = (
        RoundRobinPolicy(service_estimate_ms=historical_mean_ms),
        LeastJobsPolicy(service_estimate_ms=historical_mean_ms),
        MeanWorkPolicy(mean_service_ms=historical_mean_ms),
        InputRegressionPolicy(intercept_ms=0, slope_ms_per_unit=10),
        OracleWorkPolicy.from_jobs(jobs),
    )
    report = benchmark_policies(jobs, ["worker-a", "worker-b"], policies)
    print(json.dumps(report.to_dict(include_executions=False), indent=2))


if __name__ == "__main__":
    main()
