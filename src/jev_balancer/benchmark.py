"""Replay one trace across policies and assemble comparison reports."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from jev_balancer.metrics import BenchmarkMetrics, calculate_metrics
from jev_balancer.models import TraceJob
from jev_balancer.scheduler import SchedulingPolicy
from jev_balancer.simulation import SimulationResult, simulate


@dataclass(frozen=True, slots=True)
class PolicyBenchmark:
    """Simulation output and metrics for one scheduling policy."""

    result: SimulationResult
    metrics: BenchmarkMetrics

    def to_dict(self, *, include_executions: bool = True) -> dict[str, object]:
        """Return a JSON-ready policy report.

        Args:
            include_executions: Include per-request traces when true.
        """

        report: dict[str, object] = {"metrics": self.metrics.to_dict()}
        if include_executions:
            report["simulation"] = self.result.to_dict()
        return report


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Comparable results produced from one shared trace and worker pool."""

    request_ids: tuple[str, ...]
    worker_ids: tuple[str, ...]
    policies: tuple[PolicyBenchmark, ...]

    def to_dict(self, *, include_executions: bool = True) -> dict[str, object]:
        """Return a JSON-ready benchmark report."""

        return {
            "request_ids": list(self.request_ids),
            "worker_ids": list(self.worker_ids),
            "policies": {
                policy.metrics.policy_name: policy.to_dict(
                    include_executions=include_executions
                )
                for policy in self.policies
            },
        }


def benchmark_policies(
    jobs: Iterable[TraceJob],
    worker_ids: Iterable[str],
    policies: Iterable[SchedulingPolicy],
) -> BenchmarkReport:
    """Replay identical jobs through every policy and calculate metrics.

    Args:
        jobs: Trace replayed unchanged for each policy.
        worker_ids: Shared worker pool used by each simulation.
        policies: Policies with unique stable names.

    Returns:
        A comparison report in the supplied policy order.

    Raises:
        ValueError: If no policies are provided or names are duplicated.
    """

    trace = tuple(jobs)
    stable_worker_ids = tuple(worker_ids)
    configured_policies = tuple(policies)
    if not configured_policies:
        raise ValueError("at least one policy is required")

    policy_names = [policy.name for policy in configured_policies]
    if len(set(policy_names)) != len(policy_names):
        raise ValueError("policy names must be unique")

    policy_results_list: list[PolicyBenchmark] = []
    for policy in configured_policies:
        result = simulate(trace, stable_worker_ids, policy)
        policy_results_list.append(
            PolicyBenchmark(
                result=result,
                metrics=calculate_metrics(result),
            )
        )
    policy_results = tuple(policy_results_list)
    ordered_request_ids = tuple(
        job.item.request_id
        for job in sorted(
            trace,
            key=lambda job: (job.item.arrival_ms, job.item.request_id),
        )
    )
    report_worker_ids = (
        policy_results[0].result.worker_ids
        if policy_results
        else tuple(sorted(stable_worker_ids))
    )
    return BenchmarkReport(
        request_ids=ordered_request_ids,
        worker_ids=report_worker_ids,
        policies=policy_results,
    )
