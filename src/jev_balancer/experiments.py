"""Reproducible multi-seed experiments for uncertainty-aware scheduling."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from math import isfinite
from statistics import fmean

from jev_balancer.benchmark import BenchmarkReport, benchmark_policies
from jev_balancer.metrics import BenchmarkMetrics
from jev_balancer.models import ServiceEstimateMethod, TraceJob
from jev_balancer.policies import (
    EntropyFallbackPolicy,
    InputRegressionPolicy,
    LeastJobsPolicy,
    MeanWorkPolicy,
    OracleWorkPolicy,
    RoundRobinPolicy,
    SemanticWorkPolicy,
    UncertaintyAwarePolicy,
)
from jev_balancer.predictors import (
    PredictionMetrics,
    SemanticNaiveBayesPredictor,
    evaluate_predictions,
)
from jev_balancer.scheduler import SchedulingPolicy
from jev_balancer.workloads import (
    ArrivalPattern,
    ServiceDistribution,
    SyntheticWorkloadConfig,
    chronological_split,
    generate_workload,
)


@dataclass(frozen=True, slots=True)
class ExperimentRegime:
    """One controlled workload and predictor-overhead configuration.

    Attributes:
        name: Unique human-readable regime identifier.
        arrival_pattern: Steady or bursty arrivals.
        service_distribution: Uniform or heavy-tailed service times.
        interarrival_ms: Steady spacing or within-burst spacing.
        predictor_overhead_ms: Simulated semantic prediction latency.
        training_jobs: Historical labeled jobs used only for fitting.
        evaluation_jobs: Later jobs used only for evaluation.
        worker_count: Number of identical FIFO workers.
        burst_size: Jobs per burst when arrivals are bursty.
        burst_gap_ms: Time between burst starts.
        min_service_ms: Minimum generated service time.
        max_service_ms: Maximum generated service time.
        pareto_shape: Heavy-tail shape parameter.
        sla_ms: Optional response-time objective.
        semantic_weight: Strength of the synthetic text signal.
    """

    name: str
    arrival_pattern: ArrivalPattern
    service_distribution: ServiceDistribution
    interarrival_ms: float
    predictor_overhead_ms: float
    training_jobs: int = 400
    evaluation_jobs: int = 300
    worker_count: int = 4
    burst_size: int = 10
    burst_gap_ms: float = 150.0
    min_service_ms: float = 5.0
    max_service_ms: float = 120.0
    pareto_shape: float = 0.6
    sla_ms: float | None = 150.0
    semantic_weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate regime identity, counts, and workload parameters."""

        if not self.name.strip():
            raise ValueError("name must not be empty")
        if type(self.training_jobs) is not int or self.training_jobs <= 0:
            raise ValueError("training_jobs must be a positive integer")
        if type(self.evaluation_jobs) is not int or self.evaluation_jobs <= 0:
            raise ValueError("evaluation_jobs must be a positive integer")
        if type(self.worker_count) is not int or self.worker_count <= 0:
            raise ValueError("worker_count must be a positive integer")
        if not isfinite(self.predictor_overhead_ms) or self.predictor_overhead_ms < 0:
            raise ValueError("predictor_overhead_ms must be finite and non-negative")
        if not isfinite(self.semantic_weight) or not 0 <= self.semantic_weight <= 1:
            raise ValueError("semantic_weight must be from 0 to 1")
        self.workload_config(job_count=self.training_jobs, seed=0)

    def workload_config(
        self,
        *,
        job_count: int,
        seed: int,
        start_ms: float = 0.0,
    ) -> SyntheticWorkloadConfig:
        """Build a validated generator configuration for this regime."""

        return SyntheticWorkloadConfig(
            job_count=job_count,
            seed=seed,
            arrival_pattern=self.arrival_pattern,
            service_distribution=self.service_distribution,
            start_ms=start_ms,
            interarrival_ms=self.interarrival_ms,
            burst_size=self.burst_size,
            burst_gap_ms=self.burst_gap_ms,
            min_service_ms=self.min_service_ms,
            max_service_ms=self.max_service_ms,
            pareto_shape=self.pareto_shape,
            sla_ms=self.sla_ms,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready description of the controlled regime."""

        return {
            "name": self.name,
            "arrival_pattern": self.arrival_pattern.value,
            "service_distribution": self.service_distribution.value,
            "interarrival_ms": self.interarrival_ms,
            "predictor_overhead_ms": self.predictor_overhead_ms,
            "training_jobs": self.training_jobs,
            "evaluation_jobs": self.evaluation_jobs,
            "worker_count": self.worker_count,
            "burst_size": self.burst_size,
            "burst_gap_ms": self.burst_gap_ms,
            "min_service_ms": self.min_service_ms,
            "max_service_ms": self.max_service_ms,
            "pareto_shape": self.pareto_shape,
            "sla_ms": self.sla_ms,
            "semantic_weight": self.semantic_weight,
        }


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """Prediction and scheduling results for one regime seed."""

    seed: int
    training_job_count: int
    evaluation_job_count: int
    prediction_metrics: PredictionMetrics
    benchmark: BenchmarkReport

    def to_dict(self) -> dict[str, object]:
        """Return a compact JSON-ready run report without job traces."""

        return {
            "seed": self.seed,
            "training_jobs": self.training_job_count,
            "evaluation_jobs": self.evaluation_job_count,
            "prediction_quality": self.prediction_metrics.to_dict(),
            "scheduling": self.benchmark.to_dict(include_executions=False),
        }


@dataclass(frozen=True, slots=True)
class AggregatePolicyMetrics:
    """Mean scheduling metrics across independent deterministic seeds."""

    policy_name: str
    runs: int
    response_p95_ms: float
    queue_wait_p95_ms: float
    sla_violation_rate: float | None
    throughput_per_second: float
    decision_latency_mean_ms: float

    @classmethod
    def from_metrics(
        cls,
        metrics: Iterable[BenchmarkMetrics],
    ) -> AggregatePolicyMetrics:
        """Average one policy's metrics across a non-empty run sequence."""

        values = tuple(metrics)
        if not values:
            raise ValueError("at least one metric set is required")
        policy_name = values[0].policy_name
        if any(value.policy_name != policy_name for value in values):
            raise ValueError("all metric sets must belong to one policy")
        sla_rates = tuple(
            value.sla_violation_rate
            for value in values
            if value.sla_violation_rate is not None
        )
        return cls(
            policy_name=policy_name,
            runs=len(values),
            response_p95_ms=fmean(value.response_time.p95_ms for value in values),
            queue_wait_p95_ms=fmean(value.queue_wait.p95_ms for value in values),
            sla_violation_rate=fmean(sla_rates) if sla_rates else None,
            throughput_per_second=fmean(
                value.throughput_per_second for value in values
            ),
            decision_latency_mean_ms=fmean(
                value.decision_latency.mean_ms for value in values
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready aggregate."""

        return {
            "policy": self.policy_name,
            "runs": self.runs,
            "response_p95_ms": self.response_p95_ms,
            "queue_wait_p95_ms": self.queue_wait_p95_ms,
            "sla_violation_rate": self.sla_violation_rate,
            "throughput_per_second": self.throughput_per_second,
            "decision_latency_mean_ms": self.decision_latency_mean_ms,
        }


@dataclass(frozen=True, slots=True)
class RegimeExperimentResult:
    """All seeded runs and aggregates for one experimental regime."""

    regime: ExperimentRegime
    runs: tuple[ExperimentRun, ...]
    policy_aggregates: tuple[AggregatePolicyMetrics, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready regime report."""

        return {
            "regime": self.regime.to_dict(),
            "runs": [run.to_dict() for run in self.runs],
            "policy_aggregates": {
                aggregate.policy_name: aggregate.to_dict()
                for aggregate in self.policy_aggregates
            },
        }


@dataclass(frozen=True, slots=True)
class UncertaintyExperimentReport:
    """Complete deterministic report across regimes and random seeds."""

    seeds: tuple[int, ...]
    regimes: tuple[RegimeExperimentResult, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready multi-regime report."""

        return {
            "seeds": list(self.seeds),
            "regimes": {
                result.regime.name: result.to_dict() for result in self.regimes
            },
        }


def _rename_jobs(jobs: Iterable[TraceJob], prefix: str) -> tuple[TraceJob, ...]:
    """Make generated identifiers unique across training and evaluation phases."""

    return tuple(
        replace(
            job,
            item=replace(
                job.item,
                request_id=f"{prefix}-{job.item.request_id}",
            ),
        )
        for job in jobs
    )


def _shift_arrivals_to_zero(jobs: Iterable[TraceJob]) -> tuple[TraceJob, ...]:
    """Preserve evaluation spacing while using a local simulation clock."""

    trace = tuple(jobs)
    start_ms = min(job.item.arrival_ms for job in trace)
    return tuple(
        replace(job, item=replace(job.item, arrival_ms=job.item.arrival_ms - start_ms))
        for job in trace
    )


def _generate_split(
    regime: ExperimentRegime,
    seed: int,
) -> tuple[tuple[TraceJob, ...], tuple[TraceJob, ...]]:
    """Generate label-safe chronological training and evaluation phases."""

    training = _rename_jobs(
        generate_workload(
            regime.workload_config(job_count=regime.training_jobs, seed=seed)
        ),
        "train",
    )
    evaluation_start_ms = (
        max(job.item.arrival_ms + job.service_ms for job in training) + 1
    )
    evaluation = _rename_jobs(
        generate_workload(
            regime.workload_config(
                job_count=regime.evaluation_jobs,
                seed=seed + 1_000_003,
                start_ms=evaluation_start_ms,
            )
        ),
        "evaluation",
    )
    combined = (*training, *evaluation)
    split = chronological_split(
        combined,
        training_fraction=(len(training) + 0.5) / len(combined),
        require_observed_labels=True,
    )
    return split.training, _shift_arrivals_to_zero(split.evaluation)


def _policies_for_run(
    training: tuple[TraceJob, ...],
    evaluation: tuple[TraceJob, ...],
    regime: ExperimentRegime,
) -> tuple[SemanticNaiveBayesPredictor, tuple[SchedulingPolicy, ...]]:
    """Fit one predictor and construct all comparable policies."""

    mean_policy = MeanWorkPolicy.from_jobs(training)
    regression_policy = InputRegressionPolicy.from_jobs(training)
    predictor = SemanticNaiveBayesPredictor.fit(
        training,
        semantic_weight=regime.semantic_weight,
        overhead_ms=regime.predictor_overhead_ms,
    )
    policies: tuple[SchedulingPolicy, ...] = (
        RoundRobinPolicy(service_estimate_ms=mean_policy.mean_service_ms),
        LeastJobsPolicy(service_estimate_ms=mean_policy.mean_service_ms),
        mean_policy,
        regression_policy,
        SemanticWorkPolicy(predictor, name="semantic-expected"),
        UncertaintyAwarePolicy(
            predictor,
            estimate_method=ServiceEstimateMethod.QUANTILE,
            risk_quantile=0.9,
            name="semantic-quantile-90",
        ),
        UncertaintyAwarePolicy(
            predictor,
            estimate_method=ServiceEstimateMethod.CVAR,
            risk_quantile=0.9,
            name="semantic-cvar-90",
        ),
        UncertaintyAwarePolicy(
            predictor,
            estimate_method=ServiceEstimateMethod.EXPECTED_CVAR_BLEND,
            risk_quantile=0.9,
            blend_weight=0.5,
            name="semantic-expected-cvar-50",
        ),
        EntropyFallbackPolicy(
            predictor,
            regression_policy,
            name="semantic-entropy-fallback",
        ),
        OracleWorkPolicy.from_jobs(evaluation),
    )
    return predictor, policies


def _aggregate_runs(
    runs: tuple[ExperimentRun, ...],
) -> tuple[AggregatePolicyMetrics, ...]:
    """Group matching policy metrics across seeded runs."""

    policy_names = tuple(
        policy.metrics.policy_name for policy in runs[0].benchmark.policies
    )
    return tuple(
        AggregatePolicyMetrics.from_metrics(
            next(
                policy.metrics
                for policy in run.benchmark.policies
                if policy.metrics.policy_name == policy_name
            )
            for run in runs
        )
        for policy_name in policy_names
    )


def run_uncertainty_experiment(
    regimes: Iterable[ExperimentRegime],
    seeds: Iterable[int],
) -> UncertaintyExperimentReport:
    """Run all policies on independent seeded traces for every regime.

    Training always precedes evaluation and all training labels are observable
    before the holdout begins. Every policy receives the same evaluation trace
    within a run, and prediction overhead is included in response time.

    Args:
        regimes: Unique workload and overhead configurations.
        seeds: Unique integer seeds reused across all regimes.

    Returns:
        Per-run prediction/scheduling results and per-policy regime means.

    Raises:
        ValueError: If regimes or seeds are empty, duplicated, or invalid.
    """

    configured_regimes = tuple(regimes)
    configured_seeds = tuple(seeds)
    if not configured_regimes:
        raise ValueError("at least one experiment regime is required")
    if len({regime.name for regime in configured_regimes}) != len(configured_regimes):
        raise ValueError("experiment regime names must be unique")
    if not configured_seeds:
        raise ValueError("at least one seed is required")
    if any(type(seed) is not int for seed in configured_seeds):
        raise ValueError("seeds must be integers")
    if len(set(configured_seeds)) != len(configured_seeds):
        raise ValueError("seeds must be unique")

    regime_results: list[RegimeExperimentResult] = []
    for regime in configured_regimes:
        run_results: list[ExperimentRun] = []
        for seed in configured_seeds:
            training, evaluation = _generate_split(regime, seed)
            predictor, policies = _policies_for_run(training, evaluation, regime)
            run_results.append(
                ExperimentRun(
                    seed=seed,
                    training_job_count=len(training),
                    evaluation_job_count=len(evaluation),
                    prediction_metrics=evaluate_predictions(predictor, evaluation),
                    benchmark=benchmark_policies(
                        evaluation,
                        tuple(
                            f"worker-{index + 1}"
                            for index in range(regime.worker_count)
                        ),
                        policies,
                    ),
                )
            )
        stable_runs = tuple(run_results)
        regime_results.append(
            RegimeExperimentResult(
                regime=regime,
                runs=stable_runs,
                policy_aggregates=_aggregate_runs(stable_runs),
            )
        )
    return UncertaintyExperimentReport(
        seeds=configured_seeds,
        regimes=tuple(regime_results),
    )
