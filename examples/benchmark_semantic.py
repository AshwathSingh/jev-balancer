"""Compare probabilistic semantic scheduling with conventional baselines."""

from __future__ import annotations

import json

from jev_balancer import (
    ArrivalPattern,
    InputRegressionPolicy,
    LeastJobsPolicy,
    MeanWorkPolicy,
    OracleWorkPolicy,
    RoundRobinPolicy,
    SchedulingPolicy,
    SemanticNaiveBayesPredictor,
    SemanticWorkPolicy,
    ServiceDistribution,
    SyntheticWorkloadConfig,
    benchmark_policies,
    chronological_split,
    evaluate_predictions,
    generate_workload,
)


def main() -> None:
    """Fit on early jobs and benchmark policies on the later holdout."""

    jobs = generate_workload(
        SyntheticWorkloadConfig(
            job_count=150,
            seed=11,
            arrival_pattern=ArrivalPattern.BURSTY,
            service_distribution=ServiceDistribution.HEAVY_TAILED,
            interarrival_ms=2,
            burst_size=10,
            burst_gap_ms=50,
            min_service_ms=5,
            max_service_ms=120,
            sla_multiplier=6,
        )
    )
    split = chronological_split(jobs, training_fraction=0.7)
    mean_policy = MeanWorkPolicy.from_jobs(split.training)
    regression_policy = InputRegressionPolicy.from_jobs(split.training)
    predictors = tuple(
        SemanticNaiveBayesPredictor.fit(
            split.training,
            semantic_weight=weight,
            overhead_ms=0.25,
        )
        for weight in (0.0, 0.5, 1.0)
    )
    semantic_policies = tuple(
        SemanticWorkPolicy(
            predictor,
            name=f"semantic-work-{predictor.semantic_weight:.2f}",
        )
        for predictor in predictors
    )
    policies: tuple[SchedulingPolicy, ...] = (
        RoundRobinPolicy(service_estimate_ms=mean_policy.mean_service_ms),
        LeastJobsPolicy(service_estimate_ms=mean_policy.mean_service_ms),
        mean_policy,
        regression_policy,
        *semantic_policies,
        OracleWorkPolicy.from_jobs(split.evaluation),
    )
    benchmark = benchmark_policies(
        split.evaluation,
        ("worker-a", "worker-b", "worker-c", "worker-d"),
        policies,
    )
    report = {
        "training_jobs": len(split.training),
        "evaluation_jobs": len(split.evaluation),
        "prediction_quality": {
            predictor.name: evaluate_predictions(
                predictor,
                split.evaluation,
            ).to_dict()
            for predictor in predictors
        },
        "scheduling": benchmark.to_dict(include_executions=False),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
