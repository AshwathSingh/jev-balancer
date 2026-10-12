"""Measure human-written prompts locally and benchmark scheduling policies."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from math import ceil
from pathlib import Path
from statistics import fmean
from urllib.parse import urlencode
from urllib.request import urlopen

from jev_balancer import (
    DurationClassThresholds,
    EntropyFallbackPolicy,
    InputRegressionPolicy,
    LeastJobsPolicy,
    MeanWorkPolicy,
    OllamaClient,
    OracleWorkPolicy,
    PromptRecord,
    RoundRobinPolicy,
    SchedulingPolicy,
    SemanticNaiveBayesPredictor,
    SemanticWorkPolicy,
    ServiceEstimateMethod,
    TraceJob,
    UncertaintyAwarePolicy,
    benchmark_policies,
    chronological_split,
    collect_measured_trace,
    evaluate_predictions,
    extract_oasst_prompts,
)

_OASST_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
_OASST_DATASET_URL = "https://huggingface.co/datasets/OpenAssistant/oasst1"


def _fetch_oasst_prompts(limit: int) -> tuple[PromptRecord, ...]:
    """Fetch enough validation rows to obtain ``limit`` reviewed root prompts."""

    prompts: tuple[PromptRecord, ...] = ()
    offset = 0
    page_size = 100
    while len(prompts) < limit:
        query = urlencode(
            {
                "dataset": "OpenAssistant/oasst1",
                "config": "default",
                "split": "validation",
                "offset": offset,
                "length": page_size,
            }
        )
        with urlopen(f"{_OASST_ROWS_ENDPOINT}?{query}", timeout=30) as response:
            page = json.load(response)
        entries = page.get("rows")
        if not isinstance(entries, list) or not entries:
            raise RuntimeError("OASST1 validation rows ended before prompt limit")
        rows = []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("row"), dict):
                raise RuntimeError("OASST1 row response has an unexpected schema")
            rows.append(entry["row"])
        prompts = (*prompts, *extract_oasst_prompts(rows))
        offset += len(entries)
    return prompts[:limit]


def _percentile(values: tuple[float, ...], quantile: float) -> float:
    """Return a nearest-rank percentile for non-empty measured values."""

    ordered = tuple(sorted(values))
    return ordered[max(ceil(len(ordered) * quantile) - 1, 0)]


def _replay_evaluation(
    jobs: tuple[TraceJob, ...],
    *,
    interarrival_ms: float,
    sla_ms: float,
) -> tuple[TraceJob, ...]:
    """Apply controlled offered load without changing measured service time."""

    return tuple(
        replace(
            job,
            item=replace(
                job.item,
                arrival_ms=index * interarrival_ms,
                sla_ms=sla_ms,
            ),
        )
        for index, job in enumerate(jobs)
    )


def _build_policies(
    training: tuple[TraceJob, ...],
    evaluation: tuple[TraceJob, ...],
) -> tuple[SemanticNaiveBayesPredictor, tuple[SchedulingPolicy, ...]]:
    """Fit the same conventional, semantic, risk, and oracle comparisons."""

    mean_policy = MeanWorkPolicy.from_jobs(training)
    regression_policy = InputRegressionPolicy.from_jobs(training)
    predictor = SemanticNaiveBayesPredictor.fit(training)
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


def parse_args() -> argparse.Namespace:
    """Parse local measurement and replay options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama2")
    parser.add_argument("--prompts", type=int, default=80)
    parser.add_argument("--num-predict", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--target-utilization", type=float, default=0.85)
    parser.add_argument("--training-fraction", type=float, default=0.7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-executions", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Collect a real-prompt trace and emit a machine-readable benchmark."""

    args = parse_args()
    if args.prompts < 10:
        raise ValueError("prompts must be at least 10")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.target_utilization <= 0:
        raise ValueError("target-utilization must be positive")

    prompts = _fetch_oasst_prompts(args.prompts)
    client = OllamaClient(model=args.model, num_predict=args.num_predict)
    measured = collect_measured_trace(prompts, client)
    raw_split = chronological_split(
        measured.jobs,
        training_fraction=args.training_fraction,
    )
    thresholds = DurationClassThresholds.fit(raw_split.training)
    training = thresholds.label(raw_split.training)
    labeled_evaluation = thresholds.label(raw_split.evaluation)

    mean_training_service_ms = fmean(job.service_ms for job in training)
    interarrival_ms = mean_training_service_ms / (
        args.workers * args.target_utilization
    )
    sla_ms = 2 * _percentile(
        tuple(job.service_ms for job in training),
        0.95,
    )
    evaluation = _replay_evaluation(
        labeled_evaluation,
        interarrival_ms=interarrival_ms,
        sla_ms=sla_ms,
    )
    predictor, policies = _build_policies(training, evaluation)
    benchmark = benchmark_policies(
        evaluation,
        tuple(f"worker-{index + 1}" for index in range(args.workers)),
        policies,
    )
    measurement_values = measured.measurements
    report = {
        "schema_version": 1,
        "source": {
            "name": "OpenAssistant OASST1",
            "url": _OASST_DATASET_URL,
            "license": "Apache-2.0",
            "split": "validation",
            "selection": "reviewed English root prompter messages in source order",
        },
        "measurement": {
            "runtime": "Ollama",
            "model": args.model,
            "num_predict": args.num_predict,
            "temperature": 0.0,
            "seed": 0,
            "service_time_definition": "prompt_eval_duration + eval_duration",
            "cold_load_excluded": True,
            "measured_prompts": len(measured.jobs),
            "mean_service_ms": fmean(
                measurement.service_ms for measurement in measurement_values
            ),
            "p95_service_ms": _percentile(
                tuple(measurement.service_ms for measurement in measurement_values),
                0.95,
            ),
            "mean_prompt_tokens": fmean(
                measurement.prompt_tokens for measurement in measurement_values
            ),
            "mean_output_tokens": fmean(
                measurement.output_tokens for measurement in measurement_values
            ),
        },
        "split": {
            "training_jobs": len(training),
            "evaluation_jobs": len(evaluation),
            "training_fraction": args.training_fraction,
            "duration_thresholds": thresholds.to_dict(),
        },
        "replay": {
            "workers": args.workers,
            "target_utilization": args.target_utilization,
            "interarrival_ms": interarrival_ms,
            "sla_ms": sla_ms,
            "arrival_times_are_controlled": True,
        },
        "prediction_quality": evaluate_predictions(predictor, evaluation).to_dict(),
        "scheduling": benchmark.to_dict(include_executions=args.include_executions),
    }
    serialized = json.dumps(report, indent=2)
    if args.output is None:
        print(serialized)
    else:
        args.output.write_text(f"{serialized}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
