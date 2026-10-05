from __future__ import annotations

import json
from dataclasses import asdict
from math import inf, nan

import pytest

from jev_balancer import (
    DurationClass,
    DurationPrediction,
    RoutingDecision,
    TraceJob,
    WorkerSnapshot,
    WorkItem,
)


def make_item(**overrides: object) -> WorkItem:
    values: dict[str, object] = {
        "request_id": "request-1",
        "arrival_ms": 10.0,
        "payload": "Summarize this request",
        "input_units": 12,
        "sla_ms": 1_000.0,
    }
    values.update(overrides)
    return WorkItem(**values)  # type: ignore[arg-type]


def make_prediction(**overrides: object) -> DurationPrediction:
    values: dict[str, object] = {
        "probabilities": {
            DurationClass.SHORT: 0.7,
            DurationClass.MEDIUM: 0.2,
            DurationClass.LONG: 0.1,
        },
        "expected_service_ms": 125.0,
        "confidence": 0.8,
        "predictor": "test",
        "overhead_ms": 5.0,
    }
    values.update(overrides)
    return DurationPrediction(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("request_id", ["", "   "])
def test_work_item_requires_an_identifier(request_id: str) -> None:
    with pytest.raises(ValueError, match="request_id"):
        make_item(request_id=request_id)


@pytest.mark.parametrize("arrival_ms", [-1.0, inf, nan])
def test_work_item_requires_valid_arrival(arrival_ms: float) -> None:
    with pytest.raises(ValueError, match="arrival_ms"):
        make_item(arrival_ms=arrival_ms)


@pytest.mark.parametrize("input_units", [-1, 1.5, True])
def test_work_item_requires_a_nonnegative_integer_input_count(
    input_units: object,
) -> None:
    with pytest.raises(ValueError, match="input_units"):
        make_item(input_units=input_units)


@pytest.mark.parametrize("sla_ms", [0.0, -1.0, inf, nan])
def test_work_item_requires_a_positive_sla(sla_ms: float) -> None:
    with pytest.raises(ValueError, match="sla_ms"):
        make_item(sla_ms=sla_ms)


def test_trace_truth_is_separate_from_scheduler_visible_item() -> None:
    item = make_item()
    trace_job = TraceJob(item=item, service_ms=250.0, duration_class=DurationClass.LONG)

    assert not hasattr(item, "service_ms")
    assert trace_job.service_ms == 250.0


@pytest.mark.parametrize("service_ms", [0.0, -1.0, inf, nan])
def test_trace_job_requires_positive_service_time(service_ms: float) -> None:
    with pytest.raises(ValueError, match="service_ms"):
        TraceJob(item=make_item(), service_ms=service_ms)


@pytest.mark.parametrize("job_count", [-1, 1.5, True])
def test_worker_snapshot_requires_a_nonnegative_integer_job_count(
    job_count: object,
) -> None:
    with pytest.raises(ValueError, match="job_count"):
        WorkerSnapshot(
            worker_id="worker-1",
            job_count=job_count,  # type: ignore[arg-type]
            estimated_backlog_ms=0,
        )


@pytest.mark.parametrize("estimated_backlog_ms", [-1.0, inf, nan])
def test_worker_snapshot_requires_valid_backlog(
    estimated_backlog_ms: float,
) -> None:
    with pytest.raises(ValueError, match="estimated_backlog_ms"):
        WorkerSnapshot(
            worker_id="worker-1",
            job_count=0,
            estimated_backlog_ms=estimated_backlog_ms,
        )


def test_duration_prediction_copies_and_freezes_probabilities() -> None:
    probabilities = {
        DurationClass.SHORT: 0.7,
        DurationClass.MEDIUM: 0.2,
        DurationClass.LONG: 0.1,
    }

    prediction = make_prediction(probabilities=probabilities)
    probabilities[DurationClass.SHORT] = 0.0

    assert prediction.probabilities[DurationClass.SHORT] == 0.7
    with pytest.raises(TypeError):
        prediction.probabilities[DurationClass.SHORT] = 0.0  # type: ignore[index]


def test_duration_prediction_rejects_plain_string_probability_keys() -> None:
    with pytest.raises(ValueError, match="DurationClass"):
        make_prediction(
            probabilities={"short": 0.7, "medium": 0.2, "long": 0.1},
        )


def test_duration_prediction_supports_dataclass_and_json_serialization() -> None:
    serialized = asdict(make_prediction())

    assert serialized["probabilities"] == {
        DurationClass.SHORT: 0.7,
        DurationClass.MEDIUM: 0.2,
        DurationClass.LONG: 0.1,
    }
    json.dumps(serialized)


@pytest.mark.parametrize(
    "probabilities",
    [
        {DurationClass.SHORT: 1.0},
        {
            DurationClass.SHORT: 0.8,
            DurationClass.MEDIUM: 0.3,
            DurationClass.LONG: -0.1,
        },
        {
            DurationClass.SHORT: 0.8,
            DurationClass.MEDIUM: 0.3,
            DurationClass.LONG: 0.1,
        },
    ],
)
def test_duration_prediction_rejects_invalid_distributions(
    probabilities: dict[DurationClass, float],
) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        make_prediction(probabilities=probabilities)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, inf, nan])
def test_duration_prediction_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        make_prediction(confidence=confidence)


@pytest.mark.parametrize("overhead_ms", [-1.0, inf, nan])
def test_duration_prediction_rejects_invalid_overhead(overhead_ms: float) -> None:
    with pytest.raises(ValueError, match="overhead_ms"):
        make_prediction(overhead_ms=overhead_ms)


def test_routing_decision_exposes_estimated_finish_time() -> None:
    decision = RoutingDecision(
        request_id="request-1",
        worker_id="worker-2",
        policy="semantic-work",
        decided_at_ms=25.0,
        estimated_service_ms=100.0,
        estimated_backlog_ms=200.0,
        prediction=make_prediction(),
    )

    assert decision.estimated_finish_ms == 325.0
