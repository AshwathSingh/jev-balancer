from __future__ import annotations

import json
from copy import deepcopy
from math import inf, nan

import pytest

from jev_balancer import (
    DurationClass,
    DurationPrediction,
    RoutingDecision,
    ServiceEstimateMethod,
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
        "class_service_ms": {
            DurationClass.SHORT: 50.0,
            DurationClass.MEDIUM: 200.0,
            DurationClass.LONG: 500.0,
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


def test_duration_prediction_copies_and_freezes_class_service_times() -> None:
    class_service_ms = {
        DurationClass.SHORT: 50.0,
        DurationClass.MEDIUM: 200.0,
        DurationClass.LONG: 500.0,
    }

    prediction = make_prediction(class_service_ms=class_service_ms)
    class_service_ms[DurationClass.SHORT] = 1.0

    assert prediction.class_service_ms[DurationClass.SHORT] == 50
    with pytest.raises(TypeError):
        prediction.class_service_ms[DurationClass.SHORT] = 1.0  # type: ignore[index]


def test_duration_prediction_rejects_plain_string_probability_keys() -> None:
    with pytest.raises(ValueError, match="DurationClass"):
        make_prediction(
            probabilities={"short": 0.7, "medium": 0.2, "long": 0.1},
        )


def test_duration_prediction_remains_immutable_after_deepcopy() -> None:
    prediction = deepcopy(make_prediction())

    with pytest.raises(TypeError):
        prediction.probabilities[DurationClass.SHORT] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        prediction.class_service_ms[DurationClass.SHORT] = 1.0  # type: ignore[index]


def test_duration_prediction_supports_json_serialization() -> None:
    serialized = make_prediction().to_dict()

    assert serialized["probabilities"] == {
        "short": 0.7,
        "medium": 0.2,
        "long": 0.1,
    }
    assert serialized["class_service_ms"] == {
        "short": 50.0,
        "medium": 200.0,
        "long": 500.0,
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


def test_duration_prediction_requires_complete_positive_class_service_times() -> None:
    with pytest.raises(ValueError, match="class_service_ms"):
        make_prediction(class_service_ms={DurationClass.SHORT: 1})
    with pytest.raises(ValueError, match="class service time"):
        make_prediction(
            class_service_ms={
                DurationClass.SHORT: 0,
                DurationClass.MEDIUM: 200,
                DurationClass.LONG: 500,
            }
        )


def test_duration_prediction_requires_consistent_expected_service_time() -> None:
    with pytest.raises(ValueError, match="distribution mean"):
        make_prediction(expected_service_ms=124)


def test_duration_prediction_calculates_quantile_and_tail_risk() -> None:
    prediction = make_prediction()

    assert prediction.quantile_service_ms(0.7) == 50
    assert prediction.quantile_service_ms(0.8) == 200
    assert prediction.quantile_service_ms(0.9) == 200
    assert prediction.quantile_service_ms(0.95) == 500
    assert prediction.conditional_value_at_risk_ms(0) == pytest.approx(125)
    assert prediction.conditional_value_at_risk_ms(0.8) == pytest.approx(350)
    assert prediction.conditional_value_at_risk_ms(0.9) == pytest.approx(500)
    assert prediction.normalized_entropy == pytest.approx(0.7298466991620975)


def test_duration_prediction_entropy_is_bounded_for_uniform_distribution() -> None:
    prediction = make_prediction(
        probabilities={duration_class: 1 / 3 for duration_class in DurationClass},
        expected_service_ms=250,
    )

    assert 0 <= prediction.normalized_entropy <= 1
    assert prediction.normalized_entropy == pytest.approx(1)


def test_duration_prediction_transforms_distribution_into_work_estimates() -> None:
    prediction = make_prediction()

    assert prediction.service_estimate_ms(ServiceEstimateMethod.EXPECTED) == 125
    assert (
        prediction.service_estimate_ms(
            ServiceEstimateMethod.QUANTILE,
            risk_quantile=0.9,
        )
        == 200
    )
    assert prediction.service_estimate_ms(
        ServiceEstimateMethod.CVAR,
        risk_quantile=0.9,
    ) == pytest.approx(500)
    assert prediction.service_estimate_ms(
        ServiceEstimateMethod.EXPECTED_CVAR_BLEND,
        risk_quantile=0.8,
        blend_weight=0.25,
    ) == pytest.approx(181.25)
    assert prediction.service_estimate_ms(
        ServiceEstimateMethod.SEMANTIC_FALLBACK_BLEND,
        blend_weight=0.4,
        fallback_service_ms=275,
    ) == pytest.approx(185)


@pytest.mark.parametrize(
    "method, parameters",
    [
        (ServiceEstimateMethod.EXPECTED, {"risk_quantile": 0.9}),
        (ServiceEstimateMethod.QUANTILE, {}),
        (ServiceEstimateMethod.CVAR, {"risk_quantile": 0.9, "blend_weight": 0.5}),
        (
            ServiceEstimateMethod.EXPECTED_CVAR_BLEND,
            {"risk_quantile": 0.9},
        ),
        (
            ServiceEstimateMethod.SEMANTIC_FALLBACK_BLEND,
            {"blend_weight": 0.5},
        ),
    ],
)
def test_duration_prediction_rejects_incomplete_estimate_parameters(
    method: ServiceEstimateMethod,
    parameters: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        make_prediction().service_estimate_ms(method, **parameters)


@pytest.mark.parametrize("quantile", [0.0, -0.1, 1.1, inf, nan])
def test_duration_prediction_rejects_invalid_quantile(quantile: float) -> None:
    with pytest.raises(ValueError, match="quantile"):
        make_prediction().quantile_service_ms(quantile)


@pytest.mark.parametrize("quantile", [-0.1, 1.0, 1.1, inf, nan])
def test_duration_prediction_rejects_invalid_cvar_quantile(
    quantile: float,
) -> None:
    with pytest.raises(ValueError, match="quantile"):
        make_prediction().conditional_value_at_risk_ms(quantile)


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
        estimated_service_ms=125.0,
        estimated_backlog_ms=200.0,
        prediction=make_prediction(),
    )

    assert decision.estimated_finish_ms == 350.0


def test_routing_decision_supports_json_serialization() -> None:
    decision = RoutingDecision(
        request_id="request-1",
        worker_id="worker-2",
        policy="semantic-work",
        decided_at_ms=25.0,
        estimated_service_ms=125.0,
        estimated_backlog_ms=200.0,
        prediction=make_prediction(),
    )

    serialized = decision.to_dict()

    assert serialized["estimated_finish_ms"] == 350.0
    assert serialized["estimate_method"] == "expected"
    json.dumps(serialized)
