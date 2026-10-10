from __future__ import annotations

from math import inf, nan

import pytest

from jev_balancer import DurationClass, DurationPrediction, SchedulingChoice


def make_prediction(
    *,
    expected_service_ms: float = 100,
    overhead_ms: float = 5,
) -> DurationPrediction:
    """Create a valid semantic prediction for choice validation tests."""

    return DurationPrediction(
        probabilities={
            DurationClass.SHORT: 0.7,
            DurationClass.MEDIUM: 0.2,
            DurationClass.LONG: 0.1,
        },
        class_service_ms={
            DurationClass.SHORT: 50,
            DurationClass.MEDIUM: 200,
            DurationClass.LONG: 250,
        },
        expected_service_ms=expected_service_ms,
        confidence=0.8,
        predictor="test",
        overhead_ms=overhead_ms,
    )


@pytest.mark.parametrize("worker_id", ["", "   "])
def test_scheduling_choice_requires_worker_identifier(worker_id: str) -> None:
    with pytest.raises(ValueError, match="worker_id"):
        SchedulingChoice(worker_id=worker_id, estimated_service_ms=1)


@pytest.mark.parametrize("estimated_service_ms", [0.0, -1.0, inf, nan])
def test_scheduling_choice_requires_positive_service_estimate(
    estimated_service_ms: float,
) -> None:
    with pytest.raises(ValueError, match="estimated_service_ms"):
        SchedulingChoice(
            worker_id="worker-a",
            estimated_service_ms=estimated_service_ms,
        )


@pytest.mark.parametrize("decision_latency_ms", [-1.0, inf, nan])
def test_scheduling_choice_requires_valid_decision_latency(
    decision_latency_ms: float,
) -> None:
    with pytest.raises(ValueError, match="decision_latency_ms"):
        SchedulingChoice(
            worker_id="worker-a",
            estimated_service_ms=1,
            decision_latency_ms=decision_latency_ms,
        )


def test_scheduling_choice_requires_prediction_service_estimate() -> None:
    with pytest.raises(ValueError, match="must match"):
        SchedulingChoice(
            worker_id="worker-a",
            estimated_service_ms=99,
            decision_latency_ms=5,
            prediction=make_prediction(expected_service_ms=100),
        )


def test_scheduling_choice_includes_prediction_overhead() -> None:
    with pytest.raises(ValueError, match="prediction overhead"):
        SchedulingChoice(
            worker_id="worker-a",
            estimated_service_ms=100,
            decision_latency_ms=4,
            prediction=make_prediction(overhead_ms=5),
        )
