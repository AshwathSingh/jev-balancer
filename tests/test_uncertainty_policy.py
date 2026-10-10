from __future__ import annotations

from dataclasses import dataclass

import pytest

from jev_balancer import (
    DurationClass,
    DurationPrediction,
    EntropyFallbackPolicy,
    ServiceEstimateMethod,
    TraceJob,
    UncertaintyAwarePolicy,
    WorkerSnapshot,
    WorkItem,
    simulate,
)


@dataclass(frozen=True)
class FixedPredictor:
    """Return a stable, intentionally uncertain duration distribution."""

    name: str = "fixed-uncertainty-test"

    def predict(self, item: WorkItem) -> DurationPrediction:
        """Return a prediction with a 125 ms mean and 2 ms overhead."""

        return DurationPrediction(
            probabilities={
                DurationClass.SHORT: 0.7,
                DurationClass.MEDIUM: 0.2,
                DurationClass.LONG: 0.1,
            },
            class_service_ms={
                DurationClass.SHORT: 50,
                DurationClass.MEDIUM: 200,
                DurationClass.LONG: 500,
            },
            expected_service_ms=125,
            confidence=0.7,
            predictor=self.name,
            overhead_ms=2,
        )


@dataclass(frozen=True)
class FixedEstimator:
    """Return one conventional estimate for fallback tests."""

    estimate_ms: float = 275

    def estimate_service_ms(self, item: WorkItem) -> float:
        """Return the configured estimate."""

        return self.estimate_ms


def make_item() -> WorkItem:
    """Create one scheduler-visible request."""

    return WorkItem("request", 0, "payload", 10)


def make_workers() -> tuple[WorkerSnapshot, ...]:
    """Create workers that exercise all deterministic tie breakers."""

    return (
        WorkerSnapshot("worker-a", 2, 10),
        WorkerSnapshot("worker-b", 2, 0),
        WorkerSnapshot("worker-c", 1, 0),
    )


@pytest.mark.parametrize(
    "method, risk_quantile, blend_weight, expected_ms",
    [
        (ServiceEstimateMethod.QUANTILE, 0.9, None, 200),
        (ServiceEstimateMethod.CVAR, 0.9, None, 500),
        (ServiceEstimateMethod.EXPECTED_CVAR_BLEND, 0.8, 0.25, 181.25),
    ],
)
def test_uncertainty_policy_records_selected_risk_estimate(
    method: ServiceEstimateMethod,
    risk_quantile: float,
    blend_weight: float | None,
    expected_ms: float,
) -> None:
    policy = UncertaintyAwarePolicy(
        FixedPredictor(),
        estimate_method=method,
        risk_quantile=risk_quantile,
        blend_weight=blend_weight,
        scheduler_overhead_ms=0.5,
    )

    choice = policy.choose(make_item(), make_workers(), 0)

    assert choice.worker_id == "worker-c"
    assert choice.estimated_service_ms == pytest.approx(expected_ms)
    assert choice.decision_latency_ms == 2.5
    assert choice.estimate_method is method
    assert choice.risk_quantile == risk_quantile
    assert choice.blend_weight == blend_weight


def test_entropy_fallback_uses_normalized_entropy_as_blend_weight() -> None:
    policy = EntropyFallbackPolicy(
        FixedPredictor(),
        FixedEstimator(),
        scheduler_overhead_ms=0.5,
    )

    choice = policy.choose(make_item(), make_workers(), 0)
    prediction = FixedPredictor().predict(make_item())
    expected_ms = (
        1 - prediction.normalized_entropy
    ) * prediction.expected_service_ms + prediction.normalized_entropy * 275

    assert choice.worker_id == "worker-c"
    assert choice.estimated_service_ms == pytest.approx(expected_ms)
    assert choice.decision_latency_ms == 2.5
    assert choice.estimate_method is ServiceEstimateMethod.SEMANTIC_FALLBACK_BLEND
    assert choice.blend_weight == pytest.approx(prediction.normalized_entropy)
    assert choice.fallback_service_ms == 275


def test_simulator_preserves_uncertainty_decision_metadata() -> None:
    policy = UncertaintyAwarePolicy(
        FixedPredictor(),
        estimate_method=ServiceEstimateMethod.CVAR,
        risk_quantile=0.9,
        scheduler_overhead_ms=1,
    )
    job = TraceJob(make_item(), service_ms=25, duration_class=DurationClass.SHORT)

    execution = simulate([job], ["worker-a"], policy).executions[0]

    assert execution.decision.estimate_method is ServiceEstimateMethod.CVAR
    assert execution.decision.risk_quantile == 0.9
    assert execution.decision.estimated_service_ms == pytest.approx(500)
    assert execution.decision_latency_ms == 3
    assert execution.response_time_ms == 28


def test_uncertainty_policies_reject_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="estimate_method"):
        UncertaintyAwarePolicy(
            FixedPredictor(),
            estimate_method=ServiceEstimateMethod.EXPECTED,
        )
    with pytest.raises(ValueError, match="risk_quantile"):
        UncertaintyAwarePolicy(FixedPredictor(), risk_quantile=1)
    with pytest.raises(ValueError, match="blend_weight"):
        UncertaintyAwarePolicy(
            FixedPredictor(),
            estimate_method=ServiceEstimateMethod.EXPECTED_CVAR_BLEND,
        )
    with pytest.raises(ValueError, match="only valid"):
        UncertaintyAwarePolicy(FixedPredictor(), blend_weight=0.5)
    with pytest.raises(ValueError, match="scheduler_overhead_ms"):
        EntropyFallbackPolicy(FixedPredictor(), FixedEstimator(), -1)


@pytest.mark.parametrize(
    "policy",
    [
        UncertaintyAwarePolicy(FixedPredictor()),
        EntropyFallbackPolicy(FixedPredictor(), FixedEstimator()),
    ],
)
def test_uncertainty_policies_require_workers(
    policy: UncertaintyAwarePolicy | EntropyFallbackPolicy,
) -> None:
    with pytest.raises(ValueError, match="worker snapshot"):
        policy.choose(make_item(), (), 0)
