from __future__ import annotations

from dataclasses import dataclass

import pytest

from jev_balancer import (
    DurationClass,
    DurationPrediction,
    SemanticWorkPolicy,
    TraceJob,
    WorkerSnapshot,
    WorkItem,
    simulate,
)


@dataclass(frozen=True)
class FixedPredictor:
    """Return one auditable prediction for policy integration tests."""

    name: str = "fixed-test"

    def predict(self, item: WorkItem) -> DurationPrediction:
        """Return a fixed expected duration with two milliseconds overhead."""

        return DurationPrediction(
            probabilities={
                DurationClass.SHORT: 0.5,
                DurationClass.MEDIUM: 0.3,
                DurationClass.LONG: 0.2,
            },
            expected_service_ms=12,
            confidence=0.5,
            predictor=self.name,
            overhead_ms=2,
        )


def test_semantic_policy_uses_prediction_and_least_backlog() -> None:
    policy = SemanticWorkPolicy(FixedPredictor(), scheduler_overhead_ms=0.5)
    workers = (
        WorkerSnapshot("worker-a", 1, 10),
        WorkerSnapshot("worker-b", 2, 0),
        WorkerSnapshot("worker-c", 1, 0),
    )

    choice = policy.choose(WorkItem("request", 0, "payload", 1), workers, 0)

    assert choice.worker_id == "worker-c"
    assert choice.estimated_service_ms == 12
    assert choice.decision_latency_ms == 2.5
    assert choice.prediction is not None
    assert choice.prediction.predictor == "fixed-test"


def test_semantic_prediction_overhead_counts_toward_response_time() -> None:
    policy = SemanticWorkPolicy(FixedPredictor(), scheduler_overhead_ms=1)
    job = TraceJob(
        WorkItem("request", 0, "payload", 1),
        service_ms=5,
        duration_class=DurationClass.SHORT,
    )

    execution = simulate([job], ["worker-a"], policy).executions[0]

    assert execution.decision_latency_ms == 3
    assert execution.response_time_ms == 8
    assert execution.decision.prediction is not None


def test_semantic_policy_rejects_invalid_configuration_and_workers() -> None:
    with pytest.raises(ValueError, match="name"):
        SemanticWorkPolicy(FixedPredictor(), name=" ")
    with pytest.raises(ValueError, match="scheduler_overhead_ms"):
        SemanticWorkPolicy(FixedPredictor(), scheduler_overhead_ms=-1)
    with pytest.raises(ValueError, match="worker snapshot"):
        SemanticWorkPolicy(FixedPredictor()).choose(
            WorkItem("request", 0, "payload", 1),
            (),
            0,
        )
