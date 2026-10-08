from __future__ import annotations

import json
from dataclasses import dataclass
from math import inf, nan

import pytest

from jev_balancer import (
    DurationClass,
    DurationPrediction,
    SemanticNaiveBayesPredictor,
    TraceJob,
    WorkItem,
    evaluate_predictions,
)


def make_job(
    request_id: str,
    arrival_ms: float,
    payload: str,
    service_ms: float,
    duration_class: DurationClass | None,
) -> TraceJob:
    """Create one labeled training or evaluation record."""

    return TraceJob(
        item=WorkItem(request_id, arrival_ms, payload, input_units=1),
        service_ms=service_ms,
        duration_class=duration_class,
    )


def training_jobs() -> tuple[TraceJob, ...]:
    """Return a balanced trace with distinct semantic vocabulary."""

    return (
        make_job("short", 0, "classify compact status", 5, DurationClass.SHORT),
        make_job("medium", 1, "summarize customer case", 30, DurationClass.MEDIUM),
        make_job("long", 2, "analyze production incident", 90, DurationClass.LONG),
    )


def test_semantic_predictor_learns_text_probabilities_and_class_means() -> None:
    predictor = SemanticNaiveBayesPredictor.fit(
        training_jobs(),
        overhead_ms=0.25,
    )

    prediction = predictor.predict(
        WorkItem("held-out", 10, "analyze another production incident", 2)
    )

    assert max(prediction.probabilities, key=prediction.probabilities.__getitem__) is (
        DurationClass.LONG
    )
    assert predictor.class_service_means_ms == {
        DurationClass.SHORT: 5,
        DurationClass.MEDIUM: 30,
        DurationClass.LONG: 90,
    }
    assert prediction.predictor == "semantic-naive-bayes-1.0"
    assert prediction.overhead_ms == 0.25
    assert prediction.expected_service_ms == pytest.approx(
        sum(
            probability * predictor.class_service_means_ms[duration_class]
            for duration_class, probability in prediction.probabilities.items()
        )
    )


def test_zero_semantic_weight_returns_learned_priors() -> None:
    predictor = SemanticNaiveBayesPredictor.fit(
        training_jobs(),
        semantic_weight=0,
    )

    prediction = predictor.predict(
        WorkItem("held-out", 10, "analyze production incident", 2)
    )

    assert prediction.probabilities == predictor.class_priors
    assert all(
        probability == pytest.approx(1 / 3)
        for probability in prediction.probabilities.values()
    )


def test_semantic_predictor_fit_is_independent_of_input_order() -> None:
    jobs = training_jobs()
    first = SemanticNaiveBayesPredictor.fit(jobs)
    second = SemanticNaiveBayesPredictor.fit(reversed(jobs))
    item = WorkItem("held-out", 10, "summarize customer case", 2)

    assert first.predict(item) == second.predict(item)


def test_missing_training_class_uses_global_service_mean() -> None:
    predictor = SemanticNaiveBayesPredictor.fit(training_jobs()[:2])

    assert predictor.class_service_means_ms[DurationClass.LONG] == 17.5


@pytest.mark.parametrize("alpha", [0.0, -1.0, inf, nan])
def test_semantic_predictor_requires_positive_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        SemanticNaiveBayesPredictor.fit(training_jobs(), alpha=alpha)


def test_semantic_predictor_handles_large_finite_alpha() -> None:
    predictor = SemanticNaiveBayesPredictor.fit(training_jobs(), alpha=1e308)

    prediction = predictor.predict(
        WorkItem("held-out", 10, "analyze production incident", 2)
    )

    assert sum(prediction.probabilities.values()) == pytest.approx(1)
    assert prediction.expected_service_ms > 0


def test_semantic_predictor_rejects_underflowing_alpha() -> None:
    with pytest.raises(ValueError, match="alpha is too small"):
        SemanticNaiveBayesPredictor.fit(training_jobs(), alpha=5e-324)


def test_predictor_names_preserve_distinct_signal_weights() -> None:
    first = SemanticNaiveBayesPredictor.fit(
        training_jobs(),
        semantic_weight=0.001,
    )
    second = SemanticNaiveBayesPredictor.fit(
        training_jobs(),
        semantic_weight=0.004,
    )

    assert first.name != second.name


@pytest.mark.parametrize("semantic_weight", [-0.1, 1.1, inf, nan])
def test_semantic_predictor_requires_bounded_weight(
    semantic_weight: float,
) -> None:
    with pytest.raises(ValueError, match="semantic_weight"):
        SemanticNaiveBayesPredictor.fit(
            training_jobs(),
            semantic_weight=semantic_weight,
        )


def test_semantic_predictor_rejects_empty_unlabeled_and_duplicate_training() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SemanticNaiveBayesPredictor.fit([])
    with pytest.raises(ValueError, match="duration_class"):
        SemanticNaiveBayesPredictor.fit([make_job("unlabeled", 0, "payload", 1, None)])
    with pytest.raises(ValueError, match="unique"):
        SemanticNaiveBayesPredictor.fit([training_jobs()[0], training_jobs()[0]])


@dataclass(frozen=True)
class PerfectPredictor:
    """Test predictor with exact probabilities and runtimes encoded by text."""

    name: str = "perfect-test"

    def predict(self, item: WorkItem) -> DurationPrediction:
        """Return the exact class named by the test payload."""

        duration_class = DurationClass(item.payload)
        service_times = {
            DurationClass.SHORT: 5.0,
            DurationClass.MEDIUM: 30.0,
            DurationClass.LONG: 90.0,
        }
        return DurationPrediction(
            probabilities={
                candidate: float(candidate is duration_class)
                for candidate in DurationClass
            },
            expected_service_ms=service_times[duration_class],
            confidence=0.0,
            predictor=self.name,
            overhead_ms=0.25,
        )


def test_evaluate_predictions_scores_probability_and_runtime_quality() -> None:
    jobs = tuple(
        make_job(
            duration_class.value,
            index,
            duration_class.value,
            service_ms,
            duration_class,
        )
        for index, (duration_class, service_ms) in enumerate(
            (
                (DurationClass.SHORT, 5.0),
                (DurationClass.MEDIUM, 30.0),
                (DurationClass.LONG, 90.0),
            )
        )
    )

    metrics = evaluate_predictions(PerfectPredictor(), jobs)

    assert metrics.evaluated_jobs == 3
    assert metrics.class_support == {
        DurationClass.SHORT: 1,
        DurationClass.MEDIUM: 1,
        DurationClass.LONG: 1,
    }
    assert all(recall == 1 for recall in metrics.class_recall.values())
    assert metrics.class_accuracy == 1
    assert metrics.brier_score == 0
    assert metrics.expected_calibration_error == 0
    assert metrics.mean_absolute_error_ms == 0
    assert metrics.mean_overhead_ms == 0.25
    json.dumps(metrics.to_dict())


def test_evaluate_predictions_rejects_invalid_inputs() -> None:
    predictor = PerfectPredictor()
    with pytest.raises(ValueError, match="at least one"):
        evaluate_predictions(predictor, [])
    with pytest.raises(ValueError, match="duration_class"):
        evaluate_predictions(
            predictor,
            [make_job("unlabeled", 0, "short", 5, None)],
        )
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_predictions(predictor, training_jobs(), calibration_bins=0)
