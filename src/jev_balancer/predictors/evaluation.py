"""Accuracy and calibration metrics for probabilistic duration predictors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from statistics import fmean
from types import MappingProxyType

from jev_balancer.models import DurationClass, TraceJob
from jev_balancer.predictors.base import DurationPredictor


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    """Aggregate quality metrics calculated on a labeled evaluation trace.

    Attributes:
        predictor_name: Stable predictor identifier.
        evaluated_jobs: Number of held-out predictions scored.
        class_support: Evaluation count for each ground-truth class.
        class_recall: Correct classification rate within each ground-truth
            class, or ``None`` when that class is absent.
        class_accuracy: Fraction whose most likely class matched truth.
        brier_score: Mean multiclass squared probability error; lower is better.
        expected_calibration_error: Confidence/accuracy gap across equal bins.
        mean_absolute_error_ms: Mean absolute expected-duration error.
        mean_overhead_ms: Mean simulated latency charged per prediction.
    """

    predictor_name: str
    evaluated_jobs: int
    class_support: Mapping[DurationClass, int]
    class_recall: Mapping[DurationClass, float | None]
    class_accuracy: float
    brier_score: float
    expected_calibration_error: float
    mean_absolute_error_ms: float
    mean_overhead_ms: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready prediction-quality report."""

        return {
            "predictor": self.predictor_name,
            "evaluated_jobs": self.evaluated_jobs,
            "class_support": {
                duration_class.value: count
                for duration_class, count in self.class_support.items()
            },
            "class_recall": {
                duration_class.value: recall
                for duration_class, recall in self.class_recall.items()
            },
            "class_accuracy": self.class_accuracy,
            "brier_score": self.brier_score,
            "expected_calibration_error": self.expected_calibration_error,
            "mean_absolute_error_ms": self.mean_absolute_error_ms,
            "mean_overhead_ms": self.mean_overhead_ms,
        }


def evaluate_predictions(
    predictor: DurationPredictor,
    jobs: Iterable[TraceJob],
    *,
    calibration_bins: int = 10,
) -> PredictionMetrics:
    """Score predictions after they are produced from visible work items.

    True service times and classes are used only for scoring. They are never
    passed to :meth:`~jev_balancer.predictors.base.DurationPredictor.predict`.

    Args:
        predictor: Fitted provider-neutral duration predictor.
        jobs: Non-empty labeled evaluation trace.
        calibration_bins: Positive count of equal-width confidence bins.

    Returns:
        Accuracy, probability, calibration, error, and overhead metrics.

    Raises:
        ValueError: If the evaluation trace or bin count is invalid.
    """

    if type(calibration_bins) is not int or calibration_bins <= 0:
        raise ValueError("calibration_bins must be a positive integer")
    evaluation_jobs = tuple(jobs)
    if not evaluation_jobs:
        raise ValueError("at least one evaluation job is required")
    if any(job.duration_class is None for job in evaluation_jobs):
        raise ValueError("every evaluation job must have a duration_class")

    correctness: list[float] = []
    brier_scores: list[float] = []
    absolute_errors_ms: list[float] = []
    overheads_ms: list[float] = []
    bin_confidences: list[list[float]] = [[] for _ in range(calibration_bins)]
    bin_correctness: list[list[float]] = [[] for _ in range(calibration_bins)]
    class_support = {duration_class: 0 for duration_class in DurationClass}
    class_correct = {duration_class: 0 for duration_class in DurationClass}

    for job in evaluation_jobs:
        prediction = predictor.predict(job.item)
        predicted_class = max(
            DurationClass,
            key=lambda duration_class: prediction.probabilities[duration_class],
        )
        actual_class = job.duration_class
        assert actual_class is not None
        is_correct = float(predicted_class is actual_class)
        correctness.append(is_correct)
        class_support[actual_class] += 1
        class_correct[actual_class] += int(is_correct)
        brier_scores.append(
            sum(
                (
                    prediction.probabilities[duration_class]
                    - float(duration_class is actual_class)
                )
                ** 2
                for duration_class in DurationClass
            )
        )
        absolute_errors_ms.append(abs(prediction.expected_service_ms - job.service_ms))
        overheads_ms.append(prediction.overhead_ms)
        top_probability = prediction.probabilities[predicted_class]
        bin_index = min(int(top_probability * calibration_bins), calibration_bins - 1)
        bin_confidences[bin_index].append(top_probability)
        bin_correctness[bin_index].append(is_correct)

    evaluated_jobs = len(evaluation_jobs)
    expected_calibration_error = sum(
        len(confidences)
        / evaluated_jobs
        * abs(fmean(confidences) - fmean(bin_correctness[index]))
        for index, confidences in enumerate(bin_confidences)
        if confidences
    )
    return PredictionMetrics(
        predictor_name=predictor.name,
        evaluated_jobs=evaluated_jobs,
        class_support=MappingProxyType(class_support),
        class_recall=MappingProxyType(
            {
                duration_class: (
                    class_correct[duration_class] / class_support[duration_class]
                    if class_support[duration_class]
                    else None
                )
                for duration_class in DurationClass
            }
        ),
        class_accuracy=fmean(correctness),
        brier_score=fmean(brier_scores),
        expected_calibration_error=expected_calibration_error,
        mean_absolute_error_ms=fmean(absolute_errors_ms),
        mean_overhead_ms=fmean(overheads_ms),
    )
