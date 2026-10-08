"""Deterministic probabilistic prediction from request text."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import exp, isfinite, log
from statistics import fmean
from types import MappingProxyType

from jev_balancer.models import (
    DurationClass,
    DurationPrediction,
    TraceJob,
    WorkItem,
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _require_positive(value: float, field_name: str) -> None:
    """Require a finite value strictly greater than zero."""

    if not isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be finite and positive")


def _require_nonnegative(value: float, field_name: str) -> None:
    """Require a finite value greater than or equal to zero."""

    if not isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")


def _tokenize(payload: str) -> tuple[str, ...]:
    """Normalize request text into deterministic lowercase word tokens."""

    return tuple(_TOKEN_PATTERN.findall(payload.lower()))


def _smoothed_probability(
    count: int,
    total: int,
    category_count: int,
    alpha: float,
) -> float:
    """Calculate additive smoothing without overflowing large ``alpha``."""

    scale = max(float(total), alpha)
    scaled_alpha = alpha / scale
    probability = (count / scale + scaled_alpha) / (
        total / scale + scaled_alpha * category_count
    )
    if probability == 0:
        raise ValueError("alpha is too small for stable additive smoothing")
    return probability


@dataclass(frozen=True, slots=True)
class SemanticNaiveBayesPredictor:
    """Multinomial Naive Bayes predictor fitted on labeled historical jobs.

    The predictor learns duration-class priors, word likelihoods, and mean
    service time per class from training jobs. ``semantic_weight`` mixes the
    text-derived posterior with the learned class prior, providing a
    deterministic sweep from no semantic signal (zero) to full signal (one).

    Use :meth:`fit` rather than constructing this class directly.

    Attributes:
        alpha: Positive additive-smoothing strength.
        semantic_weight: Blend from learned prior to text posterior.
        overhead_ms: Simulated prediction latency charged to every request.
    """

    _class_priors: Mapping[DurationClass, float]
    _class_service_means_ms: Mapping[DurationClass, float]
    _token_counts: Mapping[DurationClass, Mapping[str, int]]
    _class_token_totals: Mapping[DurationClass, int]
    _vocabulary: frozenset[str]
    alpha: float
    semantic_weight: float
    overhead_ms: float

    @classmethod
    def fit(
        cls,
        jobs: Iterable[TraceJob],
        *,
        alpha: float = 1.0,
        semantic_weight: float = 1.0,
        overhead_ms: float = 0.0,
    ) -> SemanticNaiveBayesPredictor:
        """Fit a predictor using only a labeled chronological training split.

        Args:
            jobs: Historical jobs whose service times and duration labels are
                allowed for training.
            alpha: Positive additive smoothing applied to priors and tokens.
            semantic_weight: Text-signal strength from zero through one.
            overhead_ms: Simulated latency charged for each prediction.

        Returns:
            An immutable predictor ready to evaluate or schedule requests.

        Raises:
            ValueError: If training data or configuration is invalid.
        """

        _require_positive(alpha, "alpha")
        _require_nonnegative(overhead_ms, "overhead_ms")
        if not isfinite(semantic_weight) or not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight must be from 0 to 1")

        training_jobs = tuple(
            sorted(
                jobs,
                key=lambda job: (job.item.arrival_ms, job.item.request_id),
            )
        )
        if not training_jobs:
            raise ValueError("at least one training job is required")
        request_ids = [job.item.request_id for job in training_jobs]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("training request identifiers must be unique")
        if any(job.duration_class is None for job in training_jobs):
            raise ValueError("every training job must have a duration_class")

        class_counts = {duration_class: 0 for duration_class in DurationClass}
        class_service_times: dict[DurationClass, list[float]] = {
            duration_class: [] for duration_class in DurationClass
        }
        token_counts = {
            duration_class: Counter[str]() for duration_class in DurationClass
        }
        vocabulary: set[str] = set()

        for job in training_jobs:
            duration_class = job.duration_class
            assert duration_class is not None
            tokens = _tokenize(job.item.payload)
            class_counts[duration_class] += 1
            class_service_times[duration_class].append(job.service_ms)
            token_counts[duration_class].update(tokens)
            vocabulary.update(tokens)

        class_count = len(DurationClass)
        class_priors = {
            duration_class: _smoothed_probability(
                class_counts[duration_class],
                len(training_jobs),
                class_count,
                alpha,
            )
            for duration_class in DurationClass
        }
        global_mean_ms = fmean(job.service_ms for job in training_jobs)
        class_service_means_ms = {
            duration_class: (
                fmean(class_service_times[duration_class])
                if class_service_times[duration_class]
                else global_mean_ms
            )
            for duration_class in DurationClass
        }
        immutable_token_counts = {
            duration_class: MappingProxyType(dict(token_counts[duration_class]))
            for duration_class in DurationClass
        }
        vocabulary_size = max(len(vocabulary), 1)
        for duration_class in DurationClass:
            _smoothed_probability(
                0,
                sum(token_counts[duration_class].values()),
                vocabulary_size,
                alpha,
            )

        return cls(
            _class_priors=MappingProxyType(class_priors),
            _class_service_means_ms=MappingProxyType(class_service_means_ms),
            _token_counts=MappingProxyType(immutable_token_counts),
            _class_token_totals=MappingProxyType(
                {
                    duration_class: sum(token_counts[duration_class].values())
                    for duration_class in DurationClass
                }
            ),
            _vocabulary=frozenset(vocabulary),
            alpha=float(alpha),
            semantic_weight=float(semantic_weight),
            overhead_ms=float(overhead_ms),
        )

    @property
    def name(self) -> str:
        """Return an identifier that records the configured signal strength."""

        return f"semantic-naive-bayes-{self.semantic_weight!r}"

    @property
    def class_priors(self) -> Mapping[DurationClass, float]:
        """Return immutable duration-class probabilities learned in training."""

        return self._class_priors

    @property
    def class_service_means_ms(self) -> Mapping[DurationClass, float]:
        """Return immutable representative runtimes learned in training."""

        return self._class_service_means_ms

    def predict(self, item: WorkItem) -> DurationPrediction:
        """Predict a probability distribution from scheduler-visible text."""

        tokens = tuple(
            token for token in _tokenize(item.payload) if token in self._vocabulary
        )
        vocabulary_size = max(len(self._vocabulary), 1)
        log_scores: dict[DurationClass, float] = {}
        for duration_class in DurationClass:
            score = log(self._class_priors[duration_class])
            counts = self._token_counts[duration_class]
            for token in tokens:
                score += log(
                    _smoothed_probability(
                        counts.get(token, 0),
                        self._class_token_totals[duration_class],
                        vocabulary_size,
                        self.alpha,
                    )
                )
            log_scores[duration_class] = score

        maximum_score = max(log_scores.values())
        unnormalized = {
            duration_class: exp(score - maximum_score)
            for duration_class, score in log_scores.items()
        }
        posterior_total = sum(unnormalized.values())
        posterior = {
            duration_class: unnormalized[duration_class] / posterior_total
            for duration_class in DurationClass
        }
        probabilities = {
            duration_class: (
                self.semantic_weight * posterior[duration_class]
                + (1 - self.semantic_weight) * self._class_priors[duration_class]
            )
            for duration_class in DurationClass
        }
        probability_total = sum(probabilities.values())
        probabilities = {
            duration_class: probability / probability_total
            for duration_class, probability in probabilities.items()
        }
        expected_service_ms = sum(
            probabilities[duration_class] * self._class_service_means_ms[duration_class]
            for duration_class in DurationClass
        )
        return DurationPrediction(
            probabilities=probabilities,
            expected_service_ms=expected_service_ms,
            confidence=max(probabilities.values()),
            predictor=self.name,
            overhead_ms=self.overhead_ms,
        )
