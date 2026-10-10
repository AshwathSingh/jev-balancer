"""Policies that turn semantic uncertainty into workload estimates."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from jev_balancer.models import (
    ServiceEstimateMethod,
    WorkerSnapshot,
    WorkItem,
)
from jev_balancer.predictors import DurationPredictor
from jev_balancer.scheduler import SchedulingChoice


class ServiceEstimator(Protocol):
    """Structural interface for a non-semantic service-time estimator."""

    def estimate_service_ms(self, item: WorkItem) -> float:
        """Return a positive service-time estimate for ``item``."""


def _require_policy_settings(name: str, scheduler_overhead_ms: float) -> None:
    """Validate settings shared by uncertainty-aware policies."""

    if not name.strip():
        raise ValueError("name must not be empty")
    if not isfinite(scheduler_overhead_ms) or scheduler_overhead_ms < 0:
        raise ValueError("scheduler_overhead_ms must be finite and non-negative")


def _least_backlog(workers: tuple[WorkerSnapshot, ...]) -> WorkerSnapshot:
    """Choose estimated backlog first, then job count and worker identifier."""

    if not workers:
        raise ValueError("at least one worker snapshot is required")
    return min(
        workers,
        key=lambda worker: (
            worker.estimated_backlog_ms,
            worker.job_count,
            worker.worker_id,
        ),
    )


@dataclass(frozen=True, slots=True)
class UncertaintyAwarePolicy:
    """Route using a quantile, CVaR, or expected/CVaR service estimate.

    The policy changes only the amount of predicted work recorded in each
    worker backlog. It never receives the request's true service time.

    Attributes:
        predictor: Fitted probabilistic duration predictor.
        estimate_method: Quantile, CVaR, or expected/CVaR blend strategy.
        risk_quantile: Quantile or upper-tail boundary used by the strategy.
        blend_weight: CVaR share required by the expected/CVaR blend.
        scheduler_overhead_ms: Non-prediction routing latency per request.
        name: Stable identifier used in benchmark reports.
    """

    predictor: DurationPredictor
    estimate_method: ServiceEstimateMethod = ServiceEstimateMethod.CVAR
    risk_quantile: float = 0.9
    blend_weight: float | None = None
    scheduler_overhead_ms: float = 0.0
    name: str = "uncertainty-aware"

    def __post_init__(self) -> None:
        """Validate the selected risk transformation and policy settings."""

        _require_policy_settings(self.name, self.scheduler_overhead_ms)
        supported_methods = {
            ServiceEstimateMethod.QUANTILE,
            ServiceEstimateMethod.CVAR,
            ServiceEstimateMethod.EXPECTED_CVAR_BLEND,
        }
        if self.estimate_method not in supported_methods:
            raise ValueError("estimate_method must be quantile, CVaR, or their blend")
        if not isfinite(self.risk_quantile):
            raise ValueError("risk_quantile must be finite")
        if self.estimate_method is ServiceEstimateMethod.QUANTILE:
            if not 0 < self.risk_quantile <= 1:
                raise ValueError("quantile risk_quantile must be in (0, 1]")
        elif not 0 <= self.risk_quantile < 1:
            raise ValueError("CVaR risk_quantile must be in [0, 1)")

        if self.estimate_method is ServiceEstimateMethod.EXPECTED_CVAR_BLEND:
            if self.blend_weight is None or not isfinite(self.blend_weight):
                raise ValueError("expected-CVaR blend requires a finite blend_weight")
            if not 0 <= self.blend_weight <= 1:
                raise ValueError("blend_weight must be from 0 to 1")
        elif self.blend_weight is not None:
            raise ValueError("blend_weight is only valid for expected-CVaR blends")

    def reset(self) -> None:
        """Reset state; fitted predictors and this policy are immutable."""

    def choose(
        self,
        item: WorkItem,
        workers: tuple[WorkerSnapshot, ...],
        now_ms: float,
    ) -> SchedulingChoice:
        """Predict duration and route to the least estimated backlog."""

        worker = _least_backlog(workers)
        prediction = self.predictor.predict(item)
        estimate_ms = prediction.service_estimate_ms(
            self.estimate_method,
            risk_quantile=self.risk_quantile,
            blend_weight=self.blend_weight,
        )
        return SchedulingChoice(
            worker_id=worker.worker_id,
            estimated_service_ms=estimate_ms,
            decision_latency_ms=(prediction.overhead_ms + self.scheduler_overhead_ms),
            estimate_method=self.estimate_method,
            risk_quantile=self.risk_quantile,
            blend_weight=self.blend_weight,
            prediction=prediction,
        )


@dataclass(frozen=True, slots=True)
class EntropyFallbackPolicy:
    """Blend semantic and conventional estimates based on uncertainty.

    Normalized predictive entropy is the fallback weight: a concentrated
    distribution trusts the semantic mean, while a diffuse distribution moves
    toward the supplied non-semantic estimator.

    Attributes:
        predictor: Fitted probabilistic duration predictor.
        fallback_estimator: Conventional estimator, such as input regression.
        scheduler_overhead_ms: Total non-prediction routing and fallback cost.
        name: Stable identifier used in benchmark reports.
    """

    predictor: DurationPredictor
    fallback_estimator: ServiceEstimator
    scheduler_overhead_ms: float = 0.0
    name: str = "entropy-fallback"

    def __post_init__(self) -> None:
        """Validate policy identity and modeled non-prediction overhead."""

        _require_policy_settings(self.name, self.scheduler_overhead_ms)

    def reset(self) -> None:
        """Reset state; fitted estimators and this policy are immutable."""

    def choose(
        self,
        item: WorkItem,
        workers: tuple[WorkerSnapshot, ...],
        now_ms: float,
    ) -> SchedulingChoice:
        """Predict duration and blend toward fallback as entropy rises."""

        worker = _least_backlog(workers)
        prediction = self.predictor.predict(item)
        fallback_service_ms = self.fallback_estimator.estimate_service_ms(item)
        blend_weight = prediction.normalized_entropy
        estimate_ms = prediction.service_estimate_ms(
            ServiceEstimateMethod.SEMANTIC_FALLBACK_BLEND,
            blend_weight=blend_weight,
            fallback_service_ms=fallback_service_ms,
        )
        return SchedulingChoice(
            worker_id=worker.worker_id,
            estimated_service_ms=estimate_ms,
            decision_latency_ms=(prediction.overhead_ms + self.scheduler_overhead_ms),
            estimate_method=ServiceEstimateMethod.SEMANTIC_FALLBACK_BLEND,
            blend_weight=blend_weight,
            fallback_service_ms=fallback_service_ms,
            prediction=prediction,
        )
