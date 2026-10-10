"""Interfaces shared by scheduling policies and the simulator."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Protocol

from jev_balancer.models import (
    DurationPrediction,
    ServiceEstimateMethod,
    WorkerSnapshot,
    WorkItem,
)


@dataclass(frozen=True, slots=True)
class SchedulingChoice:
    """A policy's proposed worker assignment.

    The simulator converts this proposal into a complete
    :class:`~jev_balancer.models.RoutingDecision`. Keeping request identity,
    policy identity, and timestamps out of the proposal prevents policies from
    creating inconsistent audit records.

    Attributes:
        worker_id: Identifier of the worker selected from the supplied snapshot.
        estimated_service_ms: Positive estimated runtime for the new job.
        decision_latency_ms: Non-negative time spent making this choice.
        estimate_method: Transformation used for the service estimate.
        risk_quantile: Optional quantile or tail boundary.
        blend_weight: Optional tail or fallback blend share.
        fallback_service_ms: Optional non-semantic fallback estimate.
        prediction: Optional semantic prediction supporting the estimate.
    """

    worker_id: str
    estimated_service_ms: float
    decision_latency_ms: float = 0.0
    estimate_method: ServiceEstimateMethod = ServiceEstimateMethod.EXPECTED
    risk_quantile: float | None = None
    blend_weight: float | None = None
    fallback_service_ms: float | None = None
    prediction: DurationPrediction | None = None

    def __post_init__(self) -> None:
        """Validate identifiers, timing values, and prediction consistency."""

        if not self.worker_id.strip():
            raise ValueError("worker_id must not be empty")
        if not isfinite(self.estimated_service_ms) or self.estimated_service_ms <= 0:
            raise ValueError("estimated_service_ms must be finite and positive")
        if not isfinite(self.decision_latency_ms) or self.decision_latency_ms < 0:
            raise ValueError("decision_latency_ms must be finite and non-negative")

        if not isinstance(self.estimate_method, ServiceEstimateMethod):
            raise ValueError("estimate_method must be a ServiceEstimateMethod")
        if self.prediction is None:
            if self.estimate_method is not ServiceEstimateMethod.EXPECTED:
                raise ValueError("risk-aware estimates require a prediction")
            if any(
                value is not None
                for value in (
                    self.risk_quantile,
                    self.blend_weight,
                    self.fallback_service_ms,
                )
            ):
                raise ValueError("risk parameters require a prediction")
            return
        calculated_estimate_ms = self.prediction.service_estimate_ms(
            self.estimate_method,
            risk_quantile=self.risk_quantile,
            blend_weight=self.blend_weight,
            fallback_service_ms=self.fallback_service_ms,
        )
        if not isclose(
            self.estimated_service_ms,
            calculated_estimate_ms,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("estimated_service_ms must match estimate_method")
        if self.decision_latency_ms < self.prediction.overhead_ms:
            raise ValueError("decision_latency_ms must include prediction overhead")


class SchedulingPolicy(Protocol):
    """Structural interface implemented by every simulator scheduling policy."""

    @property
    def name(self) -> str:
        """Return the stable identifier used in benchmark reports."""

    def reset(self) -> None:
        """Reset mutable policy state before a simulation run."""

    def choose(
        self,
        item: WorkItem,
        workers: tuple[WorkerSnapshot, ...],
        now_ms: float,
    ) -> SchedulingChoice:
        """Choose a worker using only scheduler-visible request and load data.

        Args:
            item: The arriving request. It never contains true service time.
            workers: Stable worker snapshots ordered by worker identifier.
            now_ms: Time at which the single dispatcher starts this decision.

        Returns:
            A worker choice, service-time estimate, and decision latency.
        """
