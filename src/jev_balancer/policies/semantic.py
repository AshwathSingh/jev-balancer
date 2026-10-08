"""Scheduling policy backed by probabilistic semantic duration estimates."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from jev_balancer.models import WorkerSnapshot, WorkItem
from jev_balancer.predictors import DurationPredictor
from jev_balancer.scheduler import SchedulingChoice


@dataclass(frozen=True, slots=True)
class SemanticWorkPolicy:
    """Route to the least estimated backlog using a duration predictor.

    Attributes:
        predictor: Fitted predictor that can inspect only a :class:`WorkItem`.
        scheduler_overhead_ms: Non-prediction routing latency per decision.
        name: Unique policy identifier used in benchmark reports.
    """

    predictor: DurationPredictor
    scheduler_overhead_ms: float = 0.0
    name: str = "semantic-work"

    def __post_init__(self) -> None:
        """Validate policy identity and scheduler-only overhead."""

        if not self.name.strip():
            raise ValueError("name must not be empty")
        if not isfinite(self.scheduler_overhead_ms) or self.scheduler_overhead_ms < 0:
            raise ValueError("scheduler_overhead_ms must be finite and non-negative")

    def reset(self) -> None:
        """Reset state; fitted v0.1 predictors are immutable."""

    def choose(
        self,
        item: WorkItem,
        workers: tuple[WorkerSnapshot, ...],
        now_ms: float,
    ) -> SchedulingChoice:
        """Predict duration and choose the smallest visible work backlog."""

        if not workers:
            raise ValueError("at least one worker snapshot is required")
        prediction = self.predictor.predict(item)
        worker = min(
            workers,
            key=lambda snapshot: (
                snapshot.estimated_backlog_ms,
                snapshot.job_count,
                snapshot.worker_id,
            ),
        )
        return SchedulingChoice(
            worker_id=worker.worker_id,
            estimated_service_ms=prediction.expected_service_ms,
            decision_latency_ms=(prediction.overhead_ms + self.scheduler_overhead_ms),
            prediction=prediction,
        )
