"""Interfaces shared by duration predictors and scheduling policies."""

from __future__ import annotations

from typing import Protocol

from jev_balancer.models import DurationPrediction, WorkItem


class DurationPredictor(Protocol):
    """Provider-neutral interface for probabilistic duration prediction."""

    @property
    def name(self) -> str:
        """Return the stable predictor identifier used in audit records."""

    def predict(self, item: WorkItem) -> DurationPrediction:
        """Predict duration using scheduler-visible request information only."""
