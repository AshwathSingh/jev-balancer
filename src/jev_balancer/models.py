"""Provider-neutral domain models used by schedulers and simulations."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from math import isclose, isfinite, log


def _require_name(value: str, field_name: str) -> None:
    """Reject identifiers that are empty or contain only whitespace."""

    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _require_finite_nonnegative(value: float, field_name: str) -> None:
    """Require a finite numeric value greater than or equal to zero."""

    if not isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")


def _require_finite_positive(value: float, field_name: str) -> None:
    """Require a finite numeric value strictly greater than zero."""

    if not isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be finite and positive")


def _require_nonnegative_integer(value: int, field_name: str) -> None:
    """Require an integer count while explicitly rejecting booleans."""

    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


class DurationClass(str, Enum):
    """Coarse service-time classes produced by a duration predictor."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class ServiceEstimateMethod(str, Enum):
    """Method used to turn a duration distribution into a work estimate."""

    EXPECTED = "expected"
    QUANTILE = "quantile"
    CVAR = "cvar"
    EXPECTED_CVAR_BLEND = "expected-cvar-blend"
    SEMANTIC_FALLBACK_BLEND = "semantic-fallback-blend"


class _FrozenDurationMapping(Mapping[DurationClass, float]):
    """Small immutable mapping ordered by declared duration class."""

    __slots__ = ("_items",)
    _items: tuple[tuple[DurationClass, float], ...]

    def __init__(self, values: Mapping[DurationClass, float]) -> None:
        self._items = tuple(
            (duration_class, values[duration_class]) for duration_class in DurationClass
        )

    def __getitem__(self, key: DurationClass) -> float:
        """Return the stored probability or service time for ``key``."""

        for duration_class, value in self._items:
            if duration_class == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[DurationClass]:
        """Iterate over duration classes in their declared order."""

        return (duration_class for duration_class, _ in self._items)

    def __len__(self) -> int:
        """Return the number of represented duration classes."""

        return len(self._items)

    def __deepcopy__(self, memo: dict[int, object]) -> _FrozenDurationMapping:
        """Return this immutable value unchanged when copied."""

        return self


@dataclass(frozen=True, slots=True)
class WorkItem:
    """A scheduler-visible request.

    True service time intentionally does not live on this type. Only the
    simulator and oracle policy may access that value through :class:`TraceJob`.

    Attributes:
        request_id: Non-empty identifier unique within a simulation run.
        arrival_ms: Non-negative arrival time on the simulation clock.
        payload: Text available to semantic predictors.
        input_units: Non-negative integer used by non-semantic predictors.
        sla_ms: Optional positive end-to-end latency objective.
    """

    request_id: str
    arrival_ms: float
    payload: str
    input_units: int
    sla_ms: float | None = None

    def __post_init__(self) -> None:
        _require_name(self.request_id, "request_id")
        _require_finite_nonnegative(self.arrival_ms, "arrival_ms")
        _require_nonnegative_integer(self.input_units, "input_units")
        if self.sla_ms is not None:
            _require_finite_positive(self.sla_ms, "sla_ms")


@dataclass(frozen=True, slots=True)
class TraceJob:
    """A work item plus simulation truth hidden from normal policies.

    Attributes:
        item: Scheduler-visible request data.
        service_ms: Positive observed execution time used by the simulator.
        duration_class: Optional ground-truth class used for evaluation.
    """

    item: WorkItem
    service_ms: float
    duration_class: DurationClass | None = None

    def __post_init__(self) -> None:
        _require_finite_positive(self.service_ms, "service_ms")


@dataclass(frozen=True, slots=True)
class WorkerSnapshot:
    """Observable worker load at a scheduling decision point.

    Attributes:
        worker_id: Non-empty stable worker identifier.
        job_count: Number of running and queued jobs.
        estimated_backlog_ms: Non-negative estimated work remaining.
    """

    worker_id: str
    job_count: int
    estimated_backlog_ms: float

    def __post_init__(self) -> None:
        _require_name(self.worker_id, "worker_id")
        _require_nonnegative_integer(self.job_count, "job_count")
        _require_finite_nonnegative(
            self.estimated_backlog_ms,
            "estimated_backlog_ms",
        )


@dataclass(frozen=True, slots=True)
class DurationPrediction:
    """A provider-neutral probabilistic service-time estimate.

    ``expected_service_ms`` is calculated from the probabilities and calibrated
    representative service times. Keeping the complete discrete distribution
    makes mean, quantile, and tail-risk policies directly comparable.

    Attributes:
        probabilities: Probability for every :class:`DurationClass`, summing
            to one.
        class_service_ms: Positive representative runtime for every class.
        expected_service_ms: Positive predicted execution time.
        confidence: Predictor confidence from zero through one.
        predictor: Non-empty name of the producing predictor.
        overhead_ms: Non-negative time spent producing the prediction.
    """

    probabilities: Mapping[DurationClass, float]
    class_service_ms: Mapping[DurationClass, float]
    expected_service_ms: float
    confidence: float
    predictor: str
    overhead_ms: float = 0.0

    def __post_init__(self) -> None:
        if any(not isinstance(key, DurationClass) for key in self.probabilities):
            raise ValueError("probability keys must be DurationClass values")

        probabilities = dict(self.probabilities)
        expected_classes = set(DurationClass)
        if set(probabilities) != expected_classes:
            raise ValueError("probabilities must contain every duration class")

        for probability in probabilities.values():
            if not isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("probabilities must be finite values from 0 to 1")

        if not isclose(sum(probabilities.values()), 1.0, abs_tol=1e-9):
            raise ValueError("probabilities must sum to 1")

        if any(not isinstance(key, DurationClass) for key in self.class_service_ms):
            raise ValueError("class service keys must be DurationClass values")
        class_service_ms = dict(self.class_service_ms)
        if set(class_service_ms) != expected_classes:
            raise ValueError("class_service_ms must contain every duration class")
        for service_ms in class_service_ms.values():
            _require_finite_positive(service_ms, "class service time")

        _require_finite_positive(self.expected_service_ms, "expected_service_ms")
        calculated_expected_ms = sum(
            probabilities[duration_class] * class_service_ms[duration_class]
            for duration_class in DurationClass
        )
        if not isclose(
            self.expected_service_ms,
            calculated_expected_ms,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("expected_service_ms must match the distribution mean")
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be a finite value from 0 to 1")
        _require_name(self.predictor, "predictor")
        _require_finite_nonnegative(self.overhead_ms, "overhead_ms")

        object.__setattr__(
            self,
            "probabilities",
            _FrozenDurationMapping(probabilities),
        )
        object.__setattr__(
            self,
            "class_service_ms",
            _FrozenDurationMapping(class_service_ms),
        )

    def quantile_service_ms(self, quantile: float) -> float:
        """Return the smallest service time whose CDF reaches ``quantile``.

        Args:
            quantile: Probability strictly above zero and at most one.

        Raises:
            ValueError: If ``quantile`` is outside ``(0, 1]``.
        """

        if not isfinite(quantile) or not 0 < quantile <= 1:
            raise ValueError("quantile must be greater than 0 and at most 1")
        cumulative_probability = 0.0
        ordered_classes = sorted(
            DurationClass,
            key=self.class_service_ms.__getitem__,
        )
        for duration_class in ordered_classes:
            cumulative_probability += self.probabilities[duration_class]
            if cumulative_probability + 1e-12 >= quantile:
                return self.class_service_ms[duration_class]
        return self.class_service_ms[ordered_classes[-1]]

    def conditional_value_at_risk_ms(self, quantile: float) -> float:
        """Return mean service time within the upper tail above ``quantile``.

        For a discrete distribution, a fraction of the boundary class is used
        when necessary so the retained upper-tail mass is exactly
        ``1 - quantile``.

        Args:
            quantile: Lower tail boundary from zero inclusive to one exclusive.

        Raises:
            ValueError: If ``quantile`` is outside ``[0, 1)``.
        """

        if not isfinite(quantile) or not 0 <= quantile < 1:
            raise ValueError("quantile must be at least 0 and less than 1")
        tail_probability = 1 - quantile
        remaining_probability = tail_probability
        weighted_tail_ms = 0.0
        ordered_classes = sorted(
            DurationClass,
            key=self.class_service_ms.__getitem__,
            reverse=True,
        )
        for duration_class in ordered_classes:
            retained_probability = min(
                self.probabilities[duration_class],
                remaining_probability,
            )
            weighted_tail_ms += (
                retained_probability * self.class_service_ms[duration_class]
            )
            remaining_probability -= retained_probability
            if remaining_probability <= 1e-12:
                break
        return weighted_tail_ms / tail_probability

    @property
    def normalized_entropy(self) -> float:
        """Return distribution entropy scaled to the inclusive range zero to one."""

        entropy = -sum(
            probability * log(probability)
            for probability in self.probabilities.values()
            if probability > 0
        )
        return min(max(entropy / log(len(DurationClass)), 0.0), 1.0)

    def service_estimate_ms(
        self,
        method: ServiceEstimateMethod,
        *,
        risk_quantile: float | None = None,
        blend_weight: float | None = None,
        fallback_service_ms: float | None = None,
    ) -> float:
        """Transform this distribution into an auditable workload estimate.

        Args:
            method: Expected-value, quantile, tail-risk, or blend strategy.
            risk_quantile: Tail boundary required by quantile and CVaR methods.
            blend_weight: CVaR or fallback share from zero through one.
            fallback_service_ms: Positive non-semantic estimate for fallback.

        Raises:
            ValueError: If method-specific parameters are missing or invalid.
        """

        if not isinstance(method, ServiceEstimateMethod):
            raise ValueError("method must be a ServiceEstimateMethod")
        if method is ServiceEstimateMethod.EXPECTED:
            if any(
                value is not None
                for value in (risk_quantile, blend_weight, fallback_service_ms)
            ):
                raise ValueError("expected estimates do not accept risk parameters")
            return self.expected_service_ms
        if method is ServiceEstimateMethod.QUANTILE:
            if risk_quantile is None:
                raise ValueError("quantile estimates require risk_quantile")
            if blend_weight is not None or fallback_service_ms is not None:
                raise ValueError("quantile estimates accept only risk_quantile")
            return self.quantile_service_ms(risk_quantile)
        if method is ServiceEstimateMethod.CVAR:
            if risk_quantile is None:
                raise ValueError("CVaR estimates require risk_quantile")
            if blend_weight is not None or fallback_service_ms is not None:
                raise ValueError("CVaR estimates accept only risk_quantile")
            return self.conditional_value_at_risk_ms(risk_quantile)
        if blend_weight is None or not isfinite(blend_weight):
            raise ValueError("blend estimates require a finite blend_weight")
        if not 0 <= blend_weight <= 1:
            raise ValueError("blend_weight must be from 0 to 1")
        if method is ServiceEstimateMethod.EXPECTED_CVAR_BLEND:
            if risk_quantile is None:
                raise ValueError("expected-CVaR blends require risk_quantile")
            if fallback_service_ms is not None:
                raise ValueError("expected-CVaR blends do not accept a fallback")
            cvar_ms = self.conditional_value_at_risk_ms(risk_quantile)
            return (
                1 - blend_weight
            ) * self.expected_service_ms + blend_weight * cvar_ms
        if risk_quantile is not None:
            raise ValueError("semantic-fallback blends do not accept risk_quantile")
        if fallback_service_ms is None:
            raise ValueError("semantic-fallback blends require a fallback estimate")
        _require_finite_positive(fallback_service_ms, "fallback_service_ms")
        return (
            1 - blend_weight
        ) * self.expected_service_ms + blend_weight * fallback_service_ms

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation for traces and benchmark output."""

        return {
            "probabilities": {
                duration_class.value: probability
                for duration_class, probability in self.probabilities.items()
            },
            "class_service_ms": {
                duration_class.value: service_ms
                for duration_class, service_ms in self.class_service_ms.items()
            },
            "expected_service_ms": self.expected_service_ms,
            "confidence": self.confidence,
            "predictor": self.predictor,
            "overhead_ms": self.overhead_ms,
        }


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """The auditable result of assigning one request to one worker.

    Attributes:
        request_id: Identifier of the assigned request.
        worker_id: Identifier of the selected worker.
        policy: Non-empty name of the scheduling policy.
        decided_at_ms: Non-negative decision time on the simulation clock.
        estimated_service_ms: Positive service-time estimate for the request.
        estimated_backlog_ms: Worker backlog observed when routing began.
        estimate_method: Transformation used to obtain the service estimate.
        risk_quantile: Optional quantile or tail boundary.
        blend_weight: Optional CVaR or fallback blend share.
        fallback_service_ms: Optional non-semantic fallback estimate.
        prediction: Optional semantic prediction supporting the decision.
    """

    request_id: str
    worker_id: str
    policy: str
    decided_at_ms: float
    estimated_service_ms: float
    estimated_backlog_ms: float
    estimate_method: ServiceEstimateMethod = ServiceEstimateMethod.EXPECTED
    risk_quantile: float | None = None
    blend_weight: float | None = None
    fallback_service_ms: float | None = None
    prediction: DurationPrediction | None = None

    def __post_init__(self) -> None:
        _require_name(self.request_id, "request_id")
        _require_name(self.worker_id, "worker_id")
        _require_name(self.policy, "policy")
        _require_finite_nonnegative(self.decided_at_ms, "decided_at_ms")
        _require_finite_positive(self.estimated_service_ms, "estimated_service_ms")
        _require_finite_nonnegative(
            self.estimated_backlog_ms,
            "estimated_backlog_ms",
        )
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
        else:
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

    @property
    def estimated_finish_ms(self) -> float:
        """Return the predicted finish time relative to the simulation clock."""

        backlog_done_at = self.decided_at_ms + self.estimated_backlog_ms
        return backlog_done_at + self.estimated_service_ms

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation for an auditable decision trace."""

        return {
            "request_id": self.request_id,
            "worker_id": self.worker_id,
            "policy": self.policy,
            "decided_at_ms": self.decided_at_ms,
            "estimated_service_ms": self.estimated_service_ms,
            "estimated_backlog_ms": self.estimated_backlog_ms,
            "estimated_finish_ms": self.estimated_finish_ms,
            "estimate_method": self.estimate_method.value,
            "risk_quantile": self.risk_quantile,
            "blend_weight": self.blend_weight,
            "fallback_service_ms": self.fallback_service_ms,
            "prediction": self.prediction.to_dict() if self.prediction else None,
        }
