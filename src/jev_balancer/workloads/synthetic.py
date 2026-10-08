"""Seeded synthetic workload generation for controlled experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from random import Random

from jev_balancer.models import DurationClass, TraceJob, WorkItem


class ArrivalPattern(str, Enum):
    """Supported request-arrival shapes."""

    STEADY = "steady"
    BURSTY = "bursty"


class ServiceDistribution(str, Enum):
    """Supported actual service-time distributions."""

    UNIFORM = "uniform"
    HEAVY_TAILED = "heavy-tailed"


def _require_nonnegative(value: float, field_name: str) -> None:
    """Require a finite value greater than or equal to zero."""

    if not isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")


def _require_positive(value: float, field_name: str) -> None:
    """Require a finite value strictly greater than zero."""

    if not isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class SyntheticWorkloadConfig:
    """Configuration for one reproducible synthetic trace.

    Attributes:
        job_count: Number of jobs to generate, including zero for an empty trace.
        seed: Integer seed controlling all random values.
        arrival_pattern: Steady spacing or groups of burst arrivals.
        service_distribution: Uniform or truncated Pareto service times.
        start_ms: Arrival time of the first job.
        interarrival_ms: Spacing between steady jobs or jobs within a burst.
        burst_size: Number of jobs in each burst.
        burst_gap_ms: Time between the start of consecutive bursts.
        min_service_ms: Smallest generated service time.
        max_service_ms: Largest generated service time.
        pareto_shape: Shape parameter used by the heavy-tailed distribution.
        sla_multiplier: Optional service-time multiplier used to create SLAs.
    """

    job_count: int
    seed: int = 0
    arrival_pattern: ArrivalPattern = ArrivalPattern.STEADY
    service_distribution: ServiceDistribution = ServiceDistribution.UNIFORM
    start_ms: float = 0.0
    interarrival_ms: float = 10.0
    burst_size: int = 5
    burst_gap_ms: float = 100.0
    min_service_ms: float = 5.0
    max_service_ms: float = 100.0
    pareto_shape: float = 1.5
    sla_multiplier: float | None = 2.0

    def __post_init__(self) -> None:
        """Validate counts, time bounds, and distribution parameters."""

        if type(self.job_count) is not int or self.job_count < 0:
            raise ValueError("job_count must be a non-negative integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if type(self.burst_size) is not int or self.burst_size <= 0:
            raise ValueError("burst_size must be a positive integer")
        if not isinstance(self.arrival_pattern, ArrivalPattern):
            raise ValueError("arrival_pattern must be an ArrivalPattern")
        if not isinstance(self.service_distribution, ServiceDistribution):
            raise ValueError("service_distribution must be a ServiceDistribution")
        _require_nonnegative(self.start_ms, "start_ms")
        _require_positive(self.interarrival_ms, "interarrival_ms")
        _require_positive(self.burst_gap_ms, "burst_gap_ms")
        _require_positive(self.min_service_ms, "min_service_ms")
        _require_positive(self.max_service_ms, "max_service_ms")
        if self.max_service_ms <= self.min_service_ms:
            raise ValueError("max_service_ms must be greater than min_service_ms")
        _require_positive(self.pareto_shape, "pareto_shape")
        if self.sla_multiplier is not None:
            _require_positive(self.sla_multiplier, "sla_multiplier")
        final_arrival_offset_ms = self.interarrival_ms * (self.burst_size - 1)
        if (
            self.arrival_pattern is ArrivalPattern.BURSTY
            and self.burst_gap_ms <= final_arrival_offset_ms
        ):
            raise ValueError("burst_gap_ms must not overlap consecutive bursts")


_PAYLOADS = {
    DurationClass.SHORT: "Classify a compact status update.",
    DurationClass.MEDIUM: "Summarize a customer support case.",
    DurationClass.LONG: "Analyze a multi-step production incident.",
}


def _arrival_ms(index: int, config: SyntheticWorkloadConfig) -> float:
    """Calculate a stable arrival time for a generated job index."""

    if config.arrival_pattern is ArrivalPattern.STEADY:
        offset_ms = index * config.interarrival_ms
    else:
        burst_index, position = divmod(index, config.burst_size)
        offset_ms = (
            burst_index * config.burst_gap_ms + position * config.interarrival_ms
        )
    return round(config.start_ms + offset_ms, 6)


def _service_ms(rng: Random, config: SyntheticWorkloadConfig) -> float:
    """Sample and bound one actual service time."""

    if config.service_distribution is ServiceDistribution.UNIFORM:
        service_ms = rng.uniform(config.min_service_ms, config.max_service_ms)
    else:
        service_ms = config.min_service_ms * rng.paretovariate(config.pareto_shape)
        service_ms = min(service_ms, config.max_service_ms)
    return round(service_ms, 6)


def _duration_class(
    service_ms: float,
    config: SyntheticWorkloadConfig,
) -> DurationClass:
    """Assign a duration class using thirds of the configured service range."""

    width_ms = config.max_service_ms - config.min_service_ms
    short_upper_ms = config.min_service_ms + width_ms / 3
    medium_upper_ms = config.min_service_ms + width_ms * 2 / 3
    if service_ms <= short_upper_ms:
        return DurationClass.SHORT
    if service_ms <= medium_upper_ms:
        return DurationClass.MEDIUM
    return DurationClass.LONG


def _input_units(rng: Random, service_ms: float) -> int:
    """Create a noisy non-semantic size signal correlated with runtime."""

    noisy_units = service_ms / 10 * rng.uniform(0.5, 1.5)
    return max(1, round(noisy_units))


def generate_workload(config: SyntheticWorkloadConfig) -> tuple[TraceJob, ...]:
    """Generate a deterministic trace from ``config``.

    The same configuration produces equal jobs and ordering on repeated calls.
    Payload wording reflects the duration class, while ``input_units`` is a
    deliberately noisy non-semantic runtime proxy for regression baselines.

    Args:
        config: Fully validated generation settings.

    Returns:
        Jobs ordered by arrival time and request identifier.
    """

    rng = Random(config.seed)
    jobs: list[TraceJob] = []
    for index in range(config.job_count):
        service_ms = _service_ms(rng, config)
        duration_class = _duration_class(service_ms, config)
        request_id = f"synthetic-{index:06d}"
        sla_ms = (
            round(service_ms * config.sla_multiplier, 6)
            if config.sla_multiplier is not None
            else None
        )
        jobs.append(
            TraceJob(
                item=WorkItem(
                    request_id=request_id,
                    arrival_ms=_arrival_ms(index, config),
                    payload=_PAYLOADS[duration_class],
                    input_units=_input_units(rng, service_ms),
                    sla_ms=sla_ms,
                ),
                service_ms=service_ms,
                duration_class=duration_class,
            )
        )
    return tuple(jobs)
