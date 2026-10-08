"""Leakage-resistant chronological splitting for workload traces."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import floor, isfinite

from jev_balancer.models import TraceJob


@dataclass(frozen=True, slots=True)
class WorkloadSplit:
    """Chronologically ordered training and evaluation traces."""

    training: tuple[TraceJob, ...]
    evaluation: tuple[TraceJob, ...]


def chronological_split(
    jobs: Iterable[TraceJob],
    *,
    training_fraction: float = 0.7,
) -> WorkloadSplit:
    """Split a trace without allowing later observations into training.

    Jobs are ordered by ``(arrival_ms, request_id)`` before splitting. The
    split index is clamped so both partitions contain at least one job.

    Args:
        jobs: At least two uniquely identified trace jobs.
        training_fraction: Desired training share strictly between zero and one.

    Returns:
        Immutable chronological training and evaluation partitions.

    Raises:
        ValueError: If the fraction, job count, or request identifiers are
            invalid.
    """

    if (
        not isfinite(training_fraction)
        or training_fraction <= 0
        or training_fraction >= 1
    ):
        raise ValueError("training_fraction must be between 0 and 1")

    ordered = tuple(
        sorted(
            jobs,
            key=lambda job: (job.item.arrival_ms, job.item.request_id),
        )
    )
    if len(ordered) < 2:
        raise ValueError("at least two jobs are required for a split")
    request_ids = [job.item.request_id for job in ordered]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("request identifiers must be unique")

    split_index = floor(len(ordered) * training_fraction)
    split_index = min(max(split_index, 1), len(ordered) - 1)
    return WorkloadSplit(
        training=ordered[:split_index],
        evaluation=ordered[split_index:],
    )
