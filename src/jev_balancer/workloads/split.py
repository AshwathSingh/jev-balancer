"""Chronological splitting with an optional label-availability guard."""

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
    require_observed_labels: bool = False,
) -> WorkloadSplit:
    """Split a trace without placing later arrivals into training.

    Jobs are ordered by ``(arrival_ms, request_id)`` before splitting. The
    split index is clamped so both partitions contain at least one job. With
    ``require_observed_labels=True``, the closest feasible boundary is chosen
    where every training job's ``arrival_ms + service_ms`` is no later than the
    first evaluation arrival. This models labels becoming available when an
    immediately started job completes.

    Args:
        jobs: At least two uniquely identified trace jobs.
        training_fraction: Desired training share strictly between zero and one.
        require_observed_labels: Prevent unresolved training outcomes from
            crossing the evaluation cutoff.

    Returns:
        Immutable chronological training and evaluation partitions.

    Raises:
        ValueError: If inputs are invalid or no label-safe boundary exists.
    """

    if (
        not isfinite(training_fraction)
        or training_fraction <= 0
        or training_fraction >= 1
    ):
        raise ValueError("training_fraction must be between 0 and 1")
    if type(require_observed_labels) is not bool:
        raise ValueError("require_observed_labels must be a boolean")

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
    if require_observed_labels:
        latest_label_ms = float("-inf")
        feasible_indices: list[int] = []
        for index in range(1, len(ordered)):
            previous_job = ordered[index - 1]
            latest_label_ms = max(
                latest_label_ms,
                previous_job.item.arrival_ms + previous_job.service_ms,
            )
            if latest_label_ms <= ordered[index].item.arrival_ms:
                feasible_indices.append(index)
        if not feasible_indices:
            raise ValueError("no split boundary has all training labels observed")
        split_index = min(
            feasible_indices,
            key=lambda index: (abs(index - split_index), index),
        )
    return WorkloadSplit(
        training=ordered[:split_index],
        evaluation=ordered[split_index:],
    )
