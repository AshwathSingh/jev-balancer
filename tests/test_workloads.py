from __future__ import annotations

import json
from dataclasses import replace
from math import inf, nan
from pathlib import Path

import pytest

from jev_balancer import (
    ArrivalPattern,
    DurationClass,
    ServiceDistribution,
    SyntheticWorkloadConfig,
    TraceJob,
    WorkItem,
    chronological_split,
    deserialize_jsonl,
    generate_workload,
    read_jsonl,
    serialize_jsonl,
    write_jsonl,
)


def make_job(request_id: str, arrival_ms: float) -> TraceJob:
    """Create a labeled trace job for persistence and split tests."""

    return TraceJob(
        item=WorkItem(
            request_id=request_id,
            arrival_ms=arrival_ms,
            payload="test payload",
            input_units=2,
            sla_ms=20,
        ),
        service_ms=10,
        duration_class=DurationClass.SHORT,
    )


def test_generate_workload_is_seed_deterministic() -> None:
    config = SyntheticWorkloadConfig(job_count=20, seed=7)

    first = generate_workload(config)
    second = generate_workload(config)
    different_seed = generate_workload(replace(config, seed=8))

    assert first == second
    assert first != different_seed


def test_generate_steady_arrivals() -> None:
    jobs = generate_workload(
        SyntheticWorkloadConfig(
            job_count=4,
            start_ms=5,
            interarrival_ms=3,
        )
    )

    assert [job.item.arrival_ms for job in jobs] == [5, 8, 11, 14]


def test_generate_bursty_arrivals() -> None:
    jobs = generate_workload(
        SyntheticWorkloadConfig(
            job_count=7,
            arrival_pattern=ArrivalPattern.BURSTY,
            start_ms=5,
            interarrival_ms=2,
            burst_size=3,
            burst_gap_ms=20,
        )
    )

    assert [job.item.arrival_ms for job in jobs] == [5, 7, 9, 25, 27, 29, 45]


@pytest.mark.parametrize(
    "distribution",
    [ServiceDistribution.UNIFORM, ServiceDistribution.HEAVY_TAILED],
)
def test_generated_service_times_and_labels_are_bounded(
    distribution: ServiceDistribution,
) -> None:
    config = SyntheticWorkloadConfig(
        job_count=100,
        seed=11,
        service_distribution=distribution,
        min_service_ms=10,
        max_service_ms=50,
    )

    jobs = generate_workload(config)

    assert all(10 <= job.service_ms <= 50 for job in jobs)
    assert all(job.duration_class is not None for job in jobs)
    assert all(job.item.input_units >= 1 for job in jobs)
    for job in jobs:
        assert job.item.sla_ms == pytest.approx(job.service_ms * 2)


def test_generate_workload_can_omit_slas() -> None:
    jobs = generate_workload(SyntheticWorkloadConfig(job_count=2, sla_multiplier=None))

    assert all(job.item.sla_ms is None for job in jobs)


@pytest.mark.parametrize("job_count", [-1, 1.5, True])
def test_config_requires_nonnegative_integer_job_count(job_count: object) -> None:
    with pytest.raises(ValueError, match="job_count"):
        SyntheticWorkloadConfig(job_count=job_count)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0.0, -1.0, inf, nan])
def test_config_requires_positive_service_bounds(value: float) -> None:
    with pytest.raises(ValueError, match="min_service_ms"):
        SyntheticWorkloadConfig(job_count=2, min_service_ms=value)


def test_config_rejects_overlapping_bursts() -> None:
    with pytest.raises(ValueError, match="overlap"):
        SyntheticWorkloadConfig(
            job_count=5,
            arrival_pattern=ArrivalPattern.BURSTY,
            interarrival_ms=10,
            burst_size=5,
            burst_gap_ms=40,
        )


def test_jsonl_round_trip_is_lossless_and_stable() -> None:
    jobs = (make_job("b", 2), make_job("a", 1))

    serialized = serialize_jsonl(jobs)
    restored = deserialize_jsonl(serialized)

    assert restored == jobs
    assert serialize_jsonl(restored) == serialized
    assert '"schema_version":1' in serialized


def test_jsonl_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "trace.jsonl"
    jobs = (make_job("a", 0), make_job("b", 1))

    write_jsonl(jobs, path)

    assert read_jsonl(path) == jobs


def test_jsonl_rejects_invalid_json_with_line_number() -> None:
    with pytest.raises(ValueError, match="line 2"):
        deserialize_jsonl(serialize_jsonl((make_job("a", 0),)) + "not-json\n")


def test_jsonl_rejects_unsupported_schema() -> None:
    record = json.loads(serialize_jsonl((make_job("a", 0),)))
    record["schema_version"] = 2

    with pytest.raises(ValueError, match="schema_version"):
        deserialize_jsonl(json.dumps(record))


def test_jsonl_rejects_duplicate_request_ids() -> None:
    job = make_job("duplicate", 0)

    with pytest.raises(ValueError, match="duplicate request_id"):
        deserialize_jsonl(serialize_jsonl((job, job)))


def test_chronological_split_orders_then_partitions() -> None:
    jobs = [
        make_job("d", 3),
        make_job("b", 1),
        make_job("a", 1),
        make_job("c", 2),
    ]

    split = chronological_split(jobs, training_fraction=0.5)

    assert [job.item.request_id for job in split.training] == ["a", "b"]
    assert [job.item.request_id for job in split.evaluation] == ["c", "d"]
    assert split.training[-1].item.arrival_ms <= split.evaluation[0].item.arrival_ms


@pytest.mark.parametrize("training_fraction", [0.0, 1.0, -0.1, 1.1, inf, nan])
def test_chronological_split_requires_valid_fraction(
    training_fraction: float,
) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        chronological_split(
            [make_job("a", 0), make_job("b", 1)],
            training_fraction=training_fraction,
        )


def test_chronological_split_requires_two_unique_jobs() -> None:
    with pytest.raises(ValueError, match="at least two"):
        chronological_split([make_job("a", 0)])
    with pytest.raises(ValueError, match="unique"):
        chronological_split([make_job("a", 0), make_job("a", 1)])
