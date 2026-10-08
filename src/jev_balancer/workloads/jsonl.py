"""Versioned JSON Lines serialization for reproducible workload traces."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from jev_balancer.models import DurationClass, TraceJob, WorkItem

_SCHEMA_VERSION = 1


def _required_str(record: Mapping[str, object], field_name: str) -> str:
    """Read a required string field from a decoded JSON object."""

    value = record.get(field_name)
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _required_float(record: Mapping[str, object], field_name: str) -> float:
    """Read a required non-boolean JSON number as a float."""

    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    return float(value)


def _required_int(record: Mapping[str, object], field_name: str) -> int:
    """Read a required non-boolean JSON integer."""

    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def trace_job_to_dict(job: TraceJob) -> dict[str, object]:
    """Convert one trace job into the current JSONL schema."""

    return {
        "schema_version": _SCHEMA_VERSION,
        "request_id": job.item.request_id,
        "arrival_ms": float(job.item.arrival_ms),
        "payload": job.item.payload,
        "input_units": job.item.input_units,
        "service_ms": float(job.service_ms),
        "sla_ms": (float(job.item.sla_ms) if job.item.sla_ms is not None else None),
        "duration_class": (
            job.duration_class.value if job.duration_class is not None else None
        ),
    }


def trace_job_from_dict(record: Mapping[str, object]) -> TraceJob:
    """Parse and validate one decoded JSONL record."""

    schema_version = record.get("schema_version")
    if schema_version != _SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {schema_version}")

    raw_sla_ms = record.get("sla_ms")
    sla_ms = None if raw_sla_ms is None else _required_float(record, "sla_ms")
    raw_duration_class = record.get("duration_class")
    if raw_duration_class is None:
        duration_class = None
    elif isinstance(raw_duration_class, str):
        try:
            duration_class = DurationClass(raw_duration_class)
        except ValueError as error:
            raise ValueError(
                f"unsupported duration_class: {raw_duration_class}"
            ) from error
    else:
        raise ValueError("duration_class must be a string or null")

    return TraceJob(
        item=WorkItem(
            request_id=_required_str(record, "request_id"),
            arrival_ms=_required_float(record, "arrival_ms"),
            payload=_required_str(record, "payload"),
            input_units=_required_int(record, "input_units"),
            sla_ms=sla_ms,
        ),
        service_ms=_required_float(record, "service_ms"),
        duration_class=duration_class,
    )


def serialize_jsonl(jobs: Sequence[TraceJob]) -> str:
    """Serialize jobs to deterministic JSON Lines text.

    Fields are sorted to make file diffs stable. The supplied job order is
    retained because trace ordering can be useful evidence during debugging.
    """

    lines = [
        json.dumps(trace_job_to_dict(job), sort_keys=True, separators=(",", ":"))
        for job in jobs
    ]
    return "" if not lines else "\n".join(lines) + "\n"


def deserialize_jsonl(text: str) -> tuple[TraceJob, ...]:
    """Parse JSON Lines text and reject malformed or duplicate records."""

    jobs: list[TraceJob] = []
    seen_request_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}") from error
        if not isinstance(decoded, dict):
            raise ValueError(f"line {line_number} must contain a JSON object")
        try:
            job = trace_job_from_dict(decoded)
        except ValueError as error:
            raise ValueError(f"invalid trace on line {line_number}: {error}") from error
        request_id = job.item.request_id
        if request_id in seen_request_ids:
            raise ValueError(
                f"duplicate request_id on line {line_number}: {request_id}"
            )
        seen_request_ids.add(request_id)
        jobs.append(job)
    return tuple(jobs)


def write_jsonl(jobs: Sequence[TraceJob], path: str | Path) -> None:
    """Write a complete trace to ``path``, replacing an existing file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(serialize_jsonl(jobs), encoding="utf-8")


def read_jsonl(path: str | Path) -> tuple[TraceJob, ...]:
    """Read and validate a trace from ``path``."""

    return deserialize_jsonl(Path(path).read_text(encoding="utf-8"))
