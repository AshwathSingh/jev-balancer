"""Local model measurement for constructing service-time-labeled traces."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from typing import Protocol
from urllib.request import Request, urlopen

from jev_balancer.models import TraceJob, WorkItem
from jev_balancer.workloads.real import PromptRecord


def _require_finite_nonnegative(value: float, field_name: str) -> None:
    """Require a finite value greater than or equal to zero."""

    if not isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class OllamaMeasurement:
    """Measured token counts and inference timing for one local request.

    Durations exclude model load time. ``service_ms`` is prompt evaluation plus
    generation time, matching the interval in which the model performs work.
    """

    prompt_tokens: int
    output_tokens: int
    prompt_eval_ms: float
    generation_ms: float
    total_ms: float
    load_ms: float

    def __post_init__(self) -> None:
        """Validate token counts and duration components."""

        if type(self.prompt_tokens) is not int or self.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be a non-negative integer")
        if type(self.output_tokens) is not int or self.output_tokens < 0:
            raise ValueError("output_tokens must be a non-negative integer")
        for value, field_name in (
            (self.prompt_eval_ms, "prompt_eval_ms"),
            (self.generation_ms, "generation_ms"),
            (self.total_ms, "total_ms"),
            (self.load_ms, "load_ms"),
        ):
            _require_finite_nonnegative(value, field_name)
        if self.service_ms <= 0:
            raise ValueError("measured service time must be positive")

    @property
    def service_ms(self) -> float:
        """Return prompt-evaluation plus generation time in milliseconds."""

        return self.prompt_eval_ms + self.generation_ms

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready measurement without response content."""

        return {
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "prompt_eval_ms": self.prompt_eval_ms,
            "generation_ms": self.generation_ms,
            "service_ms": self.service_ms,
            "total_ms": self.total_ms,
            "load_ms": self.load_ms,
        }


class PromptMeasurementClient(Protocol):
    """Interface for a warmed model endpoint that measures prompt service."""

    def warm_up(self, payload: str) -> None:
        """Load the model before recorded measurements begin."""

    def measure(self, payload: str) -> OllamaMeasurement:
        """Measure one prompt without retaining generated response text."""


@dataclass(frozen=True, slots=True)
class OllamaClient:
    """Minimal client for deterministic measurements via Ollama's local API."""

    model: str
    endpoint: str = "http://127.0.0.1:11434/api/generate"
    num_predict: int = 64
    temperature: float = 0.0
    seed: int = 0
    timeout_seconds: float = 300.0
    keep_alive: str = "10m"

    def __post_init__(self) -> None:
        """Validate model identity, generation options, and endpoint settings."""

        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not self.endpoint.strip():
            raise ValueError("endpoint must not be empty")
        if type(self.num_predict) is not int or self.num_predict <= 0:
            raise ValueError("num_predict must be a positive integer")
        if not isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if not isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if not self.keep_alive.strip():
            raise ValueError("keep_alive must not be empty")

    def _generate(self, payload: str) -> OllamaMeasurement:
        """Call Ollama once and parse nanosecond timing fields."""

        if not payload.strip():
            raise ValueError("payload must not be empty")
        request_body = json.dumps(
            {
                "model": self.model,
                "prompt": payload,
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": {
                    "num_predict": self.num_predict,
                    "temperature": self.temperature,
                    "seed": self.seed,
                },
            }
        ).encode("utf-8")
        request = Request(
            self.endpoint,
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            result = json.load(response)
        if not isinstance(result, dict):
            raise ValueError("Ollama response must be a JSON object")
        try:
            return OllamaMeasurement(
                prompt_tokens=int(result["prompt_eval_count"]),
                output_tokens=int(result["eval_count"]),
                prompt_eval_ms=float(result["prompt_eval_duration"]) / 1_000_000,
                generation_ms=float(result["eval_duration"]) / 1_000_000,
                total_ms=float(result["total_duration"]) / 1_000_000,
                load_ms=float(result["load_duration"]) / 1_000_000,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Ollama response is missing valid timing fields"
            ) from error

    def warm_up(self, payload: str) -> None:
        """Load the configured model using an unrecorded request."""

        self._generate(payload)

    def measure(self, payload: str) -> OllamaMeasurement:
        """Measure one deterministic generation request."""

        return self._generate(payload)


@dataclass(frozen=True, slots=True)
class MeasuredTrace:
    """Measured jobs and detailed timings retained in matching order."""

    jobs: tuple[TraceJob, ...]
    measurements: tuple[OllamaMeasurement, ...]

    def __post_init__(self) -> None:
        """Require one timing record for every measured job."""

        if not self.jobs:
            raise ValueError("measured trace must not be empty")
        if len(self.jobs) != len(self.measurements):
            raise ValueError("jobs and measurements must have equal length")


def collect_measured_trace(
    prompts: Iterable[PromptRecord],
    client: PromptMeasurementClient,
    *,
    arrival_interval_ms: float = 1.0,
    sla_ms: float | None = None,
) -> MeasuredTrace:
    """Measure real prompts and build a trace with controlled arrival spacing.

    The arrival clock is a replay control, not a claim about source-dataset
    traffic. Actual model service time and prompt token count come from the
    measurement client.
    """

    records = tuple(prompts)
    if not records:
        raise ValueError("at least one prompt is required")
    request_ids = [record.request_id for record in records]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("prompt request identifiers must be unique")
    if not isfinite(arrival_interval_ms) or arrival_interval_ms <= 0:
        raise ValueError("arrival_interval_ms must be finite and positive")
    if sla_ms is not None and (not isfinite(sla_ms) or sla_ms <= 0):
        raise ValueError("sla_ms must be finite and positive")

    client.warm_up(records[0].payload)
    jobs: list[TraceJob] = []
    measurements: list[OllamaMeasurement] = []
    for index, record in enumerate(records):
        measurement = client.measure(record.payload)
        measurements.append(measurement)
        jobs.append(
            TraceJob(
                item=WorkItem(
                    request_id=record.request_id,
                    arrival_ms=index * arrival_interval_ms,
                    payload=record.payload,
                    input_units=measurement.prompt_tokens,
                    sla_ms=sla_ms,
                ),
                service_ms=measurement.service_ms,
            )
        )
    return MeasuredTrace(jobs=tuple(jobs), measurements=tuple(measurements))
