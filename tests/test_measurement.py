from __future__ import annotations

import json
from dataclasses import dataclass, field
from io import BytesIO
from math import inf, nan
from typing import Any

import pytest

import jev_balancer.measurement as measurement_module
from jev_balancer import (
    OllamaClient,
    OllamaMeasurement,
    PromptRecord,
    collect_measured_trace,
)


def make_measurement(service_ms: float = 30) -> OllamaMeasurement:
    """Create a valid local inference measurement."""

    return OllamaMeasurement(
        prompt_tokens=12,
        output_tokens=4,
        prompt_eval_ms=service_ms / 3,
        generation_ms=service_ms * 2 / 3,
        total_ms=service_ms + 1,
        load_ms=1,
    )


@dataclass
class FakeClient:
    """Record warm-up and measurement calls without a model server."""

    warmed_payloads: list[str] = field(default_factory=list)
    measured_payloads: list[str] = field(default_factory=list)

    def warm_up(self, payload: str) -> None:
        """Record the warm-up payload."""

        self.warmed_payloads.append(payload)

    def measure(self, payload: str) -> OllamaMeasurement:
        """Return a duration based on call order."""

        self.measured_payloads.append(payload)
        return make_measurement(10 * len(self.measured_payloads))


def test_collect_measured_trace_warms_model_and_records_hidden_truth() -> None:
    client = FakeClient()
    prompts = (PromptRecord("a", "first"), PromptRecord("b", "second"))

    trace = collect_measured_trace(prompts, client, arrival_interval_ms=5, sla_ms=50)

    assert client.warmed_payloads == ["first"]
    assert client.measured_payloads == ["first", "second"]
    assert [job.item.arrival_ms for job in trace.jobs] == [0, 5]
    assert [job.item.input_units for job in trace.jobs] == [12, 12]
    assert [job.service_ms for job in trace.jobs] == [10, 20]
    assert all(not hasattr(job.item, "service_ms") for job in trace.jobs)
    assert len(trace.measurements) == len(trace.jobs)


def test_collect_measured_trace_rejects_invalid_inputs() -> None:
    client = FakeClient()
    prompt = PromptRecord("a", "first")

    with pytest.raises(ValueError, match="at least one"):
        collect_measured_trace([], client)
    with pytest.raises(ValueError, match="unique"):
        collect_measured_trace([prompt, prompt], client)
    with pytest.raises(ValueError, match="arrival_interval_ms"):
        collect_measured_trace([prompt], client, arrival_interval_ms=0)
    with pytest.raises(ValueError, match="sla_ms"):
        collect_measured_trace([prompt], client, sla_ms=inf)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"model": " "}, "model"),
        ({"num_predict": 0}, "num_predict"),
        ({"temperature": -1}, "temperature"),
        ({"timeout_seconds": nan}, "timeout_seconds"),
        ({"keep_alive": " "}, "keep_alive"),
    ],
)
def test_ollama_client_rejects_invalid_configuration(
    overrides: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {"model": "llama2"}
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        OllamaClient(**values)  # type: ignore[arg-type]


def test_ollama_client_parses_reported_nanosecond_timings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = {
        "prompt_eval_count": 10,
        "eval_count": 5,
        "prompt_eval_duration": 20_000_000,
        "eval_duration": 30_000_000,
        "total_duration": 60_000_000,
        "load_duration": 10_000_000,
    }

    class FakeResponse(BytesIO):
        """Bytes response implementing the URL response context manager."""

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: Any) -> None:
            self.close()

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        assert timeout == 300
        return FakeResponse(json.dumps(result).encode())

    monkeypatch.setattr(measurement_module, "urlopen", fake_urlopen)

    measured = OllamaClient("llama2").measure("hello")

    assert measured.prompt_tokens == 10
    assert measured.output_tokens == 5
    assert measured.prompt_eval_ms == 20
    assert measured.generation_ms == 30
    assert measured.service_ms == 50
