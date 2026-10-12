from __future__ import annotations

from math import inf, nan

import pytest

from jev_balancer import (
    DurationClass,
    DurationClassThresholds,
    PromptRecord,
    TraceJob,
    WorkItem,
    extract_oasst_prompts,
)


def make_job(request_id: str, service_ms: float) -> TraceJob:
    """Create one unlabeled measured job."""

    return TraceJob(
        WorkItem(request_id, 0, f"prompt {request_id}", input_units=1),
        service_ms=service_ms,
    )


def test_duration_thresholds_fit_training_tertiles_and_label_jobs() -> None:
    training = tuple(
        make_job(str(index), service_ms)
        for index, service_ms in enumerate((10, 20, 30, 40, 50, 60))
    )
    thresholds = DurationClassThresholds.fit(training)

    labeled = thresholds.label(
        (make_job("short", 5), make_job("medium", 35), make_job("long", 1000))
    )

    assert thresholds.to_dict() == {"short_max_ms": 20, "medium_max_ms": 40}
    assert [job.duration_class for job in labeled] == [
        DurationClass.SHORT,
        DurationClass.MEDIUM,
        DurationClass.LONG,
    ]
    assert [job.item.request_id for job in labeled] == ["short", "medium", "long"]


def test_duration_thresholds_do_not_use_evaluation_values() -> None:
    training = (make_job("a", 10), make_job("b", 20), make_job("c", 30))

    first = DurationClassThresholds.fit(training)
    second = DurationClassThresholds.fit(training)

    assert first == second
    assert first.classify(1_000_000) is DurationClass.LONG


@pytest.mark.parametrize("value", [0.0, -1.0, inf, nan])
def test_duration_thresholds_reject_invalid_service_time(value: float) -> None:
    with pytest.raises(ValueError, match="service_ms"):
        DurationClassThresholds(10, 20).classify(value)


def test_extract_oasst_prompts_filters_to_reviewed_root_messages() -> None:
    base: dict[str, object] = {
        "message_id": "keep",
        "text": "Explain queue scheduling.",
        "role": "prompter",
        "lang": "en",
        "parent_id": None,
        "deleted": False,
        "review_result": True,
    }
    rows = (
        base,
        {**base, "message_id": "reply", "parent_id": "keep"},
        {**base, "message_id": "deleted", "deleted": True},
        {**base, "message_id": "other-language", "lang": "de"},
        {**base, "message_id": "failed-review", "review_result": False},
    )

    prompts = extract_oasst_prompts(rows)

    assert prompts == (PromptRecord("keep", "Explain queue scheduling."),)


def test_extract_oasst_prompts_applies_limit_and_validates_rows() -> None:
    rows = tuple(
        {
            "message_id": str(index),
            "text": f"prompt {index}",
            "role": "prompter",
            "lang": "en",
            "parent_id": None,
            "deleted": False,
            "review_result": True,
        }
        for index in range(3)
    )

    assert len(extract_oasst_prompts(rows, limit=2)) == 2
    with pytest.raises(ValueError, match="limit"):
        extract_oasst_prompts(rows, limit=0)
    with pytest.raises(ValueError, match="message_id"):
        extract_oasst_prompts(({**rows[0], "message_id": 1},))
