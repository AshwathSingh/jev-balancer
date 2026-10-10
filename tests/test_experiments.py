from __future__ import annotations

import json

import pytest

from jev_balancer import (
    ArrivalPattern,
    ExperimentRegime,
    ServiceDistribution,
    run_uncertainty_experiment,
)


def make_regime(**overrides: object) -> ExperimentRegime:
    """Create a compact regime suitable for unit tests."""

    values: dict[str, object] = {
        "name": "test-regime",
        "arrival_pattern": ArrivalPattern.BURSTY,
        "service_distribution": ServiceDistribution.HEAVY_TAILED,
        "interarrival_ms": 1.0,
        "burst_gap_ms": 40.0,
        "predictor_overhead_ms": 0.25,
        "training_jobs": 30,
        "evaluation_jobs": 20,
        "worker_count": 3,
        "burst_size": 5,
    }
    values.update(overrides)
    return ExperimentRegime(**values)  # type: ignore[arg-type]


def test_uncertainty_experiment_is_deterministic_and_json_ready() -> None:
    regime = make_regime()

    first = run_uncertainty_experiment([regime], [3, 7])
    second = run_uncertainty_experiment([regime], [3, 7])

    assert first.to_dict() == second.to_dict()
    assert first.seeds == (3, 7)
    result = first.regimes[0]
    assert len(result.runs) == 2
    assert all(run.training_job_count == 30 for run in result.runs)
    assert all(run.evaluation_job_count == 20 for run in result.runs)
    assert len(result.policy_aggregates) == 10
    assert all(aggregate.runs == 2 for aggregate in result.policy_aggregates)
    assert {
        "input-regression",
        "semantic-expected",
        "semantic-quantile-90",
        "semantic-cvar-90",
        "semantic-expected-cvar-50",
        "semantic-entropy-fallback",
        "oracle-work",
    }.issubset({aggregate.policy_name for aggregate in result.policy_aggregates})
    json.dumps(first.to_dict())


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"name": " "}, "name"),
        ({"training_jobs": 0}, "training_jobs"),
        ({"evaluation_jobs": 0}, "evaluation_jobs"),
        ({"worker_count": 0}, "worker_count"),
        ({"predictor_overhead_ms": -1}, "predictor_overhead_ms"),
        ({"semantic_weight": 2}, "semantic_weight"),
    ],
)
def test_experiment_regime_rejects_invalid_settings(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make_regime(**overrides)


def test_uncertainty_experiment_rejects_invalid_matrix() -> None:
    regime = make_regime()

    with pytest.raises(ValueError, match="regime"):
        run_uncertainty_experiment([], [1])
    with pytest.raises(ValueError, match="unique"):
        run_uncertainty_experiment([regime, regime], [1])
    with pytest.raises(ValueError, match="seed"):
        run_uncertainty_experiment([regime], [])
    with pytest.raises(ValueError, match="integers"):
        run_uncertainty_experiment([regime], [True])
    with pytest.raises(ValueError, match="unique"):
        run_uncertainty_experiment([regime], [1, 1])
