"""Run a multi-seed matrix for uncertainty-aware scheduling policies."""

from __future__ import annotations

import json
from dataclasses import dataclass

from jev_balancer import (
    ArrivalPattern,
    ExperimentRegime,
    PayloadStyle,
    ServiceDistribution,
    run_uncertainty_experiment,
)


@dataclass(frozen=True, slots=True)
class _SignalProfile:
    """Training and holdout semantic conditions for one matrix dimension."""

    name: str
    training_style: PayloadStyle = PayloadStyle.STANDARD
    evaluation_style: PayloadStyle = PayloadStyle.STANDARD
    training_ambiguity: float = 0.0
    evaluation_ambiguity: float = 0.0
    training_noise: float = 0.0
    evaluation_noise: float = 0.0


def build_regimes() -> tuple[ExperimentRegime, ...]:
    """Build the load, arrival, variance, and overhead experiment matrix."""

    regimes: list[ExperimentRegime] = []
    overheads_ms = (0.0, 2.0)
    signal_profiles = (
        _SignalProfile("clear"),
        _SignalProfile(
            "ambiguous",
            training_ambiguity=0.5,
            evaluation_ambiguity=0.5,
        ),
        _SignalProfile(
            "noisy",
            training_ambiguity=0.15,
            evaluation_ambiguity=0.15,
            training_noise=0.3,
            evaluation_noise=0.3,
        ),
        _SignalProfile(
            "shifted",
            evaluation_style=PayloadStyle.SHIFTED,
            training_ambiguity=0.1,
            evaluation_ambiguity=0.35,
            training_noise=0.05,
            evaluation_noise=0.2,
        ),
    )
    for arrival_pattern in ArrivalPattern:
        for service_distribution in ServiceDistribution:
            for load_name in ("moderate", "high"):
                if arrival_pattern is ArrivalPattern.STEADY:
                    interarrival_ms = 15.0 if load_name == "moderate" else 7.0
                    burst_gap_ms = 150.0
                else:
                    interarrival_ms = 1.0
                    burst_gap_ms = 180.0 if load_name == "moderate" else 90.0
                for signal_profile in signal_profiles:
                    for overhead_ms in overheads_ms:
                        regimes.append(
                            ExperimentRegime(
                                name=(
                                    f"{arrival_pattern.value}-"
                                    f"{service_distribution.value}-{load_name}-"
                                    f"{signal_profile.name}-"
                                    f"overhead-{overhead_ms:g}ms"
                                ),
                                arrival_pattern=arrival_pattern,
                                service_distribution=service_distribution,
                                interarrival_ms=interarrival_ms,
                                burst_gap_ms=burst_gap_ms,
                                predictor_overhead_ms=overhead_ms,
                                training_jobs=300,
                                evaluation_jobs=240,
                                training_payload_style=(signal_profile.training_style),
                                evaluation_payload_style=(
                                    signal_profile.evaluation_style
                                ),
                                training_semantic_ambiguity=(
                                    signal_profile.training_ambiguity
                                ),
                                evaluation_semantic_ambiguity=(
                                    signal_profile.evaluation_ambiguity
                                ),
                                training_semantic_noise=signal_profile.training_noise,
                                evaluation_semantic_noise=(
                                    signal_profile.evaluation_noise
                                ),
                            )
                        )
    return tuple(regimes)


def main() -> None:
    """Execute the complete deterministic matrix and print JSON results."""

    report = run_uncertainty_experiment(
        build_regimes(),
        seeds=(11, 23, 37, 53, 71),
    )
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
