"""Run a multi-seed matrix for uncertainty-aware scheduling policies."""

from __future__ import annotations

import json

from jev_balancer import (
    ArrivalPattern,
    ExperimentRegime,
    ServiceDistribution,
    run_uncertainty_experiment,
)


def build_regimes() -> tuple[ExperimentRegime, ...]:
    """Build the load, arrival, variance, and overhead experiment matrix."""

    regimes: list[ExperimentRegime] = []
    overheads_ms = (0.0, 0.25, 2.0)
    for arrival_pattern in ArrivalPattern:
        for service_distribution in ServiceDistribution:
            for load_name in ("moderate", "high"):
                if arrival_pattern is ArrivalPattern.STEADY:
                    interarrival_ms = 15.0 if load_name == "moderate" else 7.0
                    burst_gap_ms = 150.0
                else:
                    interarrival_ms = 1.0
                    burst_gap_ms = 180.0 if load_name == "moderate" else 90.0
                for overhead_ms in overheads_ms:
                    regimes.append(
                        ExperimentRegime(
                            name=(
                                f"{arrival_pattern.value}-"
                                f"{service_distribution.value}-{load_name}-"
                                f"overhead-{overhead_ms:g}ms"
                            ),
                            arrival_pattern=arrival_pattern,
                            service_distribution=service_distribution,
                            interarrival_ms=interarrival_ms,
                            burst_gap_ms=burst_gap_ms,
                            predictor_overhead_ms=overhead_ms,
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
