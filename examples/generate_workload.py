"""Generate, persist, reload, and split a deterministic workload trace."""

from __future__ import annotations

import argparse
from pathlib import Path

from jev_balancer import (
    ArrivalPattern,
    ServiceDistribution,
    SyntheticWorkloadConfig,
    chronological_split,
    generate_workload,
    read_jsonl,
    write_jsonl,
)


def _at_least_two(value: str) -> int:
    """Parse a job count suitable for chronological splitting."""

    job_count = int(value)
    if job_count < 2:
        raise argparse.ArgumentTypeError("job count must be at least two")
    return job_count


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the workload example."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=_at_least_two, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--arrival",
        choices=[pattern.value for pattern in ArrivalPattern],
        default=ArrivalPattern.BURSTY.value,
    )
    parser.add_argument(
        "--service",
        choices=[distribution.value for distribution in ServiceDistribution],
        default=ServiceDistribution.HEAVY_TAILED.value,
    )
    return parser.parse_args()


def main() -> None:
    """Generate a trace and verify its persisted chronological split."""

    args = parse_args()
    config = SyntheticWorkloadConfig(
        job_count=args.jobs,
        seed=args.seed,
        arrival_pattern=ArrivalPattern(args.arrival),
        service_distribution=ServiceDistribution(args.service),
    )
    generated = generate_workload(config)
    write_jsonl(generated, args.output)
    reloaded = read_jsonl(args.output)
    if reloaded != generated:
        raise RuntimeError("persisted trace did not round-trip exactly")

    split = chronological_split(reloaded)
    print(f"wrote {len(reloaded)} jobs to {args.output}")
    print(
        f"training={len(split.training)} evaluation={len(split.evaluation)} "
        f"first_arrival_ms={reloaded[0].item.arrival_ms} "
        f"last_arrival_ms={reloaded[-1].item.arrival_ms}"
    )


if __name__ == "__main__":
    main()
