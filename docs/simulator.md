# Deterministic simulator

The simulator replays the same workload through different scheduling policies
so their results can be compared without live-system timing noise.

## Execution model

- Jobs are processed by `(arrival_ms, request_id)`, regardless of input order.
- One dispatcher makes routing decisions sequentially.
- Policy decision latency blocks the dispatcher and is included in response
  time.
- Workers execute one job at a time in FIFO order without preemption.
- Actual `service_ms` is visible to the engine, but never to normal policies.
- Worker snapshots expose job count and estimated backlog derived from earlier
  policy estimates.

This is intentionally a small model. Parallel routing, retries, worker
failures, and heterogeneous workers remain outside v0.1.

## Policy interface

A policy supplies a stable name, resets its state before each run, and chooses
a worker from immutable snapshots:

```python
from jev_balancer import SchedulingChoice, WorkerSnapshot, WorkItem


class FirstWorkerPolicy:
    """Route every request to the first worker for baseline testing."""

    name = "first-worker"

    def reset(self) -> None:
        """Reset state; this policy has no mutable state."""

    def choose(
        self,
        item: WorkItem,
        workers: tuple[WorkerSnapshot, ...],
        now_ms: float,
    ) -> SchedulingChoice:
        """Return a deterministic worker and service-time estimate."""

        return SchedulingChoice(
            worker_id=workers[0].worker_id,
            estimated_service_ms=max(float(item.input_units), 1.0),
        )
```

`WorkItem` deliberately excludes true runtime. The simulator joins the choice
with `TraceJob.service_ms` only after routing.

## Running a trace

```python
from jev_balancer import TraceJob, WorkItem, simulate

jobs = [
    TraceJob(
        item=WorkItem(
            request_id="request-1",
            arrival_ms=0,
            payload="Example request",
            input_units=10,
        ),
        service_ms=25,
    )
]

result = simulate(jobs, ["worker-a", "worker-b"], FirstWorkerPolicy())
print(result.to_dict())
```

`SimulationResult.to_dict()` and every nested execution and decision record
return JSON-ready values for later benchmark reporting.
