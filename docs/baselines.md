# Baseline policies and metrics

The first benchmark stage establishes non-semantic results that a future Jev
policy must beat. Run the included comparison with:

```bash
uv run python examples/benchmark_baselines.py
```

The example replays one immutable trace through five policies:

- `RoundRobinPolicy` ignores load and cycles through workers.
- `LeastJobsPolicy` chooses the shortest visible queue.
- `MeanWorkPolicy` converts every queued job into one historical mean runtime.
- `InputRegressionPolicy` predicts runtime from request size without semantics.
- `OracleWorkPolicy` uses actual runtime as an unattainable upper bound.

Fit the mean and regression coefficients on training data with
`MeanWorkPolicy.from_jobs()` and `InputRegressionPolicy.from_jobs()`. Never fit
them on the trace being evaluated. The oracle is not a deployable scheduler and
must not be presented as one.

## Metrics

`benchmark_policies()` returns the simulation and metrics for every policy.
The report includes:

- Mean, p50, p95, p99, and maximum response time.
- Worker queue wait, dispatcher wait, and policy decision latency.
- Throughput and makespan.
- SLA violations among requests that define an SLA.
- Per-worker utilization and the range between highest and lowest utilization.

Percentiles use linear interpolation over sorted observations. Throughput and
utilization use the interval from the earliest arrival to the final completion.
All latency values are milliseconds.

Use `report.to_dict(include_executions=False)` for compact aggregate output or
leave `include_executions=True` to retain each auditable routing decision.
