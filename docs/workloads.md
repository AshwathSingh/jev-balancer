# Workload traces

Controlled traces let every scheduler receive identical requests, arrival
times, and true runtimes. This prevents traffic variation from being mistaken
for a routing improvement.

Generate and verify a trace with:

```bash
uv run python examples/generate_workload.py \
  --output /tmp/jev-workload.jsonl \
  --jobs 20 \
  --seed 7 \
  --arrival bursty \
  --service heavy-tailed
```

The same options and seed produce an equal in-memory trace. Generated jobs
contain:

- A stable request identifier and arrival time.
- A duration-class-specific payload for future semantic prediction.
- Noisy `input_units` for the non-semantic regression baseline.
- Hidden actual `service_ms` and an optional fixed SLA target.
- A ground-truth `short`, `medium`, or `long` duration class.

## Arrival patterns

`steady` spaces every request by `interarrival_ms`. `bursty` uses the same
spacing within each burst and separates burst starts by `burst_gap_ms`.

## Service distributions

`uniform` samples evenly between the configured service bounds.
`heavy-tailed` uses a truncated Pareto sample, producing many short jobs and a
smaller number of expensive jobs.

Synthetic duration labels use thirds of the configured service-time range.
They are generator truth, not boundaries learned by the predictor. The fixed
`sla_ms` is deliberately independent of each job's actual runtime so it cannot
reveal hidden `service_ms` to a policy.

## JSONL and splitting

Every JSONL record carries `schema_version: 1`. Readers reject unsupported
versions, malformed values, and duplicate request identifiers. Serialization
preserves the supplied job order.

`chronological_split()` first orders jobs by `(arrival_ms, request_id)` and then
creates non-empty training and evaluation partitions. Use
`require_observed_labels=True` for experiments: it selects the closest boundary
where every training label would be available by the first evaluation arrival,
assuming a job completes at `arrival_ms + service_ms`. Fit class means,
predictors, and regression coefficients only on the training partition.
