# Real-prompt evaluation

This experiment adds human-written prompts and measured model runtimes to the
offline simulator. It is a bridge between synthetic traces and a future Jev
adapter, not a production load test.

## Evidence boundary

Public datasets commonly provide either request text or serving timings, but
not both. [OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1)
provides human-written prompt text under Apache-2.0. This project therefore
measures those prompts against a local Ollama model to create new service-time
labels.

The source dataset does not provide production arrival timestamps. Evaluation
arrival times are a controlled replay derived from the training mean and a
chosen utilization. Results describe this replay only; they must not be
presented as production-traffic performance.

## Measurement procedure

`benchmark_real_prompts.py` performs the following steps:

1. Fetch reviewed English root prompts from the OASST1 validation split in
   source order.
2. Warm the configured Ollama model with an unrecorded request.
3. Measure each prompt sequentially with deterministic generation settings.
4. Define service time as prompt-evaluation time plus generation time, excluding
   cold model-load time.
5. Split the completed trace chronologically into training and evaluation data.
6. Fit duration-class thresholds and scheduling predictors only on training
   jobs.
7. Replay the evaluation jobs through every policy with identical controlled
   arrivals.

True `service_ms` remains on `TraceJob`, outside the scheduler-visible
`WorkItem`. Only the oracle comparator may read it. Generated responses and raw
prompt text are omitted from the JSON result.

## Run the experiment

Start Ollama and make sure the selected model is installed, then run:

```bash
ollama list
uv run python examples/benchmark_real_prompts.py \
  --model llama2 \
  --prompts 80 \
  --num-predict 64 \
  --workers 4 \
  --target-utilization 0.85 \
  --output real-prompt-results.json
```

The output records source selection, measurement settings, chronological split,
training-fitted thresholds, prediction quality, replay load, and scheduling
metrics. Use `--include-executions` only when per-request decisions are needed.

## Interpretation limits

- Runtime values depend on the local machine, model, Ollama version, and model
  state.
- OASST1 prompts are real human text but are not a production traffic sample.
- Controlled arrivals do not reproduce real burstiness or concurrency.
- A small run is a feasibility pilot, not evidence of statistical significance.
- Local Ollama measurements test the experimental pipeline, not Jev prediction
  quality or overhead.

The next research step is repeated measurement across seeds, loads, and models,
followed by a Jev adapter evaluated on the same frozen traces.

## Initial pilot result

The first policy-discriminating pilot was run on 2026-10-12 with Ollama 0.34.4,
`llama2:latest`, 80 prompts, 64 maximum output tokens, four workers, and 1.20
controlled offered load. The chronological split contained 56 training and 24
evaluation jobs.

| Result | Value |
| --- | ---: |
| Measured mean service time | 1,190.4 ms |
| Measured p95 service time | 1,260.8 ms |
| Semantic class accuracy | 29.2% |
| Semantic Brier score | 1.054 |
| Semantic mean absolute error | 47.0 ms |
| Strongest conventional p95 response | 2,080.1 ms |
| Semantic p95 response | 2,080.1 ms |
| Semantic p95 improvement | 0.0% |
| Conventional and semantic throughput | 3.091 jobs/s |

Round robin, mean work, input regression, all semantic variants, and the oracle
had the same scheduling metrics. Least jobs was worse at 2,341.2 ms p95. This
run therefore does **not** meet the 10% improvement criterion. The narrow
service-time range left little routing headroom, and semantic class prediction
was weak. This is a negative feasibility result, not a general conclusion about
semantic scheduling.

An earlier 0.85-load diagnostic produced zero queue wait for every policy and
was excluded from the comparison because the scheduler had no opportunity to
affect latency.
