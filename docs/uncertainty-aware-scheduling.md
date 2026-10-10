# Uncertainty-aware scheduling

This stage tests whether the full semantic duration distribution improves
scheduling beyond its expected value. It does not claim that the synthetic
predictor represents Jev or production traffic.

Run the complete experiment matrix with:

```bash
uv run python examples/benchmark_uncertainty.py > uncertainty-results.json
```

The example varies arrival pattern, service-time distribution, offered load,
and prediction overhead across five fixed seeds. Each seed has a historical
training phase followed by a label-safe holdout phase. All policies replay the
same holdout requests and use the same workers within that run.

## Work estimates

`DurationPrediction` stores probabilities and a representative service time
for every duration class. It can transform that discrete distribution into:

- **Expected value**: probability-weighted mean and semantic control policy.
- **Quantile**: smallest class service time whose cumulative probability
  reaches `risk_quantile`.
- **CVaR**: mean service time within the upper tail above `risk_quantile`.
- **Expected/CVaR blend**: convex combination of the mean and CVaR.
- **Entropy fallback**: convex combination of the semantic mean and a
  conventional estimate, weighted by normalized predictive entropy.

For entropy fallback:

```text
weight = entropy(probabilities) / log(number_of_classes)
estimate = (1 - weight) * semantic_mean + weight * input_regression
```

A concentrated semantic distribution therefore receives more trust. A diffuse
distribution moves toward the non-semantic input-size regression estimate.

Every `SchedulingChoice` and `RoutingDecision` records `estimate_method`,
`risk_quantile`, `blend_weight`, and `fallback_service_ms` when applicable.
This makes the backlog estimate reproducible from the retained prediction.
Prediction overhead is included in dispatcher latency and end-to-end response
time.

## Reading the report

Each regime contains raw per-seed results and `policy_aggregates`. Compare:

- `semantic-expected` to the risk-aware policies to isolate the value of the
  uncertainty transformation.
- Risk-aware policies to `input-regression`, `mean-work`, `least-jobs`, and
  `round-robin` to test the main research question.
- `oracle-work` only as a diagnostic perfect-estimation comparator; it is not a
  deployable policy or a global scheduling optimum.

The primary fields are `response_p95_ms`, `queue_wait_p95_ms`,
`sla_violation_rate`, and `throughput_per_second`. The project threshold remains
a 10% p95 response-time or SLA improvement against the strongest conventional
policy in at least two high-variance, moderate-or-higher-load regimes, with no
more than a 2% throughput loss.

The aggregates are arithmetic means across deterministic seeds, not confidence
intervals or proof of statistical significance. Report neutral and negative
regimes alongside improvements. Before any production claim, replace synthetic
templates with held-out labeled traces and a real semantic predictor adapter.
