# Probabilistic semantic predictor

This stage tests whether request text helps estimate service time before adding
a live Jev dependency. Run the deterministic comparison with:

```bash
uv run python examples/benchmark_semantic.py
```

`SemanticNaiveBayesPredictor.fit()` learns three things from the chronological
training split:

- The prior probability of short, medium, and long requests.
- Word frequencies associated with each duration class.
- The mean observed service time for each class.

For a new `WorkItem`, it converts the payload into class probabilities and
calculates expected service time:

```text
E[service] = sum(P(class | payload) * mean_service_ms(class))
```

The predictor never receives evaluation `service_ms`. That hidden value is used
only afterward by the simulator and `evaluate_predictions()`.

## Controlled semantic signal

`semantic_weight` blends the text-derived probabilities with the training
class prior:

- `0.0` ignores the text and represents an uninformative semantic model.
- `0.5` retains half of the text signal.
- `1.0` uses the full fitted model.

This is a controlled accuracy sweep, not a claim that a specific production
model has that quality.

## Evaluation

`evaluate_predictions()` reports class accuracy, multiclass Brier score,
expected calibration error, mean absolute duration error, and simulated
prediction overhead. `SemanticWorkPolicy` includes that overhead in scheduling
latency and end-to-end response time.

The semantic policy chooses the worker with the least estimated backlog. Its
prediction and probability distribution are retained in every routing decision
for later inspection.
