# jev-balancer

An experiment testing whether probabilistic semantic information can improve
workload scheduling over conventional load-based heuristics.

The project starts with a deterministic simulator and reproducible benchmarks
before introducing live Jev calls or an HTTP gateway.

See the [v0.1 experiment design](docs/v0.1-experiment-design.md) for the
research question, scheduler policies, metrics, and acceptance criteria.
The [simulator guide](docs/simulator.md) documents the execution model and
policy interface.
The [baseline guide](docs/baselines.md) explains the conventional comparison
policies and metrics.
The [workload guide](docs/workloads.md) covers deterministic generation, JSONL
replay, and chronological train/evaluation splits.
The [semantic predictor guide](docs/semantic-predictor.md) explains training,
probability quality, overhead, and the semantic scheduling policy.
The [uncertainty-aware scheduling guide](docs/uncertainty-aware-scheduling.md)
documents quantile, CVaR, entropy fallback, and the multi-seed experiment.

Run the current baseline experiment with:

```bash
uv run python examples/benchmark_baselines.py
```

Run the semantic signal-strength comparison with:

```bash
uv run python examples/benchmark_semantic.py
```

Run the uncertainty-aware experiment matrix with:

```bash
uv run python examples/benchmark_uncertainty.py > uncertainty-results.json
```

## Development

The package requires Python 3.10 or newer and uses
[uv](https://docs.astral.sh/uv/) for reproducible development environments.

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run mypy src tests examples
```
