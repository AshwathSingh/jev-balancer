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

Run the current baseline experiment with:

```bash
uv run python examples/benchmark_baselines.py
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
