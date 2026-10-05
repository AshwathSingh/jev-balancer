"""Built-in scheduling policies used as experiment baselines."""

from jev_balancer.policies.baselines import (
    InputRegressionPolicy,
    LeastJobsPolicy,
    MeanWorkPolicy,
    OracleWorkPolicy,
    RoundRobinPolicy,
)

__all__ = [
    "InputRegressionPolicy",
    "LeastJobsPolicy",
    "MeanWorkPolicy",
    "OracleWorkPolicy",
    "RoundRobinPolicy",
]
