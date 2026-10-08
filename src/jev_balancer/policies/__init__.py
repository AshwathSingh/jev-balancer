"""Built-in scheduling policies used as experiment baselines."""

from jev_balancer.policies.baselines import (
    InputRegressionPolicy,
    LeastJobsPolicy,
    MeanWorkPolicy,
    OracleWorkPolicy,
    RoundRobinPolicy,
)
from jev_balancer.policies.semantic import SemanticWorkPolicy

__all__ = [
    "InputRegressionPolicy",
    "LeastJobsPolicy",
    "MeanWorkPolicy",
    "OracleWorkPolicy",
    "RoundRobinPolicy",
    "SemanticWorkPolicy",
]
