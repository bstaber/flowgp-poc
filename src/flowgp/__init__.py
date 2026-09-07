"""FLOWGP package: Conditioning GPs on almost anything (arXiv 2605.21041)."""

from .likelihoods import BoundedMonotonic, EqualityConstraints, ProbitInequality
from .physics import (
    GaussianResidual,
    ProductLikelihood,
    first_derivative,
    pendulum_residual,
    second_derivative,
)
from .sampler import ConditioningLikelihood, FlowGPSampler, from_gpytorch_posterior
from .schedules import alpha, beta_schedule, time_grid

__all__ = [
    "BoundedMonotonic",
    "ConditioningLikelihood",
    "EqualityConstraints",
    "FlowGPSampler",
    "GaussianResidual",
    "ProbitInequality",
    "ProductLikelihood",
    "alpha",
    "beta_schedule",
    "first_derivative",
    "from_gpytorch_posterior",
    "pendulum_residual",
    "second_derivative",
    "time_grid",
]
