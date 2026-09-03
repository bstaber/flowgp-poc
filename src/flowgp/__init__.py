"""FLOWGP package: Conditioning GPs on almost anything (arXiv 2605.21041)."""

from .likelihoods import BoundedMonotonic, EqualityConstraints, ProbitInequality
from .physics import (
    GaussianResidual,
    ProductLikelihood,
    allen_cahn_residual,
    burgers_residual,
    dirichlet_rows_residual,
    evaluate_samples_at_test_points,
    first_derivative,
    pendulum_residual,
    rmse_nlpd,
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
    "allen_cahn_residual",
    "alpha",
    "beta_schedule",
    "burgers_residual",
    "dirichlet_rows_residual",
    "evaluate_samples_at_test_points",
    "first_derivative",
    "from_gpytorch_posterior",
    "pendulum_residual",
    "rmse_nlpd",
    "second_derivative",
    "time_grid",
]
