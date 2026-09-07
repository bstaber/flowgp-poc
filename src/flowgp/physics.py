"""Physics-constrained conditioning: PDE residuals via finite differences (App. H).

Residual closures map f0 of shape (S, m) [1D] or (S, H*W) [2D, row-major (H, W)]
to residual tensors; combined with GaussianResidual they give log p(C | f0).

Boundary/edge rows are excluded from residuals where the paper excludes them
(pendulum: j = 2..m-1 interior; Allen-Cahn/Burgers: interior points, boundary
conditions handled by separate row-pinning residuals).
"""

from __future__ import annotations

import torch

from .sampler import ConditioningLikelihood

__all__ = [
    "GaussianResidual",
    "ProductLikelihood",
    "first_derivative",
    "pendulum_residual",
    "second_derivative",
]


class GaussianResidual(ConditioningLikelihood):
    """Gaussian likelihood over point-wise residuals: log p = -||r||^2/(2 sigma^2)."""

    def __init__(self, residual_fn, sigma: float):
        self.residual_fn = residual_fn
        self.sigma = sigma

    def log_likelihood(self, f0: torch.Tensor) -> torch.Tensor:
        r = self.residual_fn(f0)
        return -0.5 * (r * r).sum(dim=-1) / (self.sigma**2)


class ProductLikelihood(ConditioningLikelihood):
    """Product of independent conditionings p(C1|f0) p(C2|f0) ..."""

    def __init__(self, *likelihoods):
        self.likelihoods = likelihoods

    def log_likelihood(self, f0: torch.Tensor) -> torch.Tensor:
        ll = 0.0
        for lik in self.likelihoods:
            ll = ll + lik.log_likelihood(f0)
        return ll


def first_derivative(f0: torch.Tensor, axis: int, h: float) -> torch.Tensor:
    """Central first derivative at interior points along `axis` (keeps that axis)."""
    sl1 = [slice(None)] * f0.dim()
    sl1[axis] = slice(2, None)
    sl0 = [slice(None)] * f0.dim()
    sl0[axis] = slice(None, -2)
    return (f0[tuple(sl1)] - f0[tuple(sl0)]) / (2 * h)


def second_derivative(f0: torch.Tensor, axis: int, h: float) -> torch.Tensor:
    sl1 = [slice(None)] * f0.dim()
    sl1[axis] = slice(2, None)
    sl0 = [slice(None)] * f0.dim()
    sl0[axis] = slice(None, -2)
    slm = [slice(None)] * f0.dim()
    slm[axis] = slice(1, -1)
    return (f0[tuple(sl1)] - 2 * f0[tuple(slm)] + f0[tuple(sl0)]) / (h * h)


def _trim(f0: torch.Tensor, axis: int, n: int = 1) -> torch.Tensor:
    """Drop n slices at both ends along `axis` (to align with FD interior)."""
    sl = [slice(None)] * f0.dim()
    sl[axis] = slice(n, -n) if n > 0 else slice(None)
    return f0[tuple(sl)]


def pendulum_residual(beta_damp: float, dt: float):
    """theta'' + sin(theta) + beta theta' = 0 at all interior grid points."""

    def residual(f0: torch.Tensor) -> torch.Tensor:
        fpp = second_derivative(f0, -1, dt)  # centres 1..m-2 (0-based)
        fp = first_derivative(f0, -1, dt)
        f = f0[..., 1:-1]

        return fpp + torch.sin(f) + beta_damp * fp

    return residual
