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
    "allen_cahn_residual",
    "burgers_residual",
    "dirichlet_rows_residual",
    "evaluate_samples_at_test_points",
    "first_derivative",
    "pendulum_residual",
    "rmse_nlpd",
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


# ---------------------------------------------------------------------------
# 1D: damped pendulum (H.2, Eq. 69/71)
# ---------------------------------------------------------------------------


def pendulum_residual(beta_damp: float, dt: float):
    """theta'' + sin(theta) + beta theta' = 0 at interior points.

    Residual evaluated at grid indices j = 1..m-2 (interior); the FD stencil at
    the very first interior index j=1 is dropped as in the paper (j = 2..m-1).
    """

    def residual(f0: torch.Tensor) -> torch.Tensor:
        fpp = second_derivative(f0, -1, dt)  # (S, m-2) at j=1..m-2
        fp = first_derivative(f0, -1, dt)  # (S, m-2)
        f = _trim(f0, -1, 1)  # (S, m-2)
        r = fpp + torch.sin(f) + beta_damp * fp
        return r[:, 1:]  # drop j=1 -> evaluate at j=2..m-2 (paper: j=2..m-1)

    return residual


# ---------------------------------------------------------------------------
# 2D fields, flattened (H*W). axis -2 = x (rows, H), axis -1 = t (cols, W)
# ---------------------------------------------------------------------------


def allen_cahn_residual(eps: float, dx: float, dt: float, hw):
    """du/dt = eps d2u/dx2 + 5u - 5u^3 (Eq. 72), interior in both axes.

    f0 arrives as (S, H*W), row-major (H, W) with hw = (H, W); reshaped here.
    """
    H, W = hw

    def residual(f0: torch.Tensor) -> torch.Tensor:
        f = f0.reshape(-1, H, W)
        ut = first_derivative(f, -1, dt)[..., 1:-1, :]  # (S, H-2, W-2)
        u_c = f[..., 1:-1, 1:-1]
        uxx = second_derivative(f, -2, dx)[..., :, 1:-1]
        r = ut - (eps * uxx + 5.0 * u_c - 5.0 * u_c**3)
        return r.reshape(-1, (H - 2) * (W - 2))

    return residual


def burgers_residual(nu: float, dx: float, dt: float, hw, eps4: float = 0.0):
    """u_t + u u_x = nu u_xx (Eq. 74/80), interior in both axes.

    f0 arrives as (S, H*W), row-major (H, W) with hw = (H, W); reshaped here.
    ``eps4`` adds a 4th-order artificial-viscosity term -eps4 * u_xxxx matching a
    stabilised discretisation used to generate the reference solution (needed on
    coarse grids where near-shock solutions are otherwise unresolvable).
    """
    H, W = hw

    def residual(f0: torch.Tensor) -> torch.Tensor:
        f = f0.reshape(-1, H, W)
        ut = first_derivative(f, -1, dt)[..., 1:-1, :]
        ux = first_derivative(f, -2, dx)[..., :, 1:-1]
        u_c = f[..., 1:-1, 1:-1]
        uxx = second_derivative(f, -2, dx)[..., :, 1:-1]
        r = ut + u_c * ux - nu * uxx
        if eps4 > 0.0:
            # non-wrapped 4th-order stencil: contributions vanish within one
            # point of the boundary, matching the stabilised truth generator
            uxxxx = torch.zeros_like(f)
            uxxxx[..., 2:-2] = (
                f[..., 4:]
                - 4 * f[..., 3:-1]
                + 6 * f[..., 2:-2]
                - 4 * f[..., 1:-3]
                + f[..., :-4]
            ) / dx**4
            r = r - eps4 * uxxxx[..., 1:-1, 1:-1]
        return r.reshape(-1, (H - 2) * (W - 2))

    return residual


def dirichlet_rows_residual(hw):
    """Residual = boundary row values (pinned to zero); f0 (S, H*W) -> (S, 2W)."""
    H, W = hw

    def residual(f0: torch.Tensor) -> torch.Tensor:
        f = f0.reshape(-1, H, W)
        return torch.cat([f[:, 0, :], f[:, -1, :]], dim=-1)

    return residual


# ---------------------------------------------------------------------------
# Evaluation (App. H.1, Eq. 65-68)
# ---------------------------------------------------------------------------


def evaluate_samples_at_test_points(
    samples, test_points, mean, cov, mu_star_fn, k_cross_fn
):
    """Extend grid samples to arbitrary test locations via kernel smoothing (Eq. 65).

    f_hat(x_new) = mu_{*|y}(x_new) + k(x_new, X*) K_{**|y}^{-1} (f - m_{*|y})

    samples   : (N, m) FLOWGP samples on the grid
    test_points : (P, d) input locations
    mean      : (m,) posterior mean on the grid
    cov       : (m, m) posterior covariance on the grid
    mu_star_fn : callable (P, d) -> (P,) GP posterior mean at test points
    k_cross_fn : callable (P, d) -> (P, m) posterior cross-covariances
        k(x_new, X*) between test points and grid points.

    Returns (P, N) evaluations (one column per sample).
    """
    resid = samples.T - mean[:, None]  # (m, N)
    chol = torch.linalg.cholesky(cov)
    w = torch.cholesky_solve(resid, chol)  # (m, N)
    k_cross = k_cross_fn(test_points)  # (P, m)
    mu_new = mu_star_fn(test_points)  # (P,)
    return mu_new[:, None] + k_cross @ w  # (P, N)


def rmse_nlpd(pred_mean, pred_var, y_true, noise_var=0.0):
    """RMSE and NLPD (Eq. 66-68). pred_var should already include noise variance."""
    pred_var = pred_var + noise_var
    rmse = ((pred_mean - y_true) ** 2).mean().sqrt()
    nlpd = (
        0.5 * torch.log(2 * torch.pi * pred_var)
        + (y_true - pred_mean) ** 2 / (2 * pred_var)
    ).mean()
    return rmse.item(), nlpd.item()
