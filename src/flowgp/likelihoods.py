"""Example likelihoods for FLOWGP conditioning statements (Sections 5.2, G, H).

All operate on f0 of shape (S, m) — a batch of discretised function values —
and return log p(C | f0) of shape (S,), differentiable w.r.t. f0.
"""

import math

import torch

from .sampler import ConditioningLikelihood

_SQRT2 = math.sqrt(2.0)


def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _log_probit(z: torch.Tensor) -> torch.Tensor:
    """Numerically stable log Phi(z) for |z| large (float64-safe).

    For z < -10 uses the asymptotic tail log Phi(z) ~= -z^2/2 - log(-z) - log(sqrt(2 pi)).
    Gradients are exact through this expression (equivalent to Mills ratio phi/Phi ~= -z).
    """
    safe = z.clamp(min=-10.0)
    main = torch.log(_norm_cdf(safe).clamp_min(1e-300))
    tail = -0.5 * z * z - torch.log(-z.clamp(max=-1e-30)) - _LOG_SQRT_2PI
    return torch.where(z < -10.0, tail, main)


def _probit_loglik(margin: torch.Tensor, v: float) -> torch.Tensor:
    """log prod_i Phi(c_i / v) with stable tails (Eq. 63)."""
    return _log_probit(margin / v).sum(dim=-1)


class ProbitInequality(ConditioningLikelihood):
    """Probit relaxation for inequality constraints c_i >= 0 (Eq. 63).

    Parameters
    ----------
    margin : callable or None
        Maps f0 (S, m) to constraint margins (S, k); positive margin = satisfied.
        If None, the margins are f0 itself (i.e. f0 >= lower).
    v : float
        Sharpness parameter (paper: 1e-4 monotonicity, 1e-5 boundedness).
    """

    def __init__(self, margin=None, v: float = 1e-4):
        self.margin_fn = margin
        self.v = v

    def margins(self, f0: torch.Tensor) -> torch.Tensor:
        return f0 if self.margin_fn is None else self.margin_fn(f0)

    def log_likelihood(self, f0: torch.Tensor) -> torch.Tensor:
        c = self.margins(f0)
        return _probit_loglik(c, self.v)


class GaussianResidual(ConditioningLikelihood):
    """Gaussian likelihood over point-wise residuals (physics constraints, H.2-H.4).

    log p(C | f0) = -||r(f0)||^2 / (2 sigma_phys^2)  (+ const)
    """

    def __init__(self, residual_fn, sigma: float = 1e-5):
        self.residual_fn = residual_fn
        self.sigma = sigma

    def log_likelihood(self, f0: torch.Tensor) -> torch.Tensor:
        r = self.residual_fn(f0)  # (S, k)
        return -0.5 * (r * r).sum(dim=-1) / (self.sigma**2)


class BoundedMonotonic(ConditioningLikelihood):
    """The Section 6.1 benchmark: monotone increasing and bounded in [lower, upper].

    Monotonicity via forward finite differences (Eq. 62-63, v=1e-4), bounds via
    pointwise margins (v=1e-5). ``lower``/``upper`` may be scalars or tensors
    of shape (m,).
    """

    def __init__(
        self, lower, upper, dx: float, v_mono: float = 1e-4, v_bound: float = 1e-5
    ):
        self.v_mono, self.v_bound = v_mono, v_bound
        self.lower = lower
        self.upper = upper
        self.dx = dx

    def log_likelihood(self, f0: torch.Tensor) -> torch.Tensor:
        # monotonicity: forward differences >= 0
        diffs = (f0[..., 1:] - f0[..., :-1]) / self.dx  # (S, m-1)
        ll_mono = _probit_loglik(diffs, self.v_mono)
        # boundedness: margins (u - f) >= 0 and (f - l) >= 0
        margin = torch.cat([(self.upper - f0), (f0 - self.lower)], dim=-1)  # (S, 2m)
        ll_bound = _probit_loglik(margin, self.v_bound)
        return ll_mono + ll_bound


class EqualityConstraints(ConditioningLikelihood):
    """Equality constraints via narrow Gaussian likelihoods (Section 5.2)."""

    def __init__(self, residual_fn, sigma: float):
        self.residual_fn = residual_fn
        self.sigma = sigma

    def log_likelihood(self, f0: torch.Tensor) -> torch.Tensor:
        r = self.residual_fn(f0)
        return -0.5 * (r * r).sum(dim=-1) / (self.sigma**2)
