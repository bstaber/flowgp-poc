"""FLOWGP: guided probability-flow ODE sampling from a GP predictive distribution.

Implements Algorithm 1 of "Conditioning Gaussian Processes on Almost Anything"
(Moss, Astfalck, Cowperthwaite, Doumont, Willis, Hennig, Nemeth, Zammit-Mangion;
arXiv 2605.21041), in PyTorch, compatible with GPyTorch posteriors.

Sampling is performed in *whitened* coordinates f_hat = K^{-1/2} (f - m), under
which the Gaussian part of the velocity field vanishes and the ODE reduces to
the (smoothly-clipped) Monte-Carlo guidance term only:

    d f_hat_t / dt = -1/2 beta(t) * nabla_{f_hat} log p(C | f_hat_t, D)

with Euler integration from t=1 (white noise) to t=0 (posterior sample).
"""

from __future__ import annotations

import torch

from .schedules import alpha, beta_schedule, time_grid

__all__ = ["ConditioningLikelihood", "FlowGPSampler"]


class ConditioningLikelihood:
    """Point-wise evaluable likelihood p(C | f0) over a discretised function.

    Subclasses must implement ``log_likelihood(f0)`` -> scalar log p(C | f0)
    differentiable w.r.t. ``f0`` (shape ``(S, m)`` -> ``(S,)``).
    Non-differentiable constraints should use the smooth surrogates of
    Section 5.2 (probit for inequalities, narrow Gaussian for equalities).
    """

    def log_likelihood(self, f0: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FlowGPSampler:
    """Whitened probability-flow ODE sampler with Monte-Carlo guidance.

    Parameters
    ----------
    mean : Tensor (m,)
        Linear-Gaussian predictive mean m_{*|y} on the grid.
    cov : Tensor (m, m)
        Linear-Gaussian predictive covariance K_{**|y} on the grid.
    likelihood : ConditioningLikelihood
        Non-Gaussian conditioning statement C.
    num_steps : int
        Number of Euler ODE steps T (paper: 1000).
    num_mc : int
        Monte-Carlo samples S for the guidance estimate (paper: 5).
    beta0, beta1 : float
        Linear VP schedule endpoints (paper: 1e-5, 10.0).
    v_max : float
        Smooth-clipping norm threshold tau (paper: 1e2).
    tau_end : float
        Truncation: integrate down to t = tau_end rather than exactly 0, since
        the bridge factors are singular as t -> 0 (paper uses tau ~ 1e-3).
    """

    def __init__(
        self,
        mean: torch.Tensor,
        cov: torch.Tensor,
        likelihood: ConditioningLikelihood,
        num_steps: int = 1000,
        num_mc: int = 5,
        beta0: float = 1e-5,
        beta1: float = 10.0,
        v_max: float = 1e2,
        tau_end: float = 1e-3,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.device = torch.device(device) if device is not None else mean.device
        self.dtype = dtype
        self.likelihood = likelihood
        self.num_steps = int(num_steps)
        self.num_mc = int(num_mc)
        self.beta0, self.beta1 = float(beta0), float(beta1)
        self.v_max = float(v_max)
        self.tau_end = float(tau_end)

        m = mean.numel()
        self.mean = mean.reshape(-1).to(self.device, dtype)
        cov = cov.to(self.device, dtype)
        # Cholesky factorisation K = L L^T (one-off O(m^3)); W^{-1}(.) = L(.) + m.
        # Jitter scaled to the median diagonal for poorly conditioned posteriors;
        # escalate geometrically if factorisation fails.
        diag = torch.diag(cov)
        pos = diag[diag > 0]
        base = (
            torch.median(pos).clamp_min(1e-12)
            if pos.numel() > 0
            else torch.tensor(1e-12, device=cov.device, dtype=cov.dtype)
        )
        # add an absolute floor scaled to the overall magnitude of the matrix,
        # so tiny negative eigenvalues from lazy evaluation are also covered
        base = torch.maximum(base, cov.abs().max() * 1e-8)
        eye = torch.eye(m, device=self.device, dtype=dtype)
        last_err = None
        for factor in (1.0, 1e2, 1e4, 1e6, 1e8):
            jitter = base * factor
            try:
                self.chol = torch.linalg.cholesky(cov + jitter * eye)
                break
            except Exception as e:  # pragma: no cover - depends on input
                last_err = e
        else:
            raise last_err
        self.ts = time_grid(
            self.num_steps, self.beta0, self.beta1, device=self.device, dtype=dtype
        )

    # -- coordinate maps ---------------------------------------------------
    def unwhiten(self, f_hat: torch.Tensor) -> torch.Tensor:
        return f_hat.to(self.dtype) @ self.chol.T + self.mean  # (S, m) or (m,)

    # -- guidance (Appendix F.3) -------------------------------------------
    def _guidance(
        self, f_hat_t: torch.Tensor, t: float, eps: torch.Tensor
    ) -> torch.Tensor:
        """Monte-Carlo guidance score in whitened coordinates, shape (m,)."""
        with torch.enable_grad():
            a = alpha(
                torch.tensor(t, device=self.device, dtype=self.dtype),
                self.beta0,
                self.beta1,
            )
            # (i) S samples from f0_hat | f_hat_t, D ~ N(alpha f_hat_t, (1 - a^2) I)
            f0_hat = (
                a * f_hat_t + torch.sqrt((1.0 - a * a).clamp_min(0.0)) * eps
            )  # (S, m)
            f0_hat = f0_hat.requires_grad_(True)
            # unwhiten before evaluating the likelihood: f0 = L f0_hat + m
            f0 = self.unwhiten(f0_hat)
            # (ii)-(iii) log-likelihoods and scores via autograd THROUGH the unwhitening
            # map, so s(i) = nabla_{f0_hat} log p(C | f0), as in Algorithm 1 line 12
            ll = self.likelihood.log_likelihood(f0)  # (S,)
            (scores,) = torch.autograd.grad(ll.sum(), f0_hat)  # (S, m) whitened scores
            # (ii) self-normalised importance weights, numerically stable
            log_w = ll - torch.logsumexp(ll, dim=0)
            w = torch.exp(log_w)  # (S,)
            # (iv) guided velocity field in whitened space (Eq. 19 + Eq. 18):
            # nabla_{f_hat} log p(C | f_hat, D) ~= alpha * sum_i wbar_i s_i
            g = a * (w[:, None] * scores).sum(dim=0)  # (m,)
            # (v) full velocity: -1/2 beta * g, then smooth norm clipping
            b = beta_schedule(
                torch.tensor(t, device=self.device, dtype=self.dtype),
                self.beta0,
                self.beta1,
            )
            v = -0.5 * b * g
            norm = v.norm()
            scale = self.v_max * torch.tanh(norm / self.v_max) / (norm + 1e-8)
            return v * scale

    # -- main loop (Appendix F.4) -------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        batch_shape: torch.Size | tuple | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw sample(s) f0 ~ p(f0 | D, C).

        Returns a tensor of shape ``(*batch, m)`` (batch = () by default).
        """
        if batch_shape is None:
            batch_shape = ()
        elif isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        flat = 1
        for d in batch_shape:
            flat *= d
        m = self.mean.numel()
        out = []
        for _ in range(flat):
            f_hat = torch.randn(
                m, device=self.device, dtype=self.dtype, generator=generator
            )
            # fixed noise eps drawn once per trajectory (reparameterisation, F.3)
            eps = torch.randn(
                self.num_mc,
                m,
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            )
            for j in range(self.num_steps):
                t = self.ts[j].item()
                dt = self.ts[j] - self.ts[j + 1]
                g = self._guidance(f_hat, t, eps)
                f_hat = (
                    f_hat - dt * g
                )  # Euler step (velocity already has -1/2 beta factor)
            out.append(self.unwhiten(f_hat))
        return torch.stack(out).reshape(*batch_shape, m)


def from_gpytorch_posterior(
    posterior, likelihood: ConditioningLikelihood, **kwargs
) -> FlowGPSampler:
    """Build a FlowGPSampler from a GPyTorch posterior object.

    Works with ``model(X)`` results (gpytorch.distributions.MultivariateNormal /
    MultivariateNormalLazy): extracts m_{*|y} and K_{**|y} on the evaluated grid,
    then applies FLOWGP. Kernel hyperparameters should be fitted (e.g. by
    marginal-likelihood maximisation) on the linear-Gaussian component beforehand,
    as in the paper.
    """
    mvn = posterior
    if hasattr(mvn, "lazy_covariance_matrix"):
        cov = mvn.lazy_covariance_matrix.cpu().to_dense()
    else:
        cov = mvn.covariance_matrix
    mean = mvn.mean
    return FlowGPSampler(mean=mean, cov=cov, likelihood=likelihood, **kwargs)
