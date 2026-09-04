"""Schedules and time discretisation (Appendix F.1 / F.2)."""

import torch


def beta_schedule(
    t: torch.Tensor, beta0: float = 1e-5, beta1: float = 10.0
) -> torch.Tensor:
    """Linear VP beta-schedule beta(t) = beta0 + (beta1 - beta0) * t."""
    return beta0 + (beta1 - beta0) * t


def alpha(t: torch.Tensor, beta0: float = 1e-5, beta1: float = 10.0) -> torch.Tensor:
    """Closed-form alpha for the linear schedule (Eq. 60)."""
    return torch.exp(-0.5 * beta0 * t - 0.25 * (beta1 - beta0) * t * t)


def log_snr(t: torch.Tensor, beta0: float = 1e-5, beta1: float = 10.0) -> torch.Tensor:
    """log SNR(t) = log alpha(t)^2 / (1 - alpha(t)^2), used to build the grid."""
    a2 = alpha(t, beta0, beta1) ** 2
    return torch.log(a2 / (1.0 - a2 + 1e-8))


def time_grid(
    num_steps: int,
    beta0: float = 1e-5,
    beta1: float = 10.0,
    tau_end: float = 1e-3,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Decreasing grid 1 = t0 > t1 > ... > tT ~ 0, uniform in log-SNR (Appendix F.2).

    Built by inverting log-SNR with bisection (SNR is monotone decreasing in t).
    """
    s_hi = log_snr(
        torch.tensor(1.0, device=device, dtype=dtype or torch.float64), beta0, beta1
    )
    s_lo = log_snr(
        torch.tensor(tau_end, device=device, dtype=dtype or torch.float64), beta0, beta1
    )
    # grid uniform in log-SNR from t=1 (low SNR) down to t~0 (high SNR)
    s = torch.linspace(
        float(s_hi), float(s_lo), num_steps + 1, device=device, dtype=dtype
    )
    # invert: given target log-SNR s, solve for t by bisection on [1e-6, 1]
    lo = torch.full_like(s, tau_end)
    hi = torch.ones_like(s)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        smid = log_snr(mid, beta0, beta1)
        gt = smid > s  # SNR decreases in t
        lo = torch.where(gt, mid, lo)
        hi = torch.where(gt, hi, mid)
    ts = 0.5 * (lo + hi)
    ts[0] = 1.0
    return ts  # decreasing from ~1 to ~0
