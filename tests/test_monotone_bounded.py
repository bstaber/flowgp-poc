"""Integration test: Section 6.1 of the paper (monotone bounded regression)
using a real GPyTorch model to produce m_{*|y} and K_{**|y}.

Ground truth: f(x) = 1/3 [arctan(20x - 10) - arctan(-10)], 7 noise-free obs,
monotonicity + 0 <= f(x) <= u(x) = 1/3 log(30x+1) + 0.1.
"""

import gpytorch
import pytest
import torch

from flowgp import BoundedMonotonic, from_gpytorch_posterior

torch.manual_seed(0)


def f_true(x):
    return (torch.arctan(20 * x - 10) - torch.arctan(torch.tensor(-10.0))) / 3.0


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(), batch_shape=torch.Size([])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_gp(train_x, train_y, grid_x):
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-12)
    )
    likelihood.noise = 1e-10
    model = ExactGPModel(train_x, train_y, likelihood)
    model.covar_module.base_kernel.lengthscale = 0.1
    model.covar_module.outputscale = 0.25
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(grid_x))
    return model, likelihood, pred


@pytest.fixture(scope="module")
def setup():
    # 7 observations at x_i = 0.1 + 1/(i+1), i=1..7 (Appendix G)
    train_x = torch.tensor([0.1 + 1.0 / (i + 1) for i in range(1, 8)])
    train_y = f_true(train_x)
    m = 64
    grid = torch.linspace(0.0, 1.0, m)
    model, likelihood, pred = fit_gp(train_x, train_y, grid)
    return model, likelihood, pred, grid, train_x, train_y


def test_unconditional_sampling_matches_gp_posterior(setup):
    """Sanity check: with NO conditioning, FLOWGP samples must match the
    GPyTorch predictive mean/covariance closely (the ODE recovers N(m, K))."""
    model, likelihood, pred, grid, train_x, train_y = setup
    m = grid.numel()

    class NoC:
        def log_likelihood(self, f0):
            # constant w.r.t. f0 but connected to the graph so autograd works
            return (f0 * 0.0).sum(dim=-1)

    sampler = from_gpytorch_posterior(pred, NoC(), num_steps=300, num_mc=1)
    n = 500
    s = sampler.sample(batch_shape=n)
    assert s.shape == (n, m)
    emp_mean = s.mean(dim=0)
    emp_cov = torch.cov(s.T)
    assert (emp_mean - pred.mean).abs().max() < 0.08
    # diagonal variances vs the dense covariance the sampler was built from;
    # skip near-zero-variance points (GPyTorch clamps negative variances)
    diag = pred.covariance_matrix.diagonal()
    mask = diag > 1e-4
    rel = ((emp_cov.diagonal()[mask] - diag[mask]) / diag[mask]).abs()
    assert rel.median() < 0.35


def test_monotone_bounded_regression(setup):
    """Reproduce the shape-constrained experiment: samples must satisfy the
    constraints and remain close to the truth."""
    model, likelihood, pred, grid, train_x, train_y = setup
    m = grid.numel()
    dx = 1.0 / m
    upper = torch.log(30 * grid + 1) / 3 + 0.1
    lower = torch.zeros(m)

    lik = BoundedMonotonic(lower, upper, dx=dx, v_mono=1e-4, v_bound=1e-5)
    sampler = from_gpytorch_posterior(pred, lik, num_steps=1000, num_mc=5)
    n = 50
    s = sampler.sample(batch_shape=n)

    assert s.shape == (n, m)
    assert torch.isfinite(s).all()
    # bound constraints
    assert s.min() > lower.min() - 1e-3
    assert (s - upper.unsqueeze(0)).max() < 1e-3
    # monotonicity (allow tiny numerical slack)
    diffs = s[:, 1:] - s[:, :-1]
    assert diffs.min() > -1e-3
    # globally faithful: constrained samples must beat the unconstrained GP
    # posterior mean, whose monotonicity violations are largest far from data
    truth = f_true(grid)
    mean = s.mean(dim=0)
    rmse_grid = ((mean - truth) ** 2).mean().sqrt()
    gp_rmse_grid = ((pred.mean - truth) ** 2).mean().sqrt()
    assert rmse_grid < 0.5 * gp_rmse_grid
    # reasonably close at the observed points too
    rmse_obs = ((mean[torch.searchsorted(grid, train_x)] - train_y) ** 2).mean().sqrt()
    assert rmse_obs < 0.15
    # and the constrained predictive mean tracks the truth's S-shape at midpoints
    for xq in (0.25, 0.5, 0.75):
        i = int(round(xq * (m - 1)))
        assert (mean[i] - truth[i]).abs() < 0.15
