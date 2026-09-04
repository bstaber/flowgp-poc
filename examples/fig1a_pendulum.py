"""Figure 1a reproduction: FLOWGP enforcing the damped pendulum constraint
theta'' + sin(theta) + 0.2 theta' = 0, using the paper's *plotting* settings
(App. H.2): sigma^2 = 0.15^2, grid m = 250, 25 predictive samples.

Two panels like Fig. 1a: left = GP posterior samples conditioned on noisy
observations; right = FLOWGP samples additionally obeying the ODE.

The right panel uses the FlowGPSampler with a Gaussian physics likelihood over
the finite-difference pendulum residual. Each panel is annotated with its
predictive RMSE vs the truth, overall and in the no-data (t>6) extrapolation
region.
"""

import matplotlib

matplotlib.use("Agg")
import gpytorch
import matplotlib.pyplot as plt
import torch

from flowgp import GaussianResidual, from_gpytorch_posterior, pendulum_residual

torch.set_default_dtype(torch.float64)
torch.manual_seed(7)

BETA_DAMP = 0.2
T_HORIZON = 30.0
SIGMA2 = 0.15**2  # qualitative/plotting setting (App. H.2)
SIGMA = SIGMA2**0.5
M = 250  # plotting grid
N_SAMPLES = 25


def solve_pendulum(n_grid=4000, theta0=2.0):
    dt = T_HORIZON / (n_grid - 1)

    def deriv(y):
        return torch.stack([y[1], -torch.sin(y[0]) - BETA_DAMP * y[1]])

    ys = torch.zeros(n_grid, 2)
    ys[0] = torch.tensor([theta0, 0.0])
    for i in range(n_grid - 1):
        y = ys[i]
        k1 = deriv(y)
        k2 = deriv(y + dt / 2 * k1)
        k3 = deriv(y + dt / 2 * k2)
        k4 = deriv(y + dt * k3)
        ys[i + 1] = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return torch.linspace(0, T_HORIZON, n_grid), ys[:, 0]


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def fit_gp(train_x, train_y):
    """Multi-start ML fit, noise pinned near the known measurement variance."""
    best = None
    for seed in range(5):
        torch.manual_seed(seed)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.Interval(SIGMA2 / 2, SIGMA2 * 2)
        )
        likelihood.noise = SIGMA2
        model = ExactGPModel(train_x, train_y, likelihood)
        model.covar_module.base_kernel.lengthscale = 0.05 + 0.4 * torch.rand(1).item()
        model.covar_module.outputscale = 0.5 + 2.0 * torch.rand(1).item()
        model.mean_module.constant = train_y.mean()
        model.train()
        likelihood.train()
        opt = torch.optim.Adam(model.parameters(), lr=0.03)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(400):
            opt.zero_grad()
            loss = -mll(model(train_x), train_y)
            loss.backward()
            opt.step()
        ml = -loss.item()
        if best is None or ml > best[0]:
            best = (ml, model, likelihood)
    return best[1], best[2]


def main():
    _, theta_all = solve_pendulum()

    # irregular training observations with the plotting-level noise, restricted
    # to the early window t < 6 (as in the paper's Figure 1a): beyond that the
    # trajectory must be extrapolated by the physics constraint alone
    n_train = 12
    train_t = torch.sort(torch.rand(n_train) * 6.0)[0]
    idx = torch.round(train_t / T_HORIZON * 3999).long()
    train_y = theta_all[idx] + SIGMA * torch.randn(n_train)

    grid_t = torch.linspace(0, T_HORIZON, M)
    g_idx = torch.round(grid_t / T_HORIZON * 3999).long()

    # ---- left panel: GP posterior samples (unconstrained, non-physical) ----
    X = (train_t / T_HORIZON).reshape(-1, 1)
    model, likelihood = fit_gp(X, train_y)
    model.eval()
    likelihood.eval()
    G = (grid_t / T_HORIZON).reshape(-1, 1)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = model(G)
    mean, cov = pred.mean.detach(), pred.covariance_matrix.detach()
    chol = torch.linalg.cholesky(cov + 1e-8 * torch.eye(M, dtype=torch.float64))
    z = torch.randn(N_SAMPLES, M, dtype=torch.float64)
    gp_samples = mean + z @ chol.T

    # ---- right panel: FLOWGP samples guided by the pendulum residual ----
    dt = T_HORIZON / (M - 1)
    physics = GaussianResidual(pendulum_residual(BETA_DAMP, dt), sigma=1e-3)
    sampler = from_gpytorch_posterior(pred, physics, num_steps=1000, num_mc=5)
    samples = sampler.sample(batch_shape=N_SAMPLES)

    # ---- predictive RMSE (vs truth), overall and no-data (t>6) ----
    gp_mean = mean
    fw_mean = samples.mean(0)
    mask_nd = grid_t > 6.0

    def rmse(mn):
        return float(((mn - theta_all[g_idx]) ** 2).mean().sqrt())

    def rmse_nd(mn):
        return float(((mn[mask_nd] - theta_all[g_idx][mask_nd]) ** 2).mean().sqrt())

    gp_rmse_all, gp_rmse_nd = rmse(gp_mean), rmse_nd(gp_mean)
    fw_rmse_all, fw_rmse_nd = rmse(fw_mean), rmse_nd(fw_mean)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    axes[0].set_title("GP conditioned on observations", fontsize=10)
    for s in gp_samples:
        axes[0].plot(grid_t, s, lw=0.6, color="C0", alpha=0.6)
    axes[0].plot(grid_t, theta_all[g_idx], "k--", lw=1.2, label="true trajectory")
    axes[0].plot(train_t, train_y, "r.", ms=5, label="observations")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel(r"$\theta$")
    axes[0].legend(fontsize=8)
    axes[0].text(
        0.03,
        0.97,
        f"RMSE {gp_rmse_all:.2f}   (no-data {gp_rmse_nd:.2f})",
        transform=axes[0].transAxes,
        fontsize=9,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "fc": "w", "ec": "C0", "alpha": 0.9},
    )

    axes[1].set_title(
        r"FLOWGP: $\ddot{\theta}+\sin\theta+0.2\,\dot{\theta}=0$", fontsize=10
    )
    for s_ in samples:
        axes[1].plot(grid_t, s_, lw=0.9, color="C1", alpha=0.7)
    axes[1].plot(grid_t, theta_all[g_idx], "k--", lw=1.2)
    axes[1].plot(train_t, train_y, "r.", ms=5)
    axes[1].set_xlabel("t")
    axes[1].text(
        0.03,
        0.97,
        f"RMSE {fw_rmse_all:.2f}   (no-data {fw_rmse_nd:.2f})",
        transform=axes[1].transAxes,
        fontsize=9,
        va="top",
        color="k",
        bbox={"boxstyle": "round,pad=0.3", "fc": "w", "ec": "C1", "alpha": 0.9},
    )

    fig.tight_layout()
    fig.savefig("fig1a_pendulum.png", dpi=150)
    print("saved fig1a_pendulum.png")


if __name__ == "__main__":
    main()
