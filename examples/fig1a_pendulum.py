"""Figure 1a reproduction: FLOWGP enforcing the damped pendulum constraint
theta'' + sin(theta) + 0.2 theta' = 0, using the paper's *plotting* settings
(App. H.2): grid m = 250, 25 predictive samples. Observations follow PHYSS:
noise standard deviation 0.01 on the first 200 trajectory points.

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
import numpy as np
import torch

from flowgp import GaussianResidual, from_gpytorch_posterior, pendulum_residual

torch.set_default_dtype(torch.float64)
torch.manual_seed(7)

BETA_DAMP = 0.2
PHYSS_DT = 0.03
PHYSS_N = 1000
T_HORIZON = PHYSS_DT * (PHYSS_N - 1)
PHYSS_SIGMA = 0.15
SIGMA2 = PHYSS_SIGMA**2
SIGMA = PHYSS_SIGMA
M = 250  # plotting grid
N_SAMPLES = 25


def solve_pendulum():
    """Reproduce PHYSS's Euler-generated trajectory."""
    state = torch.tensor([3 * torch.pi / 4, 0.0])
    states = torch.zeros(PHYSS_N, 2)
    states[0] = state
    for i in range(PHYSS_N - 1):
        theta, theta_dot = state
        state = state + PHYSS_DT * torch.stack(
            [theta_dot, -torch.sin(theta) - BETA_DAMP * theta_dot]
        )
        states[i + 1] = state
    return torch.arange(PHYSS_N) * PHYSS_DT, states[:, 0]


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
        )


def fit_gp(train_x, train_y):
    """Multi-start ML fit, noise pinned near the known measurement variance."""
    best = None
    for seed in range(5):
        torch.manual_seed(seed)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = SIGMA2
        likelihood.raw_noise.requires_grad_(False)
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
    t_all, theta_all = solve_pendulum()

    # Match PHYSS: add noise to the full trajectory, then select 20 points from
    # the first 200 samples using the same NumPy random-number sequence.
    rng = np.random.RandomState(0)
    train_y_all = theta_all[:200].numpy() + PHYSS_SIGMA * rng.randn(200)
    rng.randn(PHYSS_N - 200)  # consume PHYSS's test-noise stream
    train_idx = rng.choice(np.arange(200), 20)
    train_t = t_all[train_idx]
    train_y = torch.from_numpy(train_y_all[train_idx])

    grid_t = torch.linspace(0, T_HORIZON, M)
    g_idx = torch.round(grid_t / T_HORIZON * (PHYSS_N - 1)).long()

    # ---- left panel: GP posterior samples (unconstrained, non-physical) ----
    X = (train_t / T_HORIZON).reshape(-1, 1)
    model, likelihood = fit_gp(X, train_y)
    model.eval()
    likelihood.eval()
    G = (grid_t / T_HORIZON).reshape(-1, 1)
    # with torch.no_grad(), gpytorch.settings.fast_pred_var():
    with torch.no_grad():
        pred = model(G)
    mean, cov = pred.mean.detach(), pred.covariance_matrix.detach()
    chol = torch.linalg.cholesky(cov + 1e-8 * torch.eye(M, dtype=torch.float64))
    z = torch.randn(N_SAMPLES, M, dtype=torch.float64)
    gp_samples = mean + z @ chol.T

    # ---- right panel: FLOWGP samples guided by the pendulum residual ----
    dt = T_HORIZON / (M - 1)
    physics = GaussianResidual(pendulum_residual(BETA_DAMP, dt), sigma=1e-10)
    sampler = from_gpytorch_posterior(
        pred, physics, num_steps=1000, num_mc=100, v_max=100.0
    )
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
