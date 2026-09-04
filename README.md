# flowgp

> **⚠️ Not an official implementation.** This is an independent, community
> re-implementation of *"Conditioning Gaussian Processes on Almost Anything"*
> (Moss, Astfalck, Cowperthwaite, Doumont, Willis, Hennig, Nemeth,
> Zammit-Mangion; arXiv 2605.21041), written from scratch from the paper's text.
> It is **not** the authors' code, has **not** been reviewed or endorsed by them,
> and the authors state their own code will be released separately. Treat
> anything here as unverified third-party work.

A PyTorch implementation of **FLOWGP** — sampling from a GP predictive
distribution under **arbitrary non-linear, non-Gaussian conditioning** (shape
constraints, physics/ODE-PDE residuals, semantic/LLM-derived likelihoods, …) by
integrating a whitened probability-flow ODE with Monte-Carlo guidance. Only
point-wise evaluations of the conditioning likelihood `p(C | f0)` are required —
no bespoke kernels or per-constraint samplers.

## TL;DR — how the method works

**Goal.** You want functions `f` that (a) fit some noisy observations `D`
through a GP, *and* (b) satisfy some extra, possibly non-linear condition `C`
(an ODE/PDE, a monotonicity/shape rule, a semantic constraint from language).
The GP posterior `p(f | D)` gives (a) but not (b); plain constrained optimisation
gives one function satisfying (b) but no uncertainty. FLOWGP samples from the
joint posterior `p(f | D, C) ∝ p(f | D) · p(C | f)`, where `p(C | f)` scores how
well `f` satisfies the condition (e.g. how small its ODE residual is).

**How.** Drawing from that product directly is intractable (the GP is
huge-dimensional), so FLOWGP does it along a diffusion path:

- whiten the GP so it becomes a standard normal, then run a **probability-flow
  ODE** from pure white noise at `t=1` down to the posterior sample at `t=0`;
- at each step, steer toward condition-satisfying functions with **Monte-Carlo
  guidance**: draw a few candidate functions, score them with `p(C | f)`, and
  follow the (importance-weighted) gradient of that score;
- only point-wise scores `p(C | f)` are ever used.

The result is a genuine posterior band of functions that fit the data *and*
obey the condition. Cost is roughly `O(m^3)` in the grid size (one Cholesky
factorisation), so it targets moderate-dimensional problems.

## Compatibility with GPyTorch

The sampler needs only the linear-Gaussian predictive `(m_{*|y}, K_{**|y})` on a
discretisation grid — exactly what any GPyTorch posterior provides:

```python
import gpytorch
from flowgp import from_gpytorch_posterior, BoundedMonotonic

model.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred = model(grid)                      # any GPyTorch posterior

lik = BoundedMonotonic(lower, upper, dx=1.0 / m)   # conditioning statement
sampler = from_gpytorch_posterior(pred, lik, num_steps=1000, num_mc=5)
samples = sampler.sample(batch_shape=100)          # f0 ~ p(f0 | D, C), (100, m)
```

Kernel hyperparameters should be fitted first by marginal-likelihood maximisation
on the linear-Gaussian component `p(f0 | D)` only (as in the paper); the sampler
then treats the posterior as a fixed Gaussian.

## Reproducing Figure 1a — the pendulum

This is the only example currently shipped.

```bash
uv run python examples/fig1a_pendulum.py
```

produces `fig1a_pendulum.png`: two panels like the paper's Fig. 1(a), conditioning
a GP on noisy observations of a damped pendulum

$$\ddot{\theta} + \sin\theta + 0.2\,\dot{\theta} = 0, \qquad t \in [0, 30],$$

with the paper's plotting settings (σ²=0.15², grid m=250, 25 samples).

- **Left panel** — the unconstrained GP posterior: smooth samples that fit the
  data but violate the physics, with a wide spread that wanders off as t grows.
- **Right panel** — **FLOWGP**: samples additionally constrained to obey the
  ODE. Each panel prints its RMSE vs the truth, overall and in the no-data
  (t>6) extrapolation region; the physics constraint typically cuts the
  extrapolation RMSE by an order of magnitude.

The right panel uses the implemented `FlowGPSampler` with a `GaussianResidual`
physics likelihood built from the finite-difference pendulum residual. This is
the same whitened probability-flow ODE and Monte-Carlo guidance path exposed by
the public API; the example does not use a separate on-manifold sampler.

## What is implemented

- `flowgp.sampler.FlowGPSampler` — Algorithm 1: whitened ODE, log-SNR time grid,
  Euler integration, self-normalised importance weights (log-sum-exp), smooth
  norm clipping (`v_max`), reparameterised MC noise.
- `flowgp.likelihoods`:
  - `ProbitInequality` — inequality constraints `margin(f0) >= 0` with
    float64-stable log-CDF tails;
  - `BoundedMonotonic` — monotonicity via forward differences + pointwise bounds;
  - `GaussianResidual` / `EqualityConstraints` — Gaussian likelihoods over
    point-wise residuals (non-linear PDE constraints);
  - subclass `ConditioningLikelihood` for anything else (incl. LLM scores).
- Custom conditioning: implement `log_likelihood(f0: (S, m)) -> (S,)`
  differentiable w.r.t. `f0`; guidance scores flow through the unwhitening map
  automatically.

## Tests

`uv run pytest` — includes a monotone-bounded regression run through a real
GPyTorch `ExactGP`, plus an exactness check that unconditional FLOWGP samples
reproduce the GPyTorch predictive distribution.

## Limitations

- `O(m^3)` scaling with the grid (from the Cholesky factorisation of `K_{**|y}`).
- Importance-weight collapse is possible in very high dimensions.
- Hyperparameters are fitted on the linear-Gaussian part only.
- The stochastic MC-guidance path (`FlowGPSampler`) is most reliable for smooth
  constraints; sharp PDE residuals may require tuning the residual scale and
  integration settings.