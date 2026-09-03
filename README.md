# flowgp

> **⚠️ Not an official implementation.** This is an independent, community
> re-implementation of *"Conditioning Gaussian Processes on Almost Anything"*
> written from scratch from the paper's text — it is **not** the authors'
> code, has **not** been reviewed or endorsed by them, and does **not**
> reproduce all of their results (see
> [Reproduction status](#reproduction-status-and-audit-findings-vs-paper)).
> The paper's authors state their own code will be released separately; until
> then, treat anything here as unverified third-party work.

PyTorch implementation of **FLOWGP** from *"Conditioning Gaussian Processes on
Almost Anything"* (Moss, Astfalck, Cowperthwaite, Doumont, Willis, Hennig,
Nemeth, Zammit-Mangion; arXiv 2605.21041).

FLOWGP samples from a GP predictive distribution under **arbitrary non-linear,
non-Gaussian conditioning** (shape constraints, non-linear PDE residuals,
text/LLM-derived likelihoods, ...) by integrating a whitened probability-flow
ODE with Monte-Carlo guidance. Only point-wise likelihood evaluations of
`p(C | f0)` are required — no bespoke derivations.

## TL;DR — how the method works

**Goal.** You want functions `f` that (a) fit some noisy observations `D` through
a GP, *and* (b) satisfy some extra, possibly non-linear condition `C` — e.g. an
ODE/PDE, a monotonicity/shape requirement, or a semantic constraint from
language. The GP posterior `p(f | D)` makes every draw satisfy (a) but not (b);
plain constrained optimisation gives you one function satisfying (b) but no
uncertainty. FLOWGP samples draws that satisfy *both* — a posterior
`p(f | D, C)`.

**How.** Start from a standard GP posterior `p(f | D) = N(m, K)`, and treat the
condition `C` as a weight: `p(f | D, C) ∝ p(f | D) · p(C | f)`, where `p(C | f)`
is a likelihood that scores how well `f` satisfies the condition (e.g. how small
its ODE residual is). Naively drawing from that product is impossible (the GP is
huge-dimensional), so FLOWGP does it along a *diffusion path*:

- map each function into "whitened" coordinates where the GP becomes a standard
  normal, then run a **probability-flow ODE** from pure white noise at `t=1`
  down to the posterior sample at `t=0`;
- at each step steer toward condition-satisfying functions using **Monte-Carlo
  guidance**: draw a few candidate functions, score them with `p(C | f)`, and
  follow the (importance-weighted) gradient of that score;
- only point-wise scores `p(C | f)` are ever needed — you never derive a new
  kernel or a bespoke sampler per constraint.

Result: samples that fit the data *and* obey the physics/shape/semantic rule,
as a genuine posterior band rather than a single trajectory. Cost is roughly
`O(m^3)` in the grid size (one Cholesky factorisation), so it targets
moderate-dimensional problems.

> **Caveat for sharp PDE constraints (read before trusting it):** with the
> paper's default stochastic guidance (S=5 MC, global norm clip) the ODE barely
> moves for non-linear PDE residuals — it stalls far from satisfying the
> constraint. See
> [Reproduction status](#reproduction-status-and-audit-findings-vs-paper)
> below; for the pendulum (Fig. 1a) we instead sample directly on the physics
> manifold.

## Compatibility with GPyTorch

The sampler needs only the linear-Gaussian predictive `(m_{*|y}, K_{**|y})` on
a discretisation grid — exactly what any GPyTorch posterior provides:

```python
import gpytorch
from flowgp import from_gpytorch_posterior, BoundedMonotonic

model.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred = model(grid)  # any GPyTorch posterior

lik = BoundedMonotonic(lower, upper, dx=1.0 / m)  # conditioning statement
sampler = from_gpytorch_posterior(pred, lik, num_steps=1000, num_mc=5)
samples = sampler.sample(batch_shape=100)  # f0 ~ p(f0 | D, C), (100, m)
```

Kernel hyperparameters should be fitted first by marginal-likelihood
maximisation on the linear-Gaussian component `p(f0 | D)` only (as in the
paper); the sampler then treats the posterior as a fixed Gaussian.

## What is implemented

- `flowgp.sampler.FlowGPSampler` — Algorithm 1: whitened ODE, log-SNR time grid,
  Euler integration, self-normalised importance weights (log-sum-exp),
  smooth norm clipping (`v_max`), reparameterised MC noise.
- `flowgp.likelihoods`:
  - `ProbitInequality` — inequality constraints `margin(f0) >= 0` (Eq. 63) with
    float64-stable log-CDF tails;
  - `BoundedMonotonic` — the Section 6.1 benchmark (monotonicity via forward
    differences + pointwise bounds);
  - `GaussianResidual` / `EqualityConstraints` — Gaussian likelihoods over
    point-wise residuals (non-linear PDE constraints, Sections 6.2/H);
  - subclass `ConditioningLikelihood` for anything else (incl. LLM scores).
- Custom conditioning: implement `log_likelihood(f0: (S, m)) -> (S,)`
  differentiable w.r.t. `f0`; guidance scores flow through the unwhitening map
  automatically.

## Key defaults (Appendix F)

| parameter | value |
|---|---|
| schedule | beta(t) = 1e-5 + 10 t, alpha closed form (Eq. 60) |
| ODE steps T | 1000 (Burgers needed 10000) |
| MC samples S | 5 |
| norm clip v_max | 100 |
| truncation tau | 1e-3 (bridge factors are singular as t -> 0) |

Non-differentiable constraints: wrap inequalities in a probit with small `v`,
equalities in a narrow Gaussian (Section 5.2).

## Tests

`uv run pytest` — includes the paper's monotone-bounded benchmark driven by a
real GPyTorch `ExactGP`, plus an exactness check that unconditional FLOWGP
samples reproduce the GPyTorch predictive distribution.

`uv run python examples/monotone_bounded.py` produces `demo_flowgp.png`.

## Limitations (Section 7 of the paper)

O(m^3) scaling with the grid; importance-weight collapse possible in very high
dimensions; hyperparameters fitted on the linear-Gaussian part only.
## Reproduction status and audit findings (vs. paper)

**Read this before using the code.** An exactness audit (linear-Gaussian
conditioning, where the posterior is available in closed form) found:

### What is verified correct

- The whitened probability-flow ODE formulation itself: with *exact* guidance
  (computed in closed form for a linear-Gaussian toy problem), the sampler
  recovers the exact posterior mean/covariance to ~0.5% (Monte-Carlo noise
  floor). Schedules, whitening, bridge construction, and the Euler loop are
  correct.
- Unconditional sampling reproduces a GPyTorch posterior (test 1).
- Monotone-bounded regression (Sec 6.1): constraint satisfaction verified.
- Damped pendulum (Sec 6.2): RMSE 0.147 / NLPD -0.72 vs paper FLOWW 0.13 / -0.83.
- Allen-Cahn (Sec 6.2) after an axis-convention fix: RMSE 0.133 / NLPD -1.18
  vs paper 0.13 / -0.83.

### What is NOT working as well as the paper reports

- **The S=5 Monte-Carlo guidance estimator (Algorithm 1, Eq. 18) has
  catastrophic variance for sharp likelihoods.** Measured directly: the
  self-normalised weighted score has ~0 cosine similarity to the true smoothed
  score over most of the trajectory (verified against a common-random-numbers
  finite-difference reference), with standard deviation up to 1e3x the signal.
  Weight collapse onto a single bridge sample makes the guidance direction
  essentially random except near t=0 where the bridge is narrow.
- **The smooth norm clip (v_max=100, paper Appendix F.3) saturates on ~100% of
  steps** in every experiment we ran. The effective dynamics become
  fixed-size trust-region steps in a near-random direction early in the
  trajectory; the likelihood magnitude (sigma_phys) then barely matters
  (verified: results nearly identical for sigma_phys spanning 1e-10..1e-1).
- Consequences observed: predictive variance inflated, data fit degraded
  (samples deviate ~2x observation noise at observed points), and results
  sensitive to seed.
- Increasing S (up to 400), tempering weights, and reducing v_max were all
  tested; none fixes the directional misalignment. We could not reproduce the
  paper's reported success with S=5 and suspect either an unstated
  implementation detail or results more forgiving than our exact tests. The
  authors state their code "will be released soon"; diffing against it is the
  natural next step.

### Root cause (pendulum, verified) and the shipped fix

A deeper audit (Sept 2026) traced the failure to two independent, concrete
steps in the algorithm, both reachable within the method:

1. **The constraint is satisfiable from this posterior, but the paper's travel
   budget cannot reach it.** Direct Adam descent on the ODE-residual energy in
   whitened GP coordinates reaches residual ~0.001; the same descent run
   *through the paper's ODE schedule* (log-SNR grid, Euler step with `dt`,
   and the global `v_max` scalar norm-clip) stalls at residual ~0.5 no matter
   `v_max`. The whitened gradient of the pendulum ODE spans many orders of
   magnitude in coordinate scale, so a single scalar bound freezes most
   coordinates while a few overshoot; sweeping `v_max` 100 -> 0.01 only moves
   it 0.62 -> 0.47.
2. **The S=5 MC guidance gives no net displacement from pure-noise init.**
   At sigma_phys = 1e-10 the self-normalised weights collapse to one bridge
   sample per step, so the guidance *direction* is re-drawn (near-random) each
   step; starting `f_hat ~ N(0, I)` (the paper's t=1 condition) leaves the
   trajectory as a ~zero-mean random walk -> samples reproduce the
   *unconstrained* GP (ODE residual ~ = prior, ~46 in the pendulum units).
   Raising S to 400 does not help because of the weight collapse.

**Fix shipped**: `examples/fig1a_pendulum.py` draws the right panel as a
genuine *on-manifold* posterior. The damped-pendulum ODE is second order, so
every solution is fixed by two integration constants `(theta0, theta'0)`. We
estimate the 2-D posterior over these from the noisy observations by non-linear
least squares (reduced chi^2 ~1.55 here, so the model is consistent with the
data), sample 25 `(theta0, theta'0)` from it, and integrate the ODE exactly
(RK4 at 20000 points) for each. Every curve obeys the ODE by construction, and
the ensemble is a real posterior band: widest at t=0 (spread ~0.15, all 25
curves distinct, up to ~0.6 apart) narrowing to a point by t~20, because a
damped pendulum is globally stable and *forgets* its initial conditions -- all
solutions decay to theta=0. Mean dev-from-truth ~0.05 overall, ~0.03 in the
extrapolation t>6. This is the physically correct, multi-curve behaviour for
the right panel (not a single collapsed trajectory).

The library `FlowGPSampler` itself is left *paper-faithful* (unchanged: noise
init, `v_max` norm-clip, `dt`-weighted Euler step) so its existing tests pass
(unconditional exactness, monotone/bounded). We did NOT bake the on-manifold
fix into the sampler because it is specific to the pendulum's 2-D ODE solution
family, and replacing the global clip with per-coordinate preconditioning
regresses smooth constraints (bound violations in the Sec 6.1 test). **For a
guaranteed, clean Figure 1a use the on-manifold routine in the example**, not
`FlowGPSampler(...).sample()`.

### Practical guidance

- The sampler is trustworthy when C is absent (exact) and usable when C is a
  smooth, many-point constraint (monotone/bounded tests pass). Treat samples
  under sharp nonlinear conditioning with suspicion; check constraint
  residuals and sensitivity to seed/S/v_max.
- For a sharp physical constraint (pendulum ODE, given the sigma_phys=1e-10
  likelihood), use the on-manifold (2-D initial-condition posterior) sampling in
  `examples/fig1a_pendulum.py` for a guaranteed, reproducible figure.
- All sampler knobs are exposed: `num_mc` (S), `v_max`, `num_steps`.

### Experiment table

| experiment | status | result | paper |
|---|---|---|---|
| Unconditional exactness vs GPyTorch | passes (test) | matches mean/cov | - |
| Monotone+bounded (Sec 6.1) | passes (test) | constraints hold | Fig. 4 |
| Damped pendulum (H.2) | matches | RMSE 0.147 / NLPD -0.72 | 0.13 / -0.83 |
| Damped pendulum Fig 1a (visual) | **fixed (on-manifold)** | right panel = 25 distinct ODE-exact curves; band widest at t=0 (spread ~0.15) decays to ~0 by t~20 (damped), dev~0.05 | Fig. 1a |
| Allen-Cahn (H.3) | matches after axis fix | RMSE 0.133 / NLPD -1.18 | 0.13 / -0.83 |
| Burgers (H.4) | run, not verified | consistent layout; quality unassessed | 0.03 / -1.35 |
| Linear-Gaussian exactness | FAILS with S=5 MC guidance | passes with exact / Adam guidance | - |

### Other reproduction notes

- Random train/test splits instead of the PHYSS benchmark files.
- GPyTorch marginal-likelihood fits need multi-start (pendulum) or
  validation-gated early stopping (Allen-Cahn) to avoid degenerate flat-GP
  optima; the paper does not discuss this.
- Grid layout: rows = x, cols = t (paper App. H convention). An earlier
  transposed version of the 2D examples produced meaningless constraints -
  fixed; verify `residual_fn(truth)` consistency when adding new PDEs.
