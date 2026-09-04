"""
Integration Test: Analytic Gaussian Benchmark -- Closed-Form Reference
=======================================================================

Ground-truth posteriors for the conjugate hierarchical Gaussian benchmark (autofit_workspace_test#91,
phase 1 of the graphical-ep campaign). Every other column of the benchmark -- the minimal hand-rolled
EP loop in `analytic_ep_minimal.py`, autofit's EP (`factor_graph.optimise`) and autofit's graphical
joint fit -- is judged against the numbers this module produces, so the derivation is written out in
full and every step is self-tested under `__main__`. Pure numpy/scipy: no autofit code is imported.

__Model__

    mu          ~ N(m0, t0^2)                      hyper-prior on the parent mean
    x_i | mu    ~ N(mu, sigma^2)      i = 1..N     per-dataset latent value (the `Level.x` of dataset i)
    y_ij | x_i  ~ N(x_i, s_i^2)       j = 1..n_i   data, known noise s_i

The data enter only through the sufficient statistics

    ybar_i = mean_j y_ij,      v_i = s_i^2 / n_i,      so that   ybar_i | x_i ~ N(x_i, v_i).

__Leg A (sigma known)__

Everything is Gaussian, so the joint posterior over z = (mu, x_1, ..., x_N) is Gaussian with precision
matrix Lambda and linear term h (log p(z|D) = -1/2 z^T Lambda z + h^T z + const):

    Lambda_{mu,mu} = 1/t0^2 + N/sigma^2
    Lambda_{i,i}   = 1/sigma^2 + 1/v_i
    Lambda_{mu,i}  = Lambda_{i,mu} = -1/sigma^2
    h_mu = m0/t0^2,   h_i = ybar_i/v_i

    mean = Lambda^{-1} h,     cov = Lambda^{-1}.

The same posterior in marginal form (the form the EP messages reproduce), obtained by integrating x_i
out of each dataset's factor:

    ybar_i | mu ~ N(mu, sigma^2 + v_i)
    P_mu   = 1/t0^2 + sum_i 1/(sigma^2 + v_i)                       posterior precision of mu
    E[mu]  = (m0/t0^2 + sum_i ybar_i/(sigma^2 + v_i)) / P_mu
    c_i    = 1/(1/sigma^2 + 1/v_i)                                   conditional variance of x_i | mu
    E[x_i | mu] = c_i (mu/sigma^2 + ybar_i/v_i)
    E[x_i] = c_i (E[mu]/sigma^2 + ybar_i/v_i)
    Var[x_i] = c_i + (c_i/sigma^2)^2 / P_mu                          law of total variance
    Cov[mu, x_i] = (c_i/sigma^2) / P_mu

`leg_a_reference` computes both and `__main__` checks they agree to 1e-12.

__Leg B (sigma unknown, hyper-prior p(sigma))__

Marginalising the x_i analytically gives p(mu, sigma | D) ∝ p(mu) p(sigma) prod_i N(ybar_i | mu,
sigma^2 + v_i). Conditional on sigma, mu is Gaussian:

    V(sigma) = 1/(1/t0^2 + sum_i 1/(sigma^2 + v_i))
    m(sigma) = V(sigma) (m0/t0^2 + sum_i ybar_i/(sigma^2 + v_i))

and integrating mu out (a Gaussian integral over a quadratic in mu with precision 1/V, linear term
m/V and constant C = m0^2/t0^2 + sum_i ybar_i^2/(sigma^2 + v_i)) leaves

    log p(sigma | D) = log p(sigma) - 1/2 sum_i log(sigma^2 + v_i) + 1/2 log(V(sigma)/t0^2)
                       - 1/2 [ C - m(sigma)^2 / V(sigma) ] + const.

This one-dimensional density is evaluated on a dense sigma grid (20001 points by default) and
normalised by deterministic quadrature. From it:

    sigma:      E, Std, q05, q50, q95 (q95 = `sigma_q95`, the analytic upper limit on the scatter)
    log sigma:  E, Std
    mu:         E[mu] = E_sigma[m],  Var[mu] = E_sigma[V + m^2] - E[mu]^2
    x_i:        conditional on sigma this is leg A, so with w_i = v_i/(sigma^2 + v_i)
                E[x_i | sigma]   = w_i m(sigma) + (1 - w_i) ybar_i
                Var[x_i | sigma] = sigma^2 w_i + w_i^2 V(sigma)
                and the law of total variance over the sigma grid.

Grid range: the prior support is intersected with the region where the density is non-negligible. A
coarse log-spaced scan (1e-4 to 1e4 times the data/prior scale, 4001 points, clipped to the support)
locates the peak; the dense grid spans the points where log p exceeds the peak by more than -40
(relative density e^-40 ~ 4e-18), padded by one coarse step on each side and clipped to the support.
The normaliser is cross-checked against `scipy.integrate.quad` to 1e-8 in `__main__`.

Quadrature rule: composite trapezoid weights on the uniform grid. The rule is spectrally accurate
here even when a hard truncation edge (e.g. `("truncated", 10, 5, 0, 100)`) leaves a finite density
at sigma = 0, because log p(sigma | D) depends on sigma only through sigma^2 and is therefore even in
sigma: the endpoint derivative that drives the trapezoid rule's O(h^2) error vanishes. `__main__`
reports both trapezoid and composite Simpson against `quad`. The one edge subtlety is E[log sigma]
on a grid that starts at sigma = 0 with non-zero density: log(0) is replaced by the cell average of
log sigma over the first cell, (log h - 1), which is exact for the leading term.

__Hyper-prior families__

    ("delta", value)                           sigma fixed -> returns leg A
    ("gaussian", mean, sd)                     N(mean, sd^2) restricted to sigma > 0 (autofit's
                                               GaussianPrior returns -inf for sigma <= 0)
    ("truncated", mean, sd, lower, upper)      N(mean, sd^2) restricted to [lower, upper]
    ("loggaussian", mean_of_log, sd_of_log)    log sigma ~ N(.,.), density in sigma includes the
                                               1/sigma Jacobian
    ("loggaussian_no_jacobian", ml, sl)        deliberately WRONG: the log-space normal density
                                               evaluated at log sigma without the 1/sigma term
                                               (fingerprint for PyAutoFit#1498). In log sigma this is
                                               N(ml + sl^2, sl^2), i.e. a shifted log-Gaussian.

All normalising constants of the priors are dropped: the posterior is renormalised on the grid.
"""

import sys
import time

import numpy as np
from scipy import integrate

"""
__Model__

The scalar-mean model the autofit legs fit per dataset (phase (b)). It lives here so every script of
the benchmark shares one definition.
"""


class Level:
    """`y_ij ~ N(x, s_i^2)`: a single free parameter `x`, the level of dataset i."""

    def __init__(self, x=0.0):
        self.x = x


"""
__Simulation__
"""


def simulate(
    seed,
    n_datasets=5,
    n_points=20,
    mu_true=50.0,
    sigma_true=10.0,
    noise_sigmas=None,
):
    """
    Simulate the hierarchical Gaussian dataset in memory (nothing is written to disk).

    The noise level `s_i` varies across datasets (default `linspace(3, 8)`) so the `v_i = s_i^2/n_i`
    differ and the closed form is not degenerate in i.

    Returns a dict with `ybar` (N,), `v` (N,), `x_true` (N,), `y` (list of (n_i,) arrays), `s` (N,),
    `n` (N,) and the truths `mu_true`, `sigma_true`, `seed`.
    """
    rng = np.random.default_rng(seed)

    if noise_sigmas is None:
        noise_sigmas = np.linspace(3.0, 8.0, n_datasets)

    s = np.asarray(noise_sigmas, dtype=float)
    n = np.full(n_datasets, int(n_points))

    x_true = rng.normal(mu_true, sigma_true, size=n_datasets)
    y = [rng.normal(x_true[i], s[i], size=n[i]) for i in range(n_datasets)]

    ybar = np.array([yi.mean() for yi in y])
    v = s**2 / n

    return dict(
        ybar=ybar,
        v=v,
        x_true=x_true,
        y=y,
        s=s,
        n=n,
        mu_true=float(mu_true),
        sigma_true=float(sigma_true),
        seed=seed,
    )


"""
__Leg A: sigma known__
"""


def leg_a_reference(ybar, v, sigma, m0=50.0, t0=10.0):
    """
    Exact joint Gaussian posterior over (mu, x_1..x_N) for known `sigma` (see module docstring).

    Returns a dict with `mu_mean`, `mu_std`, `x_mean` (N,), `x_std` (N,), the full `mean` (N+1,) and
    `cov` (N+1, N+1) in the order (mu, x_1..x_N), and a `marginal` sub-dict holding the independent
    marginal-form computation (`mu_mean`, `mu_std`, `x_mean`, `x_std`, `cov_mu_x`).
    """
    ybar = np.asarray(ybar, dtype=float)
    v = np.asarray(v, dtype=float)
    n = ybar.size
    s2 = float(sigma) ** 2

    # Precision matrix and linear term of the joint Gaussian.
    lam = np.zeros((n + 1, n + 1))
    idx = np.arange(1, n + 1)
    lam[0, 0] = 1.0 / t0**2 + n / s2
    lam[idx, idx] = 1.0 / s2 + 1.0 / v
    lam[0, idx] = -1.0 / s2
    lam[idx, 0] = -1.0 / s2

    h = np.concatenate([[m0 / t0**2], ybar / v])

    cov = np.linalg.inv(lam)
    mean = np.linalg.solve(lam, h)
    std = np.sqrt(np.diag(cov))

    # Independent marginal-form computation.
    p_mu = 1.0 / t0**2 + np.sum(1.0 / (s2 + v))
    mu_mean_marg = (m0 / t0**2 + np.sum(ybar / (s2 + v))) / p_mu
    mu_var_marg = 1.0 / p_mu
    c = 1.0 / (1.0 / s2 + 1.0 / v)
    x_mean_marg = c * (mu_mean_marg / s2 + ybar / v)
    x_var_marg = c + (c / s2) ** 2 * mu_var_marg
    cov_mu_x_marg = (c / s2) * mu_var_marg

    return dict(
        mu_mean=float(mean[0]),
        mu_std=float(std[0]),
        x_mean=mean[1:].copy(),
        x_std=std[1:].copy(),
        mean=mean,
        cov=cov,
        sigma=float(sigma),
        marginal=dict(
            mu_mean=float(mu_mean_marg),
            mu_std=float(np.sqrt(mu_var_marg)),
            x_mean=x_mean_marg,
            x_std=np.sqrt(x_var_marg),
            cov_mu_x=cov_mu_x_marg,
        ),
    )


"""
__Leg B: sigma unknown__
"""


def prior_support(prior):
    """Support (lower, upper) of the hyper-prior in sigma space; `upper` may be `np.inf`."""
    kind = prior[0]
    if kind in ("gaussian", "loggaussian", "loggaussian_no_jacobian"):
        return 0.0, np.inf
    if kind == "truncated":
        return float(prior[3]), float(prior[4])
    if kind == "delta":
        return float(prior[1]), float(prior[1])
    raise ValueError(f"unknown prior family {prior!r}")


def prior_scale(prior):
    """A characteristic sigma scale of the hyper-prior, used only to seed the coarse grid scan."""
    kind = prior[0]
    if kind in ("gaussian", "truncated"):
        return abs(float(prior[1])) + float(prior[2])
    if kind in ("loggaussian", "loggaussian_no_jacobian"):
        return float(np.exp(prior[1] + prior[2]))
    return float(prior[1])


def log_prior_sigma(sigma, prior):
    """
    Unnormalised log hyper-prior density in sigma (vectorised); -inf outside the support.

    gaussian / truncated:        -1/2 ((sigma - mean)/sd)^2 on the support
    loggaussian:                 -1/2 ((log sigma - ml)/sl)^2 - log sigma       (with the Jacobian)
    loggaussian_no_jacobian:     -1/2 ((log sigma - ml)/sl)^2                   (deliberately wrong)
    """
    sigma = np.asarray(sigma, dtype=float)
    kind = prior[0]
    lo, hi = prior_support(prior)

    with np.errstate(divide="ignore", invalid="ignore"):
        if kind in ("gaussian", "truncated"):
            mean, sd = float(prior[1]), float(prior[2])
            lp = -0.5 * ((sigma - mean) / sd) ** 2
            inside = (
                (sigma > lo) & (sigma <= hi)
                if kind == "gaussian"
                else (sigma >= lo) & (sigma <= hi)
            )
        elif kind in ("loggaussian", "loggaussian_no_jacobian"):
            ml, sl = float(prior[1]), float(prior[2])
            log_sigma = np.log(sigma)
            lp = -0.5 * ((log_sigma - ml) / sl) ** 2
            if kind == "loggaussian":
                lp = lp - log_sigma
            inside = sigma > 0.0
        else:
            raise ValueError(f"unknown prior family {prior!r}")

    return np.where(inside, lp, -np.inf)


def log_post_sigma(sigma, ybar, v, prior, m0=50.0, t0=10.0):
    """
    log p(sigma | D) up to a constant (module docstring, leg B), vectorised over `sigma`.

    Returns `(logp, m, V)` where `m(sigma)`, `V(sigma)` are the conditional mean and variance of mu.
    """
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    ybar = np.asarray(ybar, dtype=float)
    v = np.asarray(v, dtype=float)

    s2v = sigma[:, None] ** 2 + v[None, :]
    p = 1.0 / t0**2 + np.sum(1.0 / s2v, axis=1)
    vv = 1.0 / p
    m = vv * (m0 / t0**2 + np.sum(ybar / s2v, axis=1))
    c = m0**2 / t0**2 + np.sum(ybar**2 / s2v, axis=1)

    logp = (
        log_prior_sigma(sigma, prior)
        - 0.5 * np.sum(np.log(s2v), axis=1)
        + 0.5 * np.log(vv / t0**2)
        - 0.5 * (c - m**2 * p)
    )
    return logp, m, vv


def trapezoid_weights(x):
    """Composite trapezoid weights for a uniform grid."""
    x = np.asarray(x, dtype=float)
    h = (x[-1] - x[0]) / (x.size - 1)
    w = np.full(x.size, h)
    w[0] = w[-1] = 0.5 * h
    return w


def simpson_weights(x):
    """Composite Simpson weights for a uniform grid with an even number of intervals (odd size)."""
    x = np.asarray(x, dtype=float)
    if x.size % 2 == 0:
        raise ValueError("Simpson weights need an odd number of grid points")
    h = (x[-1] - x[0]) / (x.size - 1)
    w = np.full(x.size, 2.0)
    w[1::2] = 4.0
    w[0] = w[-1] = 1.0
    return w * h / 3.0


def sigma_grid(ybar, v, prior, m0=50.0, t0=10.0, n_grid=20001, log_drop=40.0):
    """
    Dense sigma grid covering the prior support intersected with the non-negligible posterior region
    (module docstring, "Grid range").
    """
    lo, hi = prior_support(prior)
    scale = max(float(np.std(ybar)), float(np.sqrt(np.mean(v))), prior_scale(prior))

    coarse = np.geomspace(1e-4 * scale, 1e4 * scale, 4001)
    coarse = coarse[(coarse > lo) & (coarse < hi)]
    if np.isfinite(hi):
        coarse = np.append(coarse, hi)
    if lo > 0.0:
        coarse = np.insert(coarse, 0, lo)

    logp, _, _ = log_post_sigma(coarse, ybar, v, prior, m0, t0)
    finite = np.isfinite(logp)
    if not np.any(finite):
        raise ValueError("posterior on sigma is nowhere finite on the coarse scan")

    keep = np.where(finite & (logp > logp[finite].max() - log_drop))[0]
    i_lo, i_hi = keep[0], keep[-1]

    g_lo = coarse[i_lo - 1] if i_lo > 0 else lo
    g_hi = coarse[i_hi + 1] if i_hi + 1 < coarse.size else hi
    if not np.isfinite(g_hi):
        # The scan reached 1e4 x scale without dropping by e^-40: extend geometrically.
        g_hi = coarse[-1] * 10.0

    g_lo = max(g_lo, lo)
    g_hi = min(g_hi, hi)

    return np.linspace(g_lo, g_hi, n_grid)


def leg_b_reference(ybar, v, prior, m0=50.0, t0=10.0, grid=None, n_grid=20001):
    """
    Exact leg-B posterior by deterministic quadrature over sigma (module docstring, leg B).

    `prior` is one of the tuples listed in the module docstring. `grid` overrides the automatic
    sigma grid (must be uniform with an odd number of points).

    Returns a dict with `sigma_mean`, `sigma_std`, `sigma_q05`, `sigma_q50`, `sigma_q95`,
    `log_sigma_mean`, `log_sigma_std`, `mu_mean`, `mu_std`, `x_mean` (N,), `x_std` (N,), the `grid`,
    the normalised density `pdf` on it, the log-normaliser `log_z` (of exp(logp - logp.max())) and
    `logp_max`. For `("delta", value)` it returns `leg_a_reference` with the sigma keys filled in.
    """
    ybar = np.asarray(ybar, dtype=float)
    v = np.asarray(v, dtype=float)

    if prior[0] == "delta":
        out = leg_a_reference(ybar, v, float(prior[1]), m0, t0)
        out.update(
            sigma_mean=float(prior[1]),
            sigma_std=0.0,
            sigma_q05=float(prior[1]),
            sigma_q50=float(prior[1]),
            sigma_q95=float(prior[1]),
            log_sigma_mean=float(np.log(prior[1])),
            log_sigma_std=0.0,
        )
        return out

    if grid is None:
        grid = sigma_grid(ybar, v, prior, m0, t0, n_grid=n_grid)
    grid = np.asarray(grid, dtype=float)

    logp, m, vv = log_post_sigma(grid, ybar, v, prior, m0, t0)
    logp_max = np.max(logp[np.isfinite(logp)])
    f = np.exp(logp - logp_max)  # exp(-inf) = 0 outside the support

    w = trapezoid_weights(grid)
    z = np.sum(w * f)
    pdf = f / z
    pw = w * pdf  # probability weight of each grid point

    # sigma and log sigma moments.
    sigma_mean = np.sum(pw * grid)
    sigma_var = np.sum(pw * grid**2) - sigma_mean**2
    with np.errstate(divide="ignore"):
        log_grid = np.log(grid)
    log_grid = np.where(
        pdf > 0.0, log_grid, 0.0
    )  # points of zero weight contribute nothing
    if grid[0] == 0.0 and pdf[0] > 0.0:
        # Cell average of log sigma over [0, h]: (1/h) int_0^h log s ds = log h - 1.
        log_grid[0] = np.log(grid[1] - grid[0]) - 1.0
    log_sigma_mean = np.sum(pw * log_grid)
    log_sigma_var = np.sum(pw * log_grid**2) - log_sigma_mean**2

    # Quantiles from the cumulative (trapezoid) distribution.
    cdf = integrate.cumulative_trapezoid(pdf, grid, initial=0.0)
    cdf = cdf / cdf[-1]
    q05, q50, q95 = np.interp([0.05, 0.50, 0.95], cdf, grid)

    # mu by the law of total variance.
    mu_mean = np.sum(pw * m)
    mu_var = np.sum(pw * (vv + m**2)) - mu_mean**2

    # x_i by the law of total variance (conditional on sigma this is leg A).
    s2v = grid[:, None] ** 2 + v[None, :]
    wi = v[None, :] / s2v
    x_cond_mean = wi * m[:, None] + (1.0 - wi) * ybar[None, :]
    x_cond_var = grid[:, None] ** 2 * wi + wi**2 * vv[:, None]
    x_mean = np.sum(pw[:, None] * x_cond_mean, axis=0)
    x_var = np.sum(pw[:, None] * (x_cond_var + x_cond_mean**2), axis=0) - x_mean**2

    return dict(
        sigma_mean=float(sigma_mean),
        sigma_std=float(np.sqrt(sigma_var)),
        sigma_q05=float(q05),
        sigma_q50=float(q50),
        sigma_q95=float(q95),
        log_sigma_mean=float(log_sigma_mean),
        log_sigma_std=float(np.sqrt(log_sigma_var)),
        mu_mean=float(mu_mean),
        mu_std=float(np.sqrt(mu_var)),
        x_mean=x_mean,
        x_std=np.sqrt(x_var),
        grid=grid,
        pdf=pdf,
        log_z=float(np.log(z)),
        logp_max=float(logp_max),
        prior=prior,
    )


def quad_normaliser(ybar, v, prior, grid, logp_max, m0=50.0, t0=10.0):
    """`scipy.integrate.quad` of exp(logp - logp_max) over the grid range (self-test cross-check)."""

    def integrand(s):
        lp, _, _ = log_post_sigma(s, ybar, v, prior, m0, t0)
        return float(np.exp(lp[0] - logp_max))

    z, err = integrate.quad(
        integrand,
        grid[0],
        grid[-1],
        epsabs=0.0,
        epsrel=1e-12,
        limit=500,
        points=[grid[np.argmax(log_post_sigma(grid, ybar, v, prior, m0, t0)[0])]],
    )
    return z, err


"""
__Self-tests__
"""


def _check(label, value, tol, extra=""):
    ok = value <= tol
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {label}: {value:.3e} (tol {tol:.0e}) {extra}"
    )
    return ok


def _run_self_tests():
    t_start = time.time()
    ok = True

    data = simulate(seed=0)
    ybar, v = data["ybar"], data["v"]
    sigma_true = data["sigma_true"]
    n = ybar.size

    print("Analytic Gaussian benchmark -- closed-form reference self-tests")
    print(f"  seed 0: N={n}, n_i={data['n'][0]}, s_i={np.round(data['s'], 3)}")
    print(f"  ybar = {np.round(ybar, 4)}")
    print(f"  v    = {np.round(v, 5)}")

    # (i) Leg A: dense precision-matrix form vs marginal form.
    print("\n(i) Leg A: precision-matrix form vs marginal form")
    a = leg_a_reference(ybar, v, sigma_true)
    mg = a["marginal"]
    d_mu = max(abs(a["mu_mean"] - mg["mu_mean"]), abs(a["mu_std"] - mg["mu_std"]))
    d_x = max(
        np.max(np.abs(a["x_mean"] - mg["x_mean"])),
        np.max(np.abs(a["x_std"] - mg["x_std"])),
    )
    d_cov = np.max(np.abs(a["cov"][0, 1:] - mg["cov_mu_x"]))
    print(f"  mu = {a['mu_mean']:.6f} +/- {a['mu_std']:.6f}")
    for i in range(n):
        print(f"  x_{i} = {a['x_mean'][i]:.6f} +/- {a['x_std'][i]:.6f}")
    ok &= _check("max |dense - marginal| (mu mean/std)", d_mu, 1e-12)
    ok &= _check("max |dense - marginal| (x mean/std)", d_x, 1e-12)
    ok &= _check("max |dense - marginal| (cov(mu, x_i))", d_cov, 1e-12)

    # (ii) Leg B: quad vs grid normaliser for every prior family.
    print(
        "\n(ii) Leg B: quad vs grid normaliser (trapezoid; Simpson reported for comparison)"
    )
    priors = [
        ("gaussian", 10.0, 5.0),
        ("truncated", 10.0, 5.0, 0.0, 100.0),
        ("loggaussian", np.log(10.0), 0.5),
        ("loggaussian_no_jacobian", np.log(10.0), 0.5),
    ]
    for prior in priors:
        t0_ = time.time()
        b = leg_b_reference(ybar, v, prior)
        z_grid = np.exp(b["log_z"])
        z_quad, err = quad_normaliser(ybar, v, prior, b["grid"], b["logp_max"])
        logp, _, _ = log_post_sigma(b["grid"], ybar, v, prior)
        z_simp = np.sum(simpson_weights(b["grid"]) * np.exp(logp - b["logp_max"]))
        rel = abs(z_grid - z_quad) / z_quad
        rel_simp = abs(z_simp - z_quad) / z_quad
        print(
            f"  {prior[0]:<24} grid [{b['grid'][0]:.4g}, {b['grid'][-1]:.4g}]  "
            f"sigma = {b['sigma_mean']:.4f} +/- {b['sigma_std']:.4f}  "
            f"q05/q50/q95 = {b['sigma_q05']:.4f}/{b['sigma_q50']:.4f}/{b['sigma_q95']:.4f}  "
            f"log sigma = {b['log_sigma_mean']:.4f} +/- {b['log_sigma_std']:.4f}  "
            f"mu = {b['mu_mean']:.4f} +/- {b['mu_std']:.4f}  ({time.time() - t0_:.2f}s)"
        )
        ok &= _check(
            f"{prior[0]}: |Z_trapezoid/Z_quad - 1|",
            rel,
            1e-8,
            f"(Simpson: {rel_simp:.2e}, quad err {err / z_quad:.1e})",
        )

    # (iii) Leg B with a near-delta gaussian prior reproduces leg A.
    print(
        "\n(iii) Leg B near-delta prior (gaussian, sd 1e-3 about sigma_true) vs leg A"
    )
    b = leg_b_reference(ybar, v, ("gaussian", sigma_true, 1e-3))
    rel_mu = max(
        abs(b["mu_mean"] - a["mu_mean"]) / a["mu_std"],
        abs(b["mu_std"] / a["mu_std"] - 1.0),
    )
    rel_x = max(
        np.max(np.abs(b["x_mean"] - a["x_mean"]) / a["x_std"]),
        np.max(np.abs(b["x_std"] / a["x_std"] - 1.0)),
    )
    print(
        f"  sigma = {b['sigma_mean']:.6f} +/- {b['sigma_std']:.2e}  (grid width {b['grid'][-1] - b['grid'][0]:.4f})"
    )
    print(
        f"  mu    = {b['mu_mean']:.8f} +/- {b['mu_std']:.8f}   (leg A {a['mu_mean']:.8f} +/- {a['mu_std']:.8f})"
    )
    ok &= _check("mu: max(|dmean|/std, |std ratio - 1|)", rel_mu, 1e-6)
    ok &= _check("x_i: max(|dmean|/std, |std ratio - 1|)", rel_x, 1e-6)

    # (iv) Large-data sanity: with many datasets the sigma posterior must track the moment estimator
    # sqrt(var(ybar) - mean(v)) of the data (and the sample scatter of the true draws), not the prior.
    print(
        "\n(iv) Leg B large-data limit (N=400, n_i=200): sigma posterior tracks the data's scatter"
    )
    big = simulate(seed=1, n_datasets=400, n_points=200)
    bb = leg_b_reference(big["ybar"], big["v"], ("loggaussian", np.log(10.0), 0.5))
    moment = np.sqrt(np.var(big["ybar"], ddof=1) - np.mean(big["v"]))
    pull_truth = abs(bb["sigma_mean"] - big["sigma_true"]) / bb["sigma_std"]
    pull_moment = abs(bb["sigma_mean"] - moment) / bb["sigma_std"]
    print(
        f"  sigma = {bb['sigma_mean']:.4f} +/- {bb['sigma_std']:.4f}  "
        f"(moment estimator {moment:.4f}, std(x_true) {np.std(big['x_true'], ddof=1):.4f}, "
        f"truth {big['sigma_true']}, pull vs truth {pull_truth:.2f})"
    )
    ok &= _check("|E[sigma] - moment estimator| / std", pull_moment, 0.5)

    elapsed = time.time() - t_start
    print(f"\nTotal runtime {elapsed:.2f}s")
    ok &= _check("runtime (s)", elapsed, 10.0)

    print(f"\nREFERENCE SELF-TESTS: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not _run_self_tests():
        sys.exit(1)
