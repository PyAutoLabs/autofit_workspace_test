"""
Integration Test: Analytic Gaussian Benchmark -- Minimal Hand-Rolled EP
========================================================================

An as-simple-as-possible expectation propagation (EP) loop for the conjugate hierarchical Gaussian
benchmark of autofit_workspace_test#91, written outside autofit with explicit messages, cavities and
projections. It is the method-vs-implementation referee: if this loop reproduces the closed form of
`analytic_reference.py` and autofit's EP does not, the defect is in the library, not in EP.

__Sites in natural parameters__

Every Gaussian site / marginal is a pair of natural parameters

    eta = (eta1, eta2) = (m / v, 1 / v)        for   N(m, v)

so that multiplying Gaussians is addition (`q = sum of sites`), the cavity is subtraction
(`cavity = q - site`) and projecting a tilted distribution with moments (E, Var) back onto the
family is `nat(E, Var)`; the new site is `nat(E, Var) - cavity`.

__Leg A (sigma known)__

Variables mu and x_i (i = 1..N). Fixed exact sites: the mu hyper-prior N(m0, t0^2) and each dataset's
likelihood N(x_i | ybar_i, v_i). One EP site per hierarchical factor H_i(mu, x_i) = N(x_i | mu, sigma^2),
with a component on mu and a component on x_i. For cavities N(a, A) on mu and N(b, B) on x_i the tilted
distribution N(mu | a, A) N(x_i | b, B) N(x_i | mu, sigma^2) is a 2-D Gaussian, and its marginals are

    tilted(mu)  ∝ N(mu | a, A) N(b | mu, sigma^2 + B)     =>  message H_i -> mu  = N(mu | b, sigma^2 + B)
    tilted(x_i) ∝ N(x_i | b, B) N(x_i | a, sigma^2 + A)   =>  message H_i -> x_i = N(x_i | a, sigma^2 + A)

so the moment projection is exact and the site IS the message. The factor graph is a tree, so the EP
fixed point is the exact posterior: `ep_leg_a` must match `leg_a_reference` to 1e-10 (the anchor;
any residual is the sweep tolerance).

__Leg B (sigma unknown)__

A third variable theta carries the scatter, either theta = sigma (gaussian / truncated hyper-priors)
or theta = log sigma (loggaussian hyper-priors), with a Gaussian q(theta). Two kinds of EP site:

  * the hyper-prior site on theta: tilted(theta) ∝ cavity(theta) p_theta(theta), where p_theta is the
    prior density IN THETA SPACE, i.e. p_sigma(sigma(theta)) times the Jacobian dsigma/dtheta
    (= sigma for theta = log sigma). Moment-matched by 1-D quadrature on a grid over the cavity
    mean +/- 8 cavity std, clipped to the prior support.

  * one site per hierarchical factor H_i(mu, x_i, theta) = N(x_i | mu, sigma(theta)^2) with components
    on mu, x_i and theta. Conditional on theta the (mu, x_i) block is the leg-A 2-D Gaussian, so with
    cavities N(a, A), N(b, B), N(c, C) and S(theta) = sigma(theta)^2:

        tilted(theta) ∝ N(theta | c, C) N(b | a, S + A + B)                    (the 1-D weight)
        mu  | theta ~ N( Vm (a/A + b/(S + B)),  Vm ),   1/Vm = 1/A + 1/(S + B)
        x_i | theta ~ N( Vx (b/B + a/(S + A)),  Vx ),   1/Vx = 1/B + 1/(S + A)

    and the tilted moments of mu, x_i and theta are 1-D integrals over theta of these conditional
    moments (law of total variance), on the same clipped grid.

`projection="moments"` matches moments (EP the algorithm). `projection="laplace"` instead does what
autofit's `LaplaceOptimiser` does: it locates the mode of the tilted distribution -- for H_i the joint
mode over (mu, x_i, theta), found by maximising the profile over theta with (mu, x_i) at their
conditional means -- and takes the inverse of a central finite-difference Hessian there as the
covariance; the projected marginals are the mode and the diagonal of that covariance. Sites are
damped as `site = damping * new + (1 - damping) * old`, and the loop stops when max |Delta eta| over
all sites is below `tol`. Updates that would make a marginal precision non-positive (or a Laplace
Hessian that is not negative definite, e.g. a mode on the support boundary) are skipped and counted;
the returned `converged` flag and `sweeps` count report the outcome honestly.

The natural-parameter representation makes the loop explicit; the 1-D integrals use composite Simpson
weights on a uniform grid (odd number of points).

__What the first runs showed (seeds 0-4, N=5, n_i=20, sigma_true=10, written 2026-09-02)__

Implementation anchors (all pass): leg A matches the closed form to ~6e-15; with one dataset and an
exactly-Gaussian prior site (loggaussian prior, theta = log sigma) every cavity is exact and EP
reproduces the closed form to ~1e-13, which certifies the tilted-moment quadrature; a near-delta
hyper-prior reproduces leg A to ~3e-8; the moments fixed point is independent of damping and grid size
to 1e-6.

EP-intrinsic deviations of the moments projection from the closed form (max over seeds 0-4;
a = |dmean| / std_ref, b = |std / std_ref - 1|):

    gaussian / truncated, theta = sigma:   scatter row a 0.077, b 0.145;  mu and x_i rows a 0.020, b 0.063
    loggaussian, theta = log sigma:        scatter row a 0.049, b 0.119;  mu and x_i rows a 0.022, b 0.078

E[sigma] lies inside the closed-form [q05, q95] on every seed. The scatter row is the hard one: a
Gaussian family on a skewed sigma posterior (seed 0: q05/q50/q95 = 3.0/6.0/12.2) with six
approximated sites; the mu and x_i rows are comfortably within the issue's a = 0.05, b = 0.10.

Calibration (issue #91 rule: 2x the value observed over seeds 0-4, rounded up, never above the hard
caps), applied per row because the scatter row is EP-intrinsically harder than the location rows:

    scatter row (sigma, or log sigma):   a = 0.20, b = 0.30      (observed 0.077 / 0.145)
    mu and x_i rows:                     a = 0.05, b = 0.16      (observed 0.022 / 0.078)
    hard caps (never relaxed):           E[sigma] inside the closed-form [q05, q95]; no std error > 50%

`_run_self_tests` asserts these on seeds 0-4 for all three hyper-priors (theta = sigma for the
gaussian / truncated priors, theta = log sigma for the loggaussian prior -- the parametrisation each
family's autofit message uses).

Laplace projection: with theta = sigma the joint tilted density of every hierarchical site is
unbounded as sigma -> 0 (∝ 1/sigma at x_i = mu), so the joint mode sits on the support edge, no
Laplace approximation exists and no site can update -- the conjugate model reproduces the scale
collapse deterministically. With theta = log sigma the density is bounded but the mode of
N(t | c, C) e^-t sits at c - C, so a broad cavity (gaussian / truncated hyper-priors have a
near-infinite variance in log sigma) walks the mode to -inf sweep by sweep (log sigma ~ -13 after
500 sweeps); only the loggaussian hyper-prior, whose cavity is narrow, gives a finite Laplace fixed
point (seed 0: log sigma 1.650 +/- 0.401 vs closed form 1.836 +/- 0.361, i.e. a = 0.52, b = 0.11).
"""

import sys
import time

import numpy as np
from scipy import optimize

from analytic_reference import (
    leg_a_reference,
    leg_b_reference,
    log_prior_sigma,
    prior_support,
    simpson_weights,
    simulate,
)

"""
__Natural-parameter helpers__
"""


def nat(mean, var):
    """Natural parameters (m/v, 1/v) of N(mean, var)."""
    return np.array([mean / var, 1.0 / var], dtype=float)


def moments(eta):
    """(mean, var) of the Gaussian with natural parameters `eta`; var may be non-positive (improper)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        var = 1.0 / eta[1]
        return eta[0] * var, var


def proper(*vars_):
    """True if every variance is finite and positive (a proper Gaussian cavity)."""
    return all(np.isfinite(x) and x > 0.0 for x in vars_)


# Central-difference step for the Laplace Hessians, relative to the cavity std: 1e-3 balances the
# O(h^2) truncation error against round-off (~eps / h^2), which at 1e-4 left a ~1e-6 noise floor.
FD_STEP = 1e-3


def log_normal(x, mean, var):
    """log N(x | mean, var), vectorised."""
    return -0.5 * np.log(2.0 * np.pi * var) - 0.5 * (x - mean) ** 2 / var


"""
__Leg A__
"""


def ep_leg_a(ybar, v, sigma, m0=50.0, t0=10.0, tol=1e-12, max_sweeps=100):
    """
    EP with exact 2-D Gaussian tilted moments for known `sigma` (module docstring, leg A).

    Returns `mu_mean`, `mu_std`, `x_mean` (N,), `x_std` (N,), `sweeps`, `converged`, `max_delta`.
    """
    ybar = np.asarray(ybar, dtype=float)
    v = np.asarray(v, dtype=float)
    n = ybar.size
    s2 = float(sigma) ** 2

    # Fixed exact sites and the H_i sites (initialised flat).
    prior_mu = nat(m0, t0**2)
    lik_x = np.stack([nat(ybar[i], v[i]) for i in range(n)])
    site_mu = np.zeros((n, 2))
    site_x = np.zeros((n, 2))

    q_mu = prior_mu.copy()
    q_x = lik_x.copy()

    converged = False
    max_delta = np.inf
    for sweep in range(1, max_sweeps + 1):
        max_delta = 0.0
        for i in range(n):
            a, av = moments(q_mu - site_mu[i])  # cavity on mu: N(a, A)
            b, bv = moments(q_x[i] - site_x[i])  # cavity on x_i: N(b, B)

            new_mu = nat(b, s2 + bv)  # message H_i -> mu
            new_x = nat(a, s2 + av)  # message H_i -> x_i

            max_delta = max(
                max_delta,
                np.max(np.abs(new_mu - site_mu[i])),
                np.max(np.abs(new_x - site_x[i])),
            )
            q_mu += new_mu - site_mu[i]
            q_x[i] += new_x - site_x[i]
            site_mu[i] = new_mu
            site_x[i] = new_x

        if max_delta < tol:
            converged = True
            break

    mu_mean, mu_var = moments(q_mu)
    x_mean = np.array([moments(q_x[i])[0] for i in range(n)])
    x_var = np.array([moments(q_x[i])[1] for i in range(n)])

    return dict(
        mu_mean=float(mu_mean),
        mu_std=float(np.sqrt(mu_var)),
        x_mean=x_mean,
        x_std=np.sqrt(x_var),
        sweeps=sweep,
        converged=converged,
        max_delta=float(max_delta),
    )


"""
__Leg B helpers__
"""


def theta_support(prior, theta):
    """Support of theta = sigma or log sigma, derived from the prior's sigma support."""
    lo, hi = prior_support(prior)
    if theta == "sigma":
        return lo, hi
    if theta == "log_sigma":
        return (np.log(lo) if lo > 0.0 else -np.inf), np.log(hi)
    raise ValueError(f"theta must be 'sigma' or 'log_sigma', got {theta!r}")


def sigma_of_theta(t, theta):
    return np.exp(t) if theta == "log_sigma" else t


def log_prior_theta(t, prior, theta):
    """
    Unnormalised log hyper-prior density in theta space: log p_sigma(sigma(t)) + log |dsigma/dt|.
    For theta = log sigma the Jacobian is sigma, so log p_theta = log p_sigma(e^t) + t.
    """
    t = np.asarray(t, dtype=float)
    lp = log_prior_sigma(sigma_of_theta(t, theta), prior)
    if theta == "log_sigma":
        lp = lp + t
    return lp


def theta_grid(c, cv, prior, theta, n_grid, half_width=8.0):
    """Uniform grid over c +/- half_width * sqrt(C), clipped to the theta support (open at a -inf/0 edge)."""
    lo, hi = theta_support(prior, theta)
    sd = np.sqrt(cv)
    g_lo = max(c - half_width * sd, lo)
    g_hi = min(c + half_width * sd, hi)
    if theta == "sigma":
        # sigma = 0 has zero measure and N(x | mu, 0) is undefined: keep the first node strictly positive.
        g_lo = max(g_lo, 1e-12 * max(1.0, abs(c)))
    if not g_hi > g_lo:
        return None  # the cavity window lies outside the support: caller skips the update
    return np.linspace(g_lo, g_hi, n_grid)


def _weighted_moments(w, x):
    mean = np.sum(w * x)
    return mean, np.sum(w * x**2) - mean**2


def tilted_prior_moments(c, cv, prior, theta, n_grid):
    """Moments of cavity(theta) x p_theta(theta) by quadrature (moment projection of the prior site)."""
    t = theta_grid(c, cv, prior, theta, n_grid)
    if t is None:
        return np.nan, np.nan
    lw = log_normal(t, c, cv) + log_prior_theta(t, prior, theta)
    finite = np.isfinite(lw)
    if not np.any(finite):
        return np.nan, np.nan
    w = np.where(finite, np.exp(lw - lw[finite].max()), 0.0) * simpson_weights(t)
    w = w / np.sum(w)
    return _weighted_moments(w, t)


def tilted_prior_laplace(c, cv, prior, theta, n_grid):
    """Mode and finite-difference curvature of cavity(theta) x p_theta(theta) (Laplace projection)."""
    t = theta_grid(c, cv, prior, theta, n_grid)
    if t is None:
        return np.nan, np.nan, False

    def neg_log(x):
        return -(log_normal(x, c, cv) + log_prior_theta(x, prior, theta))

    k = int(np.argmin(neg_log(t)))
    boundary = k == 0 or k == t.size - 1
    if boundary:
        return np.nan, np.nan, True  # mode on the support edge: no Laplace approximation exists
    res = optimize.minimize_scalar(neg_log, bounds=(t[k - 1], t[k + 1]), method="bounded", options=dict(xatol=1e-12))
    mode = float(res.x)
    h = FD_STEP * np.sqrt(cv)
    curv = (neg_log(mode + h) - 2.0 * neg_log(mode) + neg_log(mode - h)) / h**2
    return mode, (1.0 / curv if curv > 0.0 else np.nan), False


def conditional_block(t, theta, a, av, b, bv):
    """Leg-A conditional moments of (mu, x_i) given theta, plus log N(b | a, S + A + B) (vectorised)."""
    s2 = sigma_of_theta(t, theta) ** 2
    log_z = log_normal(b, a, s2 + av + bv)
    vm = 1.0 / (1.0 / av + 1.0 / (s2 + bv))
    mm = vm * (a / av + b / (s2 + bv))
    vx = 1.0 / (1.0 / bv + 1.0 / (s2 + av))
    mx = vx * (b / bv + a / (s2 + av))
    return log_z, mm, vm, mx, vx


def tilted_h_moments(a, av, b, bv, c, cv, prior, theta, n_grid):
    """
    Tilted moments of (mu, x_i, theta) for one hierarchical factor by 1-D quadrature over theta
    (module docstring, leg B). Returns ((E_mu, V_mu), (E_x, V_x), (E_theta, V_theta)).
    """
    t = theta_grid(c, cv, prior, theta, n_grid)
    if t is None:
        return (np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan)
    log_z, mm, vm, mx, vx = conditional_block(t, theta, a, av, b, bv)
    lw = log_normal(t, c, cv) + log_z
    w = np.exp(lw - lw.max()) * simpson_weights(t)
    w = w / np.sum(w)

    e_mu = np.sum(w * mm)
    v_mu = np.sum(w * (vm + mm**2)) - e_mu**2
    e_x = np.sum(w * mx)
    v_x = np.sum(w * (vx + mx**2)) - e_x**2
    e_t, v_t = _weighted_moments(w, t)
    return (e_mu, v_mu), (e_x, v_x), (e_t, v_t)


def tilted_h_laplace(a, av, b, bv, c, cv, prior, theta, n_grid):
    """
    Laplace projection of one hierarchical factor's tilted distribution over (mu, x_i, theta): joint
    mode (profile over theta with mu, x_i at their conditional means), then the inverse of a central
    finite-difference 3x3 Hessian of the log tilted density at the mode. Returns the same triple as
    `tilted_h_moments` plus a boundary flag; variances are NaN if the Hessian is not negative definite.
    """
    nan3 = (np.nan, np.nan)
    t = theta_grid(c, cv, prior, theta, n_grid)
    if t is None:
        return nan3, nan3, nan3, False

    def log_joint(z):
        mu, x, th = z
        s2 = sigma_of_theta(th, theta) ** 2
        return log_normal(mu, a, av) + log_normal(x, b, bv) + log_normal(th, c, cv) + log_normal(x, mu, s2)

    def neg_profile(th):
        _, mm, _, mx, _ = conditional_block(th, theta, a, av, b, bv)
        return -log_joint((mm, mx, th))

    k = int(np.argmin(neg_profile(t)))
    if k == 0 or k == t.size - 1:
        # Joint mode on the support edge (for theta = sigma the joint tilted density is unbounded as
        # sigma -> 0 at x_i = mu, ∝ 1/sigma): no Laplace approximation exists.
        return nan3, nan3, nan3, True
    res = optimize.minimize_scalar(neg_profile, bounds=(t[k - 1], t[k + 1]), method="bounded", options=dict(xatol=1e-12))
    th_mode = float(res.x)
    _, mm, _, mx, _ = conditional_block(th_mode, theta, a, av, b, bv)
    z0 = np.array([mm, mx, th_mode], dtype=float)

    steps = FD_STEP * np.sqrt([av, bv, cv])
    hess = np.zeros((3, 3))
    f0 = log_joint(z0)
    for i in range(3):
        for j in range(i, 3):
            ei = np.zeros(3)
            ej = np.zeros(3)
            ei[i] = steps[i]
            ej[j] = steps[j]
            if i == j:
                hess[i, i] = (log_joint(z0 + ei) - 2.0 * f0 + log_joint(z0 - ei)) / steps[i] ** 2
            else:
                hess[i, j] = hess[j, i] = (
                    log_joint(z0 + ei + ej) - log_joint(z0 + ei - ej) - log_joint(z0 - ei + ej) + log_joint(z0 - ei - ej)
                ) / (4.0 * steps[i] * steps[j])

    try:
        cov = np.linalg.inv(-hess)
        var = np.diag(cov)
        if not np.all(var > 0.0):
            var = np.full(3, np.nan)
    except np.linalg.LinAlgError:
        var = np.full(3, np.nan)

    return (z0[0], var[0]), (z0[1], var[1]), (z0[2], var[2]), False


def prior_theta_moments(prior, theta):
    """Moments of the hyper-prior in theta space, used to initialise q(theta)."""
    lo, hi = theta_support(prior, theta)
    c = np.log(prior[1]) if (theta == "log_sigma" and prior[0] in ("gaussian", "truncated")) else float(prior[1])
    cv = (float(prior[2]) / prior[1]) ** 2 if (theta == "log_sigma" and prior[0] in ("gaussian", "truncated")) else float(prior[2]) ** 2
    if theta == "sigma" and prior[0].startswith("loggaussian"):
        c, cv = float(np.exp(prior[1])), float(np.exp(prior[1]) * prior[2]) ** 2
    # Refine by quadrature of the density itself over a wide window (10 std), clipped to the support.
    t = theta_grid(c, cv, prior, theta, 2001, half_width=10.0)
    lw = log_prior_theta(t, prior, theta)
    finite = np.isfinite(lw)
    w = np.where(finite, np.exp(lw - lw[finite].max()), 0.0) * simpson_weights(t)
    w = w / np.sum(w)
    return _weighted_moments(w, t)


def theta_summary(c, cv, prior, theta, n_grid=2001):
    """E/std of sigma and log sigma implied by q(theta) = N(c, C), by quadrature over the support."""
    t = theta_grid(c, cv, prior, theta, n_grid)
    if t is None:
        return dict(sigma_mean=np.nan, sigma_std=np.nan, log_sigma_mean=np.nan, log_sigma_std=np.nan)
    w = np.exp(log_normal(t, c, cv)) * simpson_weights(t)
    w = w / np.sum(w)
    sig = sigma_of_theta(t, theta)
    e_s, v_s = _weighted_moments(w, sig)
    # log sigma moments over sigma > 0 only (a closed support edge at sigma = 0 has zero measure).
    pos = sig > 0.0
    e_l, v_l = _weighted_moments(w[pos] / np.sum(w[pos]), np.log(sig[pos]))
    return dict(
        sigma_mean=float(e_s),
        sigma_std=float(np.sqrt(v_s)),
        log_sigma_mean=float(e_l),
        log_sigma_std=float(np.sqrt(v_l)),
    )


"""
__Leg B__
"""


def ep_leg_b(
    ybar,
    v,
    prior,
    m0=50.0,
    t0=10.0,
    projection="moments",
    theta="sigma",
    damping=1.0,
    tol=1e-8,
    max_sweeps=500,
    n_grid=401,
):
    """
    EP for the unknown-scatter leg (module docstring, leg B).

    `prior` is a hyper-prior tuple of `analytic_reference`; `theta` chooses the parametrisation of the
    scatter variable ("sigma" or "log_sigma"); `projection` is "moments" or "laplace".

    Returns `sigma_mean`, `sigma_std`, `log_sigma_mean`, `log_sigma_std`, `theta_mean`, `theta_std`,
    `mu_mean`, `mu_std`, `x_mean` (N,), `x_std` (N,), `sweeps`, `converged`, `max_delta`,
    `skipped` (site updates skipped because they would break positivity), `boundary_hits` (Laplace
    modes found on the support edge).
    """
    if projection not in ("moments", "laplace"):
        raise ValueError(f"projection must be 'moments' or 'laplace', got {projection!r}")

    ybar = np.asarray(ybar, dtype=float)
    v = np.asarray(v, dtype=float)
    n = ybar.size

    prior_mu = nat(m0, t0**2)
    lik_x = np.stack([nat(ybar[i], v[i]) for i in range(n)])

    # q(theta) starts at the hyper-prior's own moments; that Gaussian is the initial prior site.
    c0, cv0 = prior_theta_moments(prior, theta)
    site_prior_t = nat(c0, cv0)

    site_mu = np.zeros((n, 2))
    site_x = np.zeros((n, 2))
    site_t = np.zeros((n, 2))

    q_mu = prior_mu.copy()
    q_x = lik_x.copy()
    q_t = site_prior_t.copy()

    skipped = 0
    boundary_hits = 0
    converged = False
    max_delta = np.inf

    def blend(new, old):
        return damping * new + (1.0 - damping) * old

    for sweep in range(1, max_sweeps + 1):
        max_delta = 0.0
        skipped_before = skipped

        # Hierarchical factor sites.
        for i in range(n):
            a, av = moments(q_mu - site_mu[i])
            b, bv = moments(q_x[i] - site_x[i])
            c, cv = moments(q_t - site_t[i])
            if not proper(av, bv, cv):
                skipped += 1
                continue

            if projection == "moments":
                (e_mu, v_mu), (e_x, v_x), (e_t, v_t) = tilted_h_moments(a, av, b, bv, c, cv, prior, theta, n_grid)
            else:
                (e_mu, v_mu), (e_x, v_x), (e_t, v_t), boundary = tilted_h_laplace(
                    a, av, b, bv, c, cv, prior, theta, n_grid
                )
                boundary_hits += int(boundary)
            if not np.all(np.isfinite([e_mu, v_mu, e_x, v_x, e_t, v_t])) or min(v_mu, v_x, v_t) <= 0.0:
                skipped += 1
                continue

            new_mu = blend(nat(e_mu, v_mu) - nat(a, av), site_mu[i])
            new_x = blend(nat(e_x, v_x) - nat(b, bv), site_x[i])
            new_t = blend(nat(e_t, v_t) - nat(c, cv), site_t[i])

            q_mu_new = q_mu + new_mu - site_mu[i]
            q_x_new = q_x[i] + new_x - site_x[i]
            q_t_new = q_t + new_t - site_t[i]
            if not (q_mu_new[1] > 0.0 and q_x_new[1] > 0.0 and q_t_new[1] > 0.0):
                skipped += 1
                continue

            max_delta = max(
                max_delta,
                np.max(np.abs(new_mu - site_mu[i])),
                np.max(np.abs(new_x - site_x[i])),
                np.max(np.abs(new_t - site_t[i])),
            )
            q_mu, q_t = q_mu_new, q_t_new
            q_x[i] = q_x_new
            site_mu[i], site_x[i], site_t[i] = new_mu, new_x, new_t

        # Hyper-prior site on theta.
        c, cv = moments(q_t - site_prior_t)
        if proper(cv):
            if projection == "moments":
                e_t, v_t = tilted_prior_moments(c, cv, prior, theta, n_grid)
            else:
                e_t, v_t, boundary = tilted_prior_laplace(c, cv, prior, theta, n_grid)
                boundary_hits += int(boundary)
            if np.isfinite(v_t) and v_t > 0.0:
                new_p = blend(nat(e_t, v_t) - nat(c, cv), site_prior_t)
                q_t_new = q_t + new_p - site_prior_t
                if q_t_new[1] > 0.0:
                    max_delta = max(max_delta, np.max(np.abs(new_p - site_prior_t)))
                    q_t = q_t_new
                    site_prior_t = new_p
                else:
                    skipped += 1
            else:
                skipped += 1
        else:
            skipped += 1

        # A sweep in which some site could not be updated is not a fixed point, whatever max_delta says.
        if max_delta < tol and skipped == skipped_before:
            converged = True
            break

    mu_mean, mu_var = moments(q_mu)
    x_mean = np.array([moments(q_x[i])[0] for i in range(n)])
    x_var = np.array([moments(q_x[i])[1] for i in range(n)])
    c, cv = moments(q_t)

    out = theta_summary(c, cv, prior, theta)
    out.update(
        theta=theta,
        theta_mean=float(c),
        theta_std=float(np.sqrt(cv)),
        mu_mean=float(mu_mean),
        mu_std=float(np.sqrt(mu_var)),
        x_mean=x_mean,
        x_std=np.sqrt(x_var),
        sweeps=sweep,
        converged=converged,
        max_delta=float(max_delta),
        skipped=skipped,
        boundary_hits=boundary_hits,
        projection=projection,
    )
    return out


"""
__Comparison helpers__
"""


def deviations(ep, ref, theta_row):
    """
    max |dmean| / std_ref and max |std/std_ref - 1| over the rows (theta_row, mu, x_0..x_{N-1}), where
    `theta_row` is "sigma" or "log_sigma". Returns (max_a, max_b, per_row) with per_row a list of
    (name, mean_ep, std_ep, mean_ref, std_ref, a, b).
    """
    rows = [(theta_row, ep[f"{theta_row}_mean"], ep[f"{theta_row}_std"], ref[f"{theta_row}_mean"], ref[f"{theta_row}_std"])]
    rows.append(("mu", ep["mu_mean"], ep["mu_std"], ref["mu_mean"], ref["mu_std"]))
    for i in range(len(ref["x_mean"])):
        rows.append((f"x_{i}", ep["x_mean"][i], ep["x_std"][i], ref["x_mean"][i], ref["x_std"][i]))

    per_row = []
    for name, m_ep, s_ep, m_ref, s_ref in rows:
        a = abs(m_ep - m_ref) / s_ref
        b = abs(s_ep / s_ref - 1.0)
        per_row.append((name, m_ep, s_ep, m_ref, s_ref, a, b))
    return max(r[5] for r in per_row), max(r[6] for r in per_row), per_row


def print_table(per_row, label_ep="minimal EP"):
    print(f"    {'row':<10} {'closed form':>22}   {label_ep:>22}   {'|dmean|/std':>11} {'|std ratio-1|':>13}")
    for name, m_ep, s_ep, m_ref, s_ref, a, b in per_row:
        print(f"    {name:<10} {m_ref:>10.4f} +/- {s_ref:<8.4f}   {m_ep:>10.4f} +/- {s_ep:<8.4f}   {a:>11.2e} {b:>13.2e}")


"""
__Self-tests__
"""

TOL_LEG_A = 1e-10
# Issue #91 minimal EP (moments), leg B, per row after the seeds 0-4 calibration (module docstring).
TOL_LEG_B_MOMENTS = dict(
    scatter=dict(a=0.20, b=0.30),  # the sigma / log sigma row
    other=dict(a=0.05, b=0.16),  # mu and every x_i row
)
HARD_CAP_STD_ERROR = 0.50  # no |std / std_ref - 1| above this, on any row, ever
CALIBRATION_SEEDS = (0, 1, 2, 3, 4)
LAPLACE_DAMPING = 0.5  # undamped Laplace-EP sits on a ~1e-7 limit cycle (FD noise); damped it converges


def _check(label, value, tol, extra=""):
    ok = value <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {value:.3e} (tol {tol:.0e}) {extra}")
    return ok


def row_deviations(per_row):
    """
    Split the `deviations` rows into the scatter row (first) and the rest: returns
    (a_scatter, b_scatter, a_other, b_other) with a = max |dmean| / std_ref, b = max |std/std_ref - 1|.
    """
    scatter, others = per_row[0], per_row[1:]
    return scatter[5], scatter[6], max(r[5] for r in others), max(r[6] for r in others)


def _run_self_tests():
    t_start = time.time()
    ok = True

    data = simulate(seed=0)
    ybar, v, sigma_true = data["ybar"], data["v"], data["sigma_true"]

    print("Analytic Gaussian benchmark -- minimal EP self-tests (seed 0)")

    # (i) Leg A vs closed form.
    print("\n(i) Leg A: EP with exact Gaussian messages vs closed form")
    t0_ = time.time()
    ref_a = leg_a_reference(ybar, v, sigma_true)
    ep_a = ep_leg_a(ybar, v, sigma_true)
    dev_mean = max(
        abs(ep_a["mu_mean"] - ref_a["mu_mean"]) / ref_a["mu_std"],
        np.max(np.abs(ep_a["x_mean"] - ref_a["x_mean"]) / ref_a["x_std"]),
    )
    dev_std = max(
        abs(ep_a["mu_std"] / ref_a["mu_std"] - 1.0),
        np.max(np.abs(ep_a["x_std"] / ref_a["x_std"] - 1.0)),
    )
    print(f"  sweeps {ep_a['sweeps']}, converged {ep_a['converged']}, max |delta eta| {ep_a['max_delta']:.1e}  ({time.time() - t0_:.2f}s)")
    print(f"  mu = {ep_a['mu_mean']:.10f} +/- {ep_a['mu_std']:.10f}  (closed form {ref_a['mu_mean']:.10f} +/- {ref_a['mu_std']:.10f})")
    ok &= _check("leg A max |dmean|/std", dev_mean, TOL_LEG_A)
    ok &= _check("leg A max |std/std_ref - 1|", dev_std, TOL_LEG_A)
    ok &= _check("leg A converged", 0.0 if ep_a["converged"] else 1.0, 0.0)

    # (ii) Leg B: moments projection vs closed form for three hyper-priors on seeds 0-4 (per-row
    # tolerances and hard caps); the Laplace gap is reported on seed 0 only.
    print("\n(ii) Leg B: EP (moments) vs closed form on seeds 0-4; Laplace projection reported (seed 0)")
    cases = [
        (("gaussian", 10.0, 5.0), "sigma"),
        (("truncated", 10.0, 5.0, 0.0, 100.0), "sigma"),
        (("loggaussian", np.log(10.0), 0.5), "log_sigma"),
    ]
    tol_s, tol_o = TOL_LEG_B_MOMENTS["scatter"], TOL_LEG_B_MOMENTS["other"]
    for prior, theta in cases:
        print(f"\n  prior {prior}, theta = {theta}")
        worst = dict(a_scatter=0.0, b_scatter=0.0, a_other=0.0, b_other=0.0, b_all=0.0)
        inside = True
        all_converged = True
        for seed in CALIBRATION_SEEDS:
            d = simulate(seed=seed)
            ref_s = leg_b_reference(d["ybar"], d["v"], prior)
            ep_s = ep_leg_b(d["ybar"], d["v"], prior, theta=theta, projection="moments")
            _, b_all, rows_s = deviations(ep_s, ref_s, theta)
            a_sc, b_sc, a_ot, b_ot = row_deviations(rows_s)
            in_interval = ref_s["sigma_q05"] <= ep_s["sigma_mean"] <= ref_s["sigma_q95"]
            inside &= in_interval
            all_converged &= ep_s["converged"]
            for key, val in zip(("a_scatter", "b_scatter", "a_other", "b_other", "b_all"), (a_sc, b_sc, a_ot, b_ot, b_all)):
                worst[key] = max(worst[key], val)
            print(
                f"    seed {seed}: sweeps {ep_s['sweeps']:>3}, converged {ep_s['converged']}, skipped {ep_s['skipped']}; "
                f"scatter row a {a_sc:.3f} b {b_sc:.3f}; mu/x rows a {a_ot:.3f} b {b_ot:.3f}; "
                f"E[sigma] {ep_s['sigma_mean']:.3f} in [q05, q95] = [{ref_s['sigma_q05']:.3f}, {ref_s['sigma_q95']:.3f}]: {in_interval}"
            )
            if seed == 0:
                ep_m, ref_b, rows_m = ep_s, ref_s, rows_s

        print("  seed 0 table:")
        print_table(rows_m, "EP (moments)")
        other = "log_sigma" if theta == "sigma" else "sigma"
        print(
            f"    [{other} row, info] EP {ep_m[f'{other}_mean']:.4f} +/- {ep_m[f'{other}_std']:.4f}  "
            f"closed form {ref_b[f'{other}_mean']:.4f} +/- {ref_b[f'{other}_std']:.4f};  "
            f"closed-form sigma q05/q50/q95 = {ref_b['sigma_q05']:.4f}/{ref_b['sigma_q50']:.4f}/{ref_b['sigma_q95']:.4f}"
        )
        ok &= _check(f"{prior[0]} moments, seeds 0-4: scatter row max |dmean|/std_ref", worst["a_scatter"], tol_s["a"])
        ok &= _check(f"{prior[0]} moments, seeds 0-4: scatter row max |std/std_ref - 1|", worst["b_scatter"], tol_s["b"])
        ok &= _check(f"{prior[0]} moments, seeds 0-4: mu/x rows max |dmean|/std_ref", worst["a_other"], tol_o["a"])
        ok &= _check(f"{prior[0]} moments, seeds 0-4: mu/x rows max |std/std_ref - 1|", worst["b_other"], tol_o["b"])
        ok &= _check(f"{prior[0]} moments, seeds 0-4: hard cap, any row |std/std_ref - 1|", worst["b_all"], HARD_CAP_STD_ERROR)
        ok &= _check(f"{prior[0]} moments, seeds 0-4: hard cap, E[sigma] outside [q05, q95] (count)", 0.0 if inside else 1.0, 0.0)
        ok &= _check(f"{prior[0]} moments, seeds 0-4: converged on every seed", 0.0 if all_converged else 1.0, 0.0)

        t0_ = time.time()
        ep_l = ep_leg_b(ybar, v, prior, theta=theta, projection="laplace", damping=LAPLACE_DAMPING)
        t_l = time.time() - t0_
        a_l, b_l, rows_l = deviations(ep_l, ref_b, theta)
        a_ml, b_ml, _ = deviations(ep_l, ep_m, theta)
        print(
            f"  laplace: sweeps {ep_l['sweeps']}, converged {ep_l['converged']}, "
            f"max |delta eta| {ep_l['max_delta']:.1e}, skipped {ep_l['skipped']}, "
            f"boundary hits {ep_l['boundary_hits']}  ({t_l:.2f}s)"
        )
        print_table(rows_l, "EP (laplace)")
        print(
            f"    [report] laplace vs closed form: max |dmean|/std_ref {a_l:.3e}, max |std ratio - 1| {b_l:.3e};  "
            f"laplace vs moments (std of moments): {a_ml:.3e}, {b_ml:.3e}"
        )

    elapsed = time.time() - t_start
    print(f"\nTotal runtime {elapsed:.2f}s")
    ok &= _check("runtime (s)", elapsed, 60.0)

    print(f"\nMINIMAL EP SELF-TESTS: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not _run_self_tests():
        sys.exit(1)
