"""
Integration Test: Analytic Gaussian Benchmark -- Hyper-Prior Family Sweep through autofit EP
=============================================================================================

The prior-family stress test of autofit_workspace_test#91 (leg B, sigma unknown): the same conjugate
hierarchical Gaussian model with three hyper-prior families on the parent scatter, each run through
autofit's EP (`factor_graph.optimise(af.LaplaceOptimiser(), ...)`, no sampler) and judged against its
own closed form (`analytic_reference.leg_b_reference`) and the minimal hand-rolled EP
(`analytic_ep_minimal.ep_leg_b`, moments projection):

    gaussian       af.GaussianPrior(10, 5)                          theta = sigma
    truncated      af.TruncatedGaussianPrior(10, 5, 0, 100)         theta = sigma
    loggaussian    af.LogGaussianPrior(log 10, 0.5)                 theta = log sigma

`theta` is the parametrisation of the scatter row: the minimal EP keeps theta = sigma for the
gaussian / truncated priors and theta = log sigma for the loggaussian prior, mirroring autofit's
message per family (a `NormalMessage` / `TruncatedNormalMessage` in sigma, a `TransformedMessage` over
log sigma). The scatter row of the loggaussian table is therefore log sigma, read from the
`base_message` of autofit's `TransformedMessage`. The shared wiring and the message traps are in
`analytic_autofit.py`; the drawn prior `GaussianPrior(50, 20)` on every x_i is folded exactly into the
reference (see there).

__PyAutoFit#1498 fingerprint (loggaussian)__

For the loggaussian family the closed form is also evaluated with the deliberately wrong prior density
`("loggaussian_no_jacobian", ...)` -- the log-space normal evaluated at log sigma WITHOUT the 1/sigma
Jacobian, i.e. a log-Gaussian shifted to N(ml + sl^2, sl^2) in log sigma. The attribution rule:

    minimal EP == closed form  and  autofit EP != closed form   =>  library finding (EP-the-method is fine)
    autofit EP == no-Jacobian reference                          =>  PyAutoFit#1498 confirmed
    autofit EP == closed form                                    =>  #1498 not reproduced here
    otherwise                                                    =>  library finding, not the #1498 fingerprint

where "==" means within the column's tolerance on the scatter row. The verdict is printed explicitly.
`af.LogGaussianPrior.factor(x)` itself (the density the `PriorFactor` and the joint fit use) was checked
by hand to include the Jacobian; the EP route is what this script fingerprints.

__Tolerances (issue #91; a = |dmean| / std_ref, b = |std / std_ref - 1|)__

    minimal EP (moments):   scatter row a 0.20, b 0.30; mu and x_i rows a 0.05, b 0.16
                            (per-row calibration over seeds 0-4 recorded in `analytic_ep_minimal.py`)
    autofit EP (Laplace):   a 0.15, b 0.25 on every row
    hard caps:              every EP column's E[sigma] inside the closed-form [q05, q95]; no std error > 50%

For the loggaussian family autofit's E[sigma] is exp(m + s^2/2) of its log-space message. The script
ends `PARITY: PASS|FAIL (k/n)` and exits 1 on any failure; nothing is loosened to pass. The library's
own EP diagnostics (`ep_diagnostics.results` warnings, `ep_history.csv` status flags) are printed per
family so a failure can be read against them.

__What the first run showed (seed 0, 2026-09-02, PyAutoFit 2026.8.17.1)__

The minimal EP passes every cell of every family. Autofit EP (PARITY: FAIL, 39/48):

    gaussian       sigma and mu never updated (10.0 +/- 2.2 and 50.0 +/- 14.2 are the starting mean
                   field; hierarchical factors: BAD_PROJECTION 110, FAILURE 35, no SUCCESS)
    truncated      biased-tight sigma 3.73 +/- 0.75 (closed form 6.57 +/- 2.88), mu std 69% low
    loggaussian    log sigma never updated (2.303 +/- 0.456 is the starting mean field; all 150
                   hierarchical-factor updates BAD_PROJECTION); mu and x_i within tolerance

#1498 verdict: LIBRARY FINDING, not the #1498 fingerprint -- minimal EP matches the closed form
(1.816 +/- 0.372 vs 1.834 +/- 0.361) while autofit EP matches neither the closed form (a 1.30) nor the
no-Jacobian reference (a 0.89): the log-space message is simply never projected.

__Status__

Parked NEEDS_FIX 2026-09-02 in `config/build/no_run.yaml`: the autofit-EP column fails against the
closed form because of the PyAutoFit defects tracked under PyAutoFit#1405 / autofit_workspace_test#91
(D2 Laplace covariance, D4 truncation limits and D5 log-space transform lost in projection). The
script is intentionally left exit-1-on-fail as the regression check that turns green with the fix;
the closed form, minimal EP and graphical columns pass.

"""

import sys
import time

import numpy as np

import autofit as af

from analytic_autofit import (
    MU_PRIOR,
    DRAWN_PRIOR,
    build_factor_graph,
    effective_statistics,
    ep_diagnostics_text,
    ep_flag_summary,
    parity_table,
    read_ep_posteriors,
    run_autofit_ep,
)
from analytic_ep_minimal import ep_leg_b
from analytic_reference import leg_b_reference, simulate

"""
__Settings__
"""
SEED = 0
EP_KWARGS = dict(kl_tol=1e-4, max_steps=30)

FAMILIES = [
    # (label, reference prior tuple, autofit hyper-prior factory, theta)
    (
        "gaussian",
        ("gaussian", 10.0, 5.0),
        lambda: af.GaussianPrior(mean=10.0, sigma=5.0),
        "sigma",
    ),
    (
        "truncated",
        ("truncated", 10.0, 5.0, 0.0, 100.0),
        lambda: af.TruncatedGaussianPrior(
            mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=100.0
        ),
        "sigma",
    ),
    (
        "loggaussian",
        ("loggaussian", float(np.log(10.0)), 0.5),
        lambda: af.LogGaussianPrior(mean=float(np.log(10.0)), sigma=0.5),
        "log_sigma",
    ),
]

TOLERANCES = {
    "minimal EP": dict(scatter=dict(a=0.20, b=0.30), other=dict(a=0.05, b=0.16)),
    "autofit EP": dict(a=0.15, b=0.25),
}
HARD_CAP_STD_ERROR = 0.50


def tolerance_for(column, row):
    tol = TOLERANCES[column]
    if "scatter" in tol:
        return tol["scatter"] if row in ("sigma", "log_sigma") else tol["other"]
    return tol


def within(mean, std, ref_mean, ref_std, tol):
    """True if (mean, std) is within tol of (ref_mean, ref_std) by the a / b rule."""
    return (
        abs(mean - ref_mean) / ref_std <= tol["a"]
        and abs(std / ref_std - 1.0) <= tol["b"]
    )


"""
__Dataset__
"""
sim = simulate(seed=SEED)
ybar_eff, v_eff = effective_statistics(sim)
x_rows = [f"x_{i}" for i in range(len(sim["y"]))]

print(
    f"Analytic Gaussian benchmark -- hyper-prior family sweep through autofit EP (seed {SEED}, N={len(sim['y'])})"
)
print(
    f"  ybar' = {np.round(ybar_eff, 4)}   v' = {np.round(v_eff, 5)}   (drawn prior N{DRAWN_PRIOR} folded in)"
)

passed_total = cells_total = 0
verdict_1498 = None

"""
__Sweep__
"""
for label, prior, hyper_prior_factory, theta in FAMILIES:
    scatter = "sigma" if theta == "sigma" else "log_sigma"

    t0 = time.time()
    ref = leg_b_reference(ybar_eff, v_eff, prior, m0=MU_PRIOR[0], t0=MU_PRIOR[1])
    t_ref = time.time() - t0

    t0 = time.time()
    minimal = ep_leg_b(
        ybar_eff,
        v_eff,
        prior,
        m0=MU_PRIOR[0],
        t0=MU_PRIOR[1],
        theta=theta,
        projection="moments",
    )
    t_min = time.time() - t0

    factor_graph, hf, models = build_factor_graph(
        sim,
        lambda: af.GaussianPrior(mean=DRAWN_PRIOR[0], sigma=DRAWN_PRIOR[1]),
        dict(
            mean=af.GaussianPrior(mean=MU_PRIOR[0], sigma=MU_PRIOR[1]),
            sigma=hyper_prior_factory(),
        ),
    )
    priors = {
        scatter: hf.sigma,
        "mu": hf.mean,
        **{r: m.x for r, m in zip(x_rows, models)},
    }
    name = f"analytic_gaussian_priors_{label}"

    t0 = time.time()
    ep_result = run_autofit_ep(factor_graph, name, seed=SEED, **EP_KWARGS)
    autofit_ep = read_ep_posteriors(ep_result, priors)
    t_ep = time.time() - t0

    if theta == "log_sigma":
        # autofit's message is over log sigma: E[sigma] of the implied log-normal.
        m_log, s_log = autofit_ep[scatter][:2]
        autofit_sigma_mean = float(np.exp(m_log + 0.5 * s_log**2))
    else:
        autofit_sigma_mean = autofit_ep[scatter][0]

    rows = [
        (scatter, ref[f"{scatter}_mean"], ref[f"{scatter}_std"]),
        ("mu", ref["mu_mean"], ref["mu_std"]),
    ] + [(r, ref["x_mean"][i], ref["x_std"][i]) for i, r in enumerate(x_rows)]
    columns = {
        "minimal EP": {
            scatter: (minimal[f"{scatter}_mean"], minimal[f"{scatter}_std"]),
            "mu": (minimal["mu_mean"], minimal["mu_std"]),
            **{
                r: (minimal["x_mean"][i], minimal["x_std"][i])
                for i, r in enumerate(x_rows)
            },
        },
        "autofit EP": autofit_ep,
    }
    interval = (ref["sigma_q05"], ref["sigma_q95"])
    hard_caps = [
        (
            f"hard cap: minimal EP E[sigma] = {minimal['sigma_mean']:.4f} inside [q05, q95] = [{interval[0]:.4f}, {interval[1]:.4f}]",
            interval[0] <= minimal["sigma_mean"] <= interval[1],
        ),
        (
            f"hard cap: autofit EP E[sigma] = {autofit_sigma_mean:.4f} inside [q05, q95] = [{interval[0]:.4f}, {interval[1]:.4f}]",
            interval[0] <= autofit_sigma_mean <= interval[1],
        ),
    ]
    p, n = parity_table(
        rows,
        columns,
        tolerance_for,
        f"{label}: prior {prior}, scatter row = {scatter}; closed-form sigma q05/q50/q95 = "
        f"{ref['sigma_q05']:.4f}/{ref['sigma_q50']:.4f}/{ref['sigma_q95']:.4f}",
        extra_checks=hard_caps,
        hard_cap_std_error=HARD_CAP_STD_ERROR,
    )
    passed_total += p
    cells_total += n
    print(
        f"  [info] autofit EP scatter message: {autofit_ep[scatter][2] or 'NormalMessage mean/std'}"
    )
    print(f"  [info] autofit EP flags: {ep_flag_summary(name)}")
    diagnostics = ep_diagnostics_text(name)
    warnings_block = (
        diagnostics.split("WARNINGS", 1)[1].strip()
        if "WARNINGS" in diagnostics
        else "(no warnings)"
    )
    print(f"  [info] ep_diagnostics.results warnings: {warnings_block}")
    print(
        f"  [info] minimal EP sweeps {minimal['sweeps']}, converged {minimal['converged']}, skipped {minimal['skipped']}"
    )
    print(
        f"  runtimes: closed form {t_ref:.1f}s, minimal EP {t_min:.1f}s, autofit EP {t_ep:.1f}s"
    )

    """
    __#1498 fingerprint__
    """
    if label == "loggaussian":
        ref_nj = leg_b_reference(
            ybar_eff,
            v_eff,
            ("loggaussian_no_jacobian", prior[1], prior[2]),
            m0=MU_PRIOR[0],
            t0=MU_PRIOR[1],
        )
        ep_m, ep_s = autofit_ep[scatter][:2]
        min_m, min_s = minimal["log_sigma_mean"], minimal["log_sigma_std"]
        minimal_ok = within(
            min_m,
            min_s,
            ref["log_sigma_mean"],
            ref["log_sigma_std"],
            TOLERANCES["minimal EP"]["scatter"],
        )
        autofit_ok = within(
            ep_m,
            ep_s,
            ref["log_sigma_mean"],
            ref["log_sigma_std"],
            TOLERANCES["autofit EP"],
        )
        autofit_nj = within(
            ep_m,
            ep_s,
            ref_nj["log_sigma_mean"],
            ref_nj["log_sigma_std"],
            TOLERANCES["autofit EP"],
        )
        print("\n  #1498 fingerprint (log sigma row):")
        print(
            f"    closed form (with Jacobian)   {ref['log_sigma_mean']:.4f} +/- {ref['log_sigma_std']:.4f}"
        )
        print(
            f"    closed form (no Jacobian)     {ref_nj['log_sigma_mean']:.4f} +/- {ref_nj['log_sigma_std']:.4f}"
        )
        print(
            f"    minimal EP                    {min_m:.4f} +/- {min_s:.4f}   == closed form: {minimal_ok}"
        )
        print(
            f"    autofit EP                    {ep_m:.4f} +/- {ep_s:.4f}   == closed form: {autofit_ok} "
            f"(a={abs(ep_m - ref['log_sigma_mean']) / ref['log_sigma_std']:.3f}, b={abs(ep_s / ref['log_sigma_std'] - 1):.3f}); "
            f"== no-Jacobian reference: {autofit_nj} "
            f"(a={abs(ep_m - ref_nj['log_sigma_mean']) / ref_nj['log_sigma_std']:.3f}, b={abs(ep_s / ref_nj['log_sigma_std'] - 1):.3f})"
        )
        if autofit_ok:
            verdict_1498 = "#1498 NOT reproduced: autofit EP matches the closed form (with Jacobian)"
        elif autofit_nj:
            verdict_1498 = (
                "PyAutoFit#1498 CONFIRMED: autofit EP matches the no-Jacobian reference"
            )
        elif minimal_ok:
            verdict_1498 = "LIBRARY FINDING (not the #1498 fingerprint): minimal EP matches the closed form, autofit EP matches neither reference"
        else:
            verdict_1498 = (
                "INCONCLUSIVE: neither EP matches the closed form on the log sigma row"
            )
        print(f"    VERDICT: {verdict_1498}")

"""
__Verdict__
"""
print(f"\n#1498 attribution: {verdict_1498}")
verdict = "PASS" if passed_total == cells_total else "FAIL"
print(f"PARITY: {verdict} ({passed_total}/{cells_total})")
if verdict == "FAIL":
    sys.exit(1)
