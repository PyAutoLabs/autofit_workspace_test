"""
Integration Test: Analytic Gaussian Benchmark -- The Phase-2 Scale-Collapse Configuration
==========================================================================================

The exact prior configuration of the phase-2 EP scale-collapse study (PyAutoMind
`draft/bug/autofit/ep_scale_collapse_leg2_assets/{toy,run_once}.py`, PyAutoFit#1405) run on the
conjugate benchmark of autofit_workspace_test#91, where the answer is known in closed form:

    drawn x_i      af.TruncatedGaussianPrior(50, 20, 0, 100)
    parent mean    af.TruncatedGaussianPrior(50, 10, 0, 100)
    parent scatter af.TruncatedGaussianPrior(10, 5, 0, 100)
    EP             af.LaplaceOptimiser(), af.EPHistory(kl_tol=0.05), max_steps=20, default updater
    N = 5 datasets, seeds 0-4

`toy.py` fits `af.ex.Gaussian` profiles with a `DynestyStatic` per analysis factor; `run_once.py`'s
`TOY_OPT=laplace` lever swaps that for `LaplaceOptimiser`, which is what runs here (every factor
Laplace; five seeds must fit in the ~120 s script budget). The analysis factors are exactly Gaussian
in x_i, so the per-dataset likelihood is reproduced without a sampler.

__Reference__

`analytic_reference.leg_b_reference` with the `("truncated", 10, 5, 0, 100)` hyper-prior on sigma, a
Gaussian N(50, 10^2) hyper-prior on mu and the drawn prior N(50, 20^2) folded into (ybar', v') (see
`analytic_autofit.py`). The truncations of the mu and x_i priors at [0, 100] are exact no-ops for the
reference: a truncated prior is the same density times an indicator, and the posterior mass of mu
(~51 +/- 3) and of every x_i (45-58 +/- 2) outside [0, 100] is below 1e-50, so the closed form of the
untruncated priors IS the posterior of the truncated ones.

__Per-seed classification (the phase-2 labels of `classify.py`, against the closed form)__

    PATHOLOGICAL   E_EP[sigma] < 0.2 x 10 (the hyper-prior mean) and std/mean < 0.5: near-zero scatter
                   reported with a tiny relative error (the #1405 signature)
    BIASED-TIGHT   |E_ref[sigma] - E_EP[sigma]| > 3 std_EP: wrong and over-confident, not caught by the
                   mean-fraction gate
    RECOVER        the closed-form E[sigma] lies within 3 std_EP of the EP value
    STALE          no hierarchical-factor update ever succeeded (`ep_history.csv` has no SUCCESS for the
                   HierarchicalFactor): the reported sigma is the starting mean-field prior, not a
                   posterior, whatever interval it happens to sit in (the #1405 stale-factor state; the
                   library's STALE FACTORS warning only fires for optimisers that raise, not for a
                   BAD_PROJECTION on every sweep)

alongside whether E_EP[sigma] lies inside the closed-form [q05, q95] (q95 is the analytic upper limit
on the scatter). After each run the library's own `ep_diagnostics.results` is read back from
`output/graphical/analytic_gaussian_collapse_seed<k>/<identifier>/` and its WARNINGS block printed,
with the `ep_history.csv` status-flag tally.

__Acceptance (issue #91: "sigma inside [q05, q95] or documented collapse")__

A seed passes if E_EP[sigma] is a posterior (at least one hierarchical-factor SUCCESS) lying inside
the closed-form [q05, q95], or if the run is a collapse (PATHOLOGICAL / BIASED-TIGHT / STALE) that the
library itself documents with a scale-collapse or stale-factor warning in `ep_diagnostics.results`. A
silent collapse -- outside the interval, or a stale prior, with no library warning -- fails. The
script ends `COLLAPSE CONFIG: PASS|FAIL (k/n seeds)` and exits 1 on failure; the per-seed table is
the evidence banked for phase 2, whichever way it falls.

__What the first, pre-fix run showed (seeds 0-4, 2026-09-02, PyAutoFit 2026.8.17.1 before #1558/#1560/#1562; ~35 s)__

    seed 0   STALE          10.00 +/- 3.69  (closed form 6.57, [q05, q95] = [2.98, 12.15]); 10/10 BAD_PROJECTION
    seed 1   RECOVER         7.94 +/- 1.34  (closed form 10.46, [6.26, 15.99])
    seed 2   BIASED-TIGHT    7.78 +/- 0.51  (closed form 13.57, [9.22, 18.94]) -- silent, no library warning
    seed 3   RECOVER        13.09 +/- 2.09  (closed form 14.35, [10.00, 19.65])
    seed 4   RECOVER        10.87 +/- 2.61  (closed form 12.48, [8.17, 17.89])

COLLAPSE CONFIG: FAIL (3/5 seeds). No seed reached the fully PATHOLOGICAL state at kl_tol 0.05 /
max_steps 20 (the run stops after 1-7 sweeps); `analytic_gaussian.py`'s leg B (kl_tol 1e-4, 30 steps)
on the same seed-0 data does collapse to a biased-tight 3.7 +/- 0.75, and depending on process
history to sigma ~ 1e-4 with the library's scale-collapse warning.

__What the same run shows on PyAutoFit main 5375f4d63 (#1558/#1560/#1562 merged; 2026-09-02, ~17 s)__

    seed 0   RECOVER         9.24 +/- 3.61  (closed form 6.57, [q05, q95] = [2.98, 12.15])
    seed 1   RECOVER        10.73 +/- 3.21  (closed form 10.46, [6.26, 15.99])
    seed 2   RECOVER        13.35 +/- 2.92  (closed form 13.57, [9.22, 18.94])
    seed 3   RECOVER        13.88 +/- 2.88  (closed form 14.35, [10.00, 19.65])
    seed 4   RECOVER        12.21 +/- 3.03  (closed form 12.48, [8.17, 17.89])

COLLAPSE CONFIG: PASS (5/5 seeds): every seed's E_EP[sigma] is a posterior (HierarchicalFactor
SUCCESS on every seed) inside the closed-form [q05, q95]; no library warning is needed or emitted.

__Status__

Curated into the smoke gate since 2026-09-02: with PyAutoFit#1558/#1560/#1562 every seed reads
RECOVER; a SILENT, STALE or PATHOLOGICAL verdict is a regression.

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
    read_ep_posteriors,
    run_autofit_ep,
)
from analytic_reference import leg_b_reference, simulate

"""
__Settings__
"""
SEEDS = (0, 1, 2, 3, 4)
N_DATASETS = 5
SIGMA_PRIOR = ("truncated", 10.0, 5.0, 0.0, 100.0)
EP_KWARGS = dict(kl_tol=0.05, max_steps=20)

INITIAL_SCATTER = 10.0  # the hyper-prior mean, `classify.py`'s INITIAL
GUARD_MEAN_FRACTION = 0.2
GUARD_RELATIVE_ERROR = 0.5


def classify(scatter, err, ref_mean, updated):
    """The phase-2 labels (module docstring), with the closed-form E[sigma] as the target."""
    if not updated:
        return "STALE"
    if (
        scatter < GUARD_MEAN_FRACTION * INITIAL_SCATTER
        and (err / scatter if scatter > 0.0 else 0.0) < GUARD_RELATIVE_ERROR
    ):
        return "PATHOLOGICAL"
    pull = abs(ref_mean - scatter) / err if err > 0.0 else np.inf
    return "BIASED-TIGHT" if pull > 3.0 else "RECOVER"


def truncated(mean, sigma):
    return af.TruncatedGaussianPrior(
        mean=mean, sigma=sigma, lower_limit=0.0, upper_limit=100.0
    )


"""
__Seeds__
"""
print(
    "Analytic Gaussian benchmark -- phase-2 collapse configuration (truncated priors, kl_tol 0.05, max_steps 20, Laplace)"
)
print(
    f"  {'seed':<5}{'ref E[sigma]':>13}{'q05':>8}{'q50':>8}{'q95':>8} | {'EP sigma':>10} +/- {'std':<9} {'inside':<7}{'class':<14}{'ref mu':>10} | {'EP mu':>10} +/- {'std':<8} {'time':>6}"
)

results = []
for seed in SEEDS:
    sim = simulate(seed=seed, n_datasets=N_DATASETS)
    ybar_eff, v_eff = effective_statistics(sim)
    ref = leg_b_reference(ybar_eff, v_eff, SIGMA_PRIOR, m0=MU_PRIOR[0], t0=MU_PRIOR[1])

    factor_graph, hf, models = build_factor_graph(
        sim,
        lambda: truncated(*DRAWN_PRIOR),
        dict(
            mean=truncated(*MU_PRIOR), sigma=truncated(SIGMA_PRIOR[1], SIGMA_PRIOR[2])
        ),
    )
    priors = {"sigma": hf.sigma, "mu": hf.mean}
    name = f"analytic_gaussian_collapse_seed{seed}"

    t0 = time.time()
    ep_result = run_autofit_ep(factor_graph, name, seed=seed, **EP_KWARGS)
    elapsed = time.time() - t0
    post = read_ep_posteriors(ep_result, priors)

    scatter, err = post["sigma"][:2]
    inside = ref["sigma_q05"] <= scatter <= ref["sigma_q95"]
    flags = ep_flag_summary(name)
    updated = (
        "HierarchicalFactor" in flags
        and "SUCCESS" in flags.split("HierarchicalFactor", 1)[1].split(";")[0]
    )
    label = classify(scatter, err, ref["sigma_mean"], updated)
    diagnostics = ep_diagnostics_text(name)
    warnings_block = (
        diagnostics.split("WARNINGS", 1)[1].strip() if "WARNINGS" in diagnostics else ""
    )
    documented = ("scale-collapse" in warnings_block) or (
        "STALE FACTORS" in warnings_block
    )
    ok = (inside and updated) or (label != "RECOVER" and documented)
    results.append(
        dict(
            seed=seed,
            ok=ok,
            label=label,
            inside=inside,
            documented=documented,
            scatter=scatter,
            err=err,
            ref=ref,
        )
    )

    print(
        f"  {seed:<5}{ref['sigma_mean']:>13.4f}{ref['sigma_q05']:>8.3f}{ref['sigma_q50']:>8.3f}{ref['sigma_q95']:>8.3f} | "
        f"{scatter:>10.4f} +/- {err:<9.3g} {str(inside):<7}{label:<14}{ref['mu_mean']:>10.4f} | "
        f"{post['mu'][0]:>10.4f} +/- {post['mu'][1]:<8.3g} {elapsed:>5.1f}s"
    )
    print(f"        sigma message: {post['sigma'][2]}")
    print(f"        flags: {flags}")
    print(
        f"        ep_diagnostics.results warnings: {warnings_block if warnings_block else '(none)'}"
    )
    print(
        f"        seed verdict: {'PASS' if ok else 'FAIL'} ({'posterior inside [q05, q95]' if (inside and updated) else ('documented collapse' if documented else 'SILENT ' + label)})"
    )

"""
__Verdict__
"""
tally = {
    k: sum(r["label"] == k for r in results)
    for k in ("RECOVER", "BIASED-TIGHT", "PATHOLOGICAL", "STALE")
}
n_ok = sum(r["ok"] for r in results)
print(
    f"\nclassification: "
    + ", ".join(f"{k} {v}/{len(results)}" for k, v in tally.items())
    + f"; inside [q05, q95] on {sum(r['inside'] for r in results)}/{len(results)} seeds; library warning on {sum(r['documented'] for r in results)}/{len(results)}"
)
verdict = "PASS" if n_ok == len(results) else "FAIL"
print(f"COLLAPSE CONFIG: {verdict} ({n_ok}/{len(results)} seeds)")
if verdict == "FAIL":
    sys.exit(1)
