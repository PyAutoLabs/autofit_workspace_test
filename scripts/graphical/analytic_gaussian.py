"""
Integration Test: Analytic Gaussian Benchmark -- Parity (closed form | minimal EP | graphical | EP)
=====================================================================================================

The CI parity script of autofit_workspace_test#91. One conjugate hierarchical Gaussian model is solved
four ways and every column is judged against the closed form of `analytic_reference.py`:

    closed form          exact posterior (leg A analytic, leg B by deterministic quadrature)
    minimal EP           the hand-rolled moments-projection EP of `analytic_ep_minimal.py`
    autofit graphical    a joint `DynestyStatic` fit of `factor_graph.global_prior_model`
    autofit EP           `factor_graph.optimise(af.LaplaceOptimiser(), ...)`

The autofit wiring (graph construction, posterior read-out with its message traps, the fact that the
drawn prior is an extra factor and how the reference accounts for it exactly) is documented in
`analytic_autofit.py`, which this script and its two siblings share.

__Model__

    mu ~ N(50, 10^2),   x_i | mu ~ N(mu, sigma^2),   y_ij | x_i ~ N(x_i, s_i^2)      i = 1..5, j = 1..20

Datasets are simulated in memory from `analytic_reference.simulate(SEED)` (truth mu = 50, sigma = 10,
noise s_i = linspace(3, 8)); nothing is written under `dataset/`. Every x_i carries the drawn prior
`GaussianPrior(50, 20)`, folded into the reference as (ybar', v'). Leg A fixes sigma = 10 (a
`Constant` in the `HierarchicalFactor`); leg B puts `TruncatedGaussianPrior(mean=10, sigma=5, 0, 100)`
on it -- the phase-2 collapse configuration, whose exact sigma posterior is skewed (closed-form
q05/q50/q95 on seed 0 = 3.0/6.0/12.2) with the analytic upper limit `sigma_q95` printed alongside.

__Tolerances (issue #91 table; a = |dmean| / std_ref, b = |std / std_ref - 1|)__

    column                    leg A (a, b)     leg B (a, b)
    minimal EP (moments)      1e-6, 1e-6       scatter row 0.20, 0.30; mu and x_i rows 0.05, 0.16
    autofit EP (Laplace)      0.01, 0.02       0.15, 0.25
    autofit graphical         0.10, 0.15       0.10, 0.15

The leg-B minimal-EP values are the per-row calibration recorded in `analytic_ep_minimal.py` (2x the
value observed over seeds 0-4, rounded up). Hard caps that no calibration may relax: every EP
column's E[sigma] lies inside the closed-form [q05, q95], and no std error exceeds 50% in any cell.
A cell passes iff both a and b are within tolerance; the script ends `PARITY: PASS|FAIL (k/n)` and
exits 1 on any failure. Tolerances are never loosened to make a run pass: a failing autofit cell is
reported as measured and left for the campaign ledger to classify.

__Graphical column__

The sampler cell is the weighted posterior mean and std of the samples (like for like with the closed
form and EP); `median_pdf` +/- the averaged `errors_at_sigma(1.0)` is printed as an information line.
On leg B's skewed sigma posterior the median sits 0.2 reference-std below the mean by construction
(q50 = 6.0 vs E = 6.6 on seed 0), which is not a sampler error.

`DynestyStatic` at the `ep_parity.py` budget (nlive 50, rwalk, maxcall 3000) stops on `maxcall` far
from convergence on this 6/7-parameter joint model (a = 2.5). Each likelihood call costs ~5 ms (the
hierarchical `GaussianPrior` distribution is instantiated per call), so the budget is set by the
number of calls: `sample="unif"` with `bound="multi", bootstrap=0` converges in ~7-10k calls per
leg (37 s / 50 s locally); the default bootstrap enlargement needs ~20k calls on leg B for the same
posterior. Every column prints its wall time; the script runs in ~115 s locally against the 300 s
CI cap.

__What the first run showed (seed 0, 2026-09-02, PyAutoFit 2026.8.17.1)__

    closed form vs minimal EP:      every cell PASS (leg A to 1e-15; leg B scatter a 0.078, b 0.146)
    closed form vs autofit graphical: every cell PASS (leg A max a 0.055, b 0.048; leg B max a 0.085, b 0.045)
    closed form vs autofit EP:      leg A 2/6 PASS, leg B 3/9 PASS -- PARITY: FAIL (33/41)

Autofit EP, leg A (exactly Gaussian graph, so Laplace should be exact): mu never updated -- it is
returned at its starting mean-field prior 50.0 +/- 20.6 while `ep_history.csv` shows the five
hierarchical factors ending every sweep in BAD_PROJECTION (54) or FAILURE (36) with no SUCCESS; the
x_i means are off by 0.015-0.079 std_ref against the 0.01 tolerance. The outcome depends on the prior
ids / process history: the identical graph built after 1, 3 or 6 other priors returns different
fixed points (x_0 stale; x_2 std 33% low; all rows within ~5%) -- see the phase-(b) report.

Autofit EP, leg B (truncated hyper-prior): the scatter collapses to a biased, over-confident
3.73 +/- 0.75 (closed form 6.57 +/- 2.88, still inside [q05, q95] = [2.98, 12.15]); the returned
`TruncatedNormalMessage` has lost its limits (-inf, inf); mu's std is 69% low. Depending on process
history the same graph also collapses fully to sigma ~ 1e-4 with the library's scale-collapse
warning firing. The minimal EP with a Laplace projection reproduces the collapse deterministically
(`analytic_ep_minimal.py`): the tilted density of every hierarchical site is unbounded as sigma -> 0.

__Status__

Parked NEEDS_FIX 2026-09-02 in `config/build/no_run.yaml`: the autofit-EP column fails against the
closed form because of the PyAutoFit defects tracked under PyAutoFit#1405 / autofit_workspace_test#91
(D1 id-0 prior / FactorValue collision, D2/D3 Laplace projection). The script is intentionally left
exit-1-on-fail as the regression check that turns green with the fix; the closed form, minimal EP and
graphical columns pass.

__Env__ (Developer Only)

The graphical joint-fit column runs a real DynestyStatic search; test-mode 2 would replace it with a
midpoint vector and fake the column.

ENV: real_search
"""

import sys
import time

import numpy as np

import autofit as af

from analytic_autofit import (
    DRAWN_PRIOR,
    MU_PRIOR,
    build_factor_graph,
    effective_statistics,
    ep_flag_summary,
    parity_table,
    read_ep_posteriors,
    read_joint_posteriors,
    run_autofit_ep,
    run_joint_fit,
)
from analytic_ep_minimal import ep_leg_a, ep_leg_b
from analytic_reference import leg_a_reference, leg_b_reference, simulate

"""
__Settings__
"""
SEED = 0
SIGMA_TRUE = 10.0
SIGMA_PRIOR_B = ("truncated", 10.0, 5.0, 0.0, 100.0)

EP_KWARGS = dict(kl_tol=1e-4, max_steps=30)
DYNESTY_KWARGS = dict(nlive=100, sample="unif", bound="multi", bootstrap=0, maxcall=30000, number_of_cores=1)

TOLERANCES = {
    "minimal EP": {"A": dict(a=1e-6, b=1e-6), "B": dict(scatter=dict(a=0.20, b=0.30), other=dict(a=0.05, b=0.16))},
    "autofit graphical": {"A": dict(a=0.10, b=0.15), "B": dict(a=0.10, b=0.15)},
    "autofit EP": {"A": dict(a=0.01, b=0.02), "B": dict(a=0.15, b=0.25)},
}
HARD_CAP_STD_ERROR = 0.50
EP_COLUMNS = ("minimal EP", "autofit EP")  # columns subject to the E[sigma] in [q05, q95] cap


def tolerance_for(leg):
    """The cell tolerance lookup of a leg; the leg-B minimal-EP tolerance is per row (scatter vs rest)."""

    def lookup(column, row):
        tol = TOLERANCES[column][leg]
        if "scatter" in tol:
            return tol["scatter"] if row in ("sigma", "log_sigma") else tol["other"]
        return tol

    return lookup


def drawn_prior():
    return af.GaussianPrior(mean=DRAWN_PRIOR[0], sigma=DRAWN_PRIOR[1])


def mu_prior():
    return af.GaussianPrior(mean=MU_PRIOR[0], sigma=MU_PRIOR[1])


"""
__Dataset__
"""
sim = simulate(seed=SEED)
ybar_eff, v_eff = effective_statistics(sim)
n_datasets = len(sim["y"])
x_rows = [f"x_{i}" for i in range(n_datasets)]

print(f"Analytic Gaussian benchmark -- parity (seed {SEED}, N={n_datasets}, n_i={sim['n'][0]}, s_i={np.round(sim['s'], 2)})")
print(f"  ybar  = {np.round(sim['ybar'], 4)}   v = {np.round(sim['v'], 5)}")
print(f"  ybar' = {np.round(ybar_eff, 4)}   v' = {np.round(v_eff, 5)}   (drawn prior N{DRAWN_PRIOR} folded in)")

timings = {}
passed_total = 0
cells_total = 0

"""
__Leg A: sigma known__
"""
t0 = time.time()
ref_a = leg_a_reference(ybar_eff, v_eff, SIGMA_TRUE, m0=MU_PRIOR[0], t0=MU_PRIOR[1])
timings["A closed form"] = time.time() - t0

t0 = time.time()
min_a = ep_leg_a(ybar_eff, v_eff, SIGMA_TRUE, m0=MU_PRIOR[0], t0=MU_PRIOR[1])
timings["A minimal EP"] = time.time() - t0

factor_graph_a, hf_a, models_a = build_factor_graph(sim, drawn_prior, dict(mean=mu_prior(), sigma=SIGMA_TRUE))
priors_a = {"mu": hf_a.mean, **{f"x_{i}": m.x for i, m in enumerate(models_a)}}
assert not isinstance(hf_a.sigma, af.Prior), "leg A must fix sigma (float -> Constant)"

t0 = time.time()
joint_result_a = run_joint_fit(factor_graph_a, "analytic_gaussian_joint_a", **DYNESTY_KWARGS)
joint_a = read_joint_posteriors(joint_result_a.samples, priors_a)
timings["A autofit graphical"] = time.time() - t0

t0 = time.time()
ep_result_a = run_autofit_ep(factor_graph_a, "analytic_gaussian_leg_a", seed=SEED, **EP_KWARGS)
ep_a = read_ep_posteriors(ep_result_a, priors_a)
timings["A autofit EP"] = time.time() - t0

rows_a = [("mu", ref_a["mu_mean"], ref_a["mu_std"])] + [(r, ref_a["x_mean"][i], ref_a["x_std"][i]) for i, r in enumerate(x_rows)]
columns_a = {
    "minimal EP": {"mu": (min_a["mu_mean"], min_a["mu_std"]), **{r: (min_a["x_mean"][i], min_a["x_std"][i]) for i, r in enumerate(x_rows)}},
    "autofit graphical": joint_a,
    "autofit EP": ep_a,
}
p, n = parity_table(
    rows_a,
    columns_a,
    tolerance_for("A"),
    f"Leg A (sigma = {SIGMA_TRUE} known): mean +/- std, a = |dmean|/std_ref, b = |std/std_ref - 1|",
    hard_cap_std_error=HARD_CAP_STD_ERROR,
)
passed_total += p
cells_total += n
print("  [info] graphical median_pdf +/- mean|errors_at_sigma(1)|: " + ", ".join(f"{r} {joint_a[r][2]:.4f} +/- {joint_a[r][3]:.4f}" for r in ["mu"] + x_rows))
print(f"  [info] joint fit: {len(joint_result_a.samples)} samples from {joint_result_a.samples.total_samples} calls; minimal EP sweeps {min_a['sweeps']}, converged {min_a['converged']}")
print(f"  [info] autofit EP flags: {ep_flag_summary('analytic_gaussian_leg_a')}")
print("  runtimes: " + ", ".join(f"{k[2:]} {v:.1f}s" for k, v in timings.items() if k.startswith("A ")))

"""
__Leg B: sigma unknown, TruncatedGaussianPrior(10, 5, 0, 100)__
"""
t0 = time.time()
ref_b = leg_b_reference(ybar_eff, v_eff, SIGMA_PRIOR_B, m0=MU_PRIOR[0], t0=MU_PRIOR[1])
timings["B closed form"] = time.time() - t0

t0 = time.time()
min_b = ep_leg_b(ybar_eff, v_eff, SIGMA_PRIOR_B, m0=MU_PRIOR[0], t0=MU_PRIOR[1], theta="sigma", projection="moments")
timings["B minimal EP"] = time.time() - t0

factor_graph_b, hf_b, models_b = build_factor_graph(
    sim,
    drawn_prior,
    dict(
        mean=mu_prior(),
        sigma=af.TruncatedGaussianPrior(mean=SIGMA_PRIOR_B[1], sigma=SIGMA_PRIOR_B[2], lower_limit=SIGMA_PRIOR_B[3], upper_limit=SIGMA_PRIOR_B[4]),
    ),
)
priors_b = {"sigma": hf_b.sigma, "mu": hf_b.mean, **{f"x_{i}": m.x for i, m in enumerate(models_b)}}

t0 = time.time()
joint_result_b = run_joint_fit(factor_graph_b, "analytic_gaussian_joint_b", **DYNESTY_KWARGS)
joint_b = read_joint_posteriors(joint_result_b.samples, priors_b)
timings["B autofit graphical"] = time.time() - t0

t0 = time.time()
ep_result_b = run_autofit_ep(factor_graph_b, "analytic_gaussian_leg_b", seed=SEED, **EP_KWARGS)
ep_b = read_ep_posteriors(ep_result_b, priors_b)
timings["B autofit EP"] = time.time() - t0

rows_b = [("sigma", ref_b["sigma_mean"], ref_b["sigma_std"]), ("mu", ref_b["mu_mean"], ref_b["mu_std"])] + [
    (r, ref_b["x_mean"][i], ref_b["x_std"][i]) for i, r in enumerate(x_rows)
]
columns_b = {
    "minimal EP": {
        "sigma": (min_b["sigma_mean"], min_b["sigma_std"]),
        "mu": (min_b["mu_mean"], min_b["mu_std"]),
        **{r: (min_b["x_mean"][i], min_b["x_std"][i]) for i, r in enumerate(x_rows)},
    },
    "autofit graphical": joint_b,
    "autofit EP": ep_b,
}
interval = (ref_b["sigma_q05"], ref_b["sigma_q95"])
hard_caps = [
    (
        f"hard cap: {col} E[sigma] = {columns_b[col]['sigma'][0]:.4f} inside closed-form [q05, q95] = [{interval[0]:.4f}, {interval[1]:.4f}]",
        interval[0] <= columns_b[col]["sigma"][0] <= interval[1],
    )
    for col in EP_COLUMNS
]
p, n = parity_table(
    rows_b,
    columns_b,
    tolerance_for("B"),
    f"Leg B (sigma ~ TruncatedGaussian{SIGMA_PRIOR_B[1:]}): mean +/- std; closed-form sigma q05/q50/q95 = "
    f"{ref_b['sigma_q05']:.4f}/{ref_b['sigma_q50']:.4f}/{ref_b['sigma_q95']:.4f} (sigma_q95 = analytic upper limit)",
    extra_checks=hard_caps,
    hard_cap_std_error=HARD_CAP_STD_ERROR,
)
passed_total += p
cells_total += n
print("  [info] graphical median_pdf +/- mean|errors_at_sigma(1)|: " + ", ".join(f"{r} {joint_b[r][2]:.4f} +/- {joint_b[r][3]:.4f}" for r in ["sigma", "mu"] + x_rows))
print(f"  [info] joint fit: {len(joint_result_b.samples)} samples from {joint_result_b.samples.total_samples} calls; minimal EP sweeps {min_b['sweeps']}, converged {min_b['converged']}, skipped {min_b['skipped']}")
print(f"  [info] autofit EP sigma message: {ep_b['sigma'][2]}")
print(f"  [info] autofit EP flags: {ep_flag_summary('analytic_gaussian_leg_b')}")
print("  runtimes: " + ", ".join(f"{k[2:]} {v:.1f}s" for k, v in timings.items() if k.startswith("B ")))

"""
__Verdict__
"""
print(f"\nTotal column runtime {sum(timings.values()):.1f}s")
verdict = "PASS" if passed_total == cells_total else "FAIL"
print(f"PARITY: {verdict} ({passed_total}/{cells_total})")
if verdict == "FAIL":
    sys.exit(1)
