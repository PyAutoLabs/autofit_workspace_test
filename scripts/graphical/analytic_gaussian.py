"""
Integration Test: Analytic Gaussian Benchmark -- Parity (closed form | minimal EP | graphical | EP)
=====================================================================================================

The CI parity script of autofit_workspace_test#91. One conjugate hierarchical Gaussian model is solved
four ways and every column is judged against the closed form of `analytic_reference.py`:

    closed form          exact posterior (leg A analytic, leg B by deterministic quadrature)
    minimal EP           the hand-rolled moments-projection EP of `analytic_ep_minimal.py`
    autofit graphical    a joint `DynestyStatic` fit of `factor_graph.global_prior_model`
    autofit EP           `factor_graph.optimise(af.LaplaceOptimiser(projection="moments"), ...)`
                         (nested-quadrature moment matching of each hierarchical factor)

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
    autofit EP (moments)      0.01, 0.02       0.15, 0.25
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

The joint fit is seeded (`af.DynestyStatic(seed=0)`: the search-level seed seeds dynesty's `rstate`
and the initializer), so the column is the same on every run. Unseeded, its
leg-B sigma cell passed or failed on sampler noise: in six unseeded runs on 2026-09-30 the script
exited 1 twice, with the EP column green both times. The one miss inspected was sigma
6.2453 +/- 2.5735, a 0.111 against 0.10.

`DynestyStatic` at the `ep_parity.py` budget (nlive 50, rwalk, maxcall 3000) stops on `maxcall` far
from convergence on this 6/7-parameter joint model (a = 2.5). Each likelihood call costs ~5 ms (the
hierarchical `GaussianPrior` distribution is instantiated per call), so the budget is set by the
number of calls: `sample="unif"` with `bound="multi", bootstrap=0` converges in ~7-10k calls per
leg (37 s / 50 s locally); the default bootstrap enlargement needs ~20k calls on leg B for the same
posterior. Every column prints its wall time; the script runs in ~85-130 s locally against the 300 s
CI cap (the moments-projection EP column is 34-52 s of it on leg B; the seed-140 leg below adds ~4 s and
the seed-25 leg ~3 s).

__Why the EP column runs `projection="moments"`__

Under the default Laplace (`"mode"`) projection the tilted density of every hierarchical factor in
sigma sits on the sigma = 0 boundary, so it has no interior mode and no negative-definite Hessian:
the updates end BAD_PROJECTION / FAILURE and the scatter row misses (the first run below).
`projection="moments"` (PyAutoFit#1654) moment-matches that density instead, by Gauss-Legendre
quadrature over sigma with a conditional Laplace over (mu, x_i) at each node -- the structure of
the referee `analytic_ep_minimal.py`. Leg A has no scale variable, so every factor there takes the
unchanged mode path.

__What the first run showed (seed 0, 2026-09-02, PyAutoFit 2026.8.17.1, Laplace "mode" projection)__

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
warning firing. The minimal EP with a Laplace projection reproduces the stale-factor state
deterministically -- every site update rejected, sigma returned at its prior
(`analytic_ep_minimal.py`): the tilted density of every hierarchical site is unbounded as sigma -> 0.

__What the moments run shows (seed 0, 2026-09-30, PyAutoFit 536049fa2 = main after #1656; smoke profile, 96-130 s)__

The same numbers came out on three repeated runs (seeded joint fit):

    closed form vs minimal EP:        every cell PASS (leg A a = b = 0.000; leg B scatter a 0.078, b 0.146)
    closed form vs autofit graphical: every cell PASS (leg A max a 0.056, b 0.029; leg B max a 0.070,
                                      b 0.110; sigma 6.7287 +/- 2.9072, a 0.056, b 0.008)
    closed form vs autofit EP:        every cell PASS -- PARITY: PASS (41/41)

Autofit EP, leg A: every row a = b = 0.000 at the printed precision (mu 50.8596 +/- 4.1102 vs
50.8595 +/- 4.1104); HierarchicalFactor SUCCESS=14. Leg B: sigma 6.3401 +/- 2.4609 against the
closed form's 6.5667 +/- 2.8832 (a 0.079, b 0.146 -- the Gaussian-matching bias on a skewed
posterior, the same as the minimal EP's 6.3423 +/- 2.4612), mu a 0.021 b 0.061, x_i a <= 0.018
b <= 0.010; the sigma message is a `TruncatedNormalMessage` with its limits (0, 100) kept;
HierarchicalFactor SUCCESS=25 with no BAD_PROJECTION or FAILURE. Column times (first of the three
runs): leg A graphical 19.0 s, EP 1.3 s; leg B graphical 26.9 s, EP 46.4 s.

__Leg A, seed 140: a factor that converges without taking a step (PyAutoFit#1711)__

The 200-seed ensemble of the science project `analytic_gaussian` (Cortex ruling R-20260910-04) found
one leg-A seed, 140, on which a factor ended the EP run with no SUCCESS update: the prior factor on
mu, `{NO_CHANGE: 2}`. On that dataset the posterior mean of mu (50.0018) lands within 2e-3 of the
prior's mean (50), so on the prior factor's first visit the start point of its Laplace optimiser is
already the mode of its tilted density to within the gradient tolerance (|g| 6.0e-6 < 1e-5): it stops
at `iter=0` and is flagged NO_CHANGE. That is convergence, not a skip -- the projection is accepted
and the message still moves (its curvature is recomputed at the mode: N(50, 22.4) -> N(50.0, 10.0)),
and mu matches the closed form to a = 6e-5, b = 2e-5. The last section pins that case: the EP
column alone on seed 140, leg A, judged against the closed form row by row (widths included) at the
leg-A tolerance, plus two flag-health checks -- every factor ends the run with at least one SUCCESS
or only accepted zero-iteration NO_CHANGE updates (`updated` true on every one of its
`ep_history.csv` rows), and the library's STALE FACTORS warning did not fire (neither logged nor
written to `ep_diagnostics.results`).

__Leg B, seed 25, mode projection: a non-finite evidence term is named (PyAutoFit#1713)__

The same ensemble found that seeds 25 and 172, run with the library-default `projection="mode"`,
printed numpy's `RuntimeWarning: invalid value encountered in add` from `EPMeanField.log_evidence`
over and over. One hierarchical member's message to sigma is a `TruncatedNormalMessage` centred far
below its support (mean -7374, sigma 384 on [0, 100] for seed 25); its truncation mass
`Phi(b) - Phi(a)` evaluates to 1 - 1 = 0, so its log-partition is -inf, sigma's variable evidence
is +inf and every adjacent factor term -inf, and the sum is NaN. The posteriors are bit-identical
with and without it -- only the evidence readout is affected -- and the moments projection never
produces that message. The library now returns the NaN explicitly, names the term once in a log
warning and in `ep_diagnostics.results` (`evidence_nonfinite: ...`), and emits no numpy warning.

Every EP column of this script is therefore checked for no `RuntimeWarning` from `ep_mean_field.py`
and a finite final `log_evidence`; the last section runs seed 25 leg B under the mode projection
(EP column only, no parity cells -- the mode projection is not the benchmarked one) and checks that
the evidence is either finite or NaN with the sigma variable named in the log and the results file.

__Status__

Curated into the smoke gate on 2026-09-30 (PyAutoFit#1654 / #1656): with the moments projection
every autofit-EP cell passes, and a FAIL on any column is a regression. The graphical column is
seeded and therefore deterministic, so its cells no longer pass or fail on sampler noise. Its
tolerances are unchanged.

__Env__ (Developer Only)

The graphical joint-fit column runs a real DynestyStatic search; test-mode 2 would replace it with a
midpoint vector and fake the column.

ENV: real_search
"""

import csv
import logging
import sys
import time
import warnings
from collections import defaultdict

import numpy as np

import autofit as af

from analytic_autofit import (
    DRAWN_PRIOR,
    MU_PRIOR,
    build_factor_graph,
    effective_statistics,
    ep_diagnostics_text,
    ep_flag_summary,
    output_dir,
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
DYNESTY_KWARGS = dict(
    nlive=100,
    sample="unif",
    bound="multi",
    bootstrap=0,
    maxcall=30000,
    number_of_cores=1,
)

TOLERANCES = {
    "minimal EP": {
        "A": dict(a=1e-6, b=1e-6),
        "B": dict(scatter=dict(a=0.20, b=0.30), other=dict(a=0.05, b=0.16)),
    },
    "autofit graphical": {"A": dict(a=0.10, b=0.15), "B": dict(a=0.10, b=0.15)},
    "autofit EP": {"A": dict(a=0.01, b=0.02), "B": dict(a=0.15, b=0.25)},
}
HARD_CAP_STD_ERROR = 0.50
EP_COLUMNS = (
    "minimal EP",
    "autofit EP",
)  # columns subject to the E[sigma] in [q05, q95] cap


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


def ep_evidence_warnings(caught):
    """
    The `RuntimeWarning`s in `caught` raised from `ep_mean_field.py` -- the EP evidence accumulator,
    which warned-and-summed a non-finite term before PyAutoFit#1713. Every other caught warning is
    shown again so recording does not hide it.
    """
    evidence = []
    for w in caught:
        if issubclass(w.category, RuntimeWarning) and w.filename.endswith(
            "ep_mean_field.py"
        ):
            evidence.append(w)
        else:
            warnings.showwarning(w.message, w.category, w.filename, w.lineno)
    return evidence


def ep_evidence_check(result, caught, leg):
    """The (label, ok) extra check: no evidence RuntimeWarning and a finite final EP log evidence."""
    log_evidence = result.updated_ep_mean_field.log_evidence
    evidence_warnings = ep_evidence_warnings(caught)
    return (
        f"autofit EP leg {leg}: no RuntimeWarning from ep_mean_field.py ({len(evidence_warnings)}) "
        f"and final log_evidence finite ({log_evidence:.4f}) (PyAutoFit#1713)",
        not evidence_warnings and bool(np.isfinite(log_evidence)),
    )


"""
__Dataset__
"""
sim = simulate(seed=SEED)
ybar_eff, v_eff = effective_statistics(sim)
n_datasets = len(sim["y"])
x_rows = [f"x_{i}" for i in range(n_datasets)]

print(
    f"Analytic Gaussian benchmark -- parity (seed {SEED}, N={n_datasets}, n_i={sim['n'][0]}, s_i={np.round(sim['s'], 2)})"
)
print(f"  ybar  = {np.round(sim['ybar'], 4)}   v = {np.round(sim['v'], 5)}")
print(
    f"  ybar' = {np.round(ybar_eff, 4)}   v' = {np.round(v_eff, 5)}   (drawn prior N{DRAWN_PRIOR} folded in)"
)

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

factor_graph_a, hf_a, models_a = build_factor_graph(
    sim, drawn_prior, dict(mean=mu_prior(), sigma=SIGMA_TRUE)
)
priors_a = {"mu": hf_a.mean, **{f"x_{i}": m.x for i, m in enumerate(models_a)}}
assert not isinstance(hf_a.sigma, af.Prior), "leg A must fix sigma (float -> Constant)"

t0 = time.time()
joint_result_a = run_joint_fit(
    factor_graph_a, "analytic_gaussian_joint_a", **DYNESTY_KWARGS
)
joint_a = read_joint_posteriors(joint_result_a.samples, priors_a)
timings["A autofit graphical"] = time.time() - t0

t0 = time.time()
with warnings.catch_warnings(record=True) as caught_a:
    warnings.simplefilter("always")
    ep_result_a = run_autofit_ep(
        factor_graph_a, "analytic_gaussian_leg_a", seed=SEED, **EP_KWARGS
    )
ep_a = read_ep_posteriors(ep_result_a, priors_a)
timings["A autofit EP"] = time.time() - t0

rows_a = [("mu", ref_a["mu_mean"], ref_a["mu_std"])] + [
    (r, ref_a["x_mean"][i], ref_a["x_std"][i]) for i, r in enumerate(x_rows)
]
columns_a = {
    "minimal EP": {
        "mu": (min_a["mu_mean"], min_a["mu_std"]),
        **{r: (min_a["x_mean"][i], min_a["x_std"][i]) for i, r in enumerate(x_rows)},
    },
    "autofit graphical": joint_a,
    "autofit EP": ep_a,
}
p, n = parity_table(
    rows_a,
    columns_a,
    tolerance_for("A"),
    f"Leg A (sigma = {SIGMA_TRUE} known): mean +/- std, a = |dmean|/std_ref, b = |std/std_ref - 1|",
    extra_checks=[ep_evidence_check(ep_result_a, caught_a, "A")],
    hard_cap_std_error=HARD_CAP_STD_ERROR,
)
passed_total += p
cells_total += n
print(
    "  [info] graphical median_pdf +/- mean|errors_at_sigma(1)|: "
    + ", ".join(
        f"{r} {joint_a[r][2]:.4f} +/- {joint_a[r][3]:.4f}" for r in ["mu"] + x_rows
    )
)
print(
    f"  [info] joint fit: {len(joint_result_a.samples)} samples from {joint_result_a.samples.total_samples} calls; minimal EP sweeps {min_a['sweeps']}, converged {min_a['converged']}"
)
print(f"  [info] autofit EP flags: {ep_flag_summary('analytic_gaussian_leg_a')}")
print(
    "  runtimes: "
    + ", ".join(f"{k[2:]} {v:.1f}s" for k, v in timings.items() if k.startswith("A "))
)

"""
__Leg B: sigma unknown, TruncatedGaussianPrior(10, 5, 0, 100)__
"""
t0 = time.time()
ref_b = leg_b_reference(ybar_eff, v_eff, SIGMA_PRIOR_B, m0=MU_PRIOR[0], t0=MU_PRIOR[1])
timings["B closed form"] = time.time() - t0

t0 = time.time()
min_b = ep_leg_b(
    ybar_eff,
    v_eff,
    SIGMA_PRIOR_B,
    m0=MU_PRIOR[0],
    t0=MU_PRIOR[1],
    theta="sigma",
    projection="moments",
)
timings["B minimal EP"] = time.time() - t0

factor_graph_b, hf_b, models_b = build_factor_graph(
    sim,
    drawn_prior,
    dict(
        mean=mu_prior(),
        sigma=af.TruncatedGaussianPrior(
            mean=SIGMA_PRIOR_B[1],
            sigma=SIGMA_PRIOR_B[2],
            lower_limit=SIGMA_PRIOR_B[3],
            upper_limit=SIGMA_PRIOR_B[4],
        ),
    ),
)
priors_b = {
    "sigma": hf_b.sigma,
    "mu": hf_b.mean,
    **{f"x_{i}": m.x for i, m in enumerate(models_b)},
}

t0 = time.time()
joint_result_b = run_joint_fit(
    factor_graph_b, "analytic_gaussian_joint_b", **DYNESTY_KWARGS
)
joint_b = read_joint_posteriors(joint_result_b.samples, priors_b)
timings["B autofit graphical"] = time.time() - t0

t0 = time.time()
with warnings.catch_warnings(record=True) as caught_b:
    warnings.simplefilter("always")
    ep_result_b = run_autofit_ep(
        factor_graph_b, "analytic_gaussian_leg_b", seed=SEED, **EP_KWARGS
    )
ep_b = read_ep_posteriors(ep_result_b, priors_b)
timings["B autofit EP"] = time.time() - t0

rows_b = [
    ("sigma", ref_b["sigma_mean"], ref_b["sigma_std"]),
    ("mu", ref_b["mu_mean"], ref_b["mu_std"]),
] + [(r, ref_b["x_mean"][i], ref_b["x_std"][i]) for i, r in enumerate(x_rows)]
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
] + [ep_evidence_check(ep_result_b, caught_b, "B")]
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
print(
    "  [info] graphical median_pdf +/- mean|errors_at_sigma(1)|: "
    + ", ".join(
        f"{r} {joint_b[r][2]:.4f} +/- {joint_b[r][3]:.4f}"
        for r in ["sigma", "mu"] + x_rows
    )
)
print(
    f"  [info] joint fit: {len(joint_result_b.samples)} samples from {joint_result_b.samples.total_samples} calls; minimal EP sweeps {min_b['sweeps']}, converged {min_b['converged']}, skipped {min_b['skipped']}"
)
print(f"  [info] autofit EP sigma message: {ep_b['sigma'][2]}")
print(f"  [info] autofit EP flags: {ep_flag_summary('analytic_gaussian_leg_b')}")
print(
    "  runtimes: "
    + ", ".join(f"{k[2:]} {v:.1f}s" for k, v in timings.items() if k.startswith("B "))
)

"""
__Leg A, seed 140: zero-iteration convergence (PyAutoFit#1711)__

The EP column only (the closed form is the reference; the other columns are the seed-0 legs' job).
`STALE FACTORS` warnings are logged by the EP optimiser at the end of the run, so a handler on its
logger records whether one fired.
"""
SEED_140 = 140


class _WarningRecorder(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def ep_history_rows(name):
    """{factor: [(flag, updated), ...]} from the run's `ep_history.csv`."""
    files = list(output_dir(name).rglob("ep_history.csv"))
    assert files, f"ep_history.csv not found for {name}"
    rows = defaultdict(list)
    with open(files[0]) as f:
        for row in csv.DictReader(f):
            rows[row["factor"]].append((row["flag"], row["updated"] == "True"))
    return rows


sim_140 = simulate(seed=SEED_140)
ybar_140, v_140 = effective_statistics(sim_140)

t0 = time.time()
ref_140 = leg_a_reference(ybar_140, v_140, SIGMA_TRUE, m0=MU_PRIOR[0], t0=MU_PRIOR[1])
timings["A140 closed form"] = time.time() - t0

factor_graph_140, hf_140, models_140 = build_factor_graph(
    sim_140, drawn_prior, dict(mean=mu_prior(), sigma=SIGMA_TRUE)
)
priors_140 = {"mu": hf_140.mean, **{f"x_{i}": m.x for i, m in enumerate(models_140)}}

recorder = _WarningRecorder()
ep_logger = logging.getLogger("autofit.graphical.expectation_propagation.optimiser")
ep_logger.addHandler(recorder)
t0 = time.time()
try:
    ep_result_140 = run_autofit_ep(
        factor_graph_140, "analytic_gaussian_leg_a_seed140", seed=SEED_140, **EP_KWARGS
    )
finally:
    ep_logger.removeHandler(recorder)
ep_140 = read_ep_posteriors(ep_result_140, priors_140)
timings["A140 autofit EP"] = time.time() - t0

history_140 = ep_history_rows("analytic_gaussian_leg_a_seed140")
zero_iteration_only = sorted(
    factor
    for factor, rows in history_140.items()
    if rows and all(flag == "NO_CHANGE" for flag, _ in rows)
)
unhealthy = sorted(
    factor
    for factor, rows in history_140.items()
    if not any(flag == "SUCCESS" for flag, _ in rows)
    and not (factor in zero_iteration_only and all(updated for _, updated in rows))
)
stale_logged = [m for m in recorder.messages if "STALE FACTORS" in m]
stale_written = "STALE FACTORS" in ep_diagnostics_text("analytic_gaussian_leg_a_seed140")

rows_140 = [("mu", ref_140["mu_mean"], ref_140["mu_std"])] + [
    (r, ref_140["x_mean"][i], ref_140["x_std"][i]) for i, r in enumerate(x_rows)
]
p, n = parity_table(
    rows_140,
    {"autofit EP": ep_140},
    tolerance_for("A"),
    f"Leg A, seed {SEED_140} (PyAutoFit#1711): mean +/- std, a = |dmean|/std_ref, b = |std/std_ref - 1|",
    extra_checks=[
        (
            "flag health: every factor has >= 1 SUCCESS or only accepted (updated) NO_CHANGE updates"
            + (f" -- failing: {', '.join(unhealthy)}" if unhealthy else ""),
            not unhealthy,
        ),
        (
            "library STALE FACTORS warning absent (log and ep_diagnostics.results)",
            not stale_logged and not stale_written,
        ),
    ],
    hard_cap_std_error=HARD_CAP_STD_ERROR,
)
passed_total += p
cells_total += n
print(
    f"  [info] zero-iteration-only (accepted NO_CHANGE) factors: {', '.join(zero_iteration_only) or 'none'}"
)
print(f"  [info] autofit EP flags: {ep_flag_summary('analytic_gaussian_leg_a_seed140')}")
print(
    "  runtimes: "
    + ", ".join(f"{k[5:]} {v:.1f}s" for k, v in timings.items() if k.startswith("A140 "))
)

"""
__Leg B, seed 25, mode projection: a non-finite evidence term is named (PyAutoFit#1713)__

The EP column only, under the library-default `projection="mode"` that the ensemble ran. Whether the
degenerate sigma message appears depends on the process history (prior ids), so the check accepts a
finite evidence, or a NaN one whose sigma variable is named by the library's `ep_mean_field` log
warning and in `ep_diagnostics.results`. Either way no numpy `RuntimeWarning` may come from
`ep_mean_field.py`.
"""
SEED_25 = 25

sim_25 = simulate(seed=SEED_25)
factor_graph_25, hf_25, models_25 = build_factor_graph(
    sim_25,
    drawn_prior,
    dict(
        mean=mu_prior(),
        sigma=af.TruncatedGaussianPrior(
            mean=SIGMA_PRIOR_B[1],
            sigma=SIGMA_PRIOR_B[2],
            lower_limit=SIGMA_PRIOR_B[3],
            upper_limit=SIGMA_PRIOR_B[4],
        ),
    ),
)
sigma_name_25 = hf_25.sigma.name

evidence_recorder = _WarningRecorder()
evidence_logger = logging.getLogger(
    "autofit.graphical.expectation_propagation.ep_mean_field"
)
evidence_logger.addHandler(evidence_recorder)
t0 = time.time()
try:
    with warnings.catch_warnings(record=True) as caught_25:
        warnings.simplefilter("always")
        ep_result_25 = run_autofit_ep(
            factor_graph_25,
            "analytic_gaussian_leg_b_seed25_mode",
            seed=SEED_25,
            projection="mode",
            **EP_KWARGS,
        )
finally:
    evidence_logger.removeHandler(evidence_recorder)
timings["B25 autofit EP (mode)"] = time.time() - t0

log_evidence_25 = ep_result_25.updated_ep_mean_field.log_evidence
evidence_warnings_25 = ep_evidence_warnings(caught_25)
named_logged = [
    m for m in evidence_recorder.messages if "non-finite" in m and sigma_name_25 in m
]
diagnostics_25 = ep_diagnostics_text("analytic_gaussian_leg_b_seed25_mode")
named_written = [
    line
    for line in diagnostics_25.splitlines()
    if line.startswith("evidence_nonfinite:") and sigma_name_25 in line
]
evidence_ok_25 = bool(np.isfinite(log_evidence_25)) or bool(
    named_logged and named_written
)
p, n = parity_table(
    [],
    {},
    tolerance_for("B"),
    f"Leg B, seed {SEED_25}, projection='mode' (PyAutoFit#1713): EP evidence reporting",
    extra_checks=[
        (
            f"no RuntimeWarning from ep_mean_field.py ({len(evidence_warnings_25)})",
            not evidence_warnings_25,
        ),
        (
            f"final log_evidence {log_evidence_25:.4f} finite, or NaN with '{sigma_name_25}' named "
            f"(logged {len(named_logged)}, ep_diagnostics.results {len(named_written)})",
            evidence_ok_25,
        ),
    ],
)
passed_total += p
cells_total += n
for line in named_written:
    print(f"  [info] {line[:160]}")
print(f"  [info] autofit EP flags: {ep_flag_summary('analytic_gaussian_leg_b_seed25_mode')}")
print(f"  runtimes: autofit EP (mode) {timings['B25 autofit EP (mode)']:.1f}s")

"""
__Verdict__
"""
print(f"\nTotal column runtime {sum(timings.values()):.1f}s")
verdict = "PASS" if passed_total == cells_total else "FAIL"
print(f"PARITY: {verdict} ({passed_total}/{cells_total})")
if verdict == "FAIL":
    sys.exit(1)
