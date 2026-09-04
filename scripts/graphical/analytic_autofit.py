"""
Integration Test: Analytic Gaussian Benchmark -- Shared autofit Wiring
=======================================================================

The autofit side of the analytic Gaussian benchmark (autofit_workspace_test#91), shared by the three
entry scripts `analytic_gaussian.py`, `analytic_gaussian_priors.py` and `analytic_gaussian_collapse.py`
(imported by sibling name: the script directory is on `sys.path` when a script is run as
`python scripts/graphical/<file>.py`). It holds no assertions of its own; run directly it only
imports.

__Graph__

    af.Model(Level) with `x = drawn_prior_factory()` per dataset
    af.AnalysisFactor(model, GaussianDataAnalysis(y_i, s_i), name=f"dataset_{i}")
    af.HierarchicalFactor(af.GaussianPrior, mean=<mu hyper-prior>, sigma=<sigma hyper-prior or float>)
        .add_drawn_variable(model.x) for every model
    af.FactorGraphModel(*analysis_factors, hierarchical_factor)

`GaussianDataAnalysis.log_likelihood_function` is the exact Gaussian log-likelihood of the scalar
level, -1/2 sum_j (y_ij - x)^2 / s_i^2, so the analysis factors are exactly Gaussian in x.

__The drawn prior is an extra factor__

Every drawn variable carries its own prior (`GaussianPrior(50, 20)` by default) and that prior is part
of the density in both autofit routes: `FactorGraphModel(include_prior_factors=True)` (the default)
attaches a `PriorFactor` to `x_i` beside its analysis factor and its hierarchical factor
(`prior_counts` = 3 per drawn variable), and `global_prior_model` samples `x_i` through the same prior
while `log_likelihood_function` adds the hierarchical term `log N(x_i | mu, sigma^2)`. Both routes
therefore describe

    p(mu) p(sigma) prod_i [ N(x_i | m_d, s_d^2) N(x_i | mu, sigma^2) N(ybar_i | x_i, v_i) ]

The extra Gaussian factor is conjugate with the data term and folds exactly into effective sufficient
statistics (`effective_statistics`):

    1/v'_i = 1/v_i + 1/s_d^2,      ybar'_i = v'_i (ybar_i / v_i + m_d / s_d^2)

on which the closed form and the minimal EP are evaluated. Nothing is approximated in the reference.

__Reading the posteriors__

EP (`read_ep_posteriors`): `result.updated_ep_mean_field.mean_field[prior]`, with two message traps.
A `TruncatedNormalMessage`'s `.mean` / `.sigma` attributes are the UNTRUNCATED location and scale (set
from `.parameters` in `__init__`, shadowing the truncated-moment properties), so the moments are
recomputed with `scipy.stats.truncnorm` from `(loc, scale, lower, upper)`. A `LogGaussianPrior`'s
message is a `TransformedMessage` whose `.mean` is the median in sigma, so the `base_message` mean and
sigma of log sigma are returned instead (and the note says so).

Graphical (`read_joint_posteriors`): the closed form and EP report posterior MEANS and standard
deviations, so the sampler column is the weighted mean and standard deviation of
`samples.parameter_lists` under `samples.weight_list`, matched to priors by identity through
`model.prior_tuples_ordered_by_id`. `median_pdf` and the averaged `errors_at_sigma(1.0)` are returned
alongside for information: on a skewed sigma posterior the median sits well below the mean by
construction. `errors_at_sigma(as_instance=True)` cannot be used on this model at all: building the
instance calls `GaussianPrior(mean=(lower, upper), ...)` for the hierarchical `distribution_model` and
raises inside `NormalMessage`.

__EP diagnostics__

`ep_flag_summary(name)` tallies the `StatusFlag`s of `ep_history.csv` per factor kind and
`ep_diagnostics_text(name)` returns the library's `ep_diagnostics.results` (mean-field summary plus
its scale-collapse / stale-factor warnings); both read the run's `output_dir(name)`, resolved through
`af.DirectoryPaths` so it follows the library's layout `output/[test_mode/]graphical/<name>/<id>/`. A variable whose factors never record SUCCESS is still at its starting prior.
"""

import csv
import logging
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import truncnorm

import autofit as af
from autofit.messages.composed_transform import TransformedMessage
from autofit.messages.truncated_normal import TruncatedNormalMessage

from analytic_reference import Level

# The per-factor EP INFO lines (17 factors x 30 steps per run) would swamp the parity tables;
# WARNINGs (the library's scale-collapse and stale-factor diagnostics) are kept.
logging.disable(logging.INFO)

MU_PRIOR = (50.0, 10.0)  # hyper-prior on mu, N(m0, t0^2): the closed form's (m0, t0)
DRAWN_PRIOR = (50.0, 20.0)  # every x_i's own GaussianPrior, folded into (ybar', v')


class GaussianDataAnalysis(af.Analysis):
    """Exact Gaussian log-likelihood of dataset i for the scalar level `instance.x` (known noise s_i)."""

    def __init__(self, y, noise_sigma):
        self.y = np.asarray(y, dtype=float)
        self.noise_sigma = float(noise_sigma)

    def log_likelihood_function(self, instance):
        return -0.5 * np.sum((self.y - instance.x) ** 2) / self.noise_sigma**2


def effective_statistics(sim, drawn_prior=DRAWN_PRIOR):
    """(ybar', v') with the drawn prior N(drawn_prior) folded into each dataset's sufficient statistics."""
    m_d, s_d = drawn_prior
    v_eff = 1.0 / (1.0 / sim["v"] + 1.0 / s_d**2)
    ybar_eff = v_eff * (sim["ybar"] / sim["v"] + m_d / s_d**2)
    return ybar_eff, v_eff


def build_factor_graph(sim, drawn_prior_factory, hf_kwargs):
    """
    The benchmark's factor graph (module docstring). Returns `(factor_graph, hierarchical_factor, models)`.
    """
    models = []
    factors = []
    for i in range(len(sim["y"])):
        model = af.Model(Level)
        model.x = drawn_prior_factory()
        models.append(model)
        factors.append(
            af.AnalysisFactor(
                prior_model=model,
                analysis=GaussianDataAnalysis(sim["y"][i], sim["s"][i]),
                name=f"dataset_{i}",
            )
        )

    hierarchical_factor = af.HierarchicalFactor(af.GaussianPrior, **hf_kwargs)
    for model in models:
        hierarchical_factor.add_drawn_variable(model.x)

    return (
        af.FactorGraphModel(*factors, hierarchical_factor),
        hierarchical_factor,
        models,
    )


def output_dir(name):
    """
    The directory an EP run named `name` writes to, resolved through `af.DirectoryPaths` itself so it
    follows the library's layout (`output/graphical/<name>/<identifier>/`, with a `test_mode` segment
    inserted after `output/` whenever `PYAUTO_TEST_MODE` is active).
    """
    return Path(af.DirectoryPaths(path_prefix=Path("graphical"), name=name).output_path)


def run_autofit_ep(factor_graph, name, kl_tol=1e-4, max_steps=30, seed=0, updater=None):
    """`factor_graph.optimise` with `af.LaplaceOptimiser()`; numpy seeded; previous output removed."""
    shutil.rmtree(output_dir(name).parent, ignore_errors=True)
    np.random.seed(seed)
    kwargs = dict(updater=updater) if updater is not None else {}
    return factor_graph.optimise(
        af.LaplaceOptimiser(),
        paths=af.DirectoryPaths(path_prefix=Path("graphical"), name=name),
        ep_history=af.EPHistory(kl_tol=kl_tol),
        max_steps=max_steps,
        **kwargs,
    )


def read_ep_posteriors(result, priors):
    """
    {name: (mean, std, note)} from an EP result for the named priors, handling the two message traps
    of the module docstring. For a `TransformedMessage` the moments are those of log sigma.
    """
    mean_field = result.updated_ep_mean_field.mean_field
    out = {}
    for name, prior in priors.items():
        message = mean_field[prior]
        if isinstance(message, TransformedMessage):
            base = message.base_message
            out[name] = (
                float(base.mean),
                float(base.sigma),
                "log-space base message (moments of log sigma)",
            )
        elif isinstance(message, TruncatedNormalMessage):
            loc, scale = (float(p) for p in message.parameters)
            lower, upper = float(message.lower_limit), float(message.upper_limit)
            a, b = (lower - loc) / scale, (upper - loc) / scale
            out[name] = (
                float(truncnorm.mean(a, b, loc=loc, scale=scale)),
                float(truncnorm.std(a, b, loc=loc, scale=scale)),
                f"truncnorm of loc {loc:.4g}, scale {scale:.4g}, limits ({lower:.4g}, {upper:.4g})",
            )
        else:
            out[name] = (float(message.mean), float(np.sqrt(message.variance)), "")
    return out


def run_joint_fit(factor_graph, name, **kwargs):
    """
    A `DynestyStatic` fit of `factor_graph.global_prior_model`. Previous output of the search is removed
    first: resuming a completed run loads a samples summary rather than the samples the parity needs.
    """
    shutil.rmtree(output_dir(name).parent, ignore_errors=True)
    search = af.DynestyStatic(path_prefix="graphical", name=name, **kwargs)
    return search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)


def read_joint_posteriors(samples, priors):
    """
    {name: (weighted mean, weighted std, median_pdf, mean |error at 1 sigma|)} for the named priors,
    matched by identity in `model.prior_tuples_ordered_by_id`.
    """
    order = [prior for _, prior in samples.model.prior_tuples_ordered_by_id]
    params = np.asarray(samples.parameter_lists, dtype=float)
    weights = np.asarray(samples.weight_list, dtype=float)
    weights = weights / np.sum(weights)
    w_mean = params.T @ weights
    w_std = np.sqrt(np.maximum((params**2).T @ weights - w_mean**2, 0.0))
    median = samples.median_pdf(as_instance=False)
    errors = samples.errors_at_sigma(sigma=1.0, as_instance=False)

    out = {}
    for name, prior in priors.items():
        k = next(i for i, p in enumerate(order) if p is prior)
        out[name] = (
            float(w_mean[k]),
            float(w_std[k]),
            float(median[k]),
            float(np.mean(np.abs(errors[k]))),
        )
    return out


def ep_flag_summary(name):
    """Per-factor-kind `StatusFlag` counts of the run's `ep_history.csv` (module docstring)."""
    files = list(output_dir(name).rglob("ep_history.csv"))
    if not files:
        return "ep_history.csv not found"
    counts = Counter()
    with open(files[0]) as f:
        for row in csv.DictReader(f):
            factor = row["factor"]
            kind = (
                factor.rstrip("0123456789")
                if factor.startswith(("Hierarchical", "Prior"))
                else "dataset"
            )
            counts[(kind, row["flag"])] += 1
    kinds = sorted({kind for kind, _ in counts})
    return "; ".join(
        f"{kind} "
        + " ".join(
            f"{flag}={n}" for (k, flag), n in sorted(counts.items()) if k == kind
        )
        for kind in kinds
    )


def ep_diagnostics_text(name):
    """The run's `ep_diagnostics.results` (mean-field summary plus library warnings), or a not-found note."""
    files = list(output_dir(name).rglob("ep_diagnostics.results"))
    if not files:
        return "ep_diagnostics.results not found"
    with open(files[0]) as f:
        return f.read()


def parity_table(
    rows, columns, tolerance_for, title, extra_checks=(), hard_cap_std_error=0.50
):
    """
    Print a parity table and return (passed, total).

    `rows` is a list of (name, ref_mean, ref_std); `columns` maps a column name to {row: (mean, std, ...)};
    `tolerance_for(column, row)` returns the cell's dict(a=..., b=...); `extra_checks` is a list of
    (label, ok) hard-cap outcomes appended to the count. A cell passes iff a = |dmean| / std_ref and
    b = |std / std_ref - 1| are within tolerance and b is within `hard_cap_std_error`.
    """
    width = 44
    print(f"\n{title}")
    print(
        f"  {'row':<9}{'closed form':>22} |"
        + " |".join(f"{col:>{width}}" for col in columns)
    )
    passed = total = 0
    for name, ref_mean, ref_std in rows:
        cells = []
        for col, values in columns.items():
            mean, std = values[name][:2]
            a = abs(mean - ref_mean) / ref_std
            b = abs(std / ref_std - 1.0)
            tol = tolerance_for(col, name)
            ok = a <= tol["a"] and b <= tol["b"] and b <= hard_cap_std_error
            passed += int(ok)
            total += 1
            cells.append(
                f"{mean:.4f} +/- {std:.4f} {'PASS' if ok else 'FAIL'} a={a:.3f} b={b:.3f}"
            )
        print(
            f"  {name:<9}{ref_mean:>11.4f} +/- {ref_std:<7.4f} |"
            + " |".join(f"{cell:>{width}}" for cell in cells)
        )
    for label, ok in extra_checks:
        passed += int(ok)
        total += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"  cells passed: {passed}/{total}")
    return passed, total
