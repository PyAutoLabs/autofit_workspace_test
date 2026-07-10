"""
Integration Test: EP Deterministic Variable (Low-Level `factor_out` API)
=========================================================================

This script exercises the LOW-LEVEL graph API used to build a hand-crafted `FactorGraph` with a
deterministic variable, and runs expectation propagation (EP) on it directly (i.e. without the
declarative `AnalysisFactor` / `FactorGraphModel` API used in `ep.py` and `simultaneous.py`).

A deterministic variable is a node on the factor graph whose value is a (deterministic) function of
other variables, rather than a free parameter with its own prior. It is declared via the `factor_out`
keyword argument of `graph.Factor`, following the canonical pattern used throughout
`PyAutoFit/test_autofit/graphical/` (e.g. `test_factor_graph.py::make_plus`,
`regression/test_exact.py::make_model`, `regression/conftest.py::make_linear_factor`).

__Graph__

We build a small graph with two free scalar variables `x` and `y`, each with an independent Gaussian
prior, and a deterministic variable `z = x + y` declared via `factor_out`. A Gaussian likelihood
message-factor is placed directly on `z`, informing the graph of an observed value of `x + y`.

    prior_x:       NormalMessage(mean=1.0, sigma=2.0).as_factor(x)
    prior_y:       NormalMessage(mean=1.0, sigma=2.0).as_factor(y)
    linear_factor: Factor(lambda x, y: x + y, x, y, factor_out=z)
    likelihood:    NormalMessage(mean=3.0, sigma=0.1).as_factor(z)

    model = likelihood * linear_factor * prior_x * prior_y

This mirrors the pattern used in `test_autofit/graphical/regression/test_linear_regression.py`
(`test_exact_updates`), which builds `likelihood_factor * linear_factor * prior_a * prior_b` and runs
`EPOptimiser.from_meanfield(model_approx, default_optimiser=laplace)`. The deterministic `linear_factor`
here is a simple Python function with no closed-form (exact) projection, so it is fit via
`af.LaplaceOptimiser()`; the Gaussian prior/likelihood factors are exact and are auto-assigned an
`ExactFactorFit` optimiser by `EPOptimiser.from_meanfield`.
"""

import numpy as np

import autofit as af
from autofit import graphical as graph
from autofit.messages.normal import NormalMessage

"""
__Variables__

`x` and `y` are free scalar variables. `z` is the deterministic output variable representing `x + y`.
"""
x_, y_, z_ = graph.variables("x, y, z")

"""
__Priors__

Independent Gaussian priors on `x` and `y`, each built via `NormalMessage(...).as_factor(variable)`,
following the pattern in `regression/test_linear_regression.py::make_normal_model_approx`.
"""
prior_mean = 1.0
prior_sigma = 2.0

prior_x = NormalMessage(prior_mean, prior_sigma).as_factor(x_)
prior_y = NormalMessage(prior_mean, prior_sigma).as_factor(y_)

"""
__Deterministic Factor__

`z = x + y`, declared via `factor_out=z_` following `test_factor_graph.py::make_plus`
(`graph.Factor(plus_two, x, factor_out=y)`) and `regression/conftest.py::make_linear_factor`
(`graph.Factor(linear, x_, a_, b_, factor_out=z_)`).
"""


def sum_of_x_and_y(x, y):
    return x + y


linear_factor = graph.Factor(sum_of_x_and_y, x_, y_, factor_out=z_)

"""
__Likelihood__

A Gaussian message-factor on `z`, centred on an observed value `z_obs = 3.0` with a tight `sigma = 0.1`,
informing the graph that `x + y` is observed to be close to `3.0`.
"""
z_obs = 3.0
z_obs_sigma = 0.1

likelihood_factor = NormalMessage(z_obs, z_obs_sigma).as_factor(z_)

"""
__Factor Graph__

Combine every factor via `*`, following every example in `test_autofit/graphical/` (e.g.
`likelihood_factor * linear_factor * prior_a * prior_b`).
"""
model = likelihood_factor * linear_factor * prior_x * prior_y

"""
__Initial Mean Field__

An initial approximating distribution is required for every variable in the graph, including the
deterministic variable `z` (see `regression/test_exact.py::make_model`, which supplies an initial dist
for the deterministic variable `f_`). The initial `z` dist is the (rough) implied distribution of
`x + y` under the priors on `x` and `y` (mean = mean_x + mean_y, variance = var_x + var_y).
"""
approx0 = {
    x_: NormalMessage(prior_mean, prior_sigma),
    y_: NormalMessage(prior_mean, prior_sigma),
    z_: NormalMessage(2.0 * prior_mean, (2.0 * prior_sigma**2) ** 0.5),
}

z_prior_variance = float(approx0[z_].variance)

model_approx = graph.EPMeanField.from_approx_dists(model, approx0)

"""
__Expectation Propagation__

`EPOptimiser.from_meanfield` auto-assigns `ExactFactorFit` to the exact Gaussian prior/likelihood
factors and the supplied `default_optimiser` (`af.LaplaceOptimiser()`) to the deterministic
`linear_factor`, following `regression/test_linear_regression.py::test_exact_updates`.
"""
laplace = af.LaplaceOptimiser()

ep_opt = graph.EPOptimiser.from_meanfield(
    model_approx, default_optimiser=laplace, paths=False
)

fit_approx = ep_opt.run(model_approx)

mean_field = fit_approx.mean_field

print(mean_field)
print()
print(f"x: mean = {mean_field[x_].mean}, variance = {mean_field[x_].variance}")
print(f"y: mean = {mean_field[y_].mean}, variance = {mean_field[y_].variance}")
print(f"z: mean = {mean_field[z_].mean}, variance = {mean_field[z_].variance}")

x_mean = float(mean_field[x_].mean)
y_mean = float(mean_field[y_].mean)
x_variance = float(mean_field[x_].variance)
y_variance = float(mean_field[y_].variance)
z_variance = float(mean_field[z_].variance)

sum_mean = x_mean + y_mean
combined_sigma = (x_variance + y_variance) ** 0.5

"""
__Assertions__

1) The posterior mean of `x + y` should be close to the observed value `z_obs = 3.0`, within 3 combined
   (x, y) sigma.
2) The posterior uncertainty on `z` should have shrunk relative to its (rough) prior-implied variance,
   since the tight likelihood on `z` (sigma = 0.1) is far more informative than the priors on `x`/`y`.
"""
assert np.isfinite(sum_mean)
assert np.isfinite(combined_sigma) and combined_sigma > 0.0
assert abs(sum_mean - z_obs) < 3.0 * combined_sigma, (
    f"posterior mean of x + y ({sum_mean}) not within 3 sigma ({3.0 * combined_sigma}) "
    f"of observed z_obs ({z_obs})"
)
assert z_variance < z_prior_variance, (
    f"posterior variance on z ({z_variance}) did not shrink relative to its prior-implied "
    f"variance ({z_prior_variance})"
)

print("ep_deterministic.py: PASS")
