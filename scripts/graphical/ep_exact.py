"""
Integration Test: EP Exact Conjugate Updates (`ExactFactorFit`)
================================================================

This script exercises the exact conjugate-update path of the expectation propagation (EP) framework.

When a factor on a hand-built `FactorGraph` is itself a message distribution (e.g. a `NormalMessage`
turned into a factor via `.as_factor(variable)`), the factor supports an EXACT projection: the EP
update is computed in closed form via conjugate Gaussian multiplication rather than by running a
numerical optimiser. `EPOptimiser.from_meanfield(...)` detects this via `factor.has_exact_projection`
and auto-assigns the `ExactFactorFit` optimiser to such factors (see
`PyAutoFit/autofit/graphical/expectation_propagation/optimiser.py::EPOptimiser.from_meanfield` and the
canonical usage in `PyAutoFit/test_autofit/graphical/regression/test_linear_regression.py::test_exact_updates`
and `regression/test_exact.py::test_probit_regression`).

__Graph__

The simplest possible conjugate problem: a single scalar variable `x` with

    prior:      NormalMessage(m1, s1).as_factor(x)
    likelihood: NormalMessage(m2, s2).as_factor(x)

    model = likelihood * prior

The exact posterior is the product of two Gaussians:

    posterior_precision = 1/s1^2 + 1/s2^2
    posterior_mean      = (m1/s1^2 + m2/s2^2) / posterior_precision

Because every factor on this graph is exact, `EPOptimiser.from_meanfield` runs the whole EP fit via
`ExactFactorFit` optimisers (no `LaplaceOptimiser` is involved) and the EP mean field should match the
analytic posterior to machine-level precision.
"""

import numpy as np

from autofit import graphical as graph
from autofit.messages.normal import NormalMessage

"""
__Variable__
"""
x_ = graph.Variable("x")

"""
__Factors__

Both the prior and the likelihood are `NormalMessage`s converted to factors via `.as_factor`, following
`regression/test_linear_regression.py::make_normal_model_approx`.
"""
m1, s1 = 1.0, 2.0  # prior
m2, s2 = 3.0, 0.5  # likelihood (observation)

prior_factor = NormalMessage(m1, s1).as_factor(x_, name="prior_x")
likelihood_factor = NormalMessage(m2, s2).as_factor(x_, name="likelihood_x")

model = likelihood_factor * prior_factor

"""
__Initial Mean Field__

A deliberately broad, offset starting approximation, so the assertions below genuinely test the exact
EP updates rather than the initialisation.
"""
approx0 = {
    x_: NormalMessage(0.0, 10.0),
}

model_approx = graph.EPMeanField.from_approx_dists(model, approx0)

"""
Both factors must support the exact projection for `ExactFactorFit` to be auto-selected.
"""
factor_mean_field = model_approx.factor_mean_field

assert prior_factor.has_exact_projection(factor_mean_field[prior_factor])
assert likelihood_factor.has_exact_projection(factor_mean_field[likelihood_factor])

"""
__Expectation Propagation__

No `default_optimiser` is passed: every factor is exact, so `EPOptimiser.from_meanfield` assigns
`ExactFactorFit` to each of them.
"""
ep_opt = graph.EPOptimiser.from_meanfield(model_approx, paths=False)

fit_approx = ep_opt.run(model_approx)

mean_field = fit_approx.mean_field

"""
__Analytic Conjugate Posterior__

The product of two Gaussians N(m1, s1) x N(m2, s2).
"""
posterior_precision = 1.0 / s1**2 + 1.0 / s2**2
posterior_mean = (m1 / s1**2 + m2 / s2**2) / posterior_precision
posterior_sigma = posterior_precision**-0.5

ep_mean = float(mean_field[x_].mean)
ep_sigma = float(mean_field[x_].sigma)

print(mean_field)
print()
print(f"EP posterior:       mean = {ep_mean}, sigma = {ep_sigma}")
print(f"Analytic posterior: mean = {posterior_mean}, sigma = {posterior_sigma}")

"""
__Assertions__

The EP mean field must match the analytic conjugate posterior to rtol 1e-6.
"""
assert np.isclose(ep_mean, posterior_mean, rtol=1e-6), (
    f"EP mean ({ep_mean}) does not match analytic posterior mean ({posterior_mean}) to rtol 1e-6"
)
assert np.isclose(ep_sigma, posterior_sigma, rtol=1e-6), (
    f"EP sigma ({ep_sigma}) does not match analytic posterior sigma ({posterior_sigma}) to rtol 1e-6"
)

print("ep_exact.py: PASS")
