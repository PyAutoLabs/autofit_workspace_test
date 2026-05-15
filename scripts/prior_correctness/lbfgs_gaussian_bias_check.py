"""
LBFGS MAP Sanity Check
======================

Empirical regression gate for the ``log_prior_from_value`` sign convention
(PyAutoLabs/PyAutoFit#1266). Drives ``scipy.optimize.minimize(method="L-BFGS-B")``
directly via ``Fitness.__call__`` with ``convert_to_chi_squared=True`` on a
1-parameter model with:

  * a flat log-likelihood (constant 0.0)
  * a single ``GaussianPrior(mean=5.0, sigma=1.0)``

Under the density convention, the MAP is at the prior mean (5.0). Under the
pre-#1266 cost-form bug, ``chi_squared = -2 * (log_lik + cost)`` is minimised
by *maximising* the cost, so L-BFGS-B diverges away from 5.0 (empirically to
~8e143 before the fix).

Run from the workspace root::

    python scripts/prior_correctness/lbfgs_gaussian_bias_check.py
"""
import os
os.environ.setdefault("PYAUTO_SKIP_WORKSPACE_VERSION_CHECK", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from scipy import optimize

import autofit as af
from autofit.example.model import Gaussian
from autofit.non_linear.fitness import Fitness


class FlatLikelihoodAnalysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return 0.0


def assert_lbfgs_converges_to_prior_mean_under_flat_likelihood():
    np.random.seed(0)

    model = af.Model(Gaussian)
    model.centre = 0.0
    model.normalization = 1.0
    model.sigma = af.GaussianPrior(mean=5.0, sigma=1.0)

    fitness = Fitness(
        model=model,
        analysis=FlatLikelihoodAnalysis(),
        paths=None,
        fom_is_log_likelihood=False,
        resample_figure_of_merit=-np.inf,
        convert_to_chi_squared=True,
    )

    res = optimize.minimize(
        fun=fitness.__call__,
        x0=np.array([5.5]),
        method="L-BFGS-B",
        options={"maxiter": 200},
    )

    assert res.success, (
        f"L-BFGS-B did not converge: {res.message}. Likely a cost-form "
        f"regression in log_prior_from_value (chi_squared diverges)."
    )
    assert abs(res.x[0] - 5.0) < 0.01, (
        f"L-BFGS-B converged to {res.x[0]:.4f}, but the flat-likelihood + "
        f"GaussianPrior(mean=5.0) MAP should be at 5.0. Likely a sign-convention "
        f"regression."
    )
    assert abs(res.fun) < 0.01, (
        f"chi_squared at MAP = {res.fun:.4e}, expected ≈ 0.0. The MAP of a "
        f"flat-likelihood + Gaussian-prior posterior is the prior mean where "
        f"log-density is 0 (constants dropped); chi_squared = -2 * 0 = 0."
    )


if __name__ == "__main__":
    assert_lbfgs_converges_to_prior_mean_under_flat_likelihood()
    print("lbfgs_gaussian_bias_check: all assertions passed")
