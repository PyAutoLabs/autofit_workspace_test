"""
Emcee MAP / Posterior Sanity Check
==================================

Empirical regression gate for the ``log_prior_from_value`` sign convention
(PyAutoLabs/PyAutoFit#1266). Drives ``emcee.EnsembleSampler`` directly via
``Fitness.call_wrap`` on a 1-parameter model with:

  * a flat log-likelihood (constant 0.0)
  * a single ``GaussianPrior(mean=5.0, sigma=1.0)``

When the prior is treated as a proper log-density (negative for low-density),
the posterior is the prior itself, so emcee samples should cluster around
mean=5.0 with std≈1.0.

When the prior is treated as a positive cost (the pre-#1266 bug),
``figure_of_merit = log_likelihood + sum(log_priors)`` adds a positive
quadratic, which acts as a repulsive potential — emcee samples diverge to
catastrophic magnitudes (empirically ~10^146 before the fix).

Run from the workspace root::

    python scripts/prior_correctness/emcee_gaussian_bias_check.py
"""

import os

os.environ.setdefault("PYAUTO_SKIP_WORKSPACE_VERSION_CHECK", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import emcee

import autofit as af
from autofit.example.model import Gaussian
from autofit.non_linear.fitness import Fitness


class FlatLikelihoodAnalysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return 0.0


def assert_gaussian_prior_recovers_prior_under_flat_likelihood():
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
    )

    nwalkers, nsteps = 32, 2000
    p0 = 5.0 + 0.1 * np.random.randn(nwalkers, 1)

    sampler = emcee.EnsembleSampler(
        nwalkers=nwalkers,
        ndim=1,
        log_prob_fn=fitness.call_wrap,
    )
    sampler.run_mcmc(p0, nsteps, progress=False)

    chain = sampler.get_chain()
    last_half = chain[nsteps // 2 :, :, 0].flatten()
    finite = last_half[np.isfinite(last_half)]

    assert finite.size == last_half.size, (
        f"emcee chain produced non-finite values: {last_half.size - finite.size} "
        f"of {last_half.size} samples non-finite. The log_prior_from_value "
        f"convention is likely back to cost form (positive quadratic acts as "
        f"repulsive potential)."
    )

    sample_mean = finite.mean()
    sample_std = finite.std()

    assert abs(sample_mean - 5.0) < 0.2, (
        f"Posterior mean {sample_mean:.4f} differs from prior mean 5.0 by "
        f"{abs(sample_mean - 5.0):.4f} (> 0.2). Suggests the prior is not "
        f"acting as an attractive density."
    )
    assert 0.7 < sample_std < 1.3, (
        f"Posterior std {sample_std:.4f} outside expected range (0.7, 1.3) "
        f"for GaussianPrior(sigma=1.0) under a flat likelihood. Suggests the "
        f"prior log-density has the wrong shape or sign."
    )


def assert_lognormal_prior_finite_under_flat_likelihood():
    """``LogGaussianPrior`` also routes through ``NormalMessage.log_prior_from_value``
    plus a Jacobian. Confirm the same flat-likelihood + density-prior posterior
    behaviour holds for it (chain stays finite, samples cluster near the
    geometric mean exp(0) = 1.0 with reasonable spread).
    """
    np.random.seed(1)

    model = af.Model(Gaussian)
    model.centre = 0.0
    model.normalization = 1.0
    model.sigma = af.LogGaussianPrior(mean=0.0, sigma=0.5)

    fitness = Fitness(
        model=model,
        analysis=FlatLikelihoodAnalysis(),
        paths=None,
        fom_is_log_likelihood=False,
        resample_figure_of_merit=-np.inf,
    )

    nwalkers, nsteps = 32, 2000
    p0 = 1.0 + 0.1 * np.random.randn(nwalkers, 1)

    sampler = emcee.EnsembleSampler(
        nwalkers=nwalkers,
        ndim=1,
        log_prob_fn=fitness.call_wrap,
    )
    sampler.run_mcmc(p0, nsteps, progress=False)

    chain = sampler.get_chain()
    last_half = chain[nsteps // 2 :, :, 0].flatten()
    finite = last_half[np.isfinite(last_half)]

    assert finite.size == last_half.size, (
        f"LogGaussianPrior emcee chain produced non-finite values: "
        f"{last_half.size - finite.size} of {last_half.size}. Cost-form "
        f"regression?"
    )
    assert finite.max() < 100.0, (
        f"LogGaussianPrior posterior diverged: max={finite.max():.4e}. "
        f"Cost-form regression?"
    )


if __name__ == "__main__":
    assert_gaussian_prior_recovers_prior_under_flat_likelihood()
    assert_lognormal_prior_finite_under_flat_likelihood()
    print("emcee_gaussian_bias_check: all assertions passed")
