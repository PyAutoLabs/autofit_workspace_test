"""
Searches: Drawer
================

Integration test for ``af.Drawer``, which draws a fixed number of points from the
priors and keeps the best one. It does no optimization, so it is used to sample
the prior or to seed another search, not to fit.

The test checks that the requested number of draws is made and that the best
draw is a valid, finite-likelihood point. The analysis is NumPy (``use_jax=False``).

This script is run on demand; it is not listed in ``smoke_tests.txt``.
"""

import numpy as np
from os import path

import autofit as af
from autofit.non_linear.test_mode import is_test_mode

"""
__Data__

Load the 1D Gaussian dataset (truth: centre=50, normalization=25, sigma=10).
Simulate it first if it is not already on disk, so this script is self-contained.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")

if not path.exists(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/simulators/simulators.py"],
        check=True,
    )

data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

"""
__Model + Analysis__

The standard N=3 ``Gaussian`` model with a NumPy analysis.
"""
model = af.Model(af.ex.Gaussian)

model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

analysis = af.ex.Analysis(data=data, noise_map=noise_map, use_jax=False)

"""
__Search__

Run ``Drawer`` with a ``name`` and ``path_prefix`` so results are written to
``output/searches/Drawer``. ``Drawer`` is not run with ``NullPaths`` here because
its ``_fit`` reads ``self.timer.time``, and the timer is ``None`` under
``NullPaths``.
"""
total_draws = 50

search = af.Drawer(path_prefix="searches", name="Drawer", total_draws=total_draws)

result = search.fit(model=model, analysis=analysis)

"""
__Assertion__

Every draw lies within the priors and the best draw has a finite likelihood.
Outside ``PYAUTO_TEST_MODE`` the number of samples equals ``total_draws``.
"""
instance = result.samples.max_log_likelihood()

print(
    f"Best draw: centre={instance.centre:.3f}, "
    f"normalization={instance.normalization:.3f}, sigma={instance.sigma:.3f}"
)
print(f"Number of samples: {len(result.samples.sample_list)}")

assert np.isfinite(result.samples.max_log_likelihood_sample.log_likelihood)
assert 0.0 <= instance.centre <= 100.0, instance.centre
assert 1e-2 <= instance.normalization <= 1e2, instance.normalization
assert 0.0 <= instance.sigma <= 30.0, instance.sigma

if not is_test_mode():
    assert len(result.samples.sample_list) == total_draws, len(
        result.samples.sample_list
    )
