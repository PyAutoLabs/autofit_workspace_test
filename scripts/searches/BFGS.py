"""
Searches: BFGS
==============

Integration test for ``af.BFGS``, the scipy ``optimize.minimize`` BFGS method
(``LBFGS.py`` covers the limited-memory variant). Gradients are estimated by
finite differences, so the analysis is NumPy (``use_jax=False``).

This script is run on demand; it is not listed in ``smoke_tests.txt``.

Information about the BFGS method can be found at the following link:

 - https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html
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

Run ``BFGS`` with ``NullPaths`` (no ``name`` / ``path_prefix``) so nothing is
written to disk. The optimizer starts from a point drawn from the priors; the
1D Gaussian likelihood is smooth enough that it converges to the truth basin.
"""
search = af.BFGS(tol=None, gtol=1e-05, maxiter=15000)

result = search.fit(model=model, analysis=analysis)

"""
__Assertion__

Under ``PYAUTO_TEST_MODE`` the search runs a cut-down schedule, so only the
wiring is checked. A real run must also land in the truth basin.
"""
instance = result.samples.max_log_likelihood()

print(
    f"Recovered: centre={instance.centre:.3f}, "
    f"normalization={instance.normalization:.3f}, sigma={instance.sigma:.3f}"
)

assert np.isfinite(result.samples.max_log_likelihood_sample.log_likelihood)

if not is_test_mode():
    assert abs(instance.centre - 50.0) < 2.0, instance.centre
    assert abs(instance.normalization - 25.0) < 3.0, instance.normalization
    assert abs(instance.sigma - 10.0) < 2.0, instance.sigma

    print("BFGS recovered the truth basin of the 1D Gaussian dataset.")
