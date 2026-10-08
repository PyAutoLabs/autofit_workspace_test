"""
Searches: SMC (BlackJAX adaptive tempered SMC, JAX-jitted likelihood)
=====================================================================

Integration test for ``af.SMC``, BlackJAX's adaptive tempered Sequential Monte
Carlo with a gradient (MALA) inner kernel, on the same 1D Gaussian dataset as the
other ``searches/`` integration tests.

SMC anneals a cloud of particles from the prior to the posterior and returns the
log evidence alongside the posterior samples. Its inner kernel uses gradients, so
the analysis is built with ``use_jax=True`` and the model is registered as a JAX
pytree (as in ``BlackJAXNUTS.py``).

This script is run on demand; it is not listed in ``smoke_tests.txt``.

References:

 - https://github.com/blackjax-devs/blackjax

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
Gradient SMC that needs JAX; TEST_MODE=1 cuts it to 16 particles and
5 tempering steps, which is too short to reach the posterior. Run it for real with JAX.

ENV: real_search jax
"""

import numpy as np
from os import path

import autofit as af
from autofit.jax.pytrees import enable_pytrees, register_model
from autofit.non_linear.test_mode import is_test_mode

enable_pytrees()

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

The standard N=3 ``Gaussian`` model with a JAX-traceable analysis
(``use_jax=True`` routes the likelihood maths through ``jax.numpy``).
"""
model = af.Model(af.ex.Gaussian)

model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

register_model(model)

analysis = af.ex.Analysis(data=data, noise_map=noise_map, use_jax=True)

"""
__Search__

Run SMC with ``NullPaths`` (no ``name`` / ``path_prefix``) so nothing is written
to disk. 256 particles and the default schedule are plenty for this 3-parameter
problem.
"""
search = af.SMC(num_particles=256, num_mcmc_steps=5, seed=42)

result = search.fit(model=model, analysis=analysis)

"""
__Assertion__

Under ``PYAUTO_TEST_MODE`` SMC runs a cut-down schedule, so only the wiring is
checked. A real run must reach ``lambda = 1``, report a finite log evidence and
recover the truth.
"""
mp = result.samples.median_pdf()
info = result.samples.samples_info

print(
    f"SMC recovered: centre={mp.centre:.3f}  normalization={mp.normalization:.3f}  sigma={mp.sigma:.3f}"
)
print(f"Truth:         centre=50.000  normalization=25.000  sigma=10.000")
print(f"SMC steps:     {info['n_smc_steps']}  converged: {info['converged']}")
print(f"log evidence:  {info['log_evidence']}")

if not is_test_mode():
    assert info["converged"], "SMC did not reach lambda = 1"
    assert np.isfinite(info["log_evidence"]), info["log_evidence"]
    assert abs(mp.centre - 50.0) < 5.0, f"centre off by too much: {mp.centre}"
    assert (
        abs(mp.normalization - 25.0) < 5.0
    ), f"normalization off by too much: {mp.normalization}"
    assert abs(mp.sigma - 10.0) < 3.0, f"sigma off by too much: {mp.sigma}"
