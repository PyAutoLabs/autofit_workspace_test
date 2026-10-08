"""
Searches: MultiStartADABelief (JAX-jitted gradient MAP optimizer)
=================================================================

End-to-end validation of ``af.MultiStartADABelief``, the multi-start gradient MAP search
using the ADABelief optax update rule. It runs ``N`` broad starts in parallel via
``jax.vmap``, takes a ADABelief step per start on the unconstrained
parameterization, and returns the best-basin start as the maximum-log-posterior
point. ``MultiStartAdam.py`` and ``MultiStartProdigy.py`` cover the other update
rules.

ADABelief scales its step by the variance of the gradient around its running
mean rather than by the raw second moment, and tied Adam in the GPU
MAP-optimizer benchmark. It takes the same learning rate as ``MultiStartAdam.py``.

Two pytree-registration calls (as in ``MultiStartAdam.py``) let
``model.instance_from_vector`` flow through ``jax.jit``:

 - ``enable_pytrees()`` registers ``Model`` / ``Collection`` / ``ModelInstance``
   and the prior classes once per process.
 - ``register_model(model)`` registers each concrete ``cls`` in the model
   (here ``af.ex.Gaussian``) so its instances become traceable pytrees.

This script is run on demand; it is not listed in ``smoke_tests.txt``.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX-native gradient MAP search; TEST_MODE=2 bypasses its JAX path. Run it for real with JAX.

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

Run ``MultiStartADABelief`` with ``NullPaths`` (no ``name`` / ``path_prefix``) so nothing is
written to disk. A modest start count and step budget keep the run cheap.
"""
search = af.MultiStartADABelief(n_starts=16, n_steps=500, learning_rate=0.5)

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

    print("MultiStartADABelief recovered the truth basin of the 1D Gaussian dataset.")
