"""
Searches: MultiStartAdam (JAX-jitted gradient MAP optimizer)
============================================================

End-to-end validation of ``af.MultiStartAdam`` — the multi-start first-order
gradient MAP search promoted from the GPU MAP-optimizer benchmark (PyAutoFit
#1369). It runs ``N`` broad starts in parallel via ``jax.vmap``, taking a fixed
self-normalised Adam step per start on the unconstrained parameterization, and
returns the best-basin start as the maximum-log-posterior point.

This is the JAX cross-backend check that deliberately lives here rather than in
PyAutoFit's unit suite (library unit tests stay NumPy-only). It asserts that the
search recovers the known truth basin of the 1D Gaussian dataset
(``centre, normalization, sigma = 50, 25, 10``).

Two pytree-registration calls (as in ``Nautilus_jax.py``) let
``model.instance_from_vector`` flow through ``jax.jit``:

 - ``enable_pytrees()`` registers ``Model`` / ``Collection`` / ``ModelInstance``
   and the prior classes once per process.
 - ``register_model(model)`` registers each concrete ``cls`` in the model
   (here ``af.ex.Gaussian``) so its instances become traceable pytrees.
"""

import numpy as np
from os import path

import autofit as af
from autofit.jax.pytrees import enable_pytrees, register_model

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

Run ``MultiStartAdam`` with ``NullPaths`` (no ``name`` / ``path_prefix``) so
nothing is written to disk. A modest start count and step budget keep the run
cheap — the goal is to verify the JAX gradient path executes end-to-end and
lands in the truth basin, not to profile it.
"""
search = af.MultiStartAdam(n_starts=16, n_steps=500, learning_rate=0.5)

result = search.fit(model=model, analysis=analysis)

"""
__Assertion__

The best-basin instance must recover the truth to within a loose tolerance
(the MAP sits near truth, offset only by the dataset noise).
"""
instance = result.samples.max_log_likelihood()

print(
    f"Recovered: centre={instance.centre:.3f}, "
    f"normalization={instance.normalization:.3f}, sigma={instance.sigma:.3f}"
)

assert abs(instance.centre - 50.0) < 2.0, instance.centre
assert abs(instance.normalization - 25.0) < 3.0, instance.normalization
assert abs(instance.sigma - 10.0) < 2.0, instance.sigma

print("MultiStartAdam recovered the truth basin of the 1D Gaussian dataset.")
