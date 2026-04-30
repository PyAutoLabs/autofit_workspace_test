"""
Searches: Nautilus (JAX-jitted likelihood)
==========================================

Companion to ``Nautilus.py`` that exercises the JAX-jitted likelihood path.

The non-linear search is identical (Nautilus on the 1D Gaussian dataset). The
difference is that the analysis is constructed with ``use_jax=True``, which:

 - Switches ``Analysis._xp`` from ``numpy`` to ``jax.numpy`` so all array maths
   in the likelihood routes through JAX.
 - Causes ``af.Nautilus`` to take its ``fit_x1_cpu`` branch and ask ``Fitness``
   to wrap the per-sample call in ``jax.vmap(jax.jit(self.call))`` (see
   ``autofit/non_linear/fitness.py::Fitness._vmap``). Nautilus then receives a
   vectorised likelihood and runs in vectorised mode (``vectorized=True``).

Two pytree-registration calls are required so ``model.instance_from_vector``
can flow through ``jax.jit``:

 - ``enable_pytrees()`` registers ``Model`` / ``Collection`` / ``ModelInstance``
   and the prior classes once per process.
 - ``register_model(model)`` walks the user's model and registers each
   concrete ``cls`` it finds (here, ``af.ex.Gaussian``) so its instances
   become traceable pytrees.

References:

 - https://nautilus-sampler.readthedocs.io/en/stable/index.html
 - https://github.com/johannesulf/nautilus
"""
import numpy as np
from os import path

import autofit as af
from autofit.jax.pytrees import enable_pytrees, register_model

enable_pytrees()

"""
__Data__

Load the same 1D Gaussian dataset used by ``Nautilus.py``. If it does not yet
exist on disk, run the simulator script so this test is self-contained.
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

Build the same N=3 ``Gaussian`` model. After construction, register it with
JAX so its instances become pytree nodes.

The analysis is the standard ``af.ex.Analysis`` with ``use_jax=True``. No
custom subclass is needed — the example analysis already routes its array
maths through ``self._xp`` and the ``Gaussian.model_data_from`` already
accepts an ``xp`` argument.
"""
model = af.Model(af.ex.Gaussian)

model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

register_model(model)

analysis = af.ex.Analysis(data=data, noise_map=noise_map, use_jax=True)

"""
__Search__

Run Nautilus with ``NullPaths`` (no ``name`` / ``path_prefix`` / ``unique_tag``)
so nothing is written to disk. ``n_live=50`` and ``n_eff=100`` keep the run
cheap — the goal is to verify the JAX-jit path executes end-to-end, not to
sample a science-quality posterior.
"""
search = af.Nautilus(n_live=50, n_eff=100)

result = search.fit(model=model, analysis=analysis)
