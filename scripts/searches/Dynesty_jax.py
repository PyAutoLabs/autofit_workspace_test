"""
Searches: DynestyStatic (JAX-jitted likelihood)
===============================================

Companion to ``Nautilus_jax.py`` that exercises the JAX-jitted likelihood path
on the dynesty nested sampler instead of Nautilus.

The non-linear search is ``af.DynestyStatic`` on the same 1D Gaussian dataset.
The analysis is constructed with ``use_jax=True``, which:

 - Switches ``Analysis._xp`` from ``numpy`` to ``jax.numpy`` so all array
   maths in the likelihood routes through JAX.
 - Causes ``AbstractDynesty._fit`` to skip its multiprocessing-Pool branch
   (jit-compiled callables don't roundtrip across multiprocessing workers)
   and ask ``Fitness`` to wrap ``self.call`` in ``jax.jit`` via
   ``autofit/non_linear/fitness.py::Fitness._jit``.

Unlike Nautilus, dynesty 2.1.5 has **no ``vectorized`` parameter** — it calls
the likelihood one sample at a time. So this path uses ``jax.jit`` only
(no ``jax.vmap``); JAX's compiled-function cache reuses the compiled
version across calls.

Two pytree-registration calls are required so ``model.instance_from_vector``
can flow through ``jax.jit``:

 - ``enable_pytrees()`` registers ``Model`` / ``Collection`` / ``ModelInstance``
   and the prior classes once per process.
 - ``register_model(model)`` walks the user's model and registers each
   concrete ``cls`` it finds (here, ``af.ex.Gaussian``) so its instances
   become traceable pytrees.

References:

 - https://dynesty.readthedocs.io/en/stable/
 - https://github.com/joshspeagle/dynesty
"""

from os import path

import autofit as af
from autofit.jax.pytrees import enable_pytrees, register_model

enable_pytrees()

"""
__Data__

Load the same 1D Gaussian dataset used by the other searches in this folder.
If it does not yet exist on disk, run the simulator script so this test is
self-contained.
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

Build the same single-``Gaussian`` model used by the other JAX-path tests.
After construction, register it with JAX so its instances become pytree nodes.
"""
model = af.Model(af.ex.Gaussian)

model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

register_model(model)

analysis = af.ex.Analysis(data=data, noise_map=noise_map, use_jax=True)

"""
__Search__

Run DynestyStatic with ``NullPaths`` (no ``name`` / ``path_prefix`` /
``unique_tag``) so nothing is written to disk. ``nlive=30`` and a loose
``dlogz=0.5`` keep the run cheap — the goal is to verify the JAX-jit path
executes end-to-end, not to sample a science-quality posterior.
"""
search = af.DynestyStatic(nlive=30, dlogz=0.5)

result = search.fit(model=model, analysis=analysis)
