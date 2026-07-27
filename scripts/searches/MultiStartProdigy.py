"""
Searches: MultiStartProdigy (JAX-jitted learning-rate-free gradient MAP optimizer)
=================================================================================

End-to-end validation of ``af.MultiStartProdigy`` — the learning-rate-free
multi-start gradient MAP search promoted from the lr-free optimizer experiment
(autolens_workspace_developer#101, Phase 1). It runs ``N`` broad starts in
parallel via ``jax.vmap``, taking a Prodigy step per start on the unconstrained
parameterization, and returns the best-basin start as the maximum-log-posterior
point.

Prodigy is resolved from ``optax.contrib`` and estimates its own step scale, so
it is run with **no** ``learning_rate``. Two properties make this the load-bearing
JAX check for the Phase-1 promotion, neither expressible in PyAutoFit's NumPy-only
unit suite:

 - **optax.contrib resolution + apply_if_finite guard** actually execute (the
   rule is not in ``optax`` itself), and
 - the optimizer state is ``jax.vmap``ed **per start**, so Prodigy's global
   distance estimate ``d`` is independent across starts rather than coupled by a
   single stacked-``(n_starts, ndim)`` state.

It asserts that the search recovers the known truth basin of the 1D Gaussian
dataset (``centre, normalization, sigma = 50, 25, 10``) with no learning rate —
on this parametric cell Prodigy matches hand-tuned Adam (the headline #101
result).

Two pytree-registration calls (as in ``MultiStartAdam.py``) let
``model.instance_from_vector`` flow through ``jax.jit``:

 - ``enable_pytrees()`` registers ``Model`` / ``Collection`` / ``ModelInstance``
   and the prior classes once per process.
 - ``register_model(model)`` registers each concrete ``cls`` in the model
   (here ``af.ex.Gaussian``) so its instances become traceable pytrees.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
Learning-rate-free JAX gradient MAP search; TEST_MODE=2 bypasses the search
entirely and returns prior midpoints, so the truth-basin assertions below fail
on ``normalization`` (the ``LogUniform(1e-2, 1e2)`` midpoint is 1.0, not 25.0).
Run it for real with JAX.

ENV: real_search jax
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

Run ``MultiStartProdigy`` with ``NullPaths`` (no ``name`` / ``path_prefix``) so
nothing is written to disk. No ``learning_rate`` is passed — Prodigy estimates
its own. A modest start count and step budget keep the run cheap — the goal is to
verify the learning-rate-free JAX gradient path (optax.contrib rule + per-start
vmapped state + apply_if_finite) executes end-to-end and lands in the truth
basin, not to profile it.
"""
search = af.MultiStartProdigy(n_starts=16, n_steps=500)

assert search.learning_rate is None, search.learning_rate

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

print(
    "MultiStartProdigy recovered the truth basin of the 1D Gaussian dataset "
    "with no learning rate."
)
