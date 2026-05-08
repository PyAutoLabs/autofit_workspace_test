"""
Searches: BlackJAXNUTS (gradient-based MCMC, JAX-jitted likelihood)
====================================================================

Integration test for ``af.BlackJAXNUTS`` — BlackJAX's No-U-Turn Sampler running on the same 1D
Gaussian dataset as the rest of the ``searches/`` integration tests.

NUTS is a gradient-based MCMC, so the analysis must be constructed with ``use_jax=True``:

 - Switches ``Analysis._xp`` from ``numpy`` to ``jax.numpy`` so all array maths in the likelihood
   routes through JAX.
 - Lets ``af.BlackJAXNUTS`` call ``jax.grad`` of the autofit ``Fitness.call`` end-to-end without
   tripping a Python-side conversion.

Two pytree-registration calls are required so ``model.instance_from_vector`` can flow through
``jax.jit``:

 - ``enable_pytrees()`` registers ``Model`` / ``Collection`` / ``ModelInstance`` and the prior
   classes once per process.
 - ``register_model(model)`` walks the user's model and registers each concrete ``cls`` it finds
   (here, ``af.ex.Gaussian``) so its instances become traceable pytrees.

Reduced settings (``num_warmup=200``, ``num_samples=500``) keep the integration run cheap — the
goal is to verify the JAX-grad path executes end-to-end and the recovered parameters land near
truth, not to sample a science-quality posterior.

References:

 - https://github.com/blackjax-devs/blackjax
 - https://arxiv.org/abs/1111.4246  (the original NUTS paper)
"""

import numpy as np
from os import path

import autofit as af
from autofit.jax.pytrees import enable_pytrees, register_model

enable_pytrees()

"""
__Data__

Load the same 1D Gaussian dataset used by the other ``searches/`` integration tests. If it does
not yet exist on disk, run the simulator script so this test is self-contained.
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

Build the same N=3 ``Gaussian`` model used by ``Nautilus_jax.py``. After construction, register
it with JAX so its instances become pytree nodes.

The analysis is the standard ``af.ex.Analysis`` with ``use_jax=True``. No custom subclass is
needed — the example analysis already routes its array maths through ``self._xp`` and
``Gaussian.model_data_from`` already accepts an ``xp`` argument.
"""
model = af.Model(af.ex.Gaussian)

model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

register_model(model)

analysis = af.ex.Analysis(data=data, noise_map=noise_map, use_jax=True)

"""
__Search__

Run BlackJAXNUTS with ``NullPaths`` (no ``name`` / ``path_prefix`` / ``unique_tag``) so the run
is purely in-memory — the integration test asserts on the result object, not disk artefacts. The
disk-output path is exercised separately by the unit test ``test_blackjax_nuts.py`` and the
worked example in ``autofit_workspace/scripts/searches/mcmc.py``.

``num_warmup=200`` is enough for window adaptation to converge on this 3-dim problem;
``num_samples=500`` gives an ESS comfortably above 100 so ``samples.median_pdf()`` is stable.
"""
search = af.BlackJAXNUTS(num_warmup=200, num_samples=500)

result = search.fit(model=model, analysis=analysis)

"""
__Sanity checks__

The chain should recover (centre, normalization, sigma) close to the simulator truth (50, 25,
10). With 500 samples and a 3-dim model these are easily within 1σ.
"""
mp = result.samples.median_pdf()
print(
    f"BlackJAXNUTS recovered: centre={mp.centre:.3f}  normalization={mp.normalization:.3f}  sigma={mp.sigma:.3f}"
)
print(f"Truth:                  centre=50.000  normalization=25.000  sigma=10.000")

info = result.samples.samples_info
print(f"ESS (min over dims):    {info['ess_min']:.1f}")
print(f"Mean acceptance:        {info['mean_acceptance']:.3f}")
print(f"Divergences:            {info['n_divergent']}")
print(f"Total leapfrog evals:   {info['n_logl_evals']}")

# Hard guard against silent regressions in the wiring: if any of these fail the chain produced
# nonsense, even if the script still ran.
assert abs(mp.centre - 50.0) < 5.0, f"centre off by too much: {mp.centre}"
assert (
    abs(mp.normalization - 25.0) < 5.0
), f"normalization off by too much: {mp.normalization}"
assert abs(mp.sigma - 10.0) < 3.0, f"sigma off by too much: {mp.sigma}"
assert info["n_divergent"] == 0, f"unexpected divergences: {info['n_divergent']}"
assert (
    info["ess_min"] > 50.0
), f"ESS too low (chain may not have mixed): {info['ess_min']}"
