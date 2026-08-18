"""
Searches: NSS (nested slice sampling, JAX-jitted likelihood)
============================================================

Integration test for ``af.NSS`` — mainline BlackJAX's nested slice sampler (merged upstream in
blackjax 1.6) running on the same 1D Gaussian dataset as the rest of the ``searches/``
integration tests. Restored alongside the search's re-mainlining (PyAutoFit#1492); the search
was previously removed with its git-fork dependencies (PyAutoFit#1356).

NSS runs both the likelihood and the prior inside ``jax.jit``, so the analysis must be
constructed with ``use_jax=True``:

 - Switches ``Analysis._xp`` from ``numpy`` to ``jax.numpy`` so all array maths in the
   likelihood routes through JAX.
 - Lets ``af.NSS``'s inline closures trace ``model.instance_from_vector`` and
   ``model.log_prior_list_from_vector`` end-to-end.

Two pytree-registration calls are required so ``model.instance_from_vector`` can flow through
``jax.jit``:

 - ``enable_pytrees()`` registers ``Model`` / ``Collection`` / ``ModelInstance`` and the prior
   classes once per process.
 - ``register_model(model)`` walks the user's model and registers each concrete ``cls`` it finds
   (here, ``af.ex.Gaussian``) so its instances become traceable pytrees.

The fit runs twice: once un-chunked and once with ``chunk_size`` below ``num_delete``, which
routes through the PyAutoFit-local chunked kernel (``_chunked_nss.py`` — the GPU-memory path
for inversion-heavy likelihoods). At a fixed seed the chunked run must reproduce the un-chunked
run exactly (``jax.lax.map`` consumes the same keys as ``jax.vmap``), so the strongest check —
bit-parity of the evidence — comes free.

Reduced settings (``n_live=100``, ``num_delete=25``) keep the integration run cheap — the goal
is to verify the mainline-blackjax wiring executes end-to-end and the recovered parameters land
near truth, not to sample a science-quality posterior.

References:

 - https://github.com/blackjax-devs/blackjax
 - https://arxiv.org/abs/2601.23252  (Nested Slice Sampling)

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
Nested sampler that needs JAX; TEST_MODE=2 bypass returns the prior midpoint
and fails truth-recovery asserts. Run the real sampler (TEST_MODE=1 loosens
NSS's termination criterion, which keeps it fast while staying real).

ENV: real_search jax
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

Build the same N=3 ``Gaussian`` model used by ``Nautilus_jax.py`` / ``BlackJAXNUTS.py``. After
construction, register it with JAX so its instances become pytree nodes.

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

Run NSS with ``NullPaths`` (no ``name`` / ``path_prefix`` / ``unique_tag``) so the run is purely
in-memory — checkpointing auto-disables under ``NullPaths`` and the test asserts on the result
object, not disk artefacts.

``num_mcmc_steps=6`` follows the upstream ``>= max(5, 2 * dim)`` inner-steps guidance at dim=3
(fewer inner steps biases the evidence upward — the blackjax 1.6 docstring states this
explicitly).
"""
search = af.NSS(n_live=100, num_mcmc_steps=6, num_delete=25, seed=1)

result = search.fit(model=model, analysis=analysis)

"""
__Sanity checks__

The posterior should recover (centre, normalization, sigma) close to the simulator truth
(50, 25, 10), and the nested-sampling evidence machinery should report a finite log-evidence
with a positive Monte-Carlo error from the simulated-volume ensemble.
"""
mp = result.samples.median_pdf()
info = result.samples.samples_info
print(
    f"NSS recovered: centre={mp.centre:.3f}  normalization={mp.normalization:.3f}  sigma={mp.sigma:.3f}"
)
print(f"Truth:         centre=50.000  normalization=25.000  sigma=10.000")
print(f"log(Z):        {info['log_evidence']:.3f} +/- {info['log_evidence_error']:.3f}")
print(f"Evals:         {info['total_samples']}   ESS: {info['ess']}")

assert abs(mp.centre - 50.0) < 5.0, f"centre off by too much: {mp.centre}"
assert (
    abs(mp.normalization - 25.0) < 5.0
), f"normalization off by too much: {mp.normalization}"
assert abs(mp.sigma - 10.0) < 3.0, f"sigma off by too much: {mp.sigma}"
assert np.isfinite(info["log_evidence"]), f"non-finite logZ: {info['log_evidence']}"
assert (
    info["log_evidence_error"] > 0.0
), f"logZ ensemble error not positive: {info['log_evidence_error']}"
assert info["total_samples"] > 0, "no likelihood evaluations recorded"
assert info["ess"] > 50, f"ESS too low (run may not have converged): {info['ess']}"

"""
__Chunked path__

Re-run at the same seed with ``chunk_size`` below ``num_delete`` so the inner MCMC fan-out and
the n_live-wide init route through ``jax.lax.map`` chunks (the A100 memory knob for
inversion-heavy likelihoods). Same seed + same key consumption ⇒ the evidence must reproduce
the un-chunked run exactly.
"""
search_chunked = af.NSS(
    n_live=100, num_mcmc_steps=6, num_delete=25, chunk_size=7, seed=1
)

result_chunked = search_chunked.fit(model=model, analysis=analysis)

info_chunked = result_chunked.samples.samples_info
print(
    f"NSS chunked:   log(Z) = {info_chunked['log_evidence']:.3f} "
    f"(unchunked {info['log_evidence']:.3f})"
)

assert abs(info_chunked["log_evidence"] - info["log_evidence"]) < 1e-6, (
    "chunked NSS diverged from the unchunked run at fixed seed: "
    f"{info_chunked['log_evidence']} vs {info['log_evidence']}"
)
