"""
Integration guard: latent variables that go NaN in an arbitrary per-sample
pattern must NOT crash the end-of-search latent summary.

This is the autofit-level guard for the bug Sam hit downstream in PyAutoLens
(``KeyError`` on ``total_lensed_source_flux``). The defect is in PyAutoFit
itself — ``autofit/non_linear/analysis/analysis.py::compute_latent_samples``
masked finite latent *columns* **per batch** on the JAX path
(``jnp.all(isfinite, axis=0)``). A single sample whose latent went NaN in one
batch dropped that latent's whole column for that batch only, so different
samples ended up with different ``Sample.kwargs`` key sets and
``Samples.summary()`` raised ``KeyError`` building its model from batch 0.

This script uses a two-latent ``af.ex.Analysis`` subclass (the shipped example
analysis has a single latent; two latents exercise the additional
``zip(LATENT_KEYS, values)`` mis-alignment that a dropped column causes). The
``PYAUTO_LATENT_NAN_INJECT=stride:N`` knob (``autoconf.test_mode``) sets NaN on
latent column 0 (``gaussian.fwhm``) for every sample whose absolute index is a
non-zero multiple of ``N``. With ``N >= batch_size`` batch 0 stays finite and
seeds the model with both keys; a later batch loses column 0.

The bug is JAX-only (the NumPy branch row-masks first), so the search runs on
the NumPy path and latents are materialised with a ``use_jax=True`` analysis.

PASS (post-fix): ``summary()`` / ``median_pdf()`` succeed and surviving latents
are finite. FAIL (pre-fix): ``KeyError``. Structural regression guard.
"""

import math
import os
from os import path

os.environ["PYAUTO_LATENT_NAN_INJECT"] = "stride:3"
# Skip the search's incidental post-fit latent pass (the NumPy path, not the
# branch under test). The explicit compute_latent_samples call below is NOT
# gated by this flag, so the JAX path we care about still runs.
os.environ["PYAUTO_SKIP_LATENTS"] = "1"

import numpy as np

import autofit as af

LATENT_BATCH_SIZE = 3  # <= the stride above, so batch 0 stays fully finite.


class TwoLatent(af.Latent):
    """Latent catalogue with two keys so a dropped column 0 mis-zips the second
    key (``gaussian.fwhm_double``), reproducing the pre-fix KeyError."""

    @staticmethod
    def keys(analysis):
        return ["gaussian.fwhm", "gaussian.fwhm_double"]

    @staticmethod
    def variables(analysis, parameters, model):
        instance = model.instance_from_vector(vector=parameters)
        fwhm = instance.gaussian.fwhm
        # Tuple positionally aligned with keys() (column 0, column 1).
        return (fwhm, 2.0 * fwhm)


class TwoLatentAnalysis(af.ex.Analysis):
    """Example analysis declaring the two-latent catalogue via ``Latent``."""

    Latent = TwoLatent


dataset_name = "gaussian_x1"
dataset_path = path.join("dataset", "example_1d", dataset_name)

if not path.exists(dataset_path):
    import subprocess
    import sys

    subprocess.run([sys.executable, "scripts/simulators/simulators.py"], check=True)

data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

model = af.Collection(gaussian=af.ex.Gaussian)
model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.gaussian.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.gaussian.sigma = af.TruncatedGaussianPrior(
    mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=np.inf
)

# Search on the NumPy path (the latent masking bug is JAX-only).
analysis = TwoLatentAnalysis(data=data, noise_map=noise_map, use_jax=False)

search = af.DynestyStatic(
    name="latent_nan_robustness",
    path_prefix=path.join("features"),
    number_of_cores=1,
    unique_tag=dataset_name,
    maxcall=1000,
    maxiter=1000,
)

result = search.fit(model=model, analysis=analysis)

assert len(result.samples.sample_list) > LATENT_BATCH_SIZE, (
    f"Need >{LATENT_BATCH_SIZE} samples for a multi-batch latent run; got "
    f"{len(result.samples.sample_list)}."
)

# Materialise latents on the JAX path — the branch the bug lives in.
analysis_jax = TwoLatentAnalysis(data=data, noise_map=noise_map, use_jax=True)

latent_samples = analysis_jax.compute_latent_samples(
    result.samples, batch_size=LATENT_BATCH_SIZE
)

assert latent_samples is not None, "compute_latent_samples returned None."

# These two calls crash pre-fix (KeyError in parameter_lists_for_paths).
summary = latent_samples.summary()
instance = latent_samples.median_pdf()


def _resolve(obj, dotted_key):
    """Walk a dotted latent path (e.g. 'gaussian.fwhm') on the instance."""
    for part in dotted_key.split("."):
        obj = getattr(obj, part)
    return obj


surviving = []
for key in TwoLatent.keys(analysis):
    try:
        value = float(_resolve(instance, key))
    except (AttributeError, TypeError):
        continue
    surviving.append((key, value))

assert surviving, "No latents survived — expected at least one finite latent."

for key, value in surviving:
    assert math.isfinite(value), f"Surviving latent '{key}' is not finite (got {value})."

print(
    f"PASSED: latent summary survived arbitrary NaN injection "
    f"({len(surviving)} latents finite, batch_size={LATENT_BATCH_SIZE})."
)
for key, value in surviving:
    print(f"  {key}: {value:.6g}")
