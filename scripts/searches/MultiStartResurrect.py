"""
Searches: MultiStart restart-on-death / resurrection (JAX)
=========================================================

Validation of the ``resurrect`` knob on the multi-start gradient MAP searches
(``af.AbstractMultiStartGradient``, Phase 2 of the multi-start gradient v2
promotion, autolens_workspace_developer#101).

When ``resurrect=True``, any start whose objective goes non-finite is redrawn
each step (fresh params from the start band + its per-start optimizer state
reinitialised), leaving the alive starts untouched. This is load-bearing on
likelihoods with broad non-finite regions (pixelized sources), where every
gradient trajectory otherwise walks into a wall within ~25-50 steps and the
``apply_if_finite`` guard alone latches the start at the cliff edge.

The pixelized-landscape validation is A100-only (the ``searches_minimal``
reference proved it — job 330598 — and the library port mirrors that exact
``reinit_starts`` logic), so it cannot run in this CPU test suite. What is
asserted here on the cheap 1D Gaussian is the property that must hold on **every**
likelihood: the knob is safe. Resurrection may fire incidentally (a broad start
drawn onto the measure-zero ``sigma≈0`` non-finite point is redrawn), but it must
never move the winning basin — ``resurrect`` on and off recover the identical MAP.

Two pytree-registration calls (as in ``MultiStartAdam.py``) let
``model.instance_from_vector`` flow through ``jax.jit``.
"""

import numpy as np
from os import path

import autofit as af
from autofit.jax.pytrees import enable_pytrees, register_model

enable_pytrees()

"""
__Data__
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
"""
model = af.Model(af.ex.Gaussian)

model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

register_model(model)

analysis = af.ex.Analysis(data=data, noise_map=noise_map, use_jax=True)

"""
__Search__

Run the same seeded search twice — ``resurrect`` off then on — and compare the
recovered MAP. Identical broad starts (seed 0) mean the only difference is
whether an incidentally-dead start is redrawn.
"""


def recovered(resurrect):
    search = af.MultiStartAdam(
        n_starts=16, n_steps=500, learning_rate=0.5, resurrect=resurrect
    )
    result = search.fit(model=model, analysis=analysis)
    instance = result.samples.max_log_likelihood()
    return (
        np.array([instance.centre, instance.normalization, instance.sigma]),
        result.samples.samples_info,
    )


off, info_off = recovered(resurrect=False)
on, info_on = recovered(resurrect=True)

print(
    f"resurrect=False: centre={off[0]:.3f}, normalization={off[1]:.3f}, "
    f"sigma={off[2]:.3f}  (n_resurrections={info_off['n_resurrections']})"
)
print(
    f"resurrect=True : centre={on[0]:.3f}, normalization={on[1]:.3f}, "
    f"sigma={on[2]:.3f}  (n_resurrections={info_on['n_resurrections']})"
)

"""
__Assertions__
"""
assert abs(on[0] - 50.0) < 2.0, on
assert abs(on[1] - 25.0) < 3.0, on
assert abs(on[2] - 10.0) < 2.0, on

assert info_off["resurrect"] is False
assert info_on["resurrect"] is True

# Safety property: resurrection never moves the winning basin.
assert np.allclose(off, on), (off, on)

print(
    "MultiStart resurrection: resurrect on/off recover the identical truth "
    "basin of the 1D Gaussian dataset."
)
