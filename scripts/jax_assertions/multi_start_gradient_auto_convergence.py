"""
JAX assertions: multi-start gradient auto-convergence (early stopping)
=====================================================================

Validation of the auto-convergence (early-stopping) mode on the multi-start
gradient MAP searches (``af.AbstractMultiStartGradient`` — ``MultiStartAdam`` /
``MultiStartADABelief`` / ``MultiStartLion`` / ``MultiStartProdigy``), phase 1 of
the auto-convergence work (PyAutoFit #1406/#1407).

The searches now stop early by default once the global-best figure-of-merit has
plateaued, so users no longer hand-tune ``n_steps`` (which stays a hard ceiling).
The convergence settings + plateau maths are unit-tested NumPy-only in PyAutoFit;
this script is the JAX cross-backend validation that deliberately lives here
(library unit tests stay NumPy-only). Two properties are asserted on the cheap 1D
Gaussian:

**A. Early stopping is real and lands at the basin.** With a generous ``n_steps``
ceiling, the fit terminates in fewer than ``n_steps`` steps
(``samples_info["total_steps"] < n_steps``) *and* still recovers the known truth
basin (``centre, normalization, sigma = 50, 25, 10``) — i.e. it stops because the
best figure-of-merit plateaued at the solution, not prematurely.

**B. Recall does not recompile (deterministic cache-key guard).** The search
builds its jitted ``value_and_grad`` identically on the fresh and resume paths, so
a resumed run must reuse the disk-cached compiled function (the persistent JAX
compilation cache, default-on via ``autonerves``) rather than recompiling. That
holds iff the traced computation is byte-identical across independent builds — the
persistent cache keys on the lowered HLO. We build the search's
``jax.jit(jax.vmap(jax.value_and_grad(fitness.call)))`` twice from scratch (as two
fresh processes would) and assert the lowered StableHLO is identical, so the
resume path hits the cache and never re-pays the compile. This is a deterministic
guard (no timing, no JAX-internal monitoring); the persistent cache mechanism
itself is separately certified (autolens_profiling/jax_compile, PyAutoConf #128).

Two pytree-registration calls (as in ``MultiStartAdam.py``) let
``model.instance_from_vector`` flow through ``jax.jit``.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX assertion scripts test JAX behaviour; disabling JAX makes their
assertions vacuous.

ENV: jax
"""

import numpy as np
from os import path

import jax

import autofit as af
from autofit.jax.pytrees import enable_pytrees, register_model
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.paths.null import NullPaths

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
__A. Early stopping lands at the basin__

``n_steps=300`` is a generous ceiling; auto-convergence (on by default) should
stop the fit well before it, once the global-best figure-of-merit has plateaued.
The fit is written to disk (``name`` / ``path_prefix``) so the results-DB
round-trip in Part C can reload this same result.
"""
n_steps = 300

name = "multi_start_gradient_auto_convergence"
path_prefix = "jax_assertions"

search = af.MultiStartAdam(
    name=name,
    path_prefix=path_prefix,
    n_starts=16,
    n_steps=n_steps,
    learning_rate=0.5,
)

# Auto-convergence is on by default.
assert search.convergence.check_for_convergence is True

result = search.fit(model=model, analysis=analysis)

total_steps = int(result.samples.samples_info["total_steps"])
print(f"Auto-convergence stopped after {total_steps} / {n_steps} steps.")

# Early stopping actually fired: fewer than the hard ceiling of steps were taken.
assert total_steps < n_steps, total_steps

instance = result.samples.max_log_likelihood()
print(
    f"Recovered: centre={instance.centre:.3f}, "
    f"normalization={instance.normalization:.3f}, sigma={instance.sigma:.3f}"
)

# It stopped at the solution, not prematurely: the truth basin is still recovered.
assert abs(instance.centre - 50.0) < 2.0, instance.centre
assert abs(instance.normalization - 25.0) < 3.0, instance.normalization
assert abs(instance.sigma - 10.0) < 2.0, instance.sigma

print("Auto-convergence early-stopped at the truth basin of the 1D Gaussian.")

"""
__B. Recall does not recompile (deterministic cache-key guard)__

Reconstruct the exact objective the search jits —
``jax.jit(jax.vmap(jax.value_and_grad(fitness.call)))`` — twice from scratch, as
two independent processes would (a cold run and a later resume). If the lowered
StableHLO is byte-identical, the two builds share a persistent-compilation-cache
key, so the resume loads the compiled ``value_and_grad`` from disk instead of
recompiling. Only shapes/dtypes affect the HLO, so representative params suffice.
"""


def lowered_value_and_grad_hlo():
    """Build the search's jitted value_and_grad from scratch and return its
    lowered StableHLO text (the persistent-cache key is a hash of this)."""
    fitness = Fitness(
        model=model,
        analysis=analysis,
        paths=NullPaths(),
        fom_is_log_likelihood=False,
        resample_figure_of_merit=-np.inf,
        convert_to_chi_squared=True,
    )
    value_and_grad = jax.jit(jax.vmap(jax.value_and_grad(fitness.call)))
    params = jax.numpy.asarray(
        np.random.default_rng(0).uniform(0.15, 0.85, size=(16, model.prior_count))
    )
    return value_and_grad.lower(params).as_text()


hlo_cold = lowered_value_and_grad_hlo()
hlo_resume = lowered_value_and_grad_hlo()

assert hlo_cold == hlo_resume, (
    "The multi-start gradient value_and_grad lowered to different HLO across two "
    "independent builds, so a resumed fit would miss the persistent compilation "
    "cache and recompile. The resume path must build a byte-identical trace."
)

print(
    "Resume path builds a byte-identical value_and_grad HLO "
    f"({len(hlo_cold)} chars) — the persistent compilation cache hits on recall, "
    "so recall never recompiles."
)

"""
__C. Results-DB round-trip of the auto-convergence outcome__

The phase-2 results contract adds ``converged`` / ``stop_reason`` / the
convergence settings / the ``fom_history`` trace to ``samples_info``. Reload the
Part-A fit (written to disk above) through the directory aggregator and assert
those survive serialisation — so a user inspecting a stored result can see whether
a run converged and verify the plateau.
"""
from pathlib import Path

from autofit import with_test_mode_segment
from autofit.aggregator.aggregator import Aggregator

output_path = with_test_mode_segment(Path("output"))

agg = Aggregator.from_directory(directory=output_path / path_prefix / name)

assert len(agg) > 0, "no results loaded from the directory aggregator"

for samples in agg.values("samples"):
    info = samples.samples_info

    assert info["converged"] is True, info.get("stop_reason")
    assert info["stop_reason"] == "converged", info["stop_reason"]

    convergence = info["convergence"]
    assert convergence["check_for_convergence"] is True

    fom_history = info["fom_history"]
    assert fom_history is not None, "fom_history trace did not round-trip"
    assert len(fom_history) == info["total_steps"], (
        len(fom_history),
        info["total_steps"],
    )
    assert (
        len(fom_history) < n_steps
    ), "early-stopped run should be shorter than the ceiling"

    print(
        "Results-DB round-trip OK: "
        f"converged={info['converged']}, stop_reason={info['stop_reason']}, "
        f"fom_history trace length {len(fom_history)} (ceiling {n_steps})."
    )
