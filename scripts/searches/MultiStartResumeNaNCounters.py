"""
Searches: MultiStart resume — NaN lane-step accounting across a kill/resume (JAX)
================================================================================

Validation that the value-NaN / gradient-NaN step counters on the multi-start
gradient MAP searches (``n_value_nan_lane_steps`` / ``n_grad_nan_lane_steps``,
PyAutoFit#1472) keep accumulating when a search is killed mid-run and resumed,
rather than restarting from zero.

Those counters restore via ``search_internal.get(..., 0)`` and are designed to be
lifetime totals across a resume. Until PyAutoFit#1474 that behaviour could not be
demonstrated end-to-end at all: ``Fitness.check_log_likelihood`` compared a stored
log likelihood against the search's own figure of merit, so *every* multi-start
resume died with a ``SearchException`` before reaching the fit loop. #1472
therefore shipped with the resume accumulation covered only by unit tests over
hand-built ``search_internal`` dicts. This script is the end-to-end cover.

__What is asserted__

The invariant is equality, not merely "the counters went up":

    a killed-and-resumed run reports the SAME NaN accounting as an
    uninterrupted one

The search is deterministic (broad starts seeded 0), so an uninterrupted run is
an exact reference. If a resume reset either counter to zero, the interrupted arm
would report strictly fewer NaN lane-steps than the reference and the equality
fails. A guard below also rejects a vacuous pass — the run must actually produce
NaN lane-steps, and the kill must land before the search finishes.

__The NaN traps__

``_broad_starts`` rejects any draw whose objective or gradient is non-finite, so
every lane begins healthy by construction and a trap out at the edges of the prior
is never reached. Both traps therefore sit ON the descent path, which is also the
realistic case (a pixelized likelihood going degenerate near its solution):

* **gradient-NaN** — ``|centre - 50| < 2``, the basin of the truth. Inside the band
  the selected branch of the ``where`` is a finite ``0.0`` while the unselected
  ``sqrt`` of a negative is NaN; reverse-mode differentiates both and
  ``0 * NaN = NaN``, so the value stays finite and only the gradient dies. This is
  the trap documented in ``Fitness.call``.
* **value-NaN** — ``sigma < 10.5``. The truth sigma is 10.0, so surviving starts
  descend across the threshold and die there.

``resurrect=False`` so dead lanes stay dead and keep counting every step, making
the totals grow monotonically and the comparison unambiguous.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX gradient MAP search driven as a subprocess so it can be killed mid-run;
TEST_MODE=2 bypasses the search entirely, so no ``search_internal`` is built and
the counters this script reports on are absent from ``samples_info``. Run it for
real with JAX.

ENV: real_search jax
"""

import glob
import json
import os
import signal
import subprocess
import sys
import time
from os import path

import numpy as np

import autofit as af
from autofit.jax.pytrees import enable_pytrees, register_model

enable_pytrees()

"""
__Data__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")

if not path.exists(dataset_path):
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

The analysis subclasses the example one and adds the two NaN traps on top of the
unmodified log likelihood, so the underlying fit is the familiar 1D Gaussian.
"""
model = af.Model(af.ex.Gaussian)

model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

register_model(model)


class NaNTrapAnalysis(af.ex.Analysis):
    def log_likelihood_function(self, instance, shared=None, xp=np):
        log_likelihood = super().log_likelihood_function(
            instance=instance, shared=shared, xp=xp
        )
        xp = self._xp

        # gradient-NaN: finite value, non-finite derivative, inside the truth basin.
        offset = 2.0 - xp.abs(instance.centre - 50.0)
        grad_trap = xp.where(offset < 0.0, xp.sqrt(-offset), 0.0)
        log_likelihood = log_likelihood + 0.0 * grad_trap

        # value-NaN: likelihood undefined below the truth sigma.
        log_likelihood = xp.where(instance.sigma < 10.5, xp.nan, log_likelihood)

        return log_likelihood


analysis = NaNTrapAnalysis(data=data, noise_map=noise_map, use_jax=True)

NAME = "multi_start_resume_nan_counters"
CHILD_FLAG = "--run-search"


def run_search(path_prefix):
    """
    One `fit`. Resumes automatically when a `search_internal` checkpoint is
    already present, which is what makes re-invoking this the resume arm.

    `iterations_per_full_update=2` so checkpoints are written *during* the run —
    the default single-chunk cadence only checkpoints at the end, and a mid-run
    kill would then leave nothing to resume from.
    """
    search = af.MultiStartAdam(
        path_prefix=path_prefix,
        name=NAME,
        n_starts=12,
        n_steps=400,
        learning_rate=0.05,
        iterations_per_full_update=2,
        resurrect=False,
    )
    search.fit(model=model, analysis=analysis)


if CHILD_FLAG in sys.argv:
    run_search(path_prefix=sys.argv[sys.argv.index(CHILD_FLAG) + 1])
    raise SystemExit(0)


"""
__Helpers__
"""


def _one(pattern):
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def checkpoint_counters(path_prefix):
    """The counters as they stand in the live mid-run checkpoint, or None."""
    import dill

    p = _one(
        f"output/{path_prefix}/{NAME}/*/files/search_internal/search_internal.dill"
    )
    if p is None:
        return None
    try:
        with open(p, "rb") as f:
            search_internal = dill.load(f)
    except Exception:
        # A checkpoint caught mid-write is not an error — just try again later.
        return None
    return {
        "total_steps": int(search_internal["total_steps"]),
        "n_value_nan_lane_steps": int(
            search_internal.get("n_value_nan_lane_steps", 0)
        ),
        "n_grad_nan_lane_steps": int(search_internal.get("n_grad_nan_lane_steps", 0)),
    }


def final_counters(path_prefix):
    """The counters recorded in `samples_info` once the fit has finished."""
    p = _one(f"output/{path_prefix}/{NAME}/*/files/samples_info.json")
    assert p is not None, f"no samples_info.json written under output/{path_prefix}"
    samples_info = json.load(open(p))
    return {
        "total_steps": int(samples_info["total_steps"]),
        "n_value_nan_lane_steps": int(samples_info["n_value_nan_lane_steps"]),
        "n_grad_nan_lane_steps": int(samples_info["n_grad_nan_lane_steps"]),
    }


def child(path_prefix):
    return subprocess.Popen(
        [sys.executable, __file__, CHILD_FLAG, path_prefix],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


"""
__Reference arm: an uninterrupted run__
"""
reference_prefix = "nan_counters_reference"

child(reference_prefix).wait()
reference = final_counters(reference_prefix)

print(
    f"uninterrupted : total_steps={reference['total_steps']}, "
    f"value_nan={reference['n_value_nan_lane_steps']}, "
    f"grad_nan={reference['n_grad_nan_lane_steps']}"
)

# Guard against a vacuous pass: if the traps never fire there is nothing to carry
# across the resume and the equality assertion below would hold trivially.
assert reference["n_value_nan_lane_steps"] > 0, reference
assert reference["n_grad_nan_lane_steps"] > 0, reference

"""
__Interrupted arm: kill mid-run, then resume__

The kill waits until the checkpoint shows a NaN lane-step already recorded, so
there is a non-zero total that the resume has to carry rather than recount.
"""
resume_prefix = "nan_counters_resume"

process = child(resume_prefix)
at_kill = None

for _ in range(600):
    if process.poll() is not None:
        break
    counters = checkpoint_counters(resume_prefix)
    if counters is not None and (
        counters["n_value_nan_lane_steps"] > 0 or counters["n_grad_nan_lane_steps"] > 0
    ):
        at_kill = counters
        break
    time.sleep(0.5)

assert at_kill is not None, (
    "never observed a mid-run checkpoint carrying a non-zero NaN counter, so the "
    "resume would have had nothing to accumulate onto"
)

os.kill(process.pid, signal.SIGKILL)
process.wait()

print(
    f"killed at     : total_steps={at_kill['total_steps']}, "
    f"value_nan={at_kill['n_value_nan_lane_steps']}, "
    f"grad_nan={at_kill['n_grad_nan_lane_steps']}"
)

# The kill must land before the search finished, or this is not a resume at all.
assert at_kill["total_steps"] < reference["total_steps"], (at_kill, reference)

# Re-invoking the identical search is the resume: it picks the checkpoint up.
child(resume_prefix).wait()
resumed = final_counters(resume_prefix)

print(
    f"killed+resumed: total_steps={resumed['total_steps']}, "
    f"value_nan={resumed['n_value_nan_lane_steps']}, "
    f"grad_nan={resumed['n_grad_nan_lane_steps']}"
)

"""
__Assertions__
"""

# The invariant. A resume that reset either counter would report strictly fewer
# NaN lane-steps than the uninterrupted reference.
assert resumed == reference, (resumed, reference)

# Stated directly as well, so a failure says which way it broke: the resumed
# totals must exceed what was on disk at the kill, i.e. they were carried, not
# recounted from zero.
assert (
    resumed["n_value_nan_lane_steps"] >= at_kill["n_value_nan_lane_steps"]
), (resumed, at_kill)
assert (
    resumed["n_grad_nan_lane_steps"] >= at_kill["n_grad_nan_lane_steps"]
), (resumed, at_kill)
assert resumed["total_steps"] > at_kill["total_steps"], (resumed, at_kill)

print(
    "MultiStart resume: value-NaN / gradient-NaN lane-step counters accumulate "
    "across a killed mid-run search, matching an uninterrupted run exactly."
)
