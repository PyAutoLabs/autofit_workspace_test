"""
Integration Test: EP vs Joint-Fit Parity (Shared Centre)
=========================================================

This script fits the same factor graph -- 3 noisy 1D Gaussian datasets whose `Gaussian` models share a
single global `centre` -- in two independent ways and checks the two posteriors on the shared centre
agree:

 (a) a JOINT fit of the `global_prior_model` of the factor graph, sampling all parameters
     simultaneously with a small `DynestyStatic` search (the pattern of `graphical/simultaneous.py`);

 (b) an EXPECTATION PROPAGATION (EP) fit of the factor graph via `factor_graph.optimise` with
     `af.LaplaceOptimiser()` (the pattern of `graphical/ep.py`).

Both fits use identical models and priors, so their posteriors on the shared `centre` should agree
within their combined uncertainties, and both should recover the true simulated value of 50.0.

__Dataset__

The first two datasets are the `gaussian_x1__low_snr` datasets on disk (true centre 50.0). Only two of
these exist in the workspace, so the third dataset is simulated in-memory with a fixed seed, using the
same generator settings as `scripts/simulators/util.py` (centre=50.0, normalization=25.0, sigma=10.0,
signal-to-noise ratio 25). It is not written to disk so the tracked `dataset` folder is unchanged.
"""

import os

cwd = os.getcwd()

import numpy as np
from os import path

import autofit as af

"""
__Dataset__
"""
total_datasets_on_disk = 2

data_list = []
noise_map_list = []

"""
__Dataset Auto-Simulation__

If the on-disk datasets do not already exist on your system, they are created by running the
corresponding simulator script.
"""
if not path.exists(
    path.join("dataset", "example_1d", "gaussian_x1__low_snr", "dataset_0")
):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/simulators/simulators.py"],
        check=True,
    )

for dataset_index in range(total_datasets_on_disk):
    dataset_name = f"dataset_{dataset_index}"

    dataset_path = path.join(
        "dataset", "example_1d", "gaussian_x1__low_snr", dataset_name
    )

    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    data_list.append(data)
    noise_map_list.append(noise_map)

"""
__Third Dataset (In-Memory)__

Simulated with the same settings as `scripts/simulators/util.py` but with a fixed seed, and not written
to disk.
"""
rng = np.random.default_rng(seed=1)

pixels = 100
signal_to_noise_ratio = 25.0
xvalues = np.arange(pixels)

gaussian_true = af.ex.Gaussian(centre=50.0, normalization=25.0, sigma=10.0)
model_data_1d = gaussian_true.model_data_from(xvalues=xvalues)

data_2 = model_data_1d + rng.normal(0.0, 1.0 / signal_to_noise_ratio, pixels)
noise_map_2 = (1.0 / signal_to_noise_ratio) * np.ones(pixels)

data_list.append(data_2)
noise_map_list.append(noise_map_2)

total_datasets = len(data_list)

"""
__Analysis__
"""
analysis_list = []

for data, noise_map in zip(data_list, noise_map_list):
    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    analysis_list.append(analysis)

"""
__Model__

All Gaussians share the same `centre`, set up as a single shared `GaussianPrior` assigned to every
model's `centre` (the pattern of `graphical/ep.py`).

The `normalization` and `sigma` priors are plain `GaussianPrior`s, following the canonical pure-Laplace
declarative EP pattern in `PyAutoFit/test_autofit/graphical/gaussian/test_declarative.py::test_gaussian`.
(The `TruncatedGaussianPrior`s used by `graphical/ep.py` are numerically unstable when the analysis
factors themselves are optimised with `LaplaceOptimiser`; `ep.py` avoids this by assigning each
`AnalysisFactor` its own `DynestyStatic` optimiser.)
"""
centre_shared_prior = af.GaussianPrior(mean=50.0, sigma=30.0)

model_list = []

for model_index in range(total_datasets):
    gaussian = af.Model(af.ex.Gaussian)

    gaussian.centre = centre_shared_prior  # This prior is used by all 3 Gaussians!

    gaussian.normalization = af.GaussianPrior(mean=25.0, sigma=10.0)
    gaussian.sigma = af.GaussianPrior(mean=10.0, sigma=10.0)

    model_list.append(gaussian)

"""
__Analysis Factors + Factor Graph__
"""
analysis_factor_list = []

for dataset_index, (model, analysis) in enumerate(zip(model_list, analysis_list)):
    analysis_factor = af.AnalysisFactor(
        prior_model=model, analysis=analysis, name=f"dataset_{dataset_index}"
    )

    analysis_factor_list.append(analysis_factor)

factor_graph = af.FactorGraphModel(*analysis_factor_list)

"""
__(a) Joint Fit__

Fit the `global_prior_model` of the factor graph with a small `DynestyStatic` budget, following
`graphical/simultaneous.py`. Parity tolerances below are loose, so the budget is kept small for speed.

Any previous output of this search is removed first: resuming a completed run loads a samples summary
whose `errors_at_sigma` are zero, which would invalidate the sigma-based parity assertions below.
"""
import shutil

shutil.rmtree(path.join("output", "graphical", "ep_parity_joint"), ignore_errors=True)

search = af.DynestyStatic(
    path_prefix=path.join("graphical"),
    name="ep_parity_joint",
    nlive=50,
    sample="rwalk",
    walks=10,
    maxcall=3000,
    maxiter=3000,
)

joint_result = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

joint_instance = joint_result.samples.median_pdf()
joint_median = joint_instance[0].centre

joint_error_instance = joint_result.samples.errors_at_sigma(sigma=1.0)
joint_sigma = float(np.mean(joint_error_instance[0].centre))

"""
__(b) EP Fit__

Fit the same factor graph with the EP framework via `factor_graph.optimise`, following
`graphical/ep.py`.

`max_steps` (forwarded by `optimise(**kwargs)` to `EPOptimiser.run`) is capped at 3. With the default
undamped updater (`SimplerUpdater(delta=1.0)`), pure-Laplace EP on this graph does not trigger the
`kl_tol` termination and repeated cycles progressively over-count shared-variable information,
collapsing every posterior sigma towards zero (~1e-15 after the default 100 steps). The mean field is
stable and sensible after 2-3 sweeps.
"""
laplace = af.LaplaceOptimiser()

paths = af.DirectoryPaths(name=path.join("graphical", "ep_parity"))

factor_graph_result = factor_graph.optimise(
    optimiser=laplace,
    paths=paths,
    ep_history=af.EPHistory(kl_tol=0.05),
    max_steps=3,
)

mean_field = factor_graph_result.updated_ep_mean_field.mean_field

ep_mean = float(mean_field.mean[centre_shared_prior])
ep_sigma = float(mean_field.scale[centre_shared_prior])

"""
__Posteriors__
"""
true_centre = 50.0

print()
print(f"True centre = {true_centre}")
print(
    f"Joint (Dynesty) centre: median = {joint_median}, sigma (1D, 1 sigma) = {joint_sigma}"
)
print(f"EP (Laplace) centre:    mean = {ep_mean}, sigma = {ep_sigma}")
print()

"""
__Assertions__

1) The EP posterior mean of the shared centre is within 3 (EP) sigma of the truth.
2) The joint posterior median of the shared centre is within 3 (joint) sigma of the truth.
3) The EP and joint posteriors agree within 3 combined sigma.
"""
assert np.isfinite(ep_mean) and np.isfinite(ep_sigma) and ep_sigma > 0.0
assert np.isfinite(joint_median) and np.isfinite(joint_sigma) and joint_sigma > 0.0

assert (
    abs(ep_mean - true_centre) < 3.0 * ep_sigma
), f"EP centre mean ({ep_mean}) not within 3 sigma ({3.0 * ep_sigma}) of truth ({true_centre})"
assert abs(joint_median - true_centre) < 3.0 * joint_sigma, (
    f"joint centre median ({joint_median}) not within 3 sigma ({3.0 * joint_sigma}) "
    f"of truth ({true_centre})"
)

combined_sigma = (ep_sigma**2 + joint_sigma**2) ** 0.5

assert abs(ep_mean - joint_median) < 3.0 * combined_sigma, (
    f"EP mean ({ep_mean}) and joint median ({joint_median}) disagree by more than "
    f"3 combined sigma ({3.0 * combined_sigma})"
)

print("ep_parity.py: PASS")
