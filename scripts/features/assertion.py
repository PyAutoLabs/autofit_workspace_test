"""
Searches: DynestyStatic
=======================

This example illustrates how to use the nested sampling algorithm DynestyStatic.

Information about Dynesty can be found at the following links:

 - https://github.com/joshspeagle/dynesty
 - https://dynesty.readthedocs.io/en/latest/
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
import numpy as np
from os import path

import autofit as af

"""
__Data__

This example fits a single 1D Gaussian, we therefore load and plot data containing one Gaussian.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
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

plt.errorbar(
    x=range(data.shape[0]),
    y=data,
    yerr=noise_map,
    color="k",
    ecolor="k",
    elinewidth=1,
    capsize=2,
)
plt.show()
plt.close()

# Assertions broken by JAX development and need fixing one day.

# """
# __Model + Analysis__
#
# We create the model and analysis, which in this example is a single `Gaussian` and therefore has dimensionality N=3.
# """
# gaussian_0 = af.Model(af.ex.Gaussian)
# gaussian_1 = af.Model(af.ex.Gaussian)
#
# gaussian_0.add_assertion(gaussian_0.centre > gaussian_1.centre)
#
# model = af.Collection(gaussian_0=gaussian_0, gaussian_1=gaussian_1)
#
# # model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
# # model.gaussian.normalization = af.UniformPrior(lower_limit=1e-2, upper_limit=1e2)
# # model.gaussian.sigma = af.UniformPrior(lower_limit=1e-2, upper_limit=1e2)
# #
# # model.add_assertion(
# #     model.gaussian.normalization > model.gaussian.sigma
# # )
#
# analysis = af.ex.Analysis(data=data, noise_map=noise_map)
#
# """
# __Search__
#
# We now create and run the `DynestyStatic` object which acts as our non-linear search.
#
# We manually specify all of the Dynesty settings, descriptions of which are provided at the following webpage:
#
#  https://dynesty.readthedocs.io/en/latest/api.html
#  https://dynesty.readthedocs.io/en/latest/api.html#module-dynesty.nestedsamplers
# """
# search = af.DynestyStatic(
#     path_prefix="searches",
#     name="DynestyStatic",
#     nlive=50,
#     bound="multi",
#     sample="auto",
#     bootstrap=None,
#     enlarge=None,
#     update_interval=None,
#     walks=25,
#     facc=0.5,
#     slices=5,
#     fmove=0.9,
#     max_move=100,
#     iterations_per_full_update=10000,
#     number_of_cores=1,
#     maxcall=10000,
#     maxiter=10000,
# )
#
# result = search.fit(model=model, analysis=analysis)
#
# """
# By iterating over samples like this an assertion error is often raised.
#
# We also explicitly check that the model instance has the correct attributes.
# """
# for sample in result.samples.sample_list:
#     instance = sample.instance_for_model(model=result.samples.model)
#
#     assert instance.gaussian_0.centre > instance.gaussian_1.centre
