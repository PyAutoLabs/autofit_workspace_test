"""
__Simulators__

These scripts simulate the 1D Gaussian datasets used to demonstrate model-fitting.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import util
from os import path

import autofit as af

"""
__Gaussian x1__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
gaussian = af.ex.Gaussian(centre=50.0, normalization=25.0, sigma=10.0)
util.simulate_dataset_1d_via_gaussian_from(gaussian=gaussian, dataset_path=dataset_path)

"""
__Gaussian x1 Low SNR (Multiple Datasets)__

Generate multiple low signal-to-noise ratio datasets used by the graphical modeling scripts
(e.g. graphical/ep.py, graphical/hierarchical.py, graphical/simultaneous.py).
"""
for i in range(2):
    dataset_path = path.join("dataset", "example_1d", "gaussian_x1__low_snr", f"dataset_{i}")
    gaussian = af.ex.Gaussian(centre=50.0, normalization=25.0, sigma=10.0)
    util.simulate_dataset_1d_via_gaussian_from(gaussian=gaussian, dataset_path=dataset_path)

"""
Finish.
"""
