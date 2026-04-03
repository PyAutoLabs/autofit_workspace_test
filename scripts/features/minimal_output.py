"""
Feature: Minimal Output
=======================

The output of a PyAutoFit model-fit can be customized in order to reduce hard-disk space use and clutter in the
output folder.

This script tests functionality when the minimal amount of samples and other files are output, including:

 1) Resuming a fit and loading a completed fit.
 2) Search chaining and prior linking.
 3) Database functionality with files missing.

 __Config__

We begin by pointing to the minimal_output config folder, which has configuration file settings update to produce
minimal output.

The following settings are listed as false in the `output.yaml` file, meaning the corresponding files will not be
outptu (which assertions test for below):

covariance
"""

import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "scripts", "features", "config"))

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
import autofit as af

"""
__Dataset Names__

Load the dataset from hard-disc, set up its `Analysis` class and fit it with a non-linear search. 
"""
dataset_name = "gaussian_x1"

"""
__Model__

Next, we create our model, which again corresponds to a single `Gaussian` with manual priors.
"""
model = af.Collection(gaussian=af.ex.Gaussian)

model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.gaussian.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.gaussian.sigma = af.TruncatedGaussianPrior(
    mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=np.inf
)

"""
The code below loads the dataset and sets up the Analysis class.
"""
dataset_path = path.join("dataset", "example_1d", dataset_name)

data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

name = "simple"

search = af.DynestyStatic(
    name=name,
    path_prefix=path.join("features", "minimal_output"),
    number_of_cores=1,
    unique_tag=dataset_name,
    maxcall=1000,
    maxiter=1000,
)

result = search.fit(model=model, analysis=analysis)

"""
__Completion__

Confirm that a completed run does not raise an exception.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Assertions__

The following files have been disabled via the ? config file.

The assertions below check that the associated files have not been output.

 - The `search_internal` folder.
"""
assert not path.exists(search.paths._files_path / "search_internal")
assert not path.exists(search.paths._files_path / "covariance.csv")

"""
__Database__
"""
from autofit.aggregator.aggregator import Aggregator

agg = Aggregator.from_directory(
    directory=path.join("output", "features", "minimal_output", dataset_name, name),
)

assert len(agg) > 0

"""
__Samples + Results__

Make sure database + agg can be used.
"""
for samples in agg.values("samples"):
    assert samples is None

"""
__Queries__
"""
path_prefix = agg.search.path_prefix
agg_query = agg.query(path_prefix == path.join("database", "session", dataset_name))
print("Total Samples Objects via `path_prefix` model query = ", len(agg_query), "\n")

name = agg.search.name
agg_query = agg.query(name == "general")
print("Total Samples Objects via `name` model query = ", len(agg_query), "\n")

gaussian = agg.model.gaussian
agg_query = agg.query(gaussian == af.ex.Gaussian)
print("Total Samples Objects via `Gaussian` model query = ", len(agg_query), "\n")

gaussian = agg.model.gaussian
agg_query = agg.query(gaussian.sigma > 3.0)
print("Total Samples Objects In Query `gaussian.sigma < 3.0` = ", len(agg_query), "\n")

gaussian = agg.model.gaussian
agg_query = agg.query((gaussian == af.ex.Gaussian) & (gaussian.sigma < 3.0))
print(
    "Total Samples Objects In Query `Gaussian & sigma < 3.0` = ", len(agg_query), "\n"
)

unique_tag = agg.search.unique_tag
agg_query = agg.query(unique_tag == "gaussian_x1_1")

print(agg_query.values("samples"))
print("Total Samples Objects via unique tag Query = ", len(agg_query), "\n")

"""
__Files__

Check that all other files stored in database (e.g. model, search) can be loaded and used.
"""

for model in agg.values("model"):
    print(f"\n****Model Info (model)****\n\n{model.info}")
    assert model.info[0] == "T"

for search in agg.values("search"):
    print(f"\n****Search (search)****\n\n{search}")
    assert search.paths.name == "simple"
    assert path.join("features") in str(search.paths.output_path)

for samples_summary in agg.values("samples_summary"):
    instance = samples_summary.max_log_likelihood()
    print(f"\n****Max Log Likelihood (samples_summary)****\n\n{instance}")
    assert instance.gaussian.centre > 0.0
    print(samples_summary.max_log_likelihood_sample.log_likelihood)

for info in agg.values("info"):
    assert info is None

for data in agg.values("dataset.data"):
    assert data is None

for noise_map in agg.values("dataset.noise_map"):
    assert noise_map is None

for covariance in agg.values("covariance"):
    assert covariance is None
