"""
Feature: Database
=================

Tests that general results can be loaded from hard-disk via a database built via a database that is written to
during a fit.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af

import os
from os import path
import numpy as np

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
___Session__

To output results directly to the database, we start a session, which includes the name of the database `.sqlite` file
where results are stored.
"""
database_file = "database_session_general.sqlite"
session = af.db.open_database(database_file)

"""
The code below loads the dataset and sets up the Analysis class.
"""
dataset_path = path.join("dataset", "example_1d", dataset_name)

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

"""
Default example Analysis does not output a .pickle file and does not test pickle loading.

We extend the Analysis class to output the data as a pickle file, which we test can be loaded below
"""


class Analysis(af.ex.Analysis):
    def save_attributes(self, paths: af.DirectoryPaths):
        super().save_attributes(paths=paths)
        paths.save_object(name="data_pickled", obj=self.data)


analysis = Analysis(data=data, noise_map=noise_map)

"""
Resultsare written directly to the `database.sqlite` file omitted hard-disc output entirely, which
can be important for performing large model-fitting tasks on high performance computing facilities where there
may be limits on the number of files allowed. The commented out code below shows how one would perform
direct output to the `.sqlite` file. 
"""
name = "general"

search = af.DynestyStatic(
    name=name,
    path_prefix=path.join("database", "session"),
    number_of_cores=1,
    unique_tag=dataset_name,
    session=session,
    maxcall=100,
    maxiter=100,
)

result = search.fit(model=model, analysis=analysis, info={"hi": "there"})

"""
__Database__
"""
agg = af.Aggregator.from_database(path.join(database_file))

assert len(agg) > 0

"""
__Samples + Results__

Make sure database + agg can be used.
"""
for samples in agg.values("samples"):
    print(samples.parameter_lists[0])

mp_instances = [samps.median_pdf() for samps in agg.values("samples")]
print(mp_instances)

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

# Session support dropped for now.

# """
# __Files__
#
# Check that all other files stored in database (e.g. model, search) can be loaded and used.
# """
#
# for model in agg.values("model"):
#     print(f"\n****Model Info (model)****\n\n{model.info}")
#     assert model.info[0] == "T"
#
# for search in agg.values("search"):
#     print(f"\n****Search (search)****\n\n{search}")
#     assert search.paths.name == "general"
#     assert path.join("database", "session", dataset_name) in str(
#         search.paths.output_path
#     )
#
# for latent_samples in agg.values("latent_samples"):
#     print(latent_samples.parameter_lists[0])
#     assert latent_samples.parameter_lists[0][0] > 0.0
#
# for samples in agg.values("samples"):
#     instance = samples.max_log_likelihood()
#     print(f"\n****Max Log Likelihood (samples)****\n\n{instance}")
#     assert instance.gaussian.centre > 0.0
#
# for info in agg.values("info"):
#     print(f"\n****Info****\n\n{info}")
#     assert info["hi"] == "there"
#
# for data in agg.values("dataset.data"):
#     print(f"\n****Data (dataset.data)****\n\n{data}")
#     assert data[0] > -1.0e8
#
# for noise_map in agg.values("dataset.noise_map"):
#     print(f"\n****Noise Map (dataset.noise_map)****\n\n{noise_map}")
#     assert noise_map[0] > 0.0
#
# for data in agg.values("data_pickled"):
#     print(f"\n****Data (data_pickled)****\n\n{data}")
#     assert data[0] > -1.0e8

# for covariance in agg.values("covariance"):
#     print(f"\n****Covariance (covariance)****\n\n{covariance}")
#     assert covariance is not None
