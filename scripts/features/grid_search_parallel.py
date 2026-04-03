"""
Feature: Database
=================

The default behaviour of **PyAutoFit** is for model-fitting results to be output to hard-disc in folders, which are
straight forward to navigate and manually check. For small model-fitting tasks this is sufficient, however many users
have a need to perform many model fits to very large datasets, making manual inspection of results time consuming.

PyAutoFit's database feature outputs all model-fitting results as a
sqlite3 (https://docs.python.org/3/library/sqlite3.html) relational database, such that all results
can be efficiently loaded into a Jupyter notebook or Python script for inspection, analysis and interpretation. This
database supports advanced querying, so that specific model-fits (e.g., which fit a certain model or dataset) can be
loaded.

This example extends our example of fitting a 1D `Gaussian` profile and fits 3 independent datasets each containing a
1D Gaussian. The results will be written to a `.sqlite` database, which we will load to demonstrate the database.

A full description of PyAutoFit's database tools is provided in the database chapter of the `HowToFit` lectures.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af

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
model = af.Model(af.ex.Gaussian)

model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.UniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

"""
___Session__

To output results directly to the database, we start a session, which includes the name of the database `.sqlite` file
where results are stored.
"""
session = None

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

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

"""
Results are written directly to the `database.sqlite` file omitted hard-disc output entirely, which
can be important for performing large model-fitting tasks on high performance computing facilities where there
may be limits on the number of files allowed. The commented out code below shows how one would perform
direct output to the `.sqlite` file. 
"""
search = af.DynestyStatic(
    name="grid_search",
    path_prefix=path.join("parallel"),
    number_of_cores=1,
    unique_tag=dataset_name,
    session=session,
    maxcall=1000,
    maxiter=1000,
    force_x1_cpu=True,  # ensures parallelizing over grid search works.
)

grid_search = af.SearchGridSearch(search=search, number_of_steps=4, number_of_cores=5)

grid_search_result = grid_search.fit(
    model=model, analysis=analysis, grid_priors=[model.centre]
)

model = af.Model(af.ex.Gaussian)

model.centre = grid_search_result.model.centre

search = af.DynestyStatic(
    name="post_grid_search",
    path_prefix=path.join("parallel"),
    number_of_cores=1,
    unique_tag=dataset_name,
    session=session,
    maxcall=1000,
    maxiter=1000,
)

result = search.fit(model=model, analysis=analysis)
