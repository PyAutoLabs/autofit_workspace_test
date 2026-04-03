"""
Feature: Database
=================

Tests that the results of a grid search of searches can be loaded from hard-disk via a database built via a scrape.
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
___Session__

To output results directly to the database, we start a session, which includes the name of the database `.sqlite` file
where results are stored.
"""
session = None

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

name = "grid_search"

search = af.DynestyStatic(
    path_prefix=path.join("database", "scrape", name),
    name="extra_search",
    unique_tag=dataset_name,
    session=session,
    maxcall=1000,
    maxiter=1000,
)

search.fit(model=model, analysis=analysis)

"""
Result sare written directly to the `database.sqlite` file omitted hard-disc output entirely, which
can be important for performing large model-fitting tasks on high performance computing facilities where there
may be limits on the number of files allowed. The commented out code below shows how one would perform
direct output to the `.sqlite` file. 
"""
search = af.DynestyStatic(
    name=name,
    path_prefix=path.join("database", "scrape", name),
    number_of_cores=1,
    unique_tag=dataset_name,
    session=session,
    maxcall=1000,
    maxiter=1000,
    force_x1_cpu=True,  # ensures parallelizing over grid search works.
)

grid_search = af.SearchGridSearch(search=search, number_of_steps=2, number_of_cores=1)

grid_search_result = grid_search.fit(
    model=model,
    analysis=analysis,
    grid_priors=[model.gaussian.centre],
)

"""
Scrape directory to create .sqlite file.
"""
import os
from autofit.database.aggregator import Aggregator

database_file = "database_directory_grid_search.sqlite"

try:
    os.remove(path.join("output", database_file))
except FileNotFoundError:
    pass

agg = Aggregator.from_database(database_file, completed_only=False)

agg.add_directory(
    directory=path.join("output", "database", "scrape", name, dataset_name)
)

assert len(agg) > 0

"""
Make sure database + agg can be used.
"""
print("\n\n***********************")
print("****RESULTS TESTING****")
print("***********************\n")

for samples in agg.values("samples"):
    print(samples.parameter_lists[0])

mp_instances = [samps.median_pdf() for samps in agg.values("samples")]
print(mp_instances)

"""
__Queries__
"""
print("\n\n***********************")
print("****QUERIES TESTING****")
print("***********************\n")

gaussian = agg.model.gaussian
agg_query = agg.query(gaussian == af.ex.Gaussian)
print("Total queries for correct model = ", len(agg_query))

name = agg.search.name
agg_query = agg.query(name == "database_grid_search")
print("Total Samples Objects via `name` model query = ", len(agg_query), "\n")

"""
__Grid Search Results__
"""
print("\n\n***********************")
print("**GRID RESULTS TESTING**")
print("***********************\n\n")

print(
    "\n****Total aggregator via `grid_searches` query = ",
    len(agg.grid_searches()),
    "****\n",
)
unique_tag = agg.grid_searches().search.unique_tag
agg_qrid = agg.grid_searches().query(unique_tag == "gaussian_x1")

print(
    "****Total aggregator via `grid_searches` & unique tag query = ",
    len(agg.grid_searches()),
    "****\n",
)

print(
    f"****Extract individual parameters on grid via (attribute_grid) {agg.grid_searches()[0]['result'].attribute_grid('gaussian.centre')}****\n\n"
)

"""
The `GridSearchResult` is accessible via the database.
"""
print(f"****Best result (list(agg.grid_searches())[0]['result'].best_samples)****\n\n")
print(f"{list(agg.grid_searches())[0]['result'].best_samples}\n")

assert list(agg.grid_searches())[0]["result"].best_samples.log_evidence > -1e8

print(
    f"****Grid Log Evidences (list(agg.grid_searches())[0]['result'].log_evidences().native)****\n\n"
)
print(f"{list(agg.grid_searches())[0]['result'].log_evidences().native}\n")

assert list(agg.grid_searches())[0]["result"].log_evidences().native[0] > -1e8

"""
From the GridSearch, get an aggregator which contains only the maximum log likelihood model. E.g. if the 10th out of the 
16 cells was the best fit:
"""
print("\n\n****MAX LH AGGREGATOR VIA GRID****\n\n")
print(
    f"Max LH Gaussian sigma (agg.grid_searches().best_fits().values('instance')[0]) \n"
)
print(f"{agg.grid_searches().best_fits().values('instance')[0].gaussian.centre}")

assert agg.grid_searches().best_fits().values("instance")[0].gaussian.centre > 0.0

print(f"Max LH samples (agg.grid_searches().best_fits().values('samples')[0])\n")
print({agg.grid_searches().best_fits().values("samples")[0]})

assert (
    agg.grid_searches().best_fits().values("samples")[0].log_likelihood_list[-1] > -1e8
)

"""
From the GridSearch, get an aggregator for any of the grid cells.
"""
print("\n\n****AGGREGATOR FROM INPUT GRID CELL****\n\n")

cell_aggregator = agg.grid_searches().cell_number(1)
print(f"Cell Aggregator (agg.grid_searches().cell_number(1)) {cell_aggregator}")
print("Size of Agg cell = ", len(cell_aggregator), "\n")

assert cell_aggregator.values("instance")[0].gaussian.centre > 0.0
