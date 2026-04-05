"""
Tutorial 4: Expectation Propagation
===================================

In the previous tutorials, we fitted graphical models to dataset comprising many noisy 1D Gaussians. These had a shared
and global value of their `centre`, or assumed their centres were hierarchically drawn from a parent Gaussian
distribution. This provides the basis of composing and fitting complex graphical models to large datasets.

We concluded by discussing that there is a ceiling scaling these graphical models up to extremely large datasets. One
would soon find that the parameter space is too complex to sample, and computational limits would ultimately cap how
many datasets one could feasibly fit.

This tutorial introduces expectation propagation (EP), the solution to this problem, which inspects a factor graph
and partitions the model-fit into many simpler fits of sub-components of the graph to individual datasets. This
overcomes the challenge of model complexity, and mitigates computational restrictions that may occur if one tries to
fit every dataset simultaneously.

This tutorial fits a global model with a shared parameter and does not use a hierarchical model. The optional tutorial
`tutorial_optional_hierarchical_ep` shows an example fit of a hierarchical model with EP.
"""

import os

cwd = os.getcwd()

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path

import autofit as af

"""
__Example Source Code (`af.ex`)__

The **PyAutoFit** source code has the following example objects (accessed via `af.ex`) used in this tutorial:

 - `Analysis`: an analysis object which fits noisy 1D datasets, including `log_likelihood_function` and 
 `visualize` functions.

 - `Gaussian`: a model component representing a 1D Gaussian profile.

 - `plot_profile_1d`: a function for plotting 1D profile datasets including their noise.

These are functionally identical to the `Analysis`, `Gaussian` and `plot_profile_1d` objects and functions you have seen 
and used elsewhere throughout the workspace.

__Dataset__

For each dataset we now set up the correct path and load it. 

We first fit the 1D Gaussians which all share the same centre, thus not requiring a hierarchical model. 

An example for fitting the hierarchical model with EP is given at the end of this tutorial.
"""
total_datasets = 2

dataset_name_list = []
data_list = []
noise_map_list = []

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
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

for dataset_index in range(total_datasets):
    dataset_name = f"dataset_{dataset_index}"

    dataset_path = path.join(
        "dataset", "example_1d", "gaussian_x1__low_snr", dataset_name
    )

    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    dataset_name_list.append(dataset_name)
    data_list.append(data)
    noise_map_list.append(noise_map)

"""
__Analysis__

For each dataset we now create a corresponding `Analysis` class, like in the previous tutorial.
"""
analysis_list = []

for data, noise_map in zip(data_list, noise_map_list):
    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    analysis_list.append(analysis)

"""
__Model__

We now compose the graphical model that we fit, using the `Model` and `Collection` objects you are now familiar with.

We will assume all Gaussians share the same centre, therefore we set up a shared prior for `centre`.
"""
centre_shared_prior = af.GaussianPrior(mean=50.0, sigma=30.0)
# centre_shared_prior = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)

model_list = []

for model_index in range(len(data_list)):
    gaussian = af.Model(af.ex.Gaussian)

    gaussian.centre = centre_shared_prior  # This prior is used by all 3 Gaussians!

    gaussian.normalization = af.TruncatedGaussianPrior(
        mean=3.0, sigma=5.0, lower_limit=0.0
    )
    gaussian.sigma = af.TruncatedGaussianPrior(mean=10.0, sigma=10.0, lower_limit=0.0)
    #
    # gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
    # gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)

    model_list.append(gaussian)

"""
__Analysis Factors__

Now we have our `Analysis` classes and graphical model, we can compose our `AnalysisFactor`'s.

However, unlike the previous tutorials, each `AnalysisFactor` is now assigned its own `search`. This is because the EP 
framework performs a model-fit to each node on the factor graph (e.g. each `AnalysisFactor`). Therefore, each node 
requires its own non-linear search, and in this tutorial we use `dynesty`. For complex graphs consisting of many 
nodes, one could easily use different searches for different nodes on the factor graph.

Each `AnalysisFactor` is also given a `name`, corresponding to the name of the dataset it fits. These names are used
to name the folders containing the results in the output directory.
"""
search = af.DynestyStatic(
    nlive=300,
    dlogz=1e-4,
    sample="rwalk",
    walks=10,
    maxcall=1000,
    maxiter=1000,
)

analysis_factor_list = []

dataset_index = 0

for model, analysis in zip(model_list, analysis_list):
    dataset_name = f"dataset_{dataset_index}"
    dataset_index += 1

    analysis_factor = af.AnalysisFactor(
        prior_model=model, analysis=analysis, optimiser=search, name=dataset_name
    )

    analysis_factor_list.append(analysis_factor)

"""
__Factor Graph__

We combine our `AnalysisFactors` into one, to compose the factor graph.
"""
factor_graph = af.FactorGraphModel(*analysis_factor_list)

"""
__Expectation Propagation__

In the previous tutorials, we used the `global_prior_model` of the `factor_graph` to fit the global model. In this 
tutorial, we instead fit the `factor_graph` using the EP framework, which fits the graphical model composed in this 
tutorial as follows:

1) Go to the first node on the factor graph (e.g. `analysis_factor_list[0]`) and fit its model to its dataset. This is 
simply a fit of the `Gaussian` model to the first 1D Gaussian dataset, the model-fit we are used to performing by now.

2) Once the model-fit is complete, inspect the model for parameters that are shared with other nodes on the factor
graph. In this example, the `centre` of the `Gaussian` fitted to the first dataset is global, and therefore connects
to two other nodes on the factor graph (the `AnalysisFactor`'s) of the second and first `Gaussian` datasets.

3) The EP framework now creates a 'message' that is to be passed to the connecting nodes on the factor graph. This
message informs them of the results of the model-fit, so they can update their priors on the `Gaussian`'s centre 
accordingly and, more importantly, update their posterior inference and therefore estimate of the global centre.

For example, the model fitted to the first Gaussian dataset includes the global centre. Therefore, after the model is 
fitted, the EP framework creates a 'message' informing the factor graph about its inference on that Gaussians's centre,
thereby updating our overall inference on this shared parameter. This is termed 'message passing'.

__Cyclic Fitting__

After every `AnalysisFactor` has been fitted (e.g. all 3 datasets in this example), we have a new estimate of the 
shared parameter `centre`. This updates our priors on the shared parameter `centre`, which needs to be reflected in 
each model-fit we perform on each `AnalysisFactor`. 

The EP framework therefore performs a second iteration of model-fits. It again cycles through each `AnalysisFactor` 
and refits the model, using updated priors on shared parameters like the `centre`. At the end of each fit, we again 
create messages that update our knowledge about other parameters on the graph.

This process is repeated multiple times, until a convergence criteria is met whereby continued cycles are expected to
produce the same estimate of the shared parameter `centre`. 

When we fit the factor graph a `name` is passed, which determines the folder all results of the factor graph are
stored in.
"""
laplace = af.LaplaceOptimiser()

paths = af.DirectoryPaths(
    name=path.join(
        "howtofit", "chapter_graphical_models", "tutorial_5_expectation_propagation"
    )
)

factor_graph_result = factor_graph.optimise(
    optimiser=laplace, paths=paths, ep_history=af.EPHistory(kl_tol=0.05)
)


"""
__Output__

The results of the factor graph, using the EP framework and message passing, are contained in the folder 
`output/howtofit/chapter_graphical_models/tutorial_5_expectation_propagation`. 

The following folders and files are worth of note:

 - `graph.info`: this provides an overall summary of the graphical model that is fitted, including every parameter, 
 how parameters are shared across `AnalysisFactor`'s and the priors associated to each individual parameter.
 
 - The 3 folders titled `gaussian_x1_#__low_snr` correspond to the three `AnalysisFactor`'s and therefore signify 
 repeated non-linear searches that are performed to fit each dataset.
 
 - Inside each of these folders are `optimization_#` folders, corresponding to each model-fit performed over cycles of
 the EP fit. A careful inspection of the `model.info` files inside each folder reveals how the priors are updated
 over each cycle, whereas the `model.results` file should indicate the improved estimate of model parameters over each
 cycle.
"""

print(factor_graph_result)

print(factor_graph_result.updated_ep_mean_field.mean_field)

"""
__Output__

The MeanField object representing the posterior.
"""
print(factor_graph_result.updated_ep_mean_field.mean_field)
print()

print(factor_graph_result.updated_ep_mean_field.mean_field.variables)
print()

"""
The logpdf of the posterior at the point specified by the dictionary values
"""
# factor_graph_result.updated_ep_mean_field.mean_field(values=None)
print()

"""
A dictionary of the mean with variables as keys.
"""
print(factor_graph_result.updated_ep_mean_field.mean_field.mean)
print()

"""
A dictionary of the variance with variables as keys.
"""
print(factor_graph_result.updated_ep_mean_field.mean_field.variance)
print()

"""
A dictionary of the s.d./variance**0.5 with variables as keys.
"""
print(factor_graph_result.updated_ep_mean_field.mean_field.scale)
print()

"""
self.updated_ep_mean_field.mean_field[v: Variable] gives the Message/approximation of the posterior for an individual variable of the model
"""
# factor_graph_result.updated_ep_mean_field.mean_field["help"]

"""
Finish.
"""
