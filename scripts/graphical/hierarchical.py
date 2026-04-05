"""
Tutorial 2: Graphical Models
============================

In the previous tutorial, we fitted a dataset containing 3 noisy 1D Gaussian which had a shared `centre` value. We
attempted to estimate the `centre` by fitting each dataset individually and combining the value of the `centre`
inferred by each fit into an overall estimate, using a joint PDF.

We concluded that this estimate of the centre was suboptimal for a number of reasons and that  our model should instead
fit the global `centre` to all 3 datasets simultaneously. In this tutorial we will do this by composing a graphical
model.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
from os import path

"""
__Dataset__

For each dataset we now set up the correct path and load it. Whereas in the previous tutorial we fitted each dataset 
one-by-one, in this tutorial we instead store each dataset in a list so that we can set up a single model-fit that 
fits the 3 datasets simultaneously.

In this tutorial we explicitly write code for loading and storing each dataset as their own Python variable (e.g. 
data_0, data_1, data_2, etc.). We do not use a for loop or a list to do this (like we did in previous tutorials), even 
though this would be syntactically cleaner code. This is to make the API for setting up a graphical model in this 
tutorial clear and explicit; in the next tutorial we will introduce  the **PyAutoFit** API for setting up a graphical 
model for large datasets concisely.
"""
total_datasets = 2

dataset_path = path.join("dataset", "example_1d")

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

dataset_name_list = []
data_list = []
noise_map_list = []

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

For each dataset we now create a corresponding `Analysis` class. 
"""
analysis_list = []

for data, noise_map in zip(data_list, noise_map_list):
    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    analysis_list.append(analysis)

"""
__Model Individual Factors__

We first set up a model for each `Gaussian` which is individually fitted to each 1D dataset, which forms the
factors on the factor graph we compose. 

This uses a nearly identical for loop to the previous tutorial, however a shared `centre` is no longer used and each 
`Gaussian` is given its own prior for the `centre`. 

We will see next how this `centre` is used to construct the hierachical model.
"""
model_list = []

for model_index in range(len(data_list)):
    gaussian = af.Model(af.ex.Gaussian)

    gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
    gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)

    model = af.Collection(gaussian=gaussian)

    model_list.append(model)

"""
__Analysis Factors__

Above, we composed a model consisting of three `Gaussian`'s with a shared `centre` prior. We also loaded three datasets
which we intend to fit with each of these `Gaussians`, setting up each in an `Analysis` class that defines how the 
model is used to fit the data.

We now simply pair each model-component to each `Analysis` class, so that:

- `gaussian` fits `data_0` via `analysis_0`.
- `gaussian_1` fits `data_1` via `analysis_1`.
- `gaussian_2` fits `data_2` via `analysis_2`.

The point where a `Model` and `Analysis` class meet is called an `AnalysisFactor`. 

This term is used to denote that we are composing a graphical model, which is commonly termed a 'factor graph'. A 
factor defines a node on this graph where we have some data, a model, and we fit the two together. The 'links' between 
these different nodes then define the global model we are fitting.
"""
analysis_factor_list = []

for model, analysis in zip(model_list, analysis_list):
    analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)

    analysis_factor_list.append(analysis_factor)


"""
__Model__

We now compose the hierarchical model that we fit, using the individual Gaussian model components created above.

We first create a `HierarchicalFactor`, which represents the parent Gaussian distribution from which we will assume 
that the `centre` of each individual `Gaussian` dataset is drawn. 

For this parent `Gaussian`, we have to place priors on its `mean` and `sigma`, given that they are parameters in our
model we are ultimately fitting for.
"""

# hierarchical_factor = af.HierarchicalFactor(
#     af.GaussianPrior,
#     mean=af.TruncatedGaussianPrior(mean=50.0, sigma=10, lower_limit=0.0, upper_limit=100.0),
#     sigma=af.TruncatedGaussianPrior(mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=100.0),
# )

hierarchical_factor = af.HierarchicalFactor(
    af.GaussianPrior,
    mean=af.UniformPrior(lower_limit=0.0, upper_limit=100.0),
    sigma=af.UniformPrior(lower_limit=0.0, upper_limit=100.0),
)

"""
We now add each of the individual model `Gaussian`'s `centre` parameters to the `hierarchical_factor`.

This composes the hierarchical model whereby the individual `centre` of every `Gaussian` in our dataset is now assumed 
to be drawn from a shared parent distribution. It is the `mean` and `sigma` of this distribution we are hoping to 
estimate.
"""
for model in model_list:
    hierarchical_factor.add_drawn_variable(model.gaussian.centre)

"""
__Factor Graph__

We now create the factor graph for this model, using the list of `AnalysisFactor`'s and the hierarchical factor.

Note that in previous tutorials, when we created the `FactorGraphModel` we only passed the list of `AnalysisFactor`'s,
which contained the necessary information on the model create the factor graph that was fitted. The `AnalysisFactor`'s
were created before we composed the `HierachicalFactor` and we pass it separately when composing the factor graph.
"""

factor_graph = af.FactorGraphModel(*analysis_factor_list, hierarchical_factor)

"""
__Search__

We can now create a non-linear search and used it to the fit the factor graph, using its `global_prior_model` property.
"""
search = af.DynestyStatic(
    path_prefix=path.join("graphical"),
    name="hierarchical",
    sample="rwalk",
    maxcall=1000,
    maxiter=1000,
)

result = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
__Result__

We can now inspect the inferred value of `centre`, and compare this to the value we estimated in the previous tutorial
via a joint PDF. 

If the graphical model was successful, we should see the errors dramatically reduce!

(The errors of the joint PDF below is what was estimated for a run on my PC, yours may be slightly different!)
"""
print(f"Value of centre via joint PDF = {50.0}")
print(f"Error on centre via joint PDF (2 sigma) = {5.0}")

instance = result.samples.median_pdf()
print(f"Value of centre via graphical model = {instance[0].gaussian.centre}")

"""
__Wrap Up__

As expected, using a graphical model allows us to infer a more precise and accurate model. Unlike the previous 
tutorial, our model-fit: 

1) Fully exploits the information we know about the global model, for example that the centre of every Gaussian in every 
dataset is aligned. Now, the fit of the Gaussian in dataset 1 informs the fits in datasets 2 and 3, and visa versa.

2) Infers a PDF on the global centre that fully accounts for the degeneracies between the models fitted to different 
datasets. This is not the case when we combine the PDFs of each individual fit.

3) Has a well defined prior on the global centre, instead of 3 independent priors on the centre of each dataset.

The tools introduced in this tutorial form the basis of building graphical models of arbitrary complexity, which may
consist of many hundreds of individual model components with thousands of parameters that are fitted to extremely large
datasets. 

However, there is a clear challenge scaling the graphical modeling framework up in this way: model complexity. As the 
model becomes more complex, an inadequate sampling of parameter space can lead one to infer local maxima. Furthermore,
one will soon hit computational limits on how many datasets can feasibly be fitted simultaneously. 

Therefore, the next tutorial introduces expectation propagation, a framework that inspects the factor graph of a 
graphical model and partitions the model-fit down into many separate fits on each graph node. When a fit is complete, 
it passes the information learned about the model to neighboring nodes. Thus, graphs comprised of hundreds of model
components (and thousands of parameters) can be fitted as many bite-sized model fits, where the model fitted at each
node consists of just tens of parameters. This makes graphical models scalable to largest datasets and most complex models!

Finish.
"""
