"""
Feature: Sensitivity Mapping
============================

Bayesian model comparison allows us to take a dataset, fit it with multiple models and use the Bayesian evidence to
quantify which model objectively gives the best-fit following the principles of Occam's Razor.

However, a complex model may not be favoured by model comparison not because it is the 'wrong' model, but simply
because the dataset being fitted is not of a sufficient quality for the more complex model to be favoured. Sensitivity
mapping addresses what quality of data would be needed for the more complex model to be favoured.

In order to do this, sensitivity mapping involves us writing a function that uses the model(s) to simulate a dataset.
We then use this function to simulate many datasets, for many different models, and fit each dataset using the same
model-fitting procedure we used to perform Bayesian model comparison. This allows us to infer how much of a Bayesian
evidence increase we should expect for datasets of varying quality and / or models with different parameters.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af

import matplotlib.pyplot as plt
import numpy as np
from os import path

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
__Data__

Load data of a 1D Gaussian from a .json file in the directory 
`autofit_workspace/dataset/gaussian_x1`.

This 1D data includes a small feature to the right of the central `Gaussian`. This feature is a second `Gaussian` 
centred on pixel 70. 
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
Lets plot the data. 

The feature on pixel 70 is clearly visible.
"""
xvalues = range(data.shape[0])

plt.errorbar(
    x=xvalues, y=data, yerr=noise_map, color="k", ecolor="k", elinewidth=1, capsize=2
)
plt.title("1D Gaussian Data With Feature at pixel 70.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
__Analysis__

Our Analysis class is described in `analysis.py` and is the same used in the `overview/complex` example. 

It fits the data as the sum of the two `Gaussian`'s in the model.
"""
analysis = af.ex.Analysis(data=data, noise_map=noise_map)

"""
__Model Comparison__

Before performing sensitivity mapping, we will quickly perform Bayesian model comparison on this data to get a sense 
for whether the `Gaussian` feature is detectable and how much the Bayesian evidence increases when it is included in
the model.

We therefore fit the data using two models, one where the model is a single `Gaussian` and one where it is 
two `Gaussians`. 

To avoid slow model-fitting and more clearly prounce the results of model comparison, we restrict the centre of 
the`gaussian_feature` to its true centre of 70 and sigma value of 0.5.
"""
model = af.Collection(gaussian_main=af.ex.Gaussian)
model.gaussian_main.centre = 50.0
model.gaussian_main.sigma = 10.0

name = "sensitivity"

search = af.DynestyStatic(
    path_prefix=path.join("database", "directory", name),
    name="single_gaussian",
    nlive=25,
    unique_tag=dataset_name,
    session=session,
    maxcall=1000,
    maxiter=1000,
    iterations_per_full_update=50000,
)

result_single = search.fit(model=model, analysis=analysis)

model = af.Collection(gaussian_main=af.ex.Gaussian, gaussian_feature=af.ex.Gaussian)
model.gaussian_main.centre = 50.0
model.gaussian_main.sigma = 10.0
model.gaussian_feature.centre = 70.0
model.gaussian_feature.sigma = 0.5

search = af.DynestyStatic(
    path_prefix=path.join("database", "directory", name),
    name=("two_gaussians"),
    nlive=25,
    unique_tag=dataset_name,
    session=session,
    maxcall=1000,
    maxiter=1000,
    iterations_per_full_update=50000,
)

result_multiple = search.fit(model=model, analysis=analysis)

"""
We can now print the `log_evidence` of each fit and confirm the model with two `Gaussians` was preferred to the model
with just one `Gaussian`.
"""
print(result_single.samples.log_evidence)
print(result_multiple.samples.log_evidence)

"""
__Sensitivity Mapping__

The model comparison above shows that in this dataset, the `Gaussian` feature was detectable and that it increased the 
Bayesian evidence by ~25. Furthermore, the normalization of this `Gaussian` was ~0.3. 

A lower value of normalization makes the `Gaussian` fainter and harder to detect. We will demonstrate sensitivity mapping 
by answering the following question, at what value of normalization does the `Gaussian` feature become undetectable and
not provide us with a noticeable increase in Bayesian evidence?

To begin, we define the `base_model` that we use to perform sensitivity mapping. This model is used to simulate every 
dataset. It is also fitted to every simulated dataset without the extra model component below, to give us the Bayesian
evidence of the every simpler model to compare to the more complex model. 

The `base_model` corresponds to the `gaussian_main` above.
"""
base_model = af.Collection(gaussian_main=af.ex.Gaussian)
base_model.gaussian_main.centre = 50.0
base_model.gaussian_main.sigma = 10.0

"""
We now define the `perturb_model`, which is the model component whose parameters we iterate over to perform 
sensitivity mapping. Many instances of the `perturb_model` are created and used to simulate the many datasets 
that we fit. However, it is only included in half of the model-fits; corresponding to the more complex models whose
Bayesian evidence we compare to the simpler model-fits consisting of just the `base_model`.

The `perturb_model` is therefore another `Gaussian` but now corresponds to the `gaussian_feature` above.

By fitting both of these models to every simulated dataset, we will therefore infer the Bayesian evidence of every
model to every dataset. Sensitivity mapping therefore maps out for what values of `normalization` in the `gaussian_feature`
 does the more complex model-fit provide higher values of Bayesian evidence than the simpler model-fit.
"""
perturb_model = af.Model(af.ex.Gaussian)

"""
Sensitivity mapping is typically performed over a large range of parameters. However, to make this demonstration quick
and clear we are going to fix the `centre` and `sigma` values to the true values of the `gaussian_feature`. We will 
also iterate over just two `normalization` values corresponding to 0.01 and 100.0, which will clearly exhaggerate the
sensitivity between the models at these two values.
"""
perturb_model.centre = 70.0
perturb_model.sigma = 0.5
perturb_model.normalization = af.UniformPrior(lower_limit=0.01, upper_limit=100.0)

"""
We are performing sensitivity mapping to determine how bright the `gaussian_feature` needs to be in order to be 
detectable. However, every simulated dataset must include the `main_gaussian`, as its presence in the data will effect
the detectability of the `gaussian_feature`.

We can pass the `main_gaussian` into the sensitivity mapping as the `simulation_instance`, meaning that it will be used 
in the simulation of every dataset. For this example we use the inferred `main_gaussian` from one of the model-fits
performed above.
"""
simulation_instance = result_single.instance

"""
We are about to write a `simulate_cls` that simulates examples of 1D `Gaussian` datasets that are fitted to
perform sensitivity mapping.

To pass each simulated data through **PyAutoFit**'s sensitivity mapping tools, the function must return a single 
Python object. We therefore define a `Dataset` class that combines the `data` and `noise_map` that are to be 
output by this `simulate_cls`.
"""


class Dataset:
    def __init__(self, data, noise_map):
        self.data = data
        self.noise_map = noise_map


"""
Each model-fit performed by sensitivity mapping creates a new instance of an `Analysis` class, which contains the
data simulated by the `simulate_cls` for that model.

This requires us to write a wrapper around the `Analysis` class that we used to fit the model above, so that is uses
the `Dataset` object above.
"""


class Analysis(af.ex.Analysis):
    def __init__(self, dataset):
        super().__init__(data=dataset.data, noise_map=dataset.noise_map)


"""
We now write the `simulate_cls`, which takes the `simulation_instance` of our model (defined above) and uses it to 
simulate a dataset which is subsequently fitted.

Note that when this dataset is simulated, the quantity `instance.perturb` is used in the `simulate_cls`.
This is an instance of the `gaussian_feature`, and it is different every time the `simulate_cls` is called. 

In this example, this `instance.perturb` corresponds to two different `gaussian_feature` with values of
`normalization` of 0.01 and 100.0, such that our simulated datasets correspond to a very faint and very bright gaussian 
features .
"""


class Simulate:
    def __init__(self):
        """
        Class used to simulate every dataset used for sensitivity mapping.

        This `__init__` constructor can be extended with new inputs which can be used to control how the dataset is
        simulated in the `__call__` simulate_function below.

        In this example we leave it empty as our `simulate_function` does not require any additional information.
        """
        pass

    def __call__(self, instance, simulate_path):
        """
        The `simulate_function` called by the `Sensitivity` class which simulates each dataset fitted
        by the sensitivity mapper.

        The simulation procedure is as follows:

        1) Use the input sensitivity `instance` to simulate the data with the small Gaussian feature.

        2) Output information about the simulation to hard-disk.

        3) Return the data for the sensitivity mapper to fit.

        Parameters
        ----------
        instance
            The sensitivity instance, which includes the Gaussian feature parameters are varied to perform sensitivity.
            The Gaussian feature in this instance changes for every iteration of the sensitivity mapping.
        simulate_path
            The path where the simulated dataset is output, contained within each sub-folder of the sensitivity
            mapping.

        Returns
        -------
        A simulated image of a Gaussian, which i input into the fits of the sensitivity mapper.
        """

        """
        Specify the number of pixels used to create the xvalues on which the 1D line of the profile is generated 
        using and thus defining the number of data-points in our data.
        """
        pixels = 100
        xvalues = np.arange(pixels)

        """
        Evaluate the `Gaussian` and Exponential model instances at every xvalues to create their model profile 
        and sum them together to create the overall model profile.
    
        This print statement will show that, when you run `Sensitivity` below the values of the perturbation 
        use fixed  values of `centre=70` and `sigma=0.5`, whereas the normalization varies over the `number_of_steps` 
        based on its prior.
        """

        print(instance.perturb.centre)
        print(instance.perturb.normalization)
        print(instance.perturb.sigma)

        model_line = instance.gaussian_main.model_data_from(
            xvalues=xvalues
        ) + instance.perturb.model_data_from(xvalues=xvalues)

        """
        Determine the noise (at a specified signal to noise level) in every pixel of our model profile.
        """
        signal_to_noise_ratio = 25.0
        noise = np.random.normal(0.0, 1.0 / signal_to_noise_ratio, pixels)

        """
        Add this noise to the model line to create the line data that is fitted, using the signal-to-noise ratio 
        to compute noise-map of our data which is required when evaluating the chi-squared value of the likelihood.
        """
        data = model_line + noise
        noise_map = (1.0 / signal_to_noise_ratio) * np.ones(pixels)

        return Dataset(data=data, noise_map=noise_map)


class BaseFit:
    def __init__(self, analysis_cls):
        """
        Class used to fit every dataset used for sensitivity mapping with the base model (the model without the
        perturbed feature sensitivity mapping maps out).

        In this example, the base model therefore does not include the extra Gaussian feature, but the simulated
        dataset includes one.

        The base fit is repeated for every parameter on the sensitivity grid and compared to the perturbed fit. This
        maps out the sensitivity of every parameter is (e.g. the sensitivity of the normalization of the Gaussian
        feature).

        The `__init__` constructor can be extended with new inputs which can be used to control how the dataset is
        fitted, below we include an input `analysis_cls` which is the `Analysis` class used to fit the model to the
        dataset.

        Parameters
        ----------
        analysis_cls
            The `Analysis` class used to fit the model to the dataset.
        """
        self.analysis_cls = analysis_cls

    def __call__(self, dataset, model, paths, instance):
        """
        The base fitting function which fits every dataset used for sensitivity mapping with the base model.

        This function receives as input each simulated dataset of the sensitivity map and fits it, in order to
        quantify how sensitive the model is to the perturbed feature.

        In this example, a full non-linear search is performed to determine how well the model fits the dataset.
        The `log_evidence` of the fit is returned which acts as the sensitivity map figure of merit.

        Parameters
        ----------
        dataset
            The dataset which is simulated with the perturbed model and which is fitted.
        model
            The model instance which is fitted to the dataset, which does not include the perturbed feature.
        paths
            The `Paths` instance which contains the path to the folder where the results of the fit are written to.
        """

        search = af.DynestyStatic(
            paths=paths,
            nlive=25,
            unique_tag=dataset_name,
            session=session,
            maxcall=1000,
            maxiter=1000,
            iterations_per_full_update=50000,
        )

        analysis = self.analysis_cls(dataset=dataset)

        return search.fit(model=model, analysis=analysis)


class PerturbFit:
    def __init__(self, analysis_cls):
        """
        Class used to fit every dataset used for sensitivity mapping with the perturbed model (the model with the
        perturbed feature sensitivity mapping maps out).

        In this example, the perturbed model therefore includes the extra Gaussian feature, which is also in the
        simulated dataset.

        The perturbed fit is repeated for every parameter on the sensitivity grid and compared to the base fit. This
        maps out the sensitivity of every parameter is (e.g. the sensitivity of the normalization of the Gaussian
        feature).

        The `__init__` constructor can be extended with new inputs which can be used to control how the dataset is
        fitted, below we include an input `analysis_cls` which is the `Analysis` class used to fit the model to the
        dataset.

        Parameters
        ----------
        analysis_cls
            The `Analysis` class used to fit the model to the dataset.
        """
        self.analysis_cls = analysis_cls

    def __call__(self, dataset, model, paths, instance):
        """
        The perturbed fitting function which fits every dataset used for sensitivity mapping with the perturbed model.

        This function receives as input each simulated dataset of the sensitivity map and fits it, in order to
        quantify how sensitive the model is to the perturbed feature.

        In this example, a full non-linear search is performed to determine how well the model fits the dataset.
        The `log_evidence` of the fit is returned which acts as the sensitivity map figure of merit.

        Parameters
        ----------
        dataset
            The dataset which is simulated with the perturbed model and which is fitted.
        model
            The model instance which is fitted to the dataset, which includes the perturbed feature.
        paths
            The `Paths` instance which contains the path to the folder where the results of the fit are written to.
        """

        search = af.DynestyStatic(
            paths=paths,
            nlive=25,
            maxcall=1000,
            maxiter=1000,
            iterations_per_full_update=50000,
        )

        analysis = self.analysis_cls(dataset=dataset)

        return search.fit(model=model, analysis=analysis)


"""
We can now combine all of the objects created above and perform sensitivity mapping. The inputs to the `Sensitivity`
object below are:

- `simulation_instance`: This is an instance of the model used to simulate every dataset that is fitted. In this 
example it contains an instance of the `gaussian_main` model component.

- `base_model`: This is the simpler model that is fitted to every simulated dataset, which in this example is composed 
of a single `Gaussian` called the `gaussian_main`.

- `perturb_model`: This is the extra model component that has two roles: (i) based on the sensitivity grid parameters
it is added to the `simulation_instance` to simulate each dataset ; (ii) it is added to the`base_model` and fitted to 
every simulated dataset (in this example every `simulation_instance` and `perturb_model` there has two `Gaussians` 
called the `gaussian_main` and `gaussian_feature`).

- `simulate_cls`: This is the function that uses the `simulation_instance` and many instances of the `perturb_model` 
to simulate many datasets which are fitted with the `base_model` and `base_model` + `perturb_model`.

- `base_fit_cls`: This is the function that fits the `base_model` to every simulated dataset and returns the
goodness-of-fit of the model to the data.

- `perturb_fit_cls`: This is the function that fits the `base_model` + `perturb_model` to every simulated dataset and
returns the goodness-of-fit of the model to the data.

- `number_of_steps`: The number of steps over which the parameters in the `perturb_model` are iterated. In this 
example, normalization has a `LogUniformPrior` with lower limit 1e-4 and upper limit 1e2, therefore the `number_of_steps` 
of 2 wills imulate and fit just 2 datasets where the intensities between 1e-4 and 1e2.

- `number_of_cores`: The number of cores over which the sensitivity mapping is performed, enabling parallel processing
if set above 1.
"""
from autofit.non_linear.grid import sensitivity as s

paths = af.DirectoryPaths(
    path_prefix=path.join("database", "directory", name),
    name="sensitivity_mapping",
)

sensitivity = s.Sensitivity(
    paths=paths,
    simulation_instance=simulation_instance,
    base_model=base_model,
    perturb_model=perturb_model,
    simulate_cls=Simulate(),
    base_fit_cls=BaseFit(analysis_cls=Analysis),
    perturb_fit_cls=PerturbFit(analysis_cls=Analysis),
    number_of_steps=2,
    number_of_cores=1,
)

sensitivity_result = sensitivity.run()

"""
You should now look at the results of the sensitivity mapping in the folder `output/features/sensitivity_mapping`. 

You will note the following 4 model-fits have been performed:

 - The `base_model` is fitted to a simulated dataset where the `simulation_instance` and 
 a `perturbation` with `normalization=0.01` are used.

 - The `base_model` + `perturb_model`  is fitted to a simulated dataset where the `simulation_instance` and 
 a `perturbation` with `normalization=0.01` are used.

 - The `base_model` is fitted to a simulated dataset where the `simulation_instance` and 
 a `perturbation` with `normalization=100.0` are used.

 - The `base_model` + `perturb_model`  is fitted to a simulated dataset where the `simulation_instance` and 
 a `perturbation` with `normalization=100.0` are used.

The fit produced a `sensitivity_result`. 

We are still developing the `SensitivityResult` class to provide a data structure that better streamlines the analysis
of results. If you intend to use sensitivity mapping, the best way to interpret the resutls is currently via
**PyAutoFit**'s database and `Aggregator` tools. 
"""
print(sensitivity_result.samples[0].log_evidence)
print(sensitivity_result.samples[1].log_evidence)

"""
Scrape directory to create .sqlite file.
"""
import os
from autofit.database.aggregator import Aggregator

database_file = "database_directory_sensitivity.sqlite"

try:
    os.remove(path.join("output", database_file))
except FileNotFoundError:
    pass

agg = Aggregator.from_database(database_file)

agg.add_directory(directory=path.join("output", "database", "directory", "sensitivity"))

assert len(agg) > 0

"""
__Samples + Results__

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
Test that we can retrieve an aggregator with only the sensitivity grid search results (I have tried using 
the `grid_searches())` API below, but I dont know if this should be `sensitivity_searches` instead.
"""
print("\n\n***********************")
print("**GRID RESULTS TESTING**")
print("***********************\n\n")

agg_grid_searches = agg.grid_searches()
print(
    "\n****Total aggregator via `grid_searches` query = ",
    len(agg_grid_searches),
    "****\n",
)
unique_tag = agg_grid_searches.search.unique_tag
agg_qrid = agg_grid_searches.query(unique_tag == "gaussian_x1")

print(
    "****Total aggregator via `grid_searches` & unique tag query = ",
    len(agg_grid_searches),
    "****\n",
)

gaussian_main = agg.model.gaussian_main
agg_query = agg.query(gaussian_main == af.ex.Gaussian)
print("Total queries for correct model = ", len(agg_query))

"""
Request 1: 

Make the `SensitivityResult` accessible via the database. Ideally, this would be accessible even when a Sensitivity 
run is mid-run (e.g. if only the first 10 of 16 runs are complete.
"""
# sensitivity_result = list(agg_grid_searches)[0]['result']
# print(sensitivity_result)

"""
Reqest 2:

From the Sensitivity, get an aggregator for the base or perturbed model of any of the grid cells.
"""
# cell_aggregator = agg_grid.cell_number(1)
# print("Size of Agg cell = ", len(cell_aggregator), "\n")

"""
Finish.
"""
