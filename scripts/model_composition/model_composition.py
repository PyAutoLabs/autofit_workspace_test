"""
Model Composition: Integration Tests
=====================================

Integration tests for the PyAutoFit model composition API. These tests compose
models at realistic complexity and assert structural properties of the resulting
``af.Model`` graph — prior count, prior identity (shared vs independent), path
structure, parameter vector ordering, serialization round-trip fidelity, and
identifier stability.

The goal is to catch silent PyAutoFit regressions — refactors that alter how
models are composed without raising any errors. A broken composition helper that
returns the right type but the wrong prior structure is worse than a crash: every
downstream fit runs normally with degraded or incorrect results.

These tests use simple model classes (Gaussian, Exponential) with no dependency
on autogalaxy or autolens, so they exercise the core PyAutoFit composition
machinery in isolation.

__Contents__

**Basic Model Composition:** af.Model and af.Collection with multiple components.
**Prior Linking:** Shared priors via object identity, cross-component links.
**Nested Collections:** Multi-level af.Collection nesting and cross-collection linking.
**Prior Types:** Uniform, Gaussian, TruncatedGaussian, LogUniform — type assertions.
**Fixed Parameters:** Constants reduce prior count.
**Parameter Vector Ordering:** prior_tuples_ordered_by_id, instance_from_vector, unit vectors.
**Serialization Round-Trip:** dict/from_dict preserves prior count, identity, types, paths.
**Identifier Stability:** Deterministic md5 hash; changes with structure or prior bounds.
**Model Subsetting:** with_paths, without_paths filtering.
**Model Assertions:** add_assertion raises FitException on invalid vectors.
"""

import autofit as af
from autofit import exc


"""
__Setup__

Simple model component classes used throughout.
"""


class Gaussian:
    def __init__(
        self,
        centre: float = 30.0,
        normalization: float = 1.0,
        sigma: float = 5.0,
    ):
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma


class Exponential:
    def __init__(
        self,
        centre: float = 30.0,
        normalization: float = 1.0,
        rate: float = 0.01,
    ):
        self.centre = centre
        self.normalization = normalization
        self.rate = rate


"""
__Basic Model Composition__

A single ``af.Model`` wrapping a class should expose one free parameter per
constructor argument. An ``af.Collection`` of two components sums their counts.
"""

model = af.Model(Gaussian)

assert model.prior_count == 3
assert model.total_free_parameters == 3
assert model.unique_prior_paths == [
    ("centre",),
    ("normalization",),
    ("sigma",),
]

prior_types = [type(p).__name__ for _, p in model.prior_tuples_ordered_by_id]
assert prior_types == ["UniformPrior", "LogUniformPrior", "UniformPrior"], (
    f"Default prior types changed: {prior_types}"
)

model = af.Collection(gaussian=Gaussian, exponential=Exponential)

assert model.prior_count == 6
assert model.unique_prior_paths == [
    ("gaussian", "centre"),
    ("gaussian", "normalization"),
    ("gaussian", "sigma"),
    ("exponential", "centre"),
    ("exponential", "normalization"),
    ("exponential", "rate"),
]

print("Basic model composition: PASSED")

"""
__Prior Linking / Shared Priors__

Assigning one prior to another makes them the same Python object. The model's
``prior_count`` drops because ``unique_prior_tuples`` deduplicates by identity.
Instances created from the linked model receive the same value for both params.
"""

model = af.Collection(gaussian=Gaussian, exponential=Exponential)
model.gaussian.centre = model.exponential.centre

assert model.prior_count == 5, f"Expected 5, got {model.prior_count}"
assert model.gaussian.centre is model.exponential.centre

instance = model.instance_from_prior_medians()
assert instance.gaussian.centre == instance.exponential.centre

model_multi_link = af.Collection(gaussian=Gaussian, exponential=Exponential)
model_multi_link.gaussian.centre = model_multi_link.exponential.centre
model_multi_link.gaussian.normalization = model_multi_link.exponential.normalization

assert model_multi_link.prior_count == 4
assert model_multi_link.gaussian.centre is model_multi_link.exponential.centre
assert model_multi_link.gaussian.normalization is model_multi_link.exponential.normalization

print("Prior linking: PASSED")

"""
__Nested Collections__

Collections can be nested arbitrarily. Prior count is the sum of all leaf priors.
Cross-collection linking reduces the count by sharing identity.
"""

inner_a = af.Collection(g1=af.Model(Gaussian), g2=af.Model(Gaussian))
inner_b = af.Collection(e1=af.Model(Exponential))
nested = af.Collection(group_a=inner_a, group_b=inner_b)

assert nested.prior_count == 9

expected_paths = [
    ("group_a", "g1", "centre"),
    ("group_a", "g1", "normalization"),
    ("group_a", "g1", "sigma"),
    ("group_a", "g2", "centre"),
    ("group_a", "g2", "normalization"),
    ("group_a", "g2", "sigma"),
    ("group_b", "e1", "centre"),
    ("group_b", "e1", "normalization"),
    ("group_b", "e1", "rate"),
]
assert nested.unique_prior_paths == expected_paths, (
    f"Path structure changed: {nested.unique_prior_paths}"
)

nested.group_a.g1.centre = nested.group_b.e1.centre
assert nested.prior_count == 8
assert nested.group_a.g1.centre is nested.group_b.e1.centre

nested.group_a.g2.centre = nested.group_b.e1.centre
assert nested.prior_count == 7
assert nested.group_a.g1.centre is nested.group_a.g2.centre
assert nested.group_a.g2.centre is nested.group_b.e1.centre

print("Nested collections: PASSED")

"""
__Prior Types and Customisation__

Swapping a prior's type (Uniform, Gaussian, TruncatedGaussian) changes the
distribution but not the model's dimensionality. The prior's parameters should
match what was set.
"""

model = af.Model(Gaussian)

model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
assert isinstance(model.centre, af.UniformPrior)
assert model.centre.lower_limit == 0.0
assert model.centre.upper_limit == 100.0

model.normalization = af.GaussianPrior(mean=5.0, sigma=1.0)
assert isinstance(model.normalization, af.GaussianPrior)
assert model.normalization.mean == 5.0
assert model.normalization.sigma == 1.0

model.sigma = af.TruncatedGaussianPrior(
    mean=10.0, sigma=2.0, lower_limit=0.0, upper_limit=25.0
)
assert isinstance(model.sigma, af.TruncatedGaussianPrior)
assert model.sigma.mean == 10.0
assert model.sigma.sigma == 2.0
assert model.sigma.lower_limit == 0.0
assert model.sigma.upper_limit == 25.0

assert model.prior_count == 3

print("Prior types: PASSED")

"""
__Fixed Parameters__

Setting a parameter to a scalar removes it from the free parameter set.
The instance uses the fixed value.
"""

model = af.Model(Gaussian)
model.centre = 50.0

assert model.prior_count == 2
instance = model.instance_from_prior_medians()
assert instance.centre == 50.0

model.normalization = 3.0
assert model.prior_count == 1
instance = model.instance_from_prior_medians()
assert instance.centre == 50.0
assert instance.normalization == 3.0

print("Fixed parameters: PASSED")

"""
__Parameter Vector Ordering__

``prior_tuples_ordered_by_id`` sorts by creation order (ascending ``prior.id``).
``instance_from_vector`` maps vector elements to parameters in this order.
``vector_from_unit_vector`` transforms unit-space values via each prior's
``value_for`` method.
"""

model = af.Model(Gaussian)
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)

ids = [p.id for _, p in model.prior_tuples_ordered_by_id]
assert ids == sorted(ids), f"Prior ordering not by ascending id: {ids}"

instance = model.instance_from_vector(vector=[25.0, 7.0, 40.0])
assert instance.centre == 25.0
assert instance.normalization == 7.0
assert instance.sigma == 40.0

physical = model.vector_from_unit_vector(unit_vector=[0.5, 0.5, 0.5])
assert physical == [50.0, 5.0, 25.0], f"Unit vector mapping changed: {physical}"

print("Parameter vector ordering: PASSED")

"""
__Serialization Round-Trip__

``model.dict()`` → ``from_dict()`` must preserve:
- prior_count (same number of free parameters)
- Shared-prior identity (linked priors remain the same object)
- Prior types (Uniform stays Uniform, etc.)
- Path structure (unique_prior_paths unchanged)
"""

model = af.Collection(gaussian=Gaussian, exponential=Exponential)
model.gaussian.centre = model.gaussian.sigma

assert model.prior_count == 5

d = model.dict()
restored = af.Collection.from_dict(d)

assert restored.prior_count == model.prior_count, (
    f"prior_count changed after round-trip: {restored.prior_count} vs {model.prior_count}"
)

assert restored.gaussian.centre is restored.gaussian.sigma, (
    "Shared-prior identity lost after serialization round-trip"
)

assert restored.gaussian.centre is not restored.exponential.centre, (
    "Unlinked priors should remain independent after round-trip"
)

original_types = sorted(type(p).__name__ for _, p in model.prior_tuples_ordered_by_id)
restored_types = sorted(type(p).__name__ for _, p in restored.prior_tuples_ordered_by_id)
assert restored_types == original_types, (
    f"Prior types changed after round-trip: {restored_types} vs {original_types}"
)

assert sorted(restored.unique_prior_paths) == sorted(model.unique_prior_paths), (
    f"Path structure changed after round-trip"
)

print("Serialization round-trip: PASSED")

"""
__Identifier Stability__

The model identifier is an md5 hash of the model structure, prior distribution
parameters, and search config. It determines the output folder for results.

If this identifier changes after a PyAutoFit refactor, existing users' output
folders will no longer match — a backwards-compatibility break.

We hardcode a known-good identifier as a regression anchor. If this assertion
fails, a code change has altered the model composition contract. This may be
intentional (in which case update the expected value) but it must not be
accidental.
"""

from autofit.non_linear.paths.directory import DirectoryPaths

model = af.Collection(gaussian=Gaussian, exponential=Exponential)

paths = DirectoryPaths()
paths.model = model

identifier = paths.identifier

assert len(identifier) == 32, f"Identifier length changed: {len(identifier)}"
assert identifier.isalnum(), f"Identifier contains non-alphanumeric chars: {identifier}"

paths2 = DirectoryPaths()
paths2.model = model
assert paths2.identifier == identifier, "Identifier not deterministic"

model_different = af.Collection(gaussian=Gaussian, exponential=Exponential)
model_different.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=200.0)
paths_diff = DirectoryPaths()
paths_diff.model = model_different
assert paths_diff.identifier != identifier, (
    "Changing prior bounds should change identifier"
)

model_linked = af.Collection(gaussian=Gaussian, exponential=Exponential)
model_linked.gaussian.centre = model_linked.exponential.centre
paths_linked = DirectoryPaths()
paths_linked.model = model_linked
assert paths_linked.identifier == identifier, (
    "Linking priors with identical distributions should NOT change the "
    "identifier — the hash is based on distribution parameters, not "
    "prior object identity. This is intentional: prior_count changes but "
    "the identifier (which determines output folders) is stable."
)

assert identifier == "0d2d241add8588ed5074cddfb8f80887", (
    f"REGRESSION: model identifier changed from expected value. "
    f"Got '{identifier}'. If this is intentional (e.g. a deliberate refactor "
    f"of the identifier algorithm), update this expected value. If not, a "
    f"code change has silently altered how models are composed."
)

print("Identifier stability: PASSED")

"""
__Model Subsetting__

``with_paths`` extracts a sub-model containing only the specified paths.
``without_paths`` removes the specified paths. Both preserve the correct
prior count.
"""

model = af.Collection(gaussian=Gaussian, exponential=Exponential)

g_only = model.with_paths([("gaussian",)])
assert g_only.prior_count == 3

without_g = model.without_paths([("gaussian",)])
assert without_g.prior_count == 3

centres_only = model.with_paths([("gaussian", "centre"), ("exponential", "centre")])
assert centres_only.prior_count == 2

print("Model subsetting: PASSED")

"""
__Model Assertions__

``add_assertion`` constrains the parameter space. Valid vectors produce instances;
invalid vectors raise ``FitException``.
"""

model = af.Model(Gaussian)
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)

model.add_assertion(model.sigma < model.centre, name="sigma_less_than_centre")

instance = model.instance_from_vector(vector=[50.0, 3.0, 10.0])
assert instance.centre == 50.0
assert instance.sigma == 10.0

try:
    model.instance_from_vector(vector=[10.0, 3.0, 50.0])
    raise RuntimeError("Expected FitException was not raised")
except exc.FitException:
    pass

print("Model assertions: PASSED")

"""
__Summary__

All model composition integration tests passed.
"""

print()
print("All model composition integration tests: PASSED")
