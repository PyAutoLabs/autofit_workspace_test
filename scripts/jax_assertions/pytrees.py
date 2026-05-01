"""
Jax Assertions: Manual Pytree Roundtrip
=======================================

Verifies that ``jax.tree_util.register_pytree_node_class`` correctly registers
PyAutoFit's ``Prior``, ``Model``, ``Collection``, and ``ModelInstance`` types
so that flatten/unflatten roundtrips preserve identity (``id``), bounds, and
nested structure.

The test pattern manually pulls the ``flatten_func`` / ``unflatten_func`` out
of ``jax._src.tree_util._registry`` and round-trips an instance — testing
the legacy direct-registration path used before
``autofit.jax.enable_pytrees()`` existed (see ``enable_pytrees.py`` for the
public API path).

Previously: ``test_autofit/jax/test_pytrees.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np

import autofit as af
from autofit import UniformPrior
from jax.tree_util import register_pytree_node_class

# Manual pytree registration for the legacy direct path.
UniformPrior = register_pytree_node_class(UniformPrior)
GaussianPrior = register_pytree_node_class(af.GaussianPrior)
TruncatedGaussianPrior = register_pytree_node_class(af.TruncatedGaussianPrior)
Collection = register_pytree_node_class(af.Collection)
Model = register_pytree_node_class(af.Model)
ModelInstance = register_pytree_node_class(af.ModelInstance)


def recreate(o):
    """jax-pytree-roundtrip helper inlined from the original test fixture."""
    flatten_func, unflatten_func = jax._src.tree_util._registry[type(o)]
    children, aux_data = flatten_func(o)
    return unflatten_func(aux_data, children)


# Original fixture monkeypatched af.example.model.np = jnp via autouse.
# Replicate globally — the active tests below don't actually call gaussian.f,
# but keep the patch to match original semantics.
af.example.model.np = jnp


model = Model(
    af.ex.Gaussian,
    centre=af.GaussianPrior(mean=1.0, sigma=1.0),
    normalization=af.GaussianPrior(mean=1.0, sigma=1.0),
    sigma=af.GaussianPrior(mean=1.0, sigma=1.0),
)

"""
__TruncatedGaussianPrior Roundtrip__
"""
prior = TruncatedGaussianPrior(mean=1.0, sigma=1.0)
new = recreate(prior)

assert new.mean == prior.mean
assert new.sigma == prior.sigma
assert new.id == prior.id
assert new.lower_limit == prior.lower_limit
assert new.upper_limit == prior.upper_limit

"""
__Model Roundtrip__
"""
new = recreate(model)
assert new.cls == af.ex.Gaussian

centre = new.centre
assert centre.mean == model.centre.mean
assert centre.sigma == model.centre.sigma
assert centre.id == model.centre.id

"""
__UniformPrior Roundtrip__
"""
prior = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
new = recreate(prior)

assert new.lower_limit == prior.lower_limit
assert new.upper_limit == prior.upper_limit
assert new.id == prior.id

"""
__ModelInstance Roundtrip__
"""
collection = Collection(gaussian=model)
instance = collection.instance_from_prior_medians()
new = recreate(instance)

assert isinstance(new, ModelInstance)
assert isinstance(new.gaussian, af.ex.Gaussian)

"""
__Collection Roundtrip__
"""
collection = Collection(gaussian=model)
new = recreate(collection)

assert isinstance(new, Collection)
assert isinstance(new.gaussian, Model)
assert new.gaussian.cls == af.ex.Gaussian

centre = new.gaussian.centre
assert centre.mean == model.centre.mean
assert centre.sigma == model.centre.sigma
assert centre.id == model.centre.id

print("pytrees: all assertions passed")
