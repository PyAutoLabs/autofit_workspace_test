"""
Jax Assertions: enable_pytrees() / register_model() Public API
==============================================================

Verifies the public ``autofit.jax.enable_pytrees`` / ``register_model`` path
that PyAutoGalaxy and PyAutoLens use to make their ``Model`` /
``ModelInstance`` types JAX-traceable. Covers:

- Model roundtrip via the public API
- Compilation under ``jax.jit``
- Constant attributes (positional float, ``**kwargs`` concrete object) stay
  in ``aux_data`` and are NOT traced — protects ``galaxy.redshift`` sorting
  and ``isinstance(x, Pixelization)`` checks under JIT
- ``TuplePrior`` paired-prior gradients flow through ``jax.value_and_grad``
- ``enable_pytrees()`` is idempotent
- ``Collection`` round-trip preserves nested model identity

Previously: ``test_autofit/jax/test_enable_pytrees.py``.
"""

"""
__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX assertion scripts test JAX behaviour; disabling JAX makes their
assertions vacuous.

ENV: jax
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

import autofit as af
from autofit.jax import enable_pytrees, register_model


def make_model():
    return af.Model(
        af.ex.Gaussian,
        centre=af.GaussianPrior(mean=1.0, sigma=1.0),
        normalization=af.GaussianPrior(mean=2.0, sigma=1.0),
        sigma=af.GaussianPrior(mean=3.0, sigma=1.0),
    )


"""
__enable_pytrees() returns True When jax Available__
"""
assert enable_pytrees() is True

"""
__Model Round-Trip via Public API__
"""
model = make_model()
register_model(model)
instance = model.instance_from_prior_medians()

leaves, treedef = jax.tree_util.tree_flatten(instance)
rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

assert isinstance(rebuilt, af.ex.Gaussian)
assert float(rebuilt.centre) == float(instance.centre)
assert float(rebuilt.normalization) == float(instance.normalization)
assert float(rebuilt.sigma) == float(instance.sigma)

"""
__Model Computable Under jax.jit__
"""
model = make_model()
register_model(model)
instance = model.instance_from_prior_medians()


@jax.jit
def total(inst):
    return inst.centre + inst.normalization + inst.sigma


npt.assert_allclose(
    float(total(instance)),
    instance.centre + instance.normalization + instance.sigma,
)

"""
__Positional Constant Stays Static (galaxy.redshift pattern)__

Constants from the model definition must NOT be traced under JIT.
``Galaxy.redshift`` is a common case: it sits in the constructor signature
but gets a fixed value via ``af.Constant``. If it became a JAX tracer, code
paths like ``sorted(galaxies, key=lambda g: g.redshift)`` would blow up with
``TracerBoolConversionError``.
"""


class Holder:
    def __init__(self, redshift, scale):
        self.redshift = redshift
        self.scale = scale


model = af.Model(
    Holder,
    redshift=0.5,
    scale=af.GaussianPrior(mean=1.0, sigma=1.0),
)
register_model(model)
instance = model.instance_from_prior_medians()
assert isinstance(instance.redshift, float)


@jax.jit
def use_redshift_for_control_flow(inst):
    if inst.redshift > 0:
        return inst.scale * 2
    return inst.scale


result = use_redshift_for_control_flow(instance)
npt.assert_allclose(float(result), 2.0 * instance.scale)

"""
__Kwarg Constants Stay Static (Pixelization-as-kwarg pattern)__

Concrete objects passed as ``**kwargs`` must stay in ``aux_data``, not
``children``. ``Galaxy.__init__(self, redshift, **kwargs)`` stores every
kwarg via ``setattr``. A ``Pixelization`` passed as a kwarg used to be
routed to ``children`` and become a JAX tracer, so downstream
``isinstance(x, Pixelization)`` checks returned False.
"""


class Marker:
    pass


class KwargHolder:
    def __init__(self, redshift, **kwargs):
        self.redshift = redshift
        for k, v in kwargs.items():
            setattr(self, k, v)


marker = Marker()
model = af.Model(
    KwargHolder,
    redshift=0.5,
    marker=marker,
    scale=af.GaussianPrior(mean=1.0, sigma=1.0),
)
register_model(model)
instance = model.instance_from_prior_medians()
assert instance.marker is marker


@jax.jit
def use_marker_isinstance(inst):
    if isinstance(inst.marker, Marker):
        return inst.scale * 2
    return inst.scale


result = use_marker_isinstance(instance)
npt.assert_allclose(float(result), 2.0 * instance.scale)

"""
__TuplePrior Paired Priors Flow Gradients Through jax.value_and_grad__

``TuplePrior``-backed attributes (``centre=(x, y)``, ``ell_comps=(e1, e2)``)
must be routed into JAX children so gradients flow. Prior to the fix,
``TuplePrior`` failed the ``(Prior, AbstractPriorModel)`` isinstance check
in ``register_model``, so the resolved tuple was frozen in ``aux_data`` and
``jax.value_and_grad`` returned gradients only for the non-tuple attributes.
"""


class Twin:
    def __init__(self, centre, amplitude):
        self.centre = centre
        self.amplitude = amplitude


model = af.Model(
    Twin,
    centre=af.TuplePrior(
        centre_0=af.GaussianPrior(mean=0.5, sigma=1.0),
        centre_1=af.GaussianPrior(mean=-0.5, sigma=1.0),
    ),
    amplitude=af.GaussianPrior(mean=1.0, sigma=1.0),
)
register_model(model)
instance = model.instance_from_prior_medians()
params_tree = jax.tree_util.tree_map(jnp.asarray, instance)

leaves = jax.tree_util.tree_leaves(params_tree)
assert len(leaves) == 3  # centre[0], centre[1], amplitude


def loss(inst):
    cx, cy = inst.centre
    return cx * cx + cy * cy + inst.amplitude


_, grad = jax.value_and_grad(loss)(params_tree)
flat_grad = jnp.concatenate(
    [jnp.asarray(l).ravel() for l in jax.tree_util.tree_leaves(grad)]
)
assert jnp.all(jnp.isfinite(flat_grad))
assert flat_grad.size == 3

"""
__enable_pytrees() Is Idempotent__
"""
assert enable_pytrees() is True
assert enable_pytrees() is True

"""
__Collection Round-Trip Preserves Nested Model Identity__
"""
model = make_model()
register_model(model)
collection = af.Collection(g1=model, g2=model)
register_model(collection)

instance = collection.instance_from_prior_medians()
leaves, treedef = jax.tree_util.tree_flatten(instance)
rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

npt.assert_allclose(float(rebuilt.g1.centre), float(instance.g1.centre))
npt.assert_allclose(float(rebuilt.g2.sigma), float(instance.g2.sigma))

print("enable_pytrees: all assertions passed")
