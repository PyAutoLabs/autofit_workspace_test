"""
Jax Assertions: Derived Attributes Cross Boundaries As Leaves
=============================================================

An instance attribute computed inside ``__init__`` from a prior-derived
parameter (e.g. an ``NFWMCRLudlowSph`` deriving ``scale_radius`` from a free
``mass_at_200``) is never declared on the model, so the instance-pytree
classifier would default it to a constant and stash it in aux data. Under a
trace that value IS a tracer, and aux survives flatten as a raw Python
reference — it re-enters nested traces (a ``custom_jvp`` rule's inner
``jax.jvp``) as a stale tracer and raises ``UnexpectedTracerError``
(PyAutoLens#678 phase B, cluster gradient cells).

The fix promotes any attribute whose *value* is a JAX array or tracer to a
dynamic child at flatten time. This script asserts the promoted behaviour:

1. Under a trace, the instance's pytree aux data contains no tracers.
2. A ``custom_jvp`` whose rule re-reads the derived attribute inside its own
   ``jax.jvp`` differentiates cleanly, and the gradient through the derived
   attribute is exact.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX assertion scripts test JAX behaviour; disabling JAX makes their
assertions vacuous.

ENV: jax
"""

import jax
import numpy as np

import autofit as af


class DerivedAttrProfile:
    def __init__(self, mass=1.0):
        self.mass = mass
        self.scale_radius = 2.0 * mass  # derived — never declared on any model


model = af.Model(DerivedAttrProfile)
model.mass = af.UniformPrior(lower_limit=0.5, upper_limit=3.0)

from autofit.jax import register_model

register_model(model)

"""
__Aux Data Carries No Tracers__

Flatten the instance inside a trace and walk the aux data: the derived
``scale_radius`` (a tracer under the trace) must be a child leaf, never aux.
"""


def _contains_tracer(node):
    if isinstance(node, jax.core.Tracer):
        return True
    if isinstance(node, (tuple, list)):
        return any(_contains_tracer(item) for item in node)
    return False


def _aux_is_clean(vector):
    instance = model.instance_from_vector([vector])
    children, aux = jax.tree_util.tree_flatten(instance)
    assert not _contains_tracer(aux), "traced value leaked into pytree aux data"
    return instance.mass


jax.jit(_aux_is_clean)(0.7)

"""
__custom_jvp Rule Re-Reads The Derived Attribute__

Mirrors the PointSolver implicit-diff structure: the rule unpacks the
instance inside its own ``jax.jvp``. With the derived attribute in aux this
raises ``UnexpectedTracerError``; as a leaf it differentiates cleanly.
"""


@jax.custom_jvp
def read_scale(profile):
    return profile.scale_radius * 1.0


@read_scale.defjvp
def read_scale_jvp(primals, tangents):
    (profile,) = primals
    (profile_dot,) = tangents
    ans = read_scale(profile)
    _, d_ans = jax.jvp(lambda p: p.scale_radius * 1.0, (profile,), (profile_dot,))
    return ans, d_ans


def objective(mass):
    instance = model.instance_from_vector([mass])
    return read_scale(instance)


value, grad = jax.value_and_grad(objective)(0.7)

assert np.isclose(float(value), 1.4), float(value)
assert np.isclose(float(grad), 2.0), float(grad)

print("derived_attribute_leaves: all assertions passed")
