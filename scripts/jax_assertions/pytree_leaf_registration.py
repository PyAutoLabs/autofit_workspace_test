"""
Jax Assertions: Pytree Leaf-Type Registration Guard
===================================================

Regression guard for PyAutoFit #1365 (2026-07-13 release-validation): JAX-fitting
a shapelet model crashed in ``jax.vmap(jax.jit(...))`` because ``register_model``
registered a *leaf type* as a pytree node.

``af.Model(ShapeletPolar)`` auto-wraps its ``int``-typed constructor args
(``n``/``m``) as ``af.Model(int)`` sub-models. ``register_model._walk`` then
registered ``builtins.int`` as an instance pytree node, so JAX called the
instance-flatten ``_partition(5)`` on every ``int`` leaf, and ``vars(5)`` raised
``TypeError: vars() argument must have __dict__ attribute``.

``int`` (like ``float``/``str``/``bool``) is a native JAX leaf and must never be
registered. The fix gates registration on ``_instances_carry_dict(cls)``. This
script reproduces the exact structure without autogalaxy: a model whose
component carries an ``af.Model(int)`` sub-model.

Previously: N/A (new guard).
"""
# ENV: jax
# JAX assertion scripts test JAX behaviour; disabling JAX makes their
# assertions vacuous.

import jax

import autofit as af
from autofit.jax import pytrees


class WithIntArg:
    """Stand-in for a shapelet-like component: an int arg autofit wraps as Model(int)."""

    def __init__(self, n: int = 2, x: float = 1.0):
        self.n = n
        self.x = x


"""
__Model with an af.Model(int) sub-model (mirrors ShapeletPolar n/m)__
"""
model = af.Model(WithIntArg, x=af.UniformPrior(lower_limit=0.0, upper_limit=1.0))
model.n = af.Model(int)

collection = af.Collection(component=model)
instance = collection.instance_from_prior_medians()

registered = pytrees.register_model(collection)
assert registered, "register_model returned False — is JAX installed?"


"""
__Leaf types are NOT registered as pytree nodes__
"""
for leaf_type in (int, float, str, bool, type(None), tuple):
    assert leaf_type not in pytrees._REGISTERED_INSTANCE_CLASSES, (
        f"{leaf_type.__name__} was registered as an instance pytree node — "
        "native JAX leaves must never be registered (the flatten does vars())."
    )

# The real component class (has a __dict__) still IS registered.
assert (
    WithIntArg in pytrees._REGISTERED_INSTANCE_CLASSES
), "WithIntArg should be registered — its instances carry a __dict__."


"""
__tree_flatten / tree_unflatten round-trips (the crash site)__
"""
leaves, treedef = jax.tree_util.tree_flatten(instance)
rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

assert isinstance(rebuilt, af.ModelInstance)
assert isinstance(rebuilt.component, WithIntArg)
assert rebuilt.component.n == instance.component.n


print("pytree_leaf_registration: all assertions passed")
