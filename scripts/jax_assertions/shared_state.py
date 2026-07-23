"""
JAX Assertions: Shared Analysis State
=====================================

JAX-only assertions for the `FactorGraphModel` cross-factor shared-state mechanism (PyAutoFit #1308). The numpy
behaviour (compute-once, exact equality, opt-in) is covered by `scripts/graphical/shared_state.py`; this script covers
the requirement that the shared object is a well-behaved JAX pytree, because the motivating consumer (the lensing
datacube likelihood) is JIT-compiled and the shared object will contain traced arrays.

The shared object must:
 - be a JAX array / pytree when the `Analysis` runs with `use_jax=True`;
 - thread through `jax.jit` as a normal traced pytree (no Python-side caching that would cache-bust or bake it in);
 - produce a likelihood that matches the eager numpy computation.

Run from the workspace root:

    python scripts/jax_assertions/shared_state.py
"""
# ENV: jax
# JAX assertion scripts test JAX behaviour; disabling JAX makes their
# assertions vacuous.

import numpy as np
import jax
import jax.numpy as jnp

import autofit as af


def _analysis(use_jax, share_model_data=True):
    data = np.ones(20)
    noise_map = np.ones(20) * 0.1
    return af.ex.Analysis(
        data=data,
        noise_map=noise_map,
        use_jax=use_jax,
        share_model_data=share_model_data,
    )


def _instance():
    model = af.Model(af.ex.Gaussian)
    return model.instance_from_unit_vector([0.5] * model.prior_count)


def assert_shared_state_is_jax_pytree_under_use_jax():
    """With `use_jax=True` the shared model data is a JAX array, i.e. a valid pytree of traced leaves."""
    analysis = _analysis(use_jax=True)
    instance = _instance()

    shared = analysis.shared_state_from(instance)

    assert shared is not None
    assert isinstance(shared, jnp.ndarray)

    leaves = jax.tree_util.tree_leaves(shared)
    assert len(leaves) == 1
    assert isinstance(leaves[0], jnp.ndarray)


def assert_shared_object_threads_through_jit():
    """
    The shared object must thread through `jax.jit` as a traced pytree argument and give the same likelihood as the
    eager computation. This is the property the datacube path relies on: the shared mapper/curvature pytree is passed
    into each channel's jitted likelihood.
    """
    analysis = _analysis(use_jax=True)
    instance = _instance()

    shared = analysis.shared_state_from(instance)

    @jax.jit
    def log_likelihood(shared_arg):
        return analysis.log_likelihood_function(instance, shared=shared_arg)

    jitted = log_likelihood(shared)
    eager = analysis.log_likelihood_function(instance, shared=shared)

    assert jnp.allclose(jitted, eager)


def assert_numpy_and_jax_shared_state_agree():
    """The mechanism is xp-agnostic: the shared model data is numerically identical under numpy and JAX."""
    instance = _instance()

    shared_numpy = _analysis(use_jax=False).shared_state_from(instance)
    shared_jax = _analysis(use_jax=True).shared_state_from(instance)

    assert np.allclose(np.asarray(shared_numpy), np.asarray(shared_jax))


def assert_no_sharing_returns_none_under_use_jax():
    """With `share_model_data=False` the JAX analysis shares nothing, exactly like the numpy default."""
    analysis = _analysis(use_jax=True, share_model_data=False)
    instance = _instance()

    assert analysis.shared_state_from(instance) is None


if __name__ == "__main__":
    assert_shared_state_is_jax_pytree_under_use_jax()
    assert_shared_object_threads_through_jit()
    assert_numpy_and_jax_shared_state_agree()
    assert_no_sharing_returns_none_under_use_jax()
    print("shared_state: all assertions passed")
