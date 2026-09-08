"""
Traced Assertions
=================

Integration-test script for model assertions (`AbstractPriorModel.add_assertion`) on the JAX
paths. An assertion signals failure by raising `FitException`, and a `raise` cannot happen inside
a trace, so `Fitness.call` builds the instance with `ignore_assertions=True` and applies the
gathered assertions as a traced boolean through `xp.where`, mapping a violating model to
`resample_figure_of_merit` — the value the numpy path reaches by catching the exception instead.

These assertions pin that equivalence end-to-end on a real analysis: the non-vmapped `call`, the
`_jit` and `_vmap` surfaces, an assertion attached to a child rather than the `Collection`, the
Euclid two-basis compound-prior shape (euclid_strong_lens_modeling_pipeline#54), and that the
numpy path still raises exactly as it did.

Run from the workspace root:

    python scripts/jax_assertions/assertions_traced.py

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX assertion scripts test JAX behaviour; disabling JAX makes their
assertions vacuous.

ENV: jax
"""

import os

# `autonerves` enables JAX x64 only if this is already set when it is imported, and JAX reads it
# at import too — the `-1e99` sentinel below overflows to `-inf` in float32.
os.environ.setdefault("JAX_ENABLE_X64", "True")

import numpy as np
import jax
import jax.numpy as jnp

import autofit as af
from autofit.non_linear.fitness import Fitness

#: The vector is ordered by prior id: `centre`, `normalization`, `sigma` for each `Gaussian` in
#: turn, so `gaussian_0.centre` is index 0 and `gaussian_1.centre` is index 3.
SATISFYING = [20.0, 1.0, 1.0, 10.0, 1.0, 1.0]
VIOLATING = [10.0, 1.0, 1.0, 20.0, 1.0, 1.0]

#: |(0.4, 0.3)|^2 = 0.25 > |(0.05, 0.05)|^2 = 0.005 — the compound-prior ordering of assertion 2.
COMPOUND_SATISFYING = [0.4, 0.3, 1.0, 0.05, 0.05, 1.0]
COMPOUND_VIOLATING = [0.05, 0.05, 1.0, 0.4, 0.3, 1.0]

#: The finite sentinel Nautilus and dynesty pass instead of `-inf`, so a rejection is
#: distinguishable from a merely terrible likelihood.
RESAMPLE = -1.0e99


def _ordering_model():
    """Two Gaussians with the ordering assertion attached to the `Collection`."""
    model = af.Collection(
        gaussian_0=af.Model(af.ex.Gaussian),
        gaussian_1=af.Model(af.ex.Gaussian),
    )
    model.add_assertion(model.gaussian_0.centre > model.gaussian_1.centre)
    return model


def _compound_model():
    """
    Ordering between two *derived* quantities — the squared radius of a pair of priors per
    component. Routes through `SumPrior` and `PowerPrior`, which must realise their operands with
    the same `xp` as the rest of the trace.
    """
    model = af.Collection(
        gaussian_0=af.Model(af.ex.Gaussian),
        gaussian_1=af.Model(af.ex.Gaussian),
    )
    model.add_assertion(
        (model.gaussian_0.centre**2 + model.gaussian_0.normalization**2)
        > (model.gaussian_1.centre**2 + model.gaussian_1.normalization**2)
    )
    return model


def _child_model():
    """The assertion attached to `gaussian_0` before the `Collection` is composed."""
    gaussian_0 = af.Model(af.ex.Gaussian)
    gaussian_1 = af.Model(af.ex.Gaussian)
    gaussian_0.add_assertion(gaussian_0.centre > gaussian_1.centre)
    return af.Collection(gaussian_0=gaussian_0, gaussian_1=gaussian_1)


def _fitness(model, use_jax, **kwargs):
    data = np.ones(20)
    noise_map = np.ones(20) * 0.1
    return Fitness(
        model=model,
        analysis=af.ex.Analysis(data=data, noise_map=noise_map, use_jax=use_jax),
        resample_figure_of_merit=RESAMPLE,
        **kwargs,
    )


def _numpy_figure_of_merit(model, vector):
    """The value the numpy path returns for the same model and vector — the reference."""
    return _fitness(model, use_jax=False).call(vector)


def _sentinel(fitness):
    """
    The resample sentinel as JAX represents it. `resample_figure_of_merit` is a Python float, but
    what comes back from a JAX surface has been through JAX's dtype: comparing to the Python value
    would pin the dtype rather than the rejection, and fail wherever x64 is off.
    """
    return float(jnp.asarray(fitness.resample_figure_of_merit))


def assert_top_level_ordering_assertion_resamples_on_jax():
    """
    The non-vmapped JAX path. Before the traced penalty the violating vector raised
    `FitException` out of `instance_from_vector`: only the numpy branch of `call` has a
    `try/except`.
    """
    model = _ordering_model()
    fitness = _fitness(model, use_jax=True)

    assert fitness._is_jax is True
    assert fitness._apply_assertions_traced is True

    assert float(fitness.call(jnp.array(VIOLATING))) == _sentinel(fitness)

    figure_of_merit = float(fitness.call(jnp.array(SATISFYING)))
    assert figure_of_merit != _sentinel(fitness)
    assert np.isclose(
        figure_of_merit,
        _numpy_figure_of_merit(model, SATISFYING),
        rtol=0.0,
        atol=1.0e-8,
    )


def assert_compound_prior_ordering_between_two_bases():
    """
    The Euclid two-basis MGE shape (euclid_strong_lens_modeling_pipeline#54): an ordering between
    two sums of squared priors rather than between two bare priors.
    """
    model = _compound_model()
    reference = _numpy_figure_of_merit(model, COMPOUND_SATISFYING)

    # `SumPrior` / `PowerPrior` realise their operands themselves, so a tracer regression in them
    # would only show under a trace: run the eager, jit and vmap surfaces, not just the first.
    for kwargs in ({}, {"use_jax_jit": True}, {"use_jax_vmap": True}):
        fitness = _fitness(model, use_jax=True, **kwargs)

        assert fitness._apply_assertions_traced is True

        if kwargs.get("use_jax_vmap"):
            values = np.asarray(
                fitness._call(jnp.array([COMPOUND_VIOLATING, COMPOUND_SATISFYING]))
            )
        else:
            values = np.array(
                [
                    float(fitness._call(jnp.array(COMPOUND_VIOLATING))),
                    float(fitness._call(jnp.array(COMPOUND_SATISFYING))),
                ]
            )

        assert values[0] == _sentinel(fitness)
        assert np.isclose(values[1], reference, rtol=0.0, atol=1.0e-8)


def assert_jit_and_vmap_surfaces_compile_and_resample():
    """
    `_jit` and `_vmap` are the strict surfaces: under `vmap` every parameter is a tracer, so an
    assertion that kept any Python branch would fail there even where jit-on-concrete passed.
    `_call` is the dispatch `Fitness.__init__` selects.
    """
    model = _ordering_model()
    reference = _numpy_figure_of_merit(model, SATISFYING)

    jit_fitness = _fitness(model, use_jax=True, use_jax_jit=True)
    assert jit_fitness._call is jit_fitness._jit
    assert float(jit_fitness._call(jnp.array(VIOLATING))) == _sentinel(jit_fitness)
    assert np.isclose(
        float(jit_fitness._call(jnp.array(SATISFYING))),
        reference,
        rtol=0.0,
        atol=1.0e-8,
    )

    vmap_fitness = _fitness(model, use_jax=True, use_jax_vmap=True)
    assert vmap_fitness._call is vmap_fitness._vmap
    values = np.asarray(vmap_fitness._call(jnp.array([VIOLATING, SATISFYING])))
    assert values.shape == (2,)
    assert values[0] == _sentinel(vmap_fitness)
    assert np.isclose(values[1], reference, rtol=0.0, atol=1.0e-8)

    # `call` itself must be jit-able by a caller, not only through the `_jit` dispatch.
    fitness = _fitness(model, use_jax=True)
    jitted = jax.jit(fitness.call)
    assert float(jitted(jnp.asarray(VIOLATING))) == _sentinel(fitness)
    assert np.isclose(
        float(jitted(jnp.asarray(SATISFYING))), reference, rtol=0.0, atol=1.0e-8
    )


def assert_child_attached_assertion_is_gathered():
    """
    The trap this closes: an assertion attached to a child `Model` is enforced on numpy, because
    that child checks its own assertions as its instance is built. The traced path is handed only
    the top-level `Collection`, whose own `_assertions` is empty — it must gather from the tree.
    """
    model = _child_model()
    fitness = _fitness(model, use_jax=True, use_jax_vmap=True)

    assert model._assertions == []
    assert len(model.gathered_assertions()) == 1
    assert fitness._apply_assertions_traced is True

    values = np.asarray(fitness._call(jnp.array([VIOLATING, SATISFYING])))
    assert values[0] == _sentinel(fitness)
    assert np.isclose(
        values[1], _numpy_figure_of_merit(model, SATISFYING), rtol=0.0, atol=1.0e-8
    )


def assert_numpy_path_unchanged():
    """
    The traced penalty is JAX-only: on numpy the `FitException` raised by `instance_from_vector`
    is still what produces `resample_figure_of_merit`, and the `xp.where` must not run at all.
    """
    model = _ordering_model()
    fitness = _fitness(model, use_jax=False)

    assert fitness._is_jax is False
    assert fitness.call(VIOLATING) == fitness.resample_figure_of_merit

    figure_of_merit = fitness.call(SATISFYING)
    assert figure_of_merit != fitness.resample_figure_of_merit
    assert np.isfinite(figure_of_merit)

    # The exception itself is unchanged: only `Fitness` catches it, the model still raises.
    try:
        model.instance_from_vector(vector=VIOLATING)
    except af.exc.FitException:
        pass
    else:
        raise AssertionError("numpy instance_from_vector did not raise FitException")


if __name__ == "__main__":
    assert_top_level_ordering_assertion_resamples_on_jax()
    assert_compound_prior_ordering_between_two_bases()
    assert_jit_and_vmap_surfaces_compile_and_resample()
    assert_child_attached_assertion_is_gathered()
    assert_numpy_path_unchanged()
    print("assertions_traced: all assertions passed")
