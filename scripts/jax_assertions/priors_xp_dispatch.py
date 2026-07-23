"""
Priors `xp` Dispatch
====================

JAX-trace and NumPy↔JAX parity assertions for every concrete
``Prior`` subclass, the underlying ``NormalMessage`` /
``TruncatedNormalMessage`` ``value_for``, and the model-level
``Model.vector_from_unit_vector`` plumbing.

Phase 0 deliverable for the ``nss_first_class_sampler`` roadmap
(PyAutoFit issue #1262). Library policy is that unit tests stay
numpy-only — cross-xp checks live here in ``autofit_workspace_test``.

Run from the workspace root:

    python scripts/jax_assertions/priors_xp_dispatch.py
"""

"""
__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX assertion scripts test JAX behaviour; disabling JAX makes their
assertions vacuous.

ENV: jax
"""

import numpy as np

import jax
import jax.numpy as jnp

import autofit as af
from autofit.messages.normal import NormalMessage
from autofit.messages.truncated_normal import TruncatedNormalMessage
from autofit.mapper.prior_model.collection import Collection


UNITS = [0.05, 0.2, 0.5, 0.75, 0.95]


def assert_uniform_prior_value_for_jax_matches_numpy():
    prior = af.UniformPrior(lower_limit=0.5, upper_limit=2.5)
    for u in UNITS:
        np_v = prior.value_for(u, xp=np)
        jnp_v = float(prior.value_for(jnp.float64(u), xp=jnp))
        assert np.isclose(np_v, jnp_v, atol=1e-12), (u, np_v, jnp_v)


def assert_uniform_prior_value_for_jax_vector_input():
    prior = af.UniformPrior(lower_limit=-1.0, upper_limit=3.0)
    units = jnp.array([0.1, 0.4, 0.6, 0.9])
    out = prior.value_for(units, xp=jnp)
    expected = np.array([prior.value_for(float(u), xp=np) for u in units])
    np.testing.assert_allclose(np.asarray(out), expected, atol=1e-12)


def assert_uniform_prior_log_prior_bounds():
    prior = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    in_bounds = float(prior.log_prior_from_value(jnp.float64(0.5), xp=jnp))
    out_bounds = float(prior.log_prior_from_value(jnp.float64(1.5), xp=jnp))
    assert in_bounds == 0.0
    assert out_bounds == -np.inf


def assert_log_uniform_prior_value_for_jax_matches_numpy():
    prior = af.LogUniformPrior(lower_limit=0.01, upper_limit=100.0)
    for u in UNITS:
        np_v = prior.value_for(u, xp=np)
        jnp_v = float(prior.value_for(jnp.float64(u), xp=jnp))
        assert np.isclose(np_v, jnp_v, atol=1e-12), (u, np_v, jnp_v)


def assert_log_uniform_prior_log_prior_jax_matches_numpy():
    prior = af.LogUniformPrior(lower_limit=0.01, upper_limit=100.0)
    value = 1.5
    np_lp = prior.log_prior_from_value(value, xp=np)
    jnp_lp = float(prior.log_prior_from_value(jnp.float64(value), xp=jnp))
    assert np.isclose(np_lp, jnp_lp, atol=1e-12)


def assert_gaussian_prior_value_for_jax_matches_numpy():
    prior = af.GaussianPrior(mean=1.0, sigma=2.0)
    for u in UNITS:
        np_v = prior.value_for(u, xp=np)
        jnp_v = float(prior.value_for(jnp.float64(u), xp=jnp))
        assert np.isclose(np_v, jnp_v, atol=1e-10), (u, np_v, jnp_v)


def assert_gaussian_prior_value_for_unit_half_is_mean():
    prior = af.GaussianPrior(mean=3.7, sigma=0.5)
    out = float(prior.value_for(jnp.float64(0.5), xp=jnp))
    assert np.isclose(out, 3.7, atol=1e-12)


def assert_gaussian_prior_log_prior_jax_matches_numpy():
    prior = af.GaussianPrior(mean=1.0, sigma=2.0)
    value = 1.5
    np_lp = prior.log_prior_from_value(value, xp=np)
    jnp_lp = float(prior.log_prior_from_value(jnp.float64(value), xp=jnp))
    assert np.isclose(np_lp, jnp_lp, atol=1e-12)


def assert_truncated_gaussian_prior_value_for_jax_matches_numpy():
    prior = af.TruncatedGaussianPrior(
        mean=1.0, sigma=2.0, lower_limit=0.0, upper_limit=3.0
    )
    for u in [0.1, 0.3, 0.5, 0.7, 0.9]:
        np_v = prior.value_for(u, xp=np)
        jnp_v = float(prior.value_for(jnp.float64(u), xp=jnp))
        assert np.isclose(np_v, jnp_v, atol=1e-8), (u, np_v, jnp_v)


def assert_truncated_gaussian_prior_log_prior_in_bounds_jax_matches_numpy():
    prior = af.TruncatedGaussianPrior(
        mean=1.0, sigma=2.0, lower_limit=0.0, upper_limit=3.0
    )
    value = 1.5
    np_lp = prior.log_prior_from_value(value, xp=np)
    jnp_lp = float(prior.log_prior_from_value(jnp.float64(value), xp=jnp))
    assert np.isclose(np_lp, jnp_lp, atol=1e-10)


def assert_truncated_gaussian_prior_log_prior_out_of_bounds_neg_inf():
    prior = af.TruncatedGaussianPrior(
        mean=1.0, sigma=2.0, lower_limit=0.0, upper_limit=3.0
    )
    out = prior.log_prior_from_value(jnp.float64(5.0), xp=jnp)
    assert float(out) == -np.inf


def assert_log_gaussian_prior_value_for_jax_matches_numpy():
    prior = af.LogGaussianPrior(mean=0.0, sigma=1.0)
    for u in UNITS:
        np_v = prior.value_for(u, xp=np)
        jnp_v = float(prior.value_for(jnp.float64(u), xp=jnp))
        assert np.isclose(np_v, jnp_v, atol=1e-10), (u, np_v, jnp_v)


def assert_log_gaussian_prior_log_prior_jax_matches_numpy():
    prior = af.LogGaussianPrior(mean=0.0, sigma=1.0)
    value = 2.0
    np_lp = prior.log_prior_from_value(value, xp=np)
    jnp_lp = float(prior.log_prior_from_value(jnp.float64(value), xp=jnp))
    assert np.isclose(np_lp, jnp_lp, atol=1e-12)


def assert_normal_message_value_for_xp_dispatch():
    message = NormalMessage(mean=1.5, sigma=0.7)
    for u in [0.1, 0.3, 0.5, 0.7, 0.9]:
        np_v = message.value_for(u, xp=np)
        jnp_v = float(message.value_for(jnp.float64(u), xp=jnp))
        assert np.isclose(np_v, jnp_v, atol=1e-10), (u, np_v, jnp_v)


def assert_normal_message_default_xp_is_numpy():
    message = NormalMessage(mean=0.0, sigma=1.0)
    out = message.value_for(0.5)
    assert isinstance(out, float)
    assert np.isclose(out, 0.0, atol=1e-12)


def assert_truncated_normal_message_value_for_xp_dispatch():
    message = TruncatedNormalMessage(
        mean=1.0, sigma=2.0, lower_limit=0.0, upper_limit=3.0
    )
    for u in [0.1, 0.3, 0.5, 0.7, 0.9]:
        np_v = message.value_for(u, xp=np)
        jnp_v = float(message.value_for(jnp.float64(u), xp=jnp))
        assert np.isclose(np_v, jnp_v, atol=1e-8), (u, np_v, jnp_v)


def _five_prior_model():
    return Collection(
        a=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
        b=af.LogUniformPrior(lower_limit=0.01, upper_limit=100.0),
        c=af.GaussianPrior(mean=0.0, sigma=1.0),
        d=af.TruncatedGaussianPrior(
            mean=0.0, sigma=1.0, lower_limit=-2.0, upper_limit=2.0
        ),
        e=af.LogGaussianPrior(mean=0.0, sigma=0.5),
    )


def assert_vector_from_unit_vector_numpy_returns_list():
    model = _five_prior_model()
    unit = [0.5] * model.prior_count
    out = model.vector_from_unit_vector(unit)
    assert isinstance(out, list)
    assert len(out) == model.prior_count
    for v in out:
        assert isinstance(v, float)


def assert_vector_from_unit_vector_jax_returns_jax_array():
    model = _five_prior_model()
    unit = jnp.array([0.2, 0.3, 0.4, 0.5, 0.6])
    out = model.vector_from_unit_vector(unit, xp=jnp)
    assert hasattr(out, "shape")
    assert out.shape == (model.prior_count,)


def assert_vector_from_unit_vector_jax_matches_numpy():
    model = _five_prior_model()
    units = [0.1, 0.4, 0.5, 0.6, 0.8]
    np_out = np.asarray(model.vector_from_unit_vector(units, xp=np))
    jnp_out = np.asarray(model.vector_from_unit_vector(jnp.array(units), xp=jnp))
    np.testing.assert_allclose(np_out, jnp_out, atol=1e-8)


def assert_vector_from_unit_vector_jit_traceable():
    model = _five_prior_model()

    @jax.jit
    def cube_to_phys(unit):
        return model.vector_from_unit_vector(unit, xp=jnp)

    out = cube_to_phys(jnp.array([0.5] * model.prior_count))
    assert out.shape == (model.prior_count,)
    assert bool(jnp.all(jnp.isfinite(out)))


def assert_log_prior_list_from_vector_jit_traceable():
    model = _five_prior_model()

    @jax.jit
    def log_priors(unit):
        physical = model.vector_from_unit_vector(unit, xp=jnp)
        return sum(model.log_prior_list_from_vector(vector=physical, xp=jnp))

    out = float(log_priors(jnp.array([0.4, 0.5, 0.5, 0.5, 0.5])))
    assert np.isfinite(out)


def assert_uniform_prior_value_for_jit_traceable():
    prior = af.UniformPrior(lower_limit=-1.0, upper_limit=4.0)

    @jax.jit
    def f(u):
        return prior.value_for(u, xp=jnp)

    out = float(f(jnp.float64(0.3)))
    assert np.isclose(out, prior.value_for(0.3, xp=np), atol=1e-12)


def assert_gaussian_prior_value_for_jit_traceable():
    prior = af.GaussianPrior(mean=2.0, sigma=0.5)

    @jax.jit
    def f(u):
        return prior.value_for(u, xp=jnp)

    out = float(f(jnp.float64(0.7)))
    assert np.isclose(out, prior.value_for(0.7, xp=np), atol=1e-10)


def assert_gaussian_prior_value_for_jax_grad_finite():
    prior = af.GaussianPrior(mean=0.0, sigma=1.0)

    @jax.jit
    def f(u):
        return prior.value_for(u, xp=jnp)

    g = float(jax.grad(f)(jnp.float64(0.4)))
    assert np.isfinite(g)


def assert_log_prior_density_convention_gaussian():
    """``GaussianPrior.log_prior_from_value`` returns ``log p(x)`` in density form.

    Maximum at value == mean (returns 0; the ``-log(sigma * sqrt(2 pi))`` constant
    is dropped), negative elsewhere. Regression gate for PyAutoLabs/PyAutoFit#1266 —
    cost-form bodies returned a positive quadratic and biased every MCMC fit
    with a non-uniform prior.
    """
    prior = af.GaussianPrior(mean=5.0, sigma=1.0)

    at_mean = prior.log_prior_from_value(value=5.0)
    one_sigma = prior.log_prior_from_value(value=6.0)
    three_sigma = prior.log_prior_from_value(value=8.0)

    assert (
        at_mean == 0.0
    ), f"log_prior at mean must be 0.0 (density form, constant dropped); got {at_mean!r}"
    assert one_sigma == -0.5, (
        f"log_prior 1σ from mean must be -0.5 (density form); got {one_sigma!r}. "
        f"Positive value indicates cost-form regression."
    )
    assert three_sigma == -4.5, (
        f"log_prior 3σ from mean must be -4.5 (density form); got {three_sigma!r}. "
        f"Positive value indicates cost-form regression."
    )


def assert_log_prior_density_convention_log_uniform():
    """``LogUniformPrior.log_prior_from_value`` returns ``-log(value)`` in density
    form, with the ``-log(log(b/a))`` normalisation constant dropped (matches
    ``UniformPrior`` dropping ``-log(b - a)`` to return 0.0).
    """
    prior = af.LogUniformPrior(lower_limit=0.01, upper_limit=100.0)

    at_one = prior.log_prior_from_value(value=1.0)
    at_e = prior.log_prior_from_value(value=np.e)

    assert at_one == 0.0, (
        f"log_prior at value=1.0 must be 0.0 (density form, -log(1)=0); got {at_one!r}. "
        f"Returning 1.0 indicates the pre-#1266 1/value Jacobian-gradient bug."
    )
    assert np.isclose(
        at_e, -1.0, atol=1e-12
    ), f"log_prior at value=e must be -1.0 (density form, -log(e)=-1); got {at_e!r}"


def assert_log_prior_density_convention_log_gaussian():
    """``LogGaussianPrior.log_prior_from_value`` returns the density-form
    ``-(log(value) - mean)**2 / (2 sigma**2) - log(value)``, with the Jacobian
    of the log-space transform contributing the ``-log(value)`` term.
    """
    prior = af.LogGaussianPrior(mean=0.0, sigma=1.0)

    out_of_support = prior.log_prior_from_value(value=0.0)
    assert out_of_support == float(
        "-inf"
    ), f"log_prior at value=0 must be -inf (out of support); got {out_of_support!r}"

    # At value=1 (i.e. log(value)=0=mean), the quadratic is zero, so the result
    # is just the Jacobian -log(1) = 0.
    at_one = prior.log_prior_from_value(value=1.0)
    assert at_one == 0.0, (
        f"log_prior at value=1 must be 0.0 (density form: quadratic=0, Jacobian=0); "
        f"got {at_one!r}"
    )


def assert_log_prior_density_jax_matches_numpy_signed():
    """Tighten the existing NumPy↔JAX parity gates with sign-direction checks.

    Confirms the JAX path returns a value with the same sign as the NumPy path
    at a point known to be away from the prior mode — guards against a partial
    sign-flip regression where one xp path is fixed but the other isn't.
    """
    prior = af.GaussianPrior(mean=0.0, sigma=1.0)
    np_lp = prior.log_prior_from_value(value=2.0, xp=np)
    jnp_lp = float(prior.log_prior_from_value(value=jnp.float64(2.0), xp=jnp))

    assert (
        np_lp < 0.0
    ), f"NumPy density-form log_prior must be < 0 at |z|=2σ; got {np_lp}"
    assert (
        jnp_lp < 0.0
    ), f"JAX density-form log_prior must be < 0 at |z|=2σ; got {jnp_lp}"
    assert np.isclose(np_lp, jnp_lp, atol=1e-12)


if __name__ == "__main__":
    assert_uniform_prior_value_for_jax_matches_numpy()
    assert_uniform_prior_value_for_jax_vector_input()
    assert_uniform_prior_log_prior_bounds()
    assert_log_uniform_prior_value_for_jax_matches_numpy()
    assert_log_uniform_prior_log_prior_jax_matches_numpy()
    assert_gaussian_prior_value_for_jax_matches_numpy()
    assert_gaussian_prior_value_for_unit_half_is_mean()
    assert_gaussian_prior_log_prior_jax_matches_numpy()
    assert_truncated_gaussian_prior_value_for_jax_matches_numpy()
    assert_truncated_gaussian_prior_log_prior_in_bounds_jax_matches_numpy()
    assert_truncated_gaussian_prior_log_prior_out_of_bounds_neg_inf()
    assert_log_gaussian_prior_value_for_jax_matches_numpy()
    assert_log_gaussian_prior_log_prior_jax_matches_numpy()
    assert_normal_message_value_for_xp_dispatch()
    assert_normal_message_default_xp_is_numpy()
    assert_truncated_normal_message_value_for_xp_dispatch()
    assert_vector_from_unit_vector_numpy_returns_list()
    assert_vector_from_unit_vector_jax_returns_jax_array()
    assert_vector_from_unit_vector_jax_matches_numpy()
    assert_vector_from_unit_vector_jit_traceable()
    assert_log_prior_list_from_vector_jit_traceable()
    assert_uniform_prior_value_for_jit_traceable()
    assert_gaussian_prior_value_for_jit_traceable()
    assert_gaussian_prior_value_for_jax_grad_finite()
    assert_log_prior_density_convention_gaussian()
    assert_log_prior_density_convention_log_uniform()
    assert_log_prior_density_convention_log_gaussian()
    assert_log_prior_density_jax_matches_numpy_signed()
    print("priors_xp_dispatch: all assertions passed")
