"""
JAX Assertions — Fitness's NaN guard contract
=============================================

Pins what ``Fitness``'s NaN/inf resample guard (``autofit/non_linear/fitness.py``,
in ``call``) actually promises: it protects the **value**, never the **gradient**.

Why this exists (PyAutoFit#1391, diagnosed in autolens_workspace_developer#104):
the guard maps a NaN/inf likelihood to ``resample_figure_of_merit``, so searches
that read only the figure of merit never select the point. Gradient consumers get
nothing: ``jax.grad`` differentiates *both* branches of an ``xp.where`` and
multiplies the unselected one by zero, so a non-finite derivative propagates as
``0 * NaN = NaN``. The value looks handled; the gradient is poison.

These assertions exist to stop that limitation silently regressing into a false
promise — and to record the disproof of the fix that looks obvious but does not
work, so nobody re-prescribes it. An output-side "double-where" **cannot** help:
by the time ``Fitness.call`` sees ``log_likelihood`` the non-finite derivative is
already on the tape. Only refusing to *evaluate* the offending op at the invalid
input works, and that decision lives at the NaN's source (for the pixelized
likelihood, autoarray's Cholesky), not in ``Fitness``.

These are jax-only assertions, so they live here rather than in ``test_autofit/``
(the library's unit tests are numpy-only).

Run from the workspace root:

    python scripts/jax_assertions/fitness_nan_gradient_contract.py
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

RESAMPLE = -jnp.inf


def _nan_derivative_likelihood(x):
    """NaN in value AND derivative for x < 0 -- the shape of a real failure.

    ``sqrt`` stands in for the site that actually bites in production: a Cholesky
    of a non-positive-definite curvature/regularization matrix, which JAX returns
    as NaN rather than raising (autolens_workspace_developer#104).
    """
    return jnp.sqrt(x)


def _finite_derivative_likelihood(x):
    """NaN in value but FINITE derivative (d/dx log = 1/x) for x < 0."""
    return jnp.log(x)


def _guarded(likelihood):
    """The guard as `Fitness.call` applies it (value path only)."""

    def call(x):
        ll = likelihood(x)
        ll = jnp.where(jnp.isnan(ll), RESAMPLE, ll)
        ll = jnp.where(jnp.isinf(ll), RESAMPLE, ll)
        return ll * -2.0  # convert_to_chi_squared

    return call


def assert_guard_repairs_the_value():
    """The guard delivers what it advertises: no NaN reaches the search."""
    call = _guarded(_nan_derivative_likelihood)
    fom = call(-1.0)
    assert not jnp.isnan(fom), "the guard must never let a NaN value through"
    assert jnp.isinf(fom), "an invalid model's figure of merit is the resample sentinel"


def assert_guard_does_not_repair_the_gradient():
    """The contract's limit: a guarded point still yields a NaN gradient.

    This assertion is deliberately pinning *broken-looking* behaviour. If it ever
    fails, the guard's reach has changed and the docstring in `Fitness.call` (plus
    every caller relying on "resampled points are safe") must be revisited.
    """
    call = _guarded(_nan_derivative_likelihood)
    grad = jax.grad(call)(-1.0)
    assert jnp.isnan(grad), (
        "expected the where-guard to leak a NaN gradient (0 * NaN); if this now "
        "passes, the value-only contract has changed -- update fitness.py's docstring"
    )


def assert_gradient_is_fine_when_the_masked_derivative_is_finite():
    """The trap is narrower than 'the guard never protects gradients'.

    It fires only when the masked branch's DERIVATIVE is non-finite. `log(x<0)` is
    NaN in value but its derivative 1/x is finite, so `0 * -1 = 0` and nothing
    leaks. Recording this stops the limitation being overstated.
    """
    call = _guarded(_finite_derivative_likelihood)
    grad = jax.grad(call)(-1.0)
    assert jnp.isfinite(
        grad
    ), "a finite masked derivative must not trigger the 0 * NaN trap"


def assert_output_side_double_where_does_not_fix_it():
    """The disproof. Sanitising the OUTPUT is too late -- the tape already has NaN."""

    def call(x):
        ll_raw = _nan_derivative_likelihood(x)
        bad = jnp.isnan(ll_raw) | jnp.isinf(ll_raw)
        safe = jnp.where(bad, 0.0, ll_raw)  # sanitise after the likelihood ran
        return jnp.where(bad, RESAMPLE, safe) * -2.0

    grad = jax.grad(call)(-1.0)
    assert jnp.isnan(grad), (
        "an output-side double-where must NOT be presented as the remedy: the "
        "non-finite derivative is already on the tape by then"
    )


def assert_input_side_safe_x_does_fix_it():
    """What a real fix looks like -- and why it cannot live in `Fitness`.

    The likelihood is never *evaluated* at the invalid input. That requires knowing
    which input is invalid, which is knowable only at the NaN's source, not at the
    Fitness boundary where the likelihood is an opaque already-computed scalar.
    """

    def call(x):
        bad = x < 0
        safe_x = jnp.where(bad, 1.0, x)  # sanitise the INPUT
        ll = _nan_derivative_likelihood(safe_x)
        return jnp.where(bad, RESAMPLE, ll) * -2.0

    grad = jax.grad(call)(-1.0)
    assert jnp.isfinite(grad), "input-side safe-x must yield a finite gradient"
    assert grad == 0.0, "an invalid point carries no gradient information"


def assert_valid_points_are_unaffected():
    """None of the above perturbs a well-behaved model."""
    call = _guarded(_nan_derivative_likelihood)
    assert jnp.isfinite(call(4.0))
    assert jnp.isfinite(jax.grad(call)(4.0))


if __name__ == "__main__":
    assert_guard_repairs_the_value()
    assert_guard_does_not_repair_the_gradient()
    assert_gradient_is_fine_when_the_masked_derivative_is_finite()
    assert_output_side_double_where_does_not_fix_it()
    assert_input_side_safe_x_does_fix_it()
    assert_valid_points_are_unaffected()
    print("fitness_nan_gradient_contract: all assertions passed")
