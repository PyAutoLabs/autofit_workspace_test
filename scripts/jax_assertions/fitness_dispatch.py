"""
JAX Assertions
==============

Integration-test script collecting PyAutoFit assertions that genuinely require
``jax``. These were moved out of the ``test_autofit/`` unit test suite as part
of making ``import autofit`` work cleanly on Pythons that lack the ``[jax]``
extra (3.9 / 3.10). The library policy is that all numerical code paths must
support pure numpy; assertions that exercise jax-only behaviour live here.

Run from the workspace root:

    python scripts/graphical/jax_assertions.py
"""

import pickle

import numpy as np
import jax.numpy as jnp

import autofit as af
from autofit.non_linear.fitness import Fitness


def _make_fitness(**kwargs):
    model = af.Model(af.ex.Gaussian)
    data = np.ones(20)
    noise_map = np.ones(20) * 0.1
    analysis = af.ex.Analysis(data=data, noise_map=noise_map)
    return Fitness(model=model, analysis=analysis, **kwargs)


def assert_jit_dispatch_sets_call_to_jit():
    fitness = _make_fitness(use_jax_jit=True)
    assert fitness.use_jax_jit is True
    assert fitness._call is fitness._jit


def assert_vmap_takes_precedence_over_jit():
    fitness = _make_fitness(use_jax_jit=True, use_jax_vmap=True)
    assert fitness._call is fitness._vmap


def assert_pickle_strips_jax_cached_attrs():
    """Dynesty's checkpoint writes pickle the loglikelihood. JIT/vmap callables
    carry C++ XLA state that cannot roundtrip through pickle.
    ``Fitness.__getstate__`` must drop them; ``Fitness.__setstate__`` re-derives
    the dispatch on resume."""
    fitness = _make_fitness(use_jax_jit=True)

    state = fitness.__getstate__()
    assert "_call" not in state
    assert "_jit" not in state
    assert "_vmap" not in state
    assert "_grad" not in state

    blob = pickle.dumps(fitness)
    restored = pickle.loads(blob)

    assert restored.use_jax_jit is True
    assert restored._call is restored._jit


class _JitFittableAnalysis(af.Analysis):
    """Analysis with a ``fit_from`` returning a JIT-traceable array."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fit_from_calls = 0

    def log_likelihood_function(self, instance):
        return 0.0

    def fit_from(self, instance):
        self.fit_from_calls += 1
        return jnp.asarray(instance) * 2.0


def assert_fit_for_visualization_follows_use_jax_contract():
    analysis = _JitFittableAnalysis(use_jax=True)

    result_1 = analysis.fit_for_visualization(instance=1.0)
    assert jnp.allclose(result_1, jnp.asarray(2.0))
    assert analysis.supports_jax_visualization is True

    result_2 = analysis.fit_for_visualization(instance=3.0)
    assert jnp.allclose(result_2, jnp.asarray(6.0))


def assert_use_jax_true_enables_jax_visualization():
    analysis = af.Analysis(use_jax=True)
    assert analysis._use_jax is True
    assert analysis.supports_jax_visualization is True


def assert_use_jax_false_keeps_visualization_numpy_safe():
    analysis = af.Analysis(use_jax=False)
    assert analysis._use_jax is False
    assert analysis.supports_jax_visualization is False


class _ArrayAnalysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return -float(
            np.mean(
                (
                    np.array(
                        [
                            [0.1, 0.2],
                            [0.3, 0.4],
                        ]
                    )
                    - instance
                )
                ** 2
            )
        )


def assert_array_optimisation_returns_jnp_instance():
    """``af.Array`` end-to-end optimisation. The original assertion required
    the resulting instance to be a ``jnp.ndarray``; with the post-2026-05
    polarity fix in ``autofit/mapper/prior_model/array.py``, plain Python
    floats route to numpy, so a default search produces an ``np.ndarray``.
    This assertion accepts either type — the integration check is that the
    fit completes and returns an array of the right shape."""
    array = af.Array(
        shape=(2, 2),
        prior=af.UniformPrior(
            lower_limit=0.0,
            upper_limit=1.0,
        ),
    )
    result = af.DynestyStatic().fit(model=array, analysis=_ArrayAnalysis())

    posterior = result.model
    array[0, 0] = posterior[0, 0]
    array[0, 1] = posterior[0, 1]

    result = af.DynestyStatic().fit(model=array, analysis=_ArrayAnalysis())

    assert isinstance(result.instance, (np.ndarray, jnp.ndarray))
    assert result.instance.shape == (2, 2)


if __name__ == "__main__":
    assert_jit_dispatch_sets_call_to_jit()
    assert_vmap_takes_precedence_over_jit()
    assert_pickle_strips_jax_cached_attrs()
    assert_fit_for_visualization_follows_use_jax_contract()
    assert_use_jax_true_enables_jax_visualization()
    assert_use_jax_false_keeps_visualization_numpy_safe()
    assert_array_optimisation_returns_jnp_instance()
    print("fitness_dispatch: all assertions passed")
