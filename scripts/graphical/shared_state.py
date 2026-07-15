"""
Graphical: Shared Analysis State
================================

Fast assertion test for the `FactorGraphModel` cross-factor shared-state mechanism (PyAutoFit #1308).

The feature lets the per-factor `Analysis` objects of a `FactorGraphModel` compute a model-dependent object once per
likelihood evaluation (via `Analysis.shared_state_from`) and reuse it across every factor (via the `shared` argument of
`log_likelihood_function`), instead of each factor recomputing identical work. See the workspace tutorial
`autofit_workspace/scripts/features/shared_analysis_state.py`.

This script asserts the behaviour that matters, deterministically and fast:

 - the shared model data is computed ONCE per evaluation, not once per factor;
 - the shared-path likelihood EXACTLY equals the un-shared per-factor sum;
 - a graph whose analyses do not opt in is unchanged (each factor computes its own model data);
 - the mechanism survives a real (tiny) non-linear search end-to-end.

Run from the workspace root:

    python scripts/graphical/shared_state.py
"""

from os import path

import autofit as af


class CountingAnalysis(af.ex.Analysis):
    """
    An `af.ex.Analysis` that counts how many times the (notionally expensive) model data computation runs, so the test
    can prove the shared-state mechanism computes it once per evaluation rather than once per factor.
    """

    def __init__(self, data, noise_map, share_model_data=True):
        super().__init__(
            data=data, noise_map=noise_map, share_model_data=share_model_data
        )
        self.model_data_calls = 0

    def model_data_1d_from(self, instance):
        self.model_data_calls += 1
        return super().model_data_1d_from(instance=instance)


total_datasets = 2


def _datasets():
    data_list = []
    noise_map_list = []

    for dataset_index in range(total_datasets):
        dataset_path = path.join(
            "dataset", "example_1d", "gaussian_x1__low_snr", f"dataset_{dataset_index}"
        )

        data_list.append(
            af.util.numpy_array_from_json(
                file_path=path.join(dataset_path, "data.json")
            )
        )
        noise_map_list.append(
            af.util.numpy_array_from_json(
                file_path=path.join(dataset_path, "noise_map.json")
            )
        )

    return data_list, noise_map_list


def _shared_gaussian_graph(analyses):
    """
    Build a `FactorGraphModel` whose factors share the ENTIRE Gaussian model via shared prior objects, so the model data
    is identical for every factor and sharing it is valid.
    """
    centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    normalization = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
    sigma = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)

    analysis_factor_list = []
    for analysis in analyses:
        gaussian = af.Model(af.ex.Gaussian)
        gaussian.centre = centre
        gaussian.normalization = normalization
        gaussian.sigma = sigma
        analysis_factor_list.append(
            af.AnalysisFactor(prior_model=gaussian, analysis=analysis)
        )

    return af.FactorGraphModel(*analysis_factor_list)


def _instance(factor_graph):
    prior_count = factor_graph.global_prior_model.prior_count
    return factor_graph.global_prior_model.instance_from_unit_vector(
        [0.5] * prior_count
    )


def _reference_log_likelihood(factor_graph, instance):
    """Sum each factor's likelihood with no sharing (each factor computes its own model data)."""
    return sum(
        factor.analysis.log_likelihood_function(instance_)
        for factor, instance_ in zip(factor_graph.model_factors, instance)
    )


def assert_shared_state_computed_once_per_evaluation():
    data_list, noise_map_list = _datasets()
    analyses = [
        CountingAnalysis(data, noise_map)
        for data, noise_map in zip(data_list, noise_map_list)
    ]
    factor_graph = _shared_gaussian_graph(analyses)
    instance = _instance(factor_graph)

    factor_graph.log_likelihood_function(instance)

    total_calls = sum(analysis.model_data_calls for analysis in analyses)
    assert total_calls == 1, (
        f"Shared model data was computed {total_calls} times for {len(analyses)} factors; "
        f"expected exactly 1 (computed once on the lead factor and reused)."
    )


def assert_shared_likelihood_equals_unshared_sum():
    data_list, noise_map_list = _datasets()
    analyses = [
        CountingAnalysis(data, noise_map)
        for data, noise_map in zip(data_list, noise_map_list)
    ]
    factor_graph = _shared_gaussian_graph(analyses)
    instance = _instance(factor_graph)

    shared_log_likelihood = factor_graph.log_likelihood_function(instance)
    reference_log_likelihood = _reference_log_likelihood(factor_graph, instance)

    assert shared_log_likelihood == reference_log_likelihood, (
        f"Shared-path log likelihood {shared_log_likelihood} != un-shared per-factor sum "
        f"{reference_log_likelihood}. The shared object must give bit-identical results."
    )


def assert_no_provider_graph_is_unchanged():
    data_list, noise_map_list = _datasets()
    analyses = [
        CountingAnalysis(data, noise_map, share_model_data=False)
        for data, noise_map in zip(data_list, noise_map_list)
    ]
    factor_graph = _shared_gaussian_graph(analyses)
    instance = _instance(factor_graph)

    log_likelihood = factor_graph.log_likelihood_function(instance)
    reference_log_likelihood = _reference_log_likelihood(factor_graph, instance)

    total_calls = sum(analysis.model_data_calls for analysis in analyses)
    # One call per factor from the graph evaluation, plus one per factor from the reference sum:
    # the graph did NOT share, so it computed each factor's model data itself.
    assert total_calls == 2 * len(analyses), (
        f"With share_model_data=False the graph should compute each factor's model data itself "
        f"({2 * len(analyses)} total calls expected), but counted {total_calls}."
    )
    assert log_likelihood == reference_log_likelihood


def assert_search_runs_end_to_end():
    data_list, noise_map_list = _datasets()
    analyses = [
        af.ex.Analysis(data=data, noise_map=noise_map, share_model_data=True)
        for data, noise_map in zip(data_list, noise_map_list)
    ]
    factor_graph = _shared_gaussian_graph(analyses)

    search = af.DynestyStatic(
        path_prefix=path.join("graphical"),
        name="shared_state",
        nlive=50,
        maxcall=1000,
        maxiter=1000,
        number_of_cores=1,
    )

    result = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

    assert result.samples.max_log_likelihood_sample.log_likelihood is not None


if __name__ == "__main__":
    assert_shared_state_computed_once_per_evaluation()
    assert_shared_likelihood_equals_unshared_sum()
    assert_no_provider_graph_is_unchanged()
    assert_search_runs_end_to_end()
    print("shared_state: all assertions passed")
