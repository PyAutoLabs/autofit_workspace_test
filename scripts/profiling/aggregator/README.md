# Aggregator profiling

Tools to measure where `af.Aggregator` result-loading scales poorly (number of
results vs samples per result vs model size), using mock output folders written
through the real `DirectoryPaths` machinery — no sampler is ever run, so thousands
of results are fabricated in seconds.

- `mock_results.py` — generate a mock result set (`--n-results/--n-samples/--n-gaussians/--zip`).
- `profile_aggregator.py` — time each loading pathway over a one-axis-at-a-time grid
  (`--quick` for a fast pass, `--n-results-max 3000` for the science-scale case,
  `--label before` to tag the JSON) and write a table + JSON under
  `output/profiling_aggregator/results/`.

Run both from the `autofit_workspace_test` root. Everything lands under `output/`
(gitignored). These scripts are profiling tools, not smoke tests — do not add them
to `smoke_tests.txt`.
