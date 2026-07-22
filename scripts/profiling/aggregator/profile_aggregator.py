"""
Profiling: Aggregator Loading Pathways
======================================

Times each aggregator loading pathway separately over a grid of mock-result sets
(varying number of results, samples per result and model size one axis at a time),
so it is explicit which axis drives poor scaling.

Stages timed per grid cell:

- `generate`: building the mock results (not an aggregator cost; reported for context).
- `from_directory`: `af.Aggregator.from_directory` scan.
- `iterate`: iterating every `SearchOutput` (no file loads).
- `values_samples_summary`: `values("samples_summary")` — the lightweight summary path.
- `values_model`: `values("model")`.
- `values_samples`: `values("samples")` — full `samples.csv` parse per result.
- `query_dataset_name`: a metadata predicate query.
- `aggregate_csv`: `af.AggregateCSV` catalogue build (the csv_make workflow pattern).

Run from the `autofit_workspace_test` root, e.g.:

    python scripts/profiling/aggregator/profile_aggregator.py --quick

Results print as a table and are written as JSON under
`output/profiling_aggregator/results/`.
"""

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from enum import Enum

import autofit as af
from autofit.aggregator.aggregator import Aggregator
from mock_results import generate_mock_results


class SubplotFit(Enum):
    """
    Panel coordinates in the mock subplot_fit.png (4x3 grid) — the png_make pattern.
    """

    Data = (0, 0)
    ModelData = (3, 0)
    ResidualMap = (0, 2)
    ChiSquaredMap = (2, 2)


class FITSFit(Enum):
    """
    HDU EXTNAMEs in the mock fit.fits — the fits_make pattern.
    """

    ModelData = "MODEL_IMAGE"
    ResidualMap = "RESIDUAL_MAP"


RESULTS_PATH = Path("output") / "profiling_aggregator" / "results"

# values("samples") holds every loaded Samples in memory (SearchOutput caches them),
# which OOMs at science scale — 3000 results x 1000 samples was killed at ~6.6 GB RSS.
# The samples stage is therefore timed on a capped slice; per-result cost extrapolates.
SAMPLES_STAGE_CAP = 500

BASE = {"n_results": 100, "n_samples": 1000, "n_gaussians": 5}
AXES = {
    "n_results": [10, 100, 1000],
    "n_samples": [100, 1000, 10000],
    "n_gaussians": [1, 5, 15],
}
AXES_QUICK = {
    "n_results": [10, 50],
    "n_samples": [100, 1000],
    "n_gaussians": [1, 5],
}


def grid_cells(axes: dict, base: dict) -> list:
    """
    One cell per axis value, holding the other axes at the base configuration.
    """
    cells = []
    for axis, values in axes.items():
        for value in values:
            cell = dict(base)
            cell[axis] = value
            if cell not in cells:
                cells.append(cell)
    return cells


def timed(func) -> float:
    """
    Wall-time a callable, swallowing its stdout (the aggregator prints progress
    messages which would drown the table).
    """
    with contextlib.redirect_stdout(io.StringIO()):
        start = time.perf_counter()
        func()
        return time.perf_counter() - start


def profile_cell(cell: dict, zip_results: bool, keep: bool) -> dict:
    timings = {}

    generate_start = time.perf_counter()
    results_root = generate_mock_results(zip_results=zip_results, **cell)
    timings["generate"] = time.perf_counter() - generate_start

    def fresh_agg():
        """
        A new Aggregator, so each stage is timed cold — SearchOutput caches model
        and samples on the instance, which would otherwise let later stages ride
        earlier stages' loads.
        """
        with contextlib.redirect_stdout(io.StringIO()):
            agg = Aggregator.from_directory(results_root)
        assert (
            len(agg) == cell["n_results"]
        ), f"Aggregator found {len(agg)} results, expected {cell['n_results']}"
        return agg

    # Warm up one-off costs (imports triggered by the first summary/search load) so
    # they don't pollute the first timed stage of the first cell.
    warm = fresh_agg()[0]
    _ = warm.samples_summary, warm.samples

    timings["from_directory"] = timed(lambda: Aggregator.from_directory(results_root))

    agg = fresh_agg()
    timings["iterate"] = timed(lambda: list(agg))

    agg = fresh_agg()
    timings["values_samples_summary"] = timed(
        lambda: list(agg.values("samples_summary"))
    )

    agg = fresh_agg()
    timings["values_model"] = timed(lambda: list(agg.values("model")))

    with contextlib.redirect_stdout(io.StringIO()):
        agg = fresh_agg()[:SAMPLES_STAGE_CAP]
    samples_stage_results = len(agg)
    timings["values_samples"] = timed(lambda: deque(agg.values("samples"), maxlen=0))

    agg = fresh_agg()
    timings["query_dataset_name"] = timed(
        lambda: agg.query(agg.dataset_name == "dataset_0000")
    )

    agg = fresh_agg()

    def aggregate_csv():
        agg_csv = af.AggregateCSV(agg)
        agg_csv.add_variable("gaussian_0.centre")
        agg_csv.add_variable("gaussian_0.sigma")
        agg_csv.save(results_root.parent / "aggregate.csv")

    timings["aggregate_csv"] = timed(aggregate_csv)

    agg = fresh_agg()

    def aggregate_images():
        agg_images = af.AggregateImages(agg)
        agg_images.output_to_folder(
            results_root.parent / "png",
            name="dataset_name",
            subplots=[
                SubplotFit.Data,
                SubplotFit.ModelData,
                SubplotFit.ResidualMap,
                SubplotFit.ChiSquaredMap,
            ],
        )

    timings["aggregate_images"] = timed(aggregate_images)

    agg = fresh_agg()

    def aggregate_fits():
        agg_fits = af.AggregateFITS(agg)
        agg_fits.output_to_folder(
            results_root.parent / "fits",
            name="dataset_name",
            hdus=[FITSFit.ModelData, FITSFit.ResidualMap],
        )

    timings["aggregate_fits"] = timed(aggregate_fits)

    if not keep:
        shutil.rmtree(results_root.parent)

    return {
        **cell,
        "zip": zip_results,
        "samples_stage_results": samples_stage_results,
        "timings": timings,
    }


def print_table(rows: list):
    stages = list(rows[0]["timings"])
    header = f"{'n_results':>9} {'n_samples':>9} {'n_gauss':>7} " + " ".join(
        f"{stage:>22}" for stage in stages
    )
    print("\nTimings in seconds (per-result milliseconds in brackets):\n")
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = []
        for stage in stages:
            seconds = row["timings"][stage]
            n = (
                row["samples_stage_results"]
                if stage == "values_samples"
                else row["n_results"]
            )
            per_result_ms = 1000.0 * seconds / n
            cells.append(f"{seconds:>12.3f} ({per_result_ms:>6.2f})")
        print(
            f"{row['n_results']:>9} {row['n_samples']:>9} {row['n_gaussians']:>7} "
            + " ".join(cells)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small grid, fast run")
    parser.add_argument("--n-results-max", type=int, default=None)
    parser.add_argument("--zip", action="store_true", dest="zip_results")
    parser.add_argument("--keep", action="store_true", help="keep mock results on disk")
    parser.add_argument("--label", type=str, default="", help="tag for the JSON output")
    parser.add_argument(
        "--representative",
        action="store_true",
        help="add the production-lens-scale cell (10k samples x 18-param model)",
    )
    args = parser.parse_args()

    axes = dict(AXES_QUICK if args.quick else AXES)
    base = dict(BASE)
    if args.quick:
        base = {"n_results": 50, "n_samples": 1000, "n_gaussians": 5}
    if os.environ.get("PYAUTO_TEST_MODE"):
        # Validation sweeps (run_all_scripts.sh) only smoke-check that the harness
        # works — profile a single tiny cell instead of a real grid.
        axes = {"n_results": [5]}
        base = {"n_results": 5, "n_samples": 50, "n_gaussians": 1}
    if args.n_results_max is not None:
        axes["n_results"] = axes["n_results"] + [args.n_results_max]

    cells = grid_cells(axes=axes, base=base)
    if args.representative and not os.environ.get("PYAUTO_TEST_MODE"):
        # ~10k rows x 21 columns (~9 MB samples.csv) — the PYAUTO_TEST_MODE_SAMPLES
        # production-parity target (PyAutoFit#1378).
        cells.append({"n_results": 25, "n_samples": 10000, "n_gaussians": 6})

    rows = []
    for cell in cells:
        print(f"profiling {cell} ...", flush=True)
        rows.append(profile_cell(cell, zip_results=args.zip_results, keep=args.keep))

    print_table(rows)

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    label = f"_{args.label}" if args.label else ""
    output_file = (
        RESULTS_PATH / f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}{label}.json"
    )
    output_file.write_text(
        json.dumps(
            {"autofit_version": af.__version__, "rows": rows},
            indent=2,
        )
    )
    print(f"\nJSON written to {output_file}")
