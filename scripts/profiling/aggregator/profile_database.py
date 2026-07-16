"""
Profiling: SQLite Database Build + Query
========================================

Times the sqlite results-database pathways (`Aggregator.from_database` +
`add_directory` build, then queries and sample loading) against the same mock result
sets as `profile_aggregator.py`, with the directory `Aggregator` timed on identical
data for comparison — answering "at what scale is building the database worth it?".

Stages timed per grid cell:

- `generate`: building the mock results (context, not a database cost).
- `database_build`: `from_database` + `add_directory` + commit on a fresh sqlite file.
- `database_query_unique_tag`: one indexed metadata query.
- `database_values_samples`: `values("samples")` on a capped slice.
- `directory_from_directory` / `directory_values_samples`: the directory Aggregator on
  the same results, same cap, for side-by-side comparison.

Run from the `autofit_workspace_test` root:

    python scripts/profiling/aggregator/profile_database.py --quick

Results print as a table and are written as JSON under
`output/profiling_aggregator/results/`.
"""

import argparse
import contextlib
import io
import json
import logging
import os
import shutil
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import autofit as af
from autofit.aggregator.aggregator import Aggregator as DirectoryAggregator
from autofit.database.aggregator import Aggregator as DatabaseAggregator
from mock_results import generate_mock_results
from profile_aggregator import RESULTS_PATH, SAMPLES_STAGE_CAP, timed

BASE = {"n_results": 100, "n_samples": 1000, "n_gaussians": 5}
AXES = {
    "n_results": [10, 100, 1000],
    "n_samples": [100, 1000, 10000],
}
AXES_QUICK = {
    "n_results": [10, 50],
    "n_samples": [100, 1000],
}


def grid_cells(axes: dict, base: dict) -> list:
    cells = []
    for axis, values in axes.items():
        for value in values:
            cell = dict(base)
            cell[axis] = value
            if cell not in cells:
                cells.append(cell)
    return cells


def profile_cell(cell: dict, keep: bool) -> dict:
    timings = {}

    generate_start = time.perf_counter()
    results_root = generate_mock_results(**cell)
    timings["generate"] = time.perf_counter() - generate_start

    # An absolute path sidesteps open_database prefixing conf.instance.output_path
    # (which generate_mock_results pushes) onto relative sqlite filenames.
    database_path = (results_root.parent / f"{results_root.name}.sqlite").resolve()
    database_path.unlink(missing_ok=True)

    aggregator = {}

    def database_build():
        agg = DatabaseAggregator.from_database(str(database_path))
        agg.add_directory(directory=str(results_root))
        aggregator["db"] = agg

    timings["database_build"] = timed(database_build)
    db_agg = aggregator["db"]
    assert len(db_agg) == cell["n_results"], (
        f"Database ingested {len(db_agg)} fits, expected {cell['n_results']}"
    )

    unique_tag = db_agg.search.unique_tag
    timings["database_query_unique_tag"] = timed(
        lambda: len(db_agg.query(unique_tag == "dataset_0000"))
    )

    db_slice = db_agg[:SAMPLES_STAGE_CAP]
    samples_stage_results = len(db_slice)
    timings["database_values_samples"] = timed(
        lambda: deque(db_slice.values("samples"), maxlen=0)
    )

    def directory_from_directory():
        with contextlib.redirect_stdout(io.StringIO()):
            aggregator["dir"] = DirectoryAggregator.from_directory(results_root)

    timings["directory_from_directory"] = timed(directory_from_directory)
    dir_slice = aggregator["dir"][:SAMPLES_STAGE_CAP]
    timings["directory_values_samples"] = timed(
        lambda: deque(dir_slice.values("samples"), maxlen=0)
    )

    database_size_mb = database_path.stat().st_size / 1024**2

    if not keep:
        database_path.unlink(missing_ok=True)
        shutil.rmtree(results_root.parent)

    return {
        **cell,
        "samples_stage_results": samples_stage_results,
        "database_size_mb": round(database_size_mb, 1),
        "timings": timings,
    }


def print_table(rows: list):
    stages = list(rows[0]["timings"])
    header = f"{'n_results':>9} {'n_samples':>9} {'db_MB':>7} " + " ".join(
        f"{stage:>28}" for stage in stages
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
                if "values_samples" in stage
                else row["n_results"]
            )
            cells.append(f"{seconds:>18.3f} ({1000.0 * seconds / n:>6.2f})")
        print(
            f"{row['n_results']:>9} {row['n_samples']:>9} {row['database_size_mb']:>7} "
            + " ".join(cells)
        )


if __name__ == "__main__":
    logging.disable(logging.WARNING)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--label", type=str, default="")
    args = parser.parse_args()

    axes = dict(AXES_QUICK if args.quick else AXES)
    base = dict(BASE)
    if args.quick:
        base = {"n_results": 50, "n_samples": 1000, "n_gaussians": 5}
    if os.environ.get("PYAUTO_TEST_MODE"):
        axes = {"n_results": [5]}
        base = {"n_results": 5, "n_samples": 50, "n_gaussians": 1}

    rows = []
    for cell in grid_cells(axes=axes, base=base):
        print(f"profiling {cell} ...", flush=True)
        rows.append(profile_cell(cell, keep=args.keep))

    print_table(rows)

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    label = f"_{args.label}" if args.label else ""
    output_file = (
        RESULTS_PATH
        / f"profile_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}{label}.json"
    )
    output_file.write_text(
        json.dumps({"autofit_version": af.__version__, "rows": rows}, indent=2)
    )
    print(f"\nJSON written to {output_file}")
