"""
Profiling: Mock Aggregator Results
==================================

Fabricates search-output directories that the real `af.Aggregator` accepts, without
running a non-linear search.

One template result per (n_gaussians, n_samples) configuration is written through the
library's own `DirectoryPaths` machinery (`save_all` / `save_samples` /
`save_samples_summary`), so file formats always track the installed autofit. The
template is then stamped `n_results` times with `shutil.copytree`, which makes
thousands of mock results in seconds.

Run from the `autofit_workspace_test` root, e.g.:

    python scripts/profiling/aggregator/mock_results.py --n-results 100 --n-samples 1000 --n-gaussians 5
"""

import argparse
import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from autoconf import conf
import autofit as af
from autofit.non_linear.paths.directory import DirectoryPaths

DEFAULT_ROOT = Path("output") / "profiling_aggregator" / "mock"


def model_from(n_gaussians: int) -> af.Collection:
    """
    A model of `n_gaussians` `af.ex.Gaussian`s (3 free parameters each) with explicit
    priors, so no prior config lookup is required.
    """
    gaussians = {}
    for i in range(n_gaussians):
        gaussian = af.Model(af.ex.Gaussian)
        gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
        gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
        gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
        gaussians[f"gaussian_{i}"] = gaussian
    return af.Collection(**gaussians)


def samples_from(model, n_samples: int, seed: int = 1) -> af.SamplesNest:
    """
    Random samples for the model. Parameter values are uniform draws in (0, 1) — the
    numbers are immaterial for profiling, only the row/column counts matter.
    """
    rng = np.random.default_rng(seed)
    n_params = model.prior_count

    parameter_lists = rng.uniform(0.01, 0.99, size=(n_samples, n_params)).tolist()
    log_likelihood_list = np.sort(rng.normal(-1000.0, 50.0, n_samples)).tolist()
    log_prior_list = [0.0] * n_samples
    weight = 1.0 / n_samples
    weight_list = [weight] * n_samples

    sample_list = af.Sample.from_lists(
        model=model,
        parameter_lists=parameter_lists,
        log_likelihood_list=log_likelihood_list,
        log_prior_list=log_prior_list,
        weight_list=weight_list,
    )

    return af.SamplesNest(
        model=model,
        sample_list=sample_list,
        samples_info={
            "number_live_points": 50,
            "log_evidence": -1000.0,
            "total_samples": n_samples,
            "total_accepted_samples": n_samples,
            "total_iterations": n_samples,
            "time": "1.0",
        },
    )


def write_template(
    root: Path,
    n_gaussians: int,
    n_samples: int,
    with_images: bool = True,
    with_latent: bool = False,
) -> Path:
    """
    Write one full search-output directory via the real paths machinery and return the
    leaf directory (the one containing the `metadata` file).
    """
    template_prefix = f"_template/g{n_gaussians}_s{n_samples}"

    conf.instance.push(new_path="config", output_path=str(root))

    model = model_from(n_gaussians)
    samples = samples_from(model=model, n_samples=n_samples)
    search = af.DynestyStatic(name="fit")

    paths = DirectoryPaths(name="fit", path_prefix=template_prefix)
    paths.model = model
    paths.search = search

    if Path(paths.output_path).exists():
        shutil.rmtree(paths.output_path)

    paths.save_all(info={})
    paths.save_samples(samples=samples)
    paths.save_samples_summary(samples_summary=samples.summary())
    if with_latent:
        paths.save_latent_samples(latent_samples=samples)
    paths.completed()

    if with_images:
        image_path = paths.image_path
        image = Image.new("RGB", (64, 64))
        for name in ("subplot_fit", "corner_pdf", "search_internal"):
            image.save(image_path / f"{name}.png")

    return Path(paths.output_path)


def _zip_directory(directory: Path):
    """
    Zip a result directory the way search output is zipped: contents at the archive
    root, so `unzip_directory` re-extracts them into a sibling folder of the same name.
    """
    with zipfile.ZipFile(directory.with_suffix(".zip"), "w") as f:
        for path in directory.rglob("*"):
            if path.is_file():
                f.write(path, path.relative_to(directory))
    shutil.rmtree(directory)


def stamp_results(
    template_leaf: Path,
    results_root: Path,
    n_results: int,
    zip_results: bool = False,
):
    """
    Copy the template `n_results` times under `results_root`, giving each copy a unique
    `dataset_name` metadata entry and a unique `unique_tag` in its search.json.

    The unique_tag matters for the sqlite database: fits are keyed by the identifier
    hashed from (search, model, unique_tag), so identical copies would collapse to a
    single database row.
    """
    search_json_text = (template_leaf / "files" / "search.json").read_text()
    search_dict = json.loads(search_json_text)

    for i in range(n_results):
        dataset_name = f"dataset_{i:04d}"
        destination = results_root / dataset_name / "fit"
        shutil.copytree(template_leaf, destination)
        with open(destination / "metadata", "a") as f:
            f.write(f"\ndataset_name={dataset_name}")
        search_dict["arguments"]["unique_tag"] = dataset_name
        (destination / "files" / "search.json").write_text(json.dumps(search_dict))
        if zip_results:
            _zip_directory(destination)


def generate_mock_results(
    root: Path = DEFAULT_ROOT,
    n_results: int = 100,
    n_samples: int = 1000,
    n_gaussians: int = 5,
    zip_results: bool = False,
    with_images: bool = True,
    with_latent: bool = False,
    fresh: bool = False,
) -> Path:
    """
    Generate a directory of `n_results` mock results and return it, ready for
    `af.Aggregator.from_directory`. Skips regeneration when an identical set already
    exists (recorded in a manifest file).
    """
    root = Path(root)
    config = {
        "n_results": n_results,
        "n_samples": n_samples,
        "n_gaussians": n_gaussians,
        "zip_results": zip_results,
        "with_images": with_images,
        "with_latent": with_latent,
    }
    tag = f"r{n_results}_s{n_samples}_g{n_gaussians}" + ("_zip" if zip_results else "")
    results_root = root / tag
    manifest_path = root / f"{tag}.manifest.json"

    if not fresh and manifest_path.exists() and results_root.exists():
        if json.loads(manifest_path.read_text()) == config:
            return results_root

    if results_root.exists():
        shutil.rmtree(results_root)

    template_leaf = write_template(
        root=root,
        n_gaussians=n_gaussians,
        n_samples=n_samples,
        with_images=with_images,
        with_latent=with_latent,
    )
    stamp_results(
        template_leaf=template_leaf,
        results_root=results_root,
        n_results=n_results,
        zip_results=zip_results,
    )

    manifest_path.write_text(json.dumps(config))
    return results_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--n-results", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-gaussians", type=int, default=5)
    parser.add_argument("--zip", action="store_true", dest="zip_results")
    parser.add_argument("--latent", action="store_true", dest="with_latent")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    results_root = generate_mock_results(
        root=args.root,
        n_results=args.n_results,
        n_samples=args.n_samples,
        n_gaussians=args.n_gaussians,
        zip_results=args.zip_results,
        with_latent=args.with_latent,
        fresh=args.fresh,
    )
    print(
        f"{args.n_results} mock results at {results_root} "
        f"in {time.perf_counter() - start:.2f}s"
    )
