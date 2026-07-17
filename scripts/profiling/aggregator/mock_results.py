"""
Profiling: Mock Aggregator Results
==================================

Fabricates search-output directories that the real `af.Aggregator` accepts, without
running a non-linear search.

One template result per (n_gaussians, n_samples) configuration is written by a real
search `fit` under the test-mode sampler bypass (`PYAUTO_TEST_MODE=3` +
`PYAUTO_TEST_MODE_SAMPLES=N`, PyAutoFit#1381), so every file — including a
`samples.csv` whose row count and byte size are representative of a production
sampler run — is produced by the canonical library write path. The template is then
stamped `n_results` times with `shutil.copytree`, which makes thousands of mock
results in seconds.

Run from the `autofit_workspace_test` root, e.g.:

    python scripts/profiling/aggregator/mock_results.py --n-results 100 --n-samples 1000 --n-gaussians 5
"""

import argparse
import json
import os
import shutil
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

from autoconf import conf
import autofit as af

DEFAULT_ROOT = Path("output") / "profiling_aggregator" / "mock"


@contextmanager
def test_mode_bypass(n_samples: int):
    """
    Run a real search `fit` as an instant sampler bypass writing `n_samples`
    size-realistic samples (PYAUTO_TEST_MODE_SAMPLES, minimum 4). Mode 3 skips the
    likelihood call entirely — sample values are irrelevant for load profiling.
    """
    previous = {
        key: os.environ.get(key)
        for key in ("PYAUTO_TEST_MODE", "PYAUTO_TEST_MODE_SAMPLES")
    }
    os.environ["PYAUTO_TEST_MODE"] = "3"
    os.environ["PYAUTO_TEST_MODE_SAMPLES"] = str(max(n_samples, 4))
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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


def write_template(
    root: Path,
    n_gaussians: int,
    n_samples: int,
    with_images: bool = True,
    with_latent: bool = False,
) -> Path:
    """
    Write one full search-output directory via a test-mode bypass fit and return the
    leaf directory (the one containing the `metadata` file).
    """
    template_prefix = f"_template/g{n_gaussians}_s{n_samples}"

    conf.instance.push(new_path="config", output_path=str(root))

    model = model_from(n_gaussians)
    analysis = af.ex.Analysis(data=np.ones(10), noise_map=np.ones(10))

    # A stale template would append duplicate metadata lines (the file is opened in
    # append mode), so remove any previous template for this config first.
    for candidate in (
        root / "test_mode" / Path(template_prefix),
        root / Path(template_prefix),
    ):
        if candidate.exists():
            shutil.rmtree(candidate)

    with test_mode_bypass(n_samples=n_samples):
        search = af.DynestyStatic(
            name="fit", path_prefix=template_prefix, number_of_cores=1
        )
        result = search.fit(model=model, analysis=analysis)

        # Resolve all paths inside the bypass context — the test-mode env var adds a
        # path segment, so a lazy access after the env restore points elsewhere.
        paths = search.paths
        if with_latent:
            paths.save_latent_samples(latent_samples=result.samples)

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
