# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

**autofit_workspace_test** is the integration test suite for PyAutoFit. It contains Python scripts that are run on the build server to verify that the core PyAutoFit functionality works end-to-end. It is not a user-facing workspace — see `../autofit_workspace` for example scripts and tutorials.

Dependencies: `autofit`. Python version: 3.11.

## Workspace Structure

```
scripts/                     Integration test scripts run on the build server
  searches/                  Non-linear search tests (Dynesty, Emcee, LBFGS, etc.)
  features/                  Feature tests (assertions, grid search, latent variables, etc.)
  database/                  Database tests (directory, scrape, session)
  graphical/                 Graphical model tests (EP, hierarchical, simultaneous)
  simulators/                Simulator utilities used by other scripts
failed/                      Failure logs written here when a script errors (one .txt per failure)
dataset/                     Input data files and example datasets
config/                      YAML configuration files
output/                      Model-fit results written here at runtime
```

## Running Tests

Scripts are run from the repository root **without** `PYAUTO_TEST_MODE=1` — the non-linear searches run for real (using sampler limits like `n_like_max` to keep runtimes short):

```bash
python scripts/imaging/model_fit.py
```

**Codex / sandboxed runs**: when running from Codex or any restricted environment, set writable cache directories so `numba` and `matplotlib` do not fail on unwritable home or source-tree paths:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python scripts/imaging/model_fit.py
```

This workspace is often imported from `/mnt/c/...` and Codex may not be able to write to module `__pycache__` directories or `/home/jammy/.cache`, which can cause import-time `numba` caching failures without this override.

To run all tests and log failures to `failed/`:

```bash
bash run_all_scripts.sh
```

Each failed script produces a `.txt` file in `failed/` named after the script path (with `/` replaced by `__`), containing the exit code and full output.

Unlike `../autolens_workspace`, there is no resume/skip logic — every run executes all scripts in `scripts/` from scratch.

## Integration Test Runner

`run_all_scripts.sh` at the repo root:
- Finds all `*.py` files under `scripts/` and runs them in order (no test mode flag)
- On failure: writes a log to `failed/<script_path_with_slashes_replaced>.txt`
- Does not skip previously-run scripts (stateless, always runs all)

## Line Endings — Always Unix (LF)

All files **must use Unix line endings (LF, `\n`)**. Never write `\r\n` line endings.
