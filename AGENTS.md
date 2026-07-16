# PyAutoFit Workspace Test — Agent Instructions

This is the integration-test suite for **PyAutoFit**, run on the build server to verify the core
library works end-to-end. It is **not** a user-facing workspace — see `../autofit_workspace` for
examples and tutorials. These are the canonical, agent-agnostic instructions for this repo.

Dependencies: `autofit` (and `autoconf`).

## Repository Structure

```
scripts/                     Integration-test scripts run on the build server
  searches/                  Non-linear search tests (Dynesty, Emcee, Nautilus, LBFGS, …)
  features/                  Feature tests (assertions, latent variables, minimal output, …)
  database/                  Database tests (directory, scrape, session)
  graphical/                 Graphical-model tests (EP, hierarchical, simultaneous)
  model_composition/         Model-composition tests
  prior_correctness/         Prior-behaviour correctness tests
  jax_assertions/            JAX assertion tests
  simulators/                Simulator utilities used by the other scripts
failed/                      One .txt log per failing script (written by run_all_scripts.sh)
dataset/ config/ output/     Input data, YAML config, runtime fit results
```

## Running Scripts

Run a single script directly from the repo root — with no env applied, the non-linear search runs
for real (sampler limits like `n_like_max` keep it short):

```bash
python scripts/searches/DynestyStatic.py
```

## Testing

On CI, `smoke_tests.yml` gates every PR on Python **3.12 and 3.13**. The gate runs the smoke runner
(the definition of green):

```bash
python .github/scripts/run_smoke.py
```

It executes the curated entries in `smoke_tests.txt`, applying per-entry environment from
`config/build/env_vars.yaml`. That file sets `PYAUTO_TEST_MODE=2` (skip the sampler) as the default,
but **search and feature tests that must validate real inference `unset` it** (or set
`PYAUTO_TEST_MODE=1`, reduced iterations with a real sampler) — those are the cases where bypassing
the sampler would defeat the test. So the test-mode story is per-script, not a blanket "runs for
real." A failure under these flags signals a real problem.

For a local **full sweep** of every script under `scripts/`, use the stateless runner (logs each
failure to `failed/<script_path>.txt`):

```bash
bash run_all_scripts.sh
```

## Sandboxed / restricted runs

If `numba` or `matplotlib` cannot write to the default cache locations, point them at writable dirs:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python scripts/searches/DynestyStatic.py
```

## Bulk-edit safety

When editing the same region across many scripts in one pass, only rewrite the targeted region.
**Never produce a whole-file write unless you have read the entire current file** — a whole-file
write from a header skim silently deletes every section below the header.

## Related Repos

- Source libs: `../PyAutoFit`, `../PyAutoConf`.
- `../autofit_workspace` — the user-facing workspace; `../HowToFit` — the tutorial series.
- `../PyAutoBuild` — CI / build tooling.

## Task Workflows

When a library change lands, run the smoke suite, read any `[FAIL]` entries, and update the affected
test scripts to the new API (preserving intent). **Never edit a script to mask a real regression** —
if a library bug surfaces, flag it for PyAutoFit rather than papering over it. Note in your PR any
change that affects sibling repos (`autofit_workspace`, the source libraries).

<!-- repos_sync:history:begin -->
## Never rewrite history

Never rewrite pushed history on any repo with a remote — no `git init` over a
tracked repo, no force-push to `main`, no fresh-start "Initial commit", no
`filter-repo` / `filter-branch` / `rebase -i` on pushed branches. To get a
clean tree: `git fetch origin && git reset --hard origin/main && git clean -fd`.
<!-- repos_sync:history:end -->
