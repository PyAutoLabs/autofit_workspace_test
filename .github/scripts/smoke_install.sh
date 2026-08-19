#!/usr/bin/env bash
# Workspace-owned install epilogue for the reusable Smoke Tests workflow
# (PyAutoHeart/.github/workflows/smoke-tests.yml). Runs with cwd at the
# checkout root (the dependency chain is cloned beside `workspace/`) and
# receives PYTHON_VERSION. Everything that differs per workspace lives
# here; the ceremony lives in the reusable workflow.
set -e

if [ "$PYTHON_VERSION" = "3.12" ]; then
  # The local [jax] extras must be listed explicitly: without them pip >= 26
  # explores PyPI release history for the autofit[jax]/autonerves[jax] extra
  # nodes and aborts (resolution-too-deep, surfacing as wcwidth 0.2.1's
  # corrupt wheel).
  pip install ./PyAutoNerves "./PyAutoNerves[jax]" "./PyAutoFit[jax]" "./PyAutoFit[optional]"
else
  pip install ./PyAutoNerves ./PyAutoFit
fi
pip install nautilus-sampler
