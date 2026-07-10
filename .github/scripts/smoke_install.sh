#!/usr/bin/env bash
# Workspace-owned install epilogue for the reusable Smoke Tests workflow
# (PyAutoHeart/.github/workflows/smoke-tests.yml). Runs with cwd at the
# checkout root (the dependency chain is cloned beside `workspace/`) and
# receives PYTHON_VERSION. Everything that differs per workspace lives
# here; the ceremony lives in the reusable workflow.
set -e

if [ "$PYTHON_VERSION" = "3.12" ]; then
  pip install ./PyAutoConf "./PyAutoFit[optional]"
else
  pip install ./PyAutoConf ./PyAutoFit
fi
pip install nautilus-sampler
