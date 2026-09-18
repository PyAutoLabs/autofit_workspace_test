"""
Run the workspace smoke test suite: the scripts listed in `smoke_tests.txt`.

Nothing about discovery, exclusion, environment resolution, per-script timeouts
or reporting is implemented here. This is a thin shim over PyAutoHands'
`autohands/run_python.py` — the same entry point PyAutoHeart's
workspace-validation uses — so the PR gate and the validation runner cannot
drift apart.

This file used to be a 198-line copy of that machinery, one of ten across the
workspace repos. Each of the last three fixes to it — the env-resolver fork
(PyAutoHands#185), the per-script timeout and process-group kill
(PyAutoHands#226/#227), the jupyter guard — had to be swept across every copy by
hand, while the HowTo repos needed none of them precisely because they hold no
logic. `--list` (PyAutoHands#261) closed the last gap: the shared runner was
opt-out only, and this workspace's coverage is opt-in.

What the shared runner provides:

  * the allowlist in `smoke_tests.txt`, run in that file's own order
  * per-script env from `config/build/profile_smoke.yaml`, via the one resolver
  * the `BUILD_SCRIPT_TIMEOUT` cap and the process-group kill on expiry
  * a structured JSON report, and a non-zero exit when anything failed

`config/build/no_run.yaml` is deliberately NOT applied here. It is policy for
the release mega-run and notebook generation; `smoke_tests.txt` is policy for
this gate, and a script legitimately appears in both (PyAutoHands#262).

`--report-dir` is REQUIRED, not cosmetic. run_python.py only propagates failures
(`sys.exit(1)`) when a report was built; without it the suite runs to completion
and always exits 0 — a vacuously green gate.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT = "autofit_workspace_test"

# pyauto:smoke-bootstrap:begin
# Generated from PyAutoMind/policy/smoke_bootstrap.py; edit the source, not copies.
# CI supplies build_util on PYTHONPATH. Discovery is only a local fallback.
try:
    import build_util
except ModuleNotFoundError as _smoke_import_error:
    if _smoke_import_error.name != "build_util":
        raise  # An installed Hands dependency is broken, not missing Hands.

    def _smoke_hands_path():
        import importlib.util

        explicit = os.environ.get("PYAUTO_ROOT")
        if explicit:
            root = Path(explicit).expanduser().resolve()
        else:
            root = next(
                (p for p in WORKSPACE.parents if (p / ".pyauto-root").is_file()),
                WORKSPACE.parent,
            )
            # Prefer the shared resolver, but never require Brain. A bundle's
            # Brain may be a symlink to canonical; reject that foreign answer.
            resolver_file = root / "PyAutoBrain" / "agents" / "_pyauto_root.py"
            if resolver_file.is_file():
                try:
                    spec = importlib.util.spec_from_file_location(
                        "_smoke_root_resolver", resolver_file
                    )
                    resolver = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(resolver)
                    resolved = Path(resolver.workspace_root_reason()[0]).resolve()
                    if resolved in WORKSPACE.parents:
                        root = resolved
                except (Exception, SystemExit):
                    pass  # Missing/broken optional resolver cannot break discovery.
        hands = root / "PyAutoHands" / "autohands"
        if not (hands / "build_util.py").is_file():
            raise ModuleNotFoundError(
                f"PyAutoHands not found for {WORKSPACE}; looked for "
                f"{hands / 'build_util.py'}. Supply PyAutoHands/autohands on "
                "PYTHONPATH or set PYAUTO_ROOT to the intended workspace root.",
                name="build_util",
            )
        return hands

    sys.path.insert(0, str(_smoke_hands_path()))
    import build_util

AUTOHANDS = Path(build_util.__file__).resolve().parent
# pyauto:smoke-bootstrap:end


def main() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(AUTOHANDS), env.get("PYTHONPATH", "")) if p
    )

    cmd = [
        sys.executable,
        str(AUTOHANDS / "run_python.py"),
        PROJECT,
        "scripts",
        "--list",
        str(WORKSPACE / "smoke_tests.txt"),
        "--report-dir",
        str(WORKSPACE / "test-results"),
    ]
    # run_python.py resolves config/build/ relative to the cwd.
    return subprocess.run(cmd, cwd=str(WORKSPACE), env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
