"""
Run the workspace smoke test suite.

Reads `smoke_tests.txt` from the workspace root and `config/build/profile_smoke.yaml`
for per-script env var overrides, then runs each listed script with the
appropriate environment. Continues through failures and exits non-zero
if any script failed.

Each script is capped at `BUILD_SCRIPT_TIMEOUT` seconds (default 300), the same
env var and default PyAutoHands's `build_util.py` uses, so the PR gate and the
release runner agree about how long a script may take. On expiry the script's
whole process group is killed and the entry is reported as TIMEOUT.

The env resolution itself is NOT implemented here: it is PyAutoHands's
`autohands/env_config.py`, imported below. This file used to carry a copy, and
the copy had already drifted (its `load_env_config` hardcoded
`config/build/profile_smoke.yaml`, so the PR gate was structurally unable to read
the release profile — the seed incident's failure mode 4/7). One resolver
means the PR gate and the release runner cannot disagree about what a script's
environment is. See PyAutoHands docs/env_profile_redesign.md §5 (#161 step 2).

Mirrors the logic of the `/smoke-test` skill so CI and local runs stay
in sync.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


# Per-script wall-clock cap, shared with PyAutoHands's build_util.py so the PR
# gate and the release runner agree about how long a script may take. Same env
# var, same 300s default; workspace-validation raises it to 1800 for
# mode=release.
TIMEOUT_SECS = int(os.environ.get("BUILD_SCRIPT_TIMEOUT", "300"))

WORKSPACE = Path(__file__).resolve().parents[2]
SMOKE_FILE = WORKSPACE / "smoke_tests.txt"
ENV_VARS_FILE = WORKSPACE / "config" / "build" / "profile_smoke.yaml"
SCRIPTS_DIR = WORKSPACE / "scripts"

# CI puts PyAutoHands/autohands on PYTHONPATH (PyAutoHeart's reusable
# smoke-tests.yml clones it alongside the dependency chain); for local runs,
# fall back to the sibling checkout.
try:
    from env_config import build_env_for_script, load_env_config
except ImportError:  # pragma: no cover - local-run fallback
    sys.path.insert(0, str(WORKSPACE.parent / "PyAutoHands" / "autohands"))
    from env_config import build_env_for_script, load_env_config

# Per-script cap resolution, from the SAME resolver the mega-run uses. The
# module-global TIMEOUT_SECS above expresses one cap for the whole run, but a
# profile may set BUILD_SCRIPT_TIMEOUT on an `overrides` pattern — and that
# value rides the per-script env, which is handed to the CHILD while the
# `communicate(timeout=...)` kill timer lives HERE in the parent. Without
# resolving it parent-side this runner would silently ignore a profile budget
# that PyAutoHands's build_util honours, so the PR gate and the mega-run would
# disagree about the same profile (PyAutoHands#226/#227).
try:
    from build_util import timeout_for
except ImportError:  # pragma: no cover - keep the gate working without Hands
    def timeout_for(env=None) -> int:
        """Fallback: whole-run cap only, matching the pre-#227 behaviour."""
        return TIMEOUT_SECS

# The group kill itself is PyAutoHands's, for the same reason the resolver is:
# one implementation, so the PR gate and the mega-run cannot disagree. The
# fallback keeps this gate working in a checkout without Hands on PYTHONPATH.
try:
    from build_util import kill_group
except ImportError:  # pragma: no cover - local-run fallback
    def kill_group(proc: subprocess.Popen) -> None:
        """SIGKILL the script's whole process group, tolerating a dead one."""
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):  # pragma: no cover - race
            proc.kill()


def load_smoke_scripts() -> list[str]:
    scripts: list[str] = []
    for line in SMOKE_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        scripts.append(line)
    return scripts


def load_cfg() -> dict | None:
    """Parsed env profile, or None when the workspace has none.

    None flows through build_env_for_script -> None -> subprocess inherits the
    parent environment, which is what the old local copy's empty-config path
    did by hand.
    """
    if not ENV_VARS_FILE.exists():
        return None
    return load_env_config(ENV_VARS_FILE)


def run_one(script_rel: str, cfg: dict | None) -> tuple[str, int, float, str, int]:
    """Run one smoke script, capped at the per-script resolved timeout.

    The script runs in its own session (``start_new_session=True``) so that a
    timeout can kill the whole process group rather than just the direct child.
    That distinction is load-bearing: capturing output means waiting for the
    stdout pipe to reach EOF, and any grandchild that inherited the pipe holds
    it open even after the child itself has exited. A script whose work has
    finished can therefore hang the runner indefinitely, which is exactly how
    smoke CI came to sit at the 6-hour GitHub Actions ceiling (issue #196) while
    reporting nothing since the last completed script. Killing the group closes
    the inherited pipe and lets the read finish.
    """
    env = build_env_for_script(Path(script_rel), cfg)
    timeout_secs = timeout_for(env)
    script_path = SCRIPTS_DIR / script_rel
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(WORKSPACE),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        output, _ = proc.communicate(timeout=timeout_secs)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_group(proc)
        # The group is gone, so this drains whatever was buffered and returns.
        output, _ = proc.communicate()
        # Always 124 (the conventional timeout code), never the signal we just
        # sent. Reporting proc.returncode here would surface -9 for a script
        # killed mid-run and mislabel a timeout as an ordinary failure; the two
        # need distinguishing because only one of them means "raise the cap or
        # SLOW-skip it".
        returncode = 124
    elapsed = time.time() - t0
    if timed_out:
        output = (output or "") + (
            f"\n::error::TIMEOUT after {timeout_secs}s — killed the process group. "
            f"Raise BUILD_SCRIPT_TIMEOUT if this script is legitimately slow, or "
            f"add it to config/build/no_run.yaml with a dated SLOW marker.\n"
        )
    # timeout_secs is returned so the caller reports the cap this script
    # actually ran under, not the run-wide default -- a quoted cap below the
    # enforced one biases every "too slow to un-skip?" call (the 60s-cap myth).
    return script_rel, returncode, elapsed, output or "", timeout_secs


def main() -> int:
    if not SMOKE_FILE.exists():
        print(f"ERROR: no smoke_tests.txt at {SMOKE_FILE}", file=sys.stderr)
        return 1
    scripts = load_smoke_scripts()
    if not scripts:
        print("No smoke test scripts listed.")
        return 0
    cfg = load_cfg()

    print(f"Running {len(scripts)} smoke test script(s) from {SMOKE_FILE.name}\n")
    failures: list[tuple[str, int, str, int]] = []
    for script_rel in scripts:
        print(f"::group::{script_rel}")
        name, rc, elapsed, output, cap = run_one(script_rel, cfg)
        print(output, end="")
        if rc == 0:
            status = "PASS"
        elif rc == 124:
            status = f"TIMEOUT ({cap}s)"
        else:
            status = f"FAIL (exit {rc})"
        print(f"\n[{status}] {name} — {elapsed:.1f}s")
        print("::endgroup::")
        if rc != 0:
            failures.append((name, rc, output, cap))

    total = len(scripts)
    passed = total - len(failures)
    print(f"\n=== Smoke test summary: {passed}/{total} passed ===")
    for name, rc, _, cap in failures:
        label = f"TIMEOUT ({cap}s)" if rc == 124 else f"FAIL  (exit {rc})"
        print(f"  {label}  {name}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
