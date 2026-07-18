"""
Run the workspace smoke test suite.

Reads `smoke_tests.txt` from the workspace root and `config/build/env_vars.yaml`
for per-script env var overrides, then runs each listed script with the
appropriate environment. Continues through failures and exits non-zero
if any script failed.

The env resolution itself is NOT implemented here: it is PyAutoHands's
`autobuild/env_config.py`, imported below. This file used to carry a copy, and
the copy had already drifted (its `load_env_config` hardcoded
`config/build/env_vars.yaml`, so the PR gate was structurally unable to read
the release profile — the seed incident's failure mode 4/7). One resolver
means the PR gate and the release runner cannot disagree about what a script's
environment is. See PyAutoHands docs/env_profile_redesign.md §5 (#161 step 2).

Mirrors the logic of the `/smoke-test` skill so CI and local runs stay
in sync.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
SMOKE_FILE = WORKSPACE / "smoke_tests.txt"
ENV_VARS_FILE = WORKSPACE / "config" / "build" / "env_vars.yaml"
SCRIPTS_DIR = WORKSPACE / "scripts"

# CI puts PyAutoHands/autobuild on PYTHONPATH (PyAutoHeart's reusable
# smoke-tests.yml clones it alongside the dependency chain); for local runs,
# fall back to the sibling checkout.
try:
    from env_config import build_env_for_script, load_env_config
except ImportError:  # pragma: no cover - local-run fallback
    sys.path.insert(0, str(WORKSPACE.parent / "PyAutoHands" / "autobuild"))
    from env_config import build_env_for_script, load_env_config


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


def run_one(script_rel: str, cfg: dict | None) -> tuple[str, int, float, str]:
    env = build_env_for_script(Path(script_rel), cfg)
    script_path = SCRIPTS_DIR / script_rel
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(WORKSPACE),
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    output = result.stdout + result.stderr
    return script_rel, result.returncode, elapsed, output


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
    failures: list[tuple[str, int, str]] = []
    for script_rel in scripts:
        print(f"::group::{script_rel}")
        name, rc, elapsed, output = run_one(script_rel, cfg)
        print(output, end="")
        status = "PASS" if rc == 0 else f"FAIL (exit {rc})"
        print(f"\n[{status}] {name} — {elapsed:.1f}s")
        print("::endgroup::")
        if rc != 0:
            failures.append((name, rc, output))

    total = len(scripts)
    passed = total - len(failures)
    print(f"\n=== Smoke test summary: {passed}/{total} passed ===")
    for name, rc, _ in failures:
        print(f"  FAIL  {name}  (exit {rc})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
