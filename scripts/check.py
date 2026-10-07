"""Run every check nexus must pass, in order; stop at the first failure.

    uv run python scripts/check.py

1. ruff: lint (bugs, imports, modern syntax)
2. ruff: formatting
3. mypy: types (config in pyproject.toml)
4. pytest: the test suite, with a coverage floor (pyproject.toml)

Exit code 0 only when all of them pass.
"""

import subprocess
import sys

CHECKS = [
    ("lint", ["ruff", "check", "src", "tests", "scripts"]),
    ("format", ["ruff", "format", "--check", "src", "tests", "scripts"]),
    ("types", ["mypy"]),
    ("tests", ["pytest", "-q", "--cov=nexus", "--cov-report=term-missing:skip-covered"]),
]


def main() -> int:
    for name, argv in CHECKS:
        print(f"== {name}: {' '.join(argv)}", flush=True)
        # Each tool from this environment, so the versions are the locked ones.
        code = subprocess.run([sys.executable, "-m", *argv]).returncode
        if code != 0:
            print(f"\nFAILED: {name} (exit code {code})")
            return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
