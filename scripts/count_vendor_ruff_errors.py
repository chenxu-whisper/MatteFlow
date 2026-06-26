"""Count Ruff diagnostics that come from vendored source files.

This is a diagnostic helper for deciding whether Ruff failures are caused by
project code or by upstream vendored code.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_TARGETS = ("src", "scripts", "tests")
VENDOR_PREFIX = ("src", "matteflow", "vendor")


def _relative_parts(filename: str, root: Path) -> tuple[str, ...]:
    path = Path(filename)
    try:
        path = path.resolve().relative_to(root)
    except ValueError:
        try:
            path = path.relative_to(root)
        except ValueError:
            pass
    return tuple(part.lower() for part in path.parts)


def _is_vendor_diagnostic(diagnostic: dict[str, Any], root: Path) -> bool:
    parts = _relative_parts(str(diagnostic.get("filename", "")), root)
    return parts[: len(VENDOR_PREFIX)] == VENDOR_PREFIX


def _run_ruff(targets: list[str]) -> tuple[int, list[dict[str, Any]], str]:
    command = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--output-format=json",
        *targets,
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        diagnostics = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Ruff did not emit valid JSON. stderr:\n" + completed.stderr
        ) from exc
    return completed.returncode, diagnostics, completed.stderr


def _print_summary(diagnostics: list[dict[str, Any]], root: Path) -> int:
    vendor = [item for item in diagnostics if _is_vendor_diagnostic(item, root)]
    non_vendor = [item for item in diagnostics if item not in vendor]

    vendor_by_code = Counter(str(item.get("code", "<unknown>")) for item in vendor)
    vendor_by_file = Counter(
        str(Path(str(item.get("filename", ""))).resolve().relative_to(root))
        for item in vendor
    )
    non_vendor_by_code = Counter(str(item.get("code", "<unknown>")) for item in non_vendor)
    non_vendor_by_file = Counter(
        str(Path(str(item.get("filename", ""))).resolve().relative_to(root))
        for item in non_vendor
    )

    print("Ruff diagnostic summary")
    print(f"  total:      {len(diagnostics)}")
    print(f"  vendor:     {len(vendor)}")
    print(f"  non-vendor: {len(non_vendor)}")
    print()

    if vendor_by_code:
        print("Vendor diagnostics by rule:")
        for code, count in vendor_by_code.most_common():
            print(f"  {code}: {count}")
        print()

    if vendor_by_file:
        print("Top vendor files:")
        for filename, count in vendor_by_file.most_common(10):
            print(f"  {count:4d}  {filename}")
        print()

    if non_vendor_by_code:
        print("Non-vendor diagnostics by rule:")
        for code, count in non_vendor_by_code.most_common():
            print(f"  {code}: {count}")
        print()

    if non_vendor_by_file:
        print("Top non-vendor files:")
        for filename, count in non_vendor_by_file.most_common(10):
            print(f"  {count:4d}  {filename}")
        print()

    if non_vendor:
        print("Release gate: BLOCKED if Ruff is required, because non-vendor diagnostics exist.")
        return 1

    if vendor:
        print(
            "Release gate: only vendor diagnostics remain. Ruff blocks only if vendor "
            "is included in the release lint gate."
        )
        return 0

    print("Release gate: clean. Ruff reported no diagnostics.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Ruff and count diagnostics under src/matteflow/vendor."
    )
    parser.add_argument(
        "targets",
        nargs="*",
        default=list(DEFAULT_TARGETS),
        help="Ruff targets to check. Defaults to: src scripts tests",
    )
    args = parser.parse_args()

    root = Path.cwd().resolve()
    ruff_returncode, diagnostics, stderr = _run_ruff(args.targets)
    if ruff_returncode not in {0, 1}:
        if stderr:
            print(stderr, file=sys.stderr)
        return ruff_returncode

    return _print_summary(diagnostics, root)


if __name__ == "__main__":
    raise SystemExit(main())
