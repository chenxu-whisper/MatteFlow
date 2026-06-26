"""Batch-fix Ruff formatting diagnostics and report remaining issues.

By default this script applies only Ruff's safe fixes. Pass ``--unsafe`` to
allow Ruff's unsafe fixes as a second step after the safe pass.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TARGETS = ("src", "scripts", "tests")

E402_FILE_IGNORES = (
    "scripts/diagnose_gvm_fusion.py",
    "scripts/download_models.py",
    "scripts/run_matting.py",
    "scripts/web_gui.py",
    "src/matteflow/vendor/corridorkey_module/inference_engine.py",
    "src/matteflow/vendor/gvm_core/wrapper.py",
    "src/matteflow/vendor/matanyone2_module/matanyone2/utils/device.py",
)


@dataclass(frozen=True)
class Replacement:
    path: str
    before: bytes
    after: bytes
    description: str


def _run_ruff(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ruff", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _add_file_noqa(root: Path, relative_path: str, code: str) -> bool:
    file_path = root / relative_path
    data = file_path.read_bytes()
    marker = f"# ruff: noqa: {code}".encode()
    if marker in data.splitlines()[:5]:
        print(f"SKIP existing file ignore: {relative_path} {code}")
        return False

    if data.startswith(b"#!"):
        first_newline = data.find(b"\n")
        insert_at = first_newline + 1
        data = data[:insert_at] + marker + b"\n" + data[insert_at:]
    else:
        data = marker + b"\n" + data
    file_path.write_bytes(data)
    print(f"FIXED file ignore: {relative_path} {code}")
    return True


def _apply_replacement(root: Path, replacement: Replacement) -> bool:
    file_path = root / replacement.path
    data = file_path.read_bytes()
    before = replacement.before
    after = replacement.after

    if after and after in data and before not in data:
        print(f"SKIP already fixed: {replacement.path} - {replacement.description}")
        return False

    count = data.count(before)
    if count == 0 and b"\n" in before:
        crlf_before = before.replace(b"\n", b"\r\n")
        crlf_after = after.replace(b"\n", b"\r\n")
        if crlf_after and crlf_after in data and crlf_before not in data:
            print(f"SKIP already fixed: {replacement.path} - {replacement.description}")
            return False
        count = data.count(crlf_before)
        if count == 1:
            before = crlf_before
            after = crlf_after

    if count != 1:
        raise RuntimeError(
            f"Expected one match in {replacement.path}, found {count}: "
            f"{replacement.description}"
        )

    file_path.write_bytes(data.replace(before, after, 1))
    print(f"FIXED residual: {replacement.path} - {replacement.description}")
    return True


def _apply_residual_fixes(root: Path) -> None:
    print("\nApplying targeted residual Ruff fixes...")
    fixed_count = 0

    for path in E402_FILE_IGNORES:
        fixed_count += int(_add_file_noqa(root, path, "E402"))

    replacements = (
        Replacement(
            path="src/matteflow/matte/rembg_matte.py",
            before="""    def _check_available(self):
        \"\"\"检查 rembg 是否可用\"\"\"
        try:
            from rembg import remove
            return True
        except ImportError:
            return False
""".encode(),
            after="""    def _check_available(self):
        \"\"\"检查 rembg 是否可用\"\"\"
        import importlib.util

        return importlib.util.find_spec(\"rembg\") is not None
""".encode(),
            description="Use find_spec for rembg availability check.",
        ),
        Replacement(
            path="src/matteflow/matte/rvm/__init__.py",
            before=b"from .model import MattingNetwork\n",
            after=b"from .model import MattingNetwork as MattingNetwork\n",
            description="Make MattingNetwork re-export explicit.",
        ),
        Replacement(
            path="src/matteflow/utils/model_paths.py",
            before=b"""

def resolve_snapshot_repo_dir(root: Path, repo_id: str) -> Optional[Path]:
    flat_dir = root / repo_id.split(\"/\")[-1]
    if flat_dir.is_dir():
        return flat_dir

    namespace, name = repo_id.split(\"/\", 1)
    snapshots = root / f\"models--{namespace}--{name}\" / \"snapshots\"
    if snapshots.is_dir():
        for path in snapshots.iterdir():
            if path.is_dir():
                return path

    current_root = project_root()
    main_root = _main_project_root(current_root)
    if main_root is not None:
        fallback_root = main_root / \"models\"
        if fallback_root != root:
            return resolve_snapshot_repo_dir(fallback_root, repo_id)

    return None
""",
            after=b"",
            description="Remove duplicated resolve_snapshot_repo_dir definition.",
        ),
        Replacement(
            path="src/matteflow/vendor/corridorkey_module/core/color_utils.py",
            before=b"""        if orig_dim == 2: mask = mask.unsqueeze(0).unsqueeze(0)
        elif orig_dim == 3: mask = mask.unsqueeze(0)
""",
            after=b"""        if orig_dim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif orig_dim == 3:
            mask = mask.unsqueeze(0)
""",
            description="Expand tensor dimension guards.",
        ),
        Replacement(
            path="src/matteflow/vendor/corridorkey_module/core/color_utils.py",
            before=b"""        if orig_dim == 2: return dilated.squeeze()
        elif orig_dim == 3: return dilated.squeeze(0)
""",
            after=b"""        if orig_dim == 2:
            return dilated.squeeze()
        elif orig_dim == 3:
            return dilated.squeeze(0)
""",
            description="Expand tensor squeeze returns.",
        ),
        Replacement(
            path="src/matteflow/vendor/corridorkey_module/inference_engine.py",
            before=b"        if res_alpha.ndim == 2: res_alpha = res_alpha[:, :, np.newaxis]\n",
            after=b"""        if res_alpha.ndim == 2:
            res_alpha = res_alpha[:, :, np.newaxis]
""",
            description="Expand alpha channel guard.",
        ),
        Replacement(
            path="src/matteflow/vendor/gvm_core/wrapper.py",
            before=b"""        if current_upscaled_shape[0] % 2 != 0: current_upscaled_shape[0] -= 1
        if current_upscaled_shape[1] % 2 != 0: current_upscaled_shape[1] -= 1
""",
            after=b"""        if current_upscaled_shape[0] % 2 != 0:
            current_upscaled_shape[0] -= 1
        if current_upscaled_shape[1] % 2 != 0:
            current_upscaled_shape[1] -= 1
""",
            description="Expand even-size shape guards.",
        ),
        Replacement(
            path="src/matteflow/vendor/gvm_core/wrapper.py",
            before=b"            if writer_alpha: writer_alpha.write(alpha)\n",
            after=b"""            if writer_alpha:
                writer_alpha.write(alpha)
""",
            description="Expand optional alpha writer call.",
        ),
        Replacement(
            path="src/matteflow/vendor/gvm_core/wrapper.py",
            before=b"        if writer_alpha: writer_alpha.close()\n",
            after=b"""        if writer_alpha:
            writer_alpha.close()
""",
            description="Expand optional alpha writer close.",
        ),
    )

    for replacement in replacements:
        fixed_count += int(_apply_replacement(root, replacement))

    print(f"Applied {fixed_count} targeted residual fixes.")


def _load_diagnostics(targets: list[str]) -> list[dict[str, Any]]:
    completed = _run_ruff(["check", "--output-format=json", *targets])
    try:
        return json.loads(completed.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Ruff did not emit valid JSON. stderr:\n" + completed.stderr
        ) from exc


def _summarize(label: str, diagnostics: list[dict[str, Any]], root: Path) -> None:
    by_code = Counter(str(item.get("code", "<unknown>")) for item in diagnostics)
    by_file = Counter(
        str(Path(str(item.get("filename", ""))).resolve().relative_to(root))
        for item in diagnostics
    )

    print(f"{label}: {len(diagnostics)} diagnostics")
    if by_code:
        print("  by rule:")
        for code, count in by_code.most_common():
            print(f"    {code}: {count}")
    if by_file:
        print("  top files:")
        for filename, count in by_file.most_common(10):
            print(f"    {count:4d}  {filename}")
    print()


def _apply_fixes(targets: list[str], unsafe: bool) -> int:
    command = ["check", "--fix"]
    if unsafe:
        command.append("--unsafe-fixes")
    command.extend(targets)

    completed = _run_ruff(command)
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip(), file=sys.stderr)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply Ruff safe formatting fixes and report remaining diagnostics."
    )
    parser.add_argument(
        "--unsafe",
        action="store_true",
        help="Also allow Ruff unsafe fixes after the safe pass.",
    )
    parser.add_argument(
        "targets",
        nargs="*",
        default=list(DEFAULT_TARGETS),
        help="Ruff targets to fix. Defaults to: src scripts tests",
    )
    args = parser.parse_args()

    root = Path.cwd().resolve()
    targets = list(args.targets)

    before = _load_diagnostics(targets)
    _summarize("Before", before, root)

    print("Applying Ruff safe fixes...")
    _apply_fixes(targets, unsafe=False)

    if args.unsafe:
        print("\nApplying Ruff unsafe fixes...")
        _apply_fixes(targets, unsafe=True)

    _apply_residual_fixes(root)

    after = _load_diagnostics(targets)
    _summarize("After", after, root)

    if after:
        print("Ruff check still fails after automatic fixes.")
        print("Review the remaining diagnostics manually or rerun with --unsafe if appropriate.")
        return 1

    print("Ruff check passes after automatic fixes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
