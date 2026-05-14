import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_vendored_matanyone2_package_exists():
    vendored_init = PROJECT_ROOT / "src" / "matteflow" / "vendor" / "matanyone2_module" / "__init__.py"

    assert vendored_init.exists()


def test_vendored_matanyone2_wrapper_file_exists():
    wrapper_file = PROJECT_ROOT / "src" / "matteflow" / "vendor" / "matanyone2_module" / "wrapper.py"

    assert wrapper_file.exists()


def test_vendored_matanyone2_python_files_have_no_external_project_imports():
    vendor_root = PROJECT_ROOT / "src" / "matteflow" / "vendor" / "matanyone2_module"
    forbidden_patterns = [
        r"^\s*from\s+modules\.MatAnyone2Module(\.|$)",
        r"^\s*import\s+modules\.MatAnyone2Module(\s|$)",
        r"^\s*from\s+backend(\.|$)",
        r"^\s*import\s+backend(\s|$)",
        r"sys\.path\.insert",
        r"E:/ByteDance/Projects/Code/EZ-CorridorKey",
    ]

    offenders = {}
    for py_file in vendor_root.rglob("*.py"):
        contents = py_file.read_text(encoding="utf-8")
        hits = [pattern for pattern in forbidden_patterns if re.search(pattern, contents, flags=re.MULTILINE)]
        if hits:
            offenders[str(py_file.relative_to(vendor_root))] = hits

    assert offenders == {}
