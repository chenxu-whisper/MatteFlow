import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_vendored_gvm_package_exists():
    vendored_init = PROJECT_ROOT / "src" / "matteflow" / "vendor" / "gvm_core" / "__init__.py"

    assert vendored_init.exists()


def test_vendored_gvm_processor_file_exists():
    wrapper_file = PROJECT_ROOT / "src" / "matteflow" / "vendor" / "gvm_core" / "wrapper.py"

    assert wrapper_file.exists()


def test_vendored_wrapper_has_no_external_backend_dependency():
    wrapper_file = PROJECT_ROOT / "src" / "matteflow" / "vendor" / "gvm_core" / "wrapper.py"

    contents = wrapper_file.read_text(encoding="utf-8")

    assert "from backend.project import get_data_dir" not in contents


def test_vendored_python_files_have_no_external_project_imports():
    vendor_root = PROJECT_ROOT / "src" / "matteflow" / "vendor" / "gvm_core"
    forbidden_tokens = [
        "from backend.",
        "import backend",
        "from gvm_core.",
        "import gvm_core",
        "sys.path.insert",
        "E:/ByteDance/Projects/Code/EZ-CorridorKey",
    ]

    offenders = {}
    for py_file in vendor_root.rglob("*.py"):
        contents = py_file.read_text(encoding="utf-8")
        hits = [token for token in forbidden_tokens if token in contents]
        if hits:
            offenders[str(py_file.relative_to(vendor_root))] = hits

    assert offenders == {}


def test_vendored_wrapper_has_no_remote_repo_fallback():
    wrapper_file = PROJECT_ROOT / "src" / "matteflow" / "vendor" / "gvm_core" / "wrapper.py"

    contents = wrapper_file.read_text(encoding="utf-8")

    assert '"geyongtao/gvm"' not in contents
