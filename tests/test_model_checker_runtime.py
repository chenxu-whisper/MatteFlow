import importlib
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matteflow.utils.model_paths as model_paths
from matteflow.utils.model_checker import ModelChecker
from matteflow.utils.model_paths import _main_project_root, model_file, models_root, project_root, resolve_snapshot_model_dir


def test_model_file_points_into_project_models_dir():
    path = model_file("corridorkey.pth")
    current_root = project_root()
    allowed_roots = {current_root / "models"}
    main_root = _main_project_root(current_root)
    if main_root is not None:
        allowed_roots.add(main_root / "models")

    assert path.name == "corridorkey.pth"
    assert path.parent in allowed_roots


def test_resolve_snapshot_model_dir_prefers_snapshot_layout(tmp_path):
    snapshot = tmp_path / "models--foo--bar" / "snapshots" / "123"
    (snapshot / "unet").mkdir(parents=True)
    (snapshot / "vae").mkdir(parents=True)
    (snapshot / "scheduler").mkdir(parents=True)

    assert resolve_snapshot_model_dir(tmp_path, "foo/bar", ("unet", "vae", "scheduler")) == snapshot


def test_models_root_falls_back_to_main_project_when_worktree_models_missing(monkeypatch, tmp_path):
    main_root = tmp_path / "MatteFlow"
    worktree_root = main_root / ".worktrees" / "transparency-layered-fusion-v1"
    fake_model_paths = worktree_root / "src" / "matteflow" / "utils" / "model_paths.py"
    fake_model_paths.parent.mkdir(parents=True)
    fake_model_paths.write_text("# test stub\n", encoding="utf-8")
    (main_root / "models").mkdir(parents=True)
    (main_root / "models" / "matanyone2.pth").write_text("stub", encoding="utf-8")

    monkeypatch.setattr(model_paths, "__file__", str(fake_model_paths), raising=False)

    assert model_paths.models_root() == main_root / "models"


def test_resolve_snapshot_model_dir_falls_back_to_main_project_when_worktree_has_partial_models(
    monkeypatch, tmp_path
):
    main_root = tmp_path / "MatteFlow"
    worktree_root = main_root / ".worktrees" / "transparency-layered-fusion-v1"
    fake_model_paths = worktree_root / "src" / "matteflow" / "utils" / "model_paths.py"
    fake_model_paths.parent.mkdir(parents=True)
    fake_model_paths.write_text("# test stub\n", encoding="utf-8")

    (worktree_root / "models").mkdir(parents=True)
    (worktree_root / "models" / "matanyone2.pth").write_text("stub", encoding="utf-8")

    snapshot = main_root / "models" / "models--geyongtao--gvm" / "snapshots" / "123"
    (snapshot / "unet").mkdir(parents=True)
    (snapshot / "vae").mkdir(parents=True)
    (snapshot / "scheduler").mkdir(parents=True)

    monkeypatch.setattr(model_paths, "__file__", str(fake_model_paths), raising=False)

    assert (
        model_paths.resolve_snapshot_model_dir(
            model_paths.models_root(), "geyongtao/gvm", ("unet", "vae", "scheduler")
        )
        == snapshot
    )


def test_gvm_checker_marks_runtime_import_failure_as_unavailable(monkeypatch, tmp_path):
    snapshot = tmp_path / "models--geyongtao--gvm" / "snapshots" / "123"
    (snapshot / "unet").mkdir(parents=True)
    (snapshot / "vae").mkdir(parents=True)
    (snapshot / "scheduler").mkdir(parents=True)

    checker = ModelChecker()
    checker.cache_dir = tmp_path
    checker.matteflow_dir = tmp_path

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "torch":
            return fake_torch
        if name == "matteflow.vendor.gvm_core.wrapper":
            raise ModuleNotFoundError("missing gvm runtime")
        return real_import_module(name, package)

    monkeypatch.setattr("importlib.import_module", fake_import_module)

    info = checker._check_gvm()

    assert info["available"] is False
    assert info["reason"] == "GVM vendored runtime 不可导入"


def test_gvm_wrapper_imports_from_vendored_runtime():
    module = importlib.import_module("matteflow.vendor.gvm_core.wrapper")

    assert hasattr(module, "GVMProcessor")


def test_collect_model_facts_exposes_reason_and_path(tmp_path):
    checker = ModelChecker()
    checker.cache_dir = tmp_path
    checker.matteflow_dir = tmp_path

    facts = checker.collect_model_facts()

    assert "gvm" in facts
    assert set(facts["gvm"]) >= {
        "model_key",
        "display_name",
        "available",
        "path",
        "reason",
        "auto_download",
    }
