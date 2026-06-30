#!/usr/bin/env python3
"""Update the matting quality regression manifest from assets samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".mov",
    ".mp4",
    ".png",
    ".webm",
}
DEFAULT_CANDIDATE_MODELS = ["traditional"]
DEFAULT_QUALITY_MODE = "standard"


def update_manifest(
    *,
    assets_dir: Path,
    manifest_path: Path,
    project_root: Path,
) -> dict[str, int]:
    """Classify assets files and merge them into a manifest."""
    assets_dir = assets_dir.resolve()
    manifest_path = manifest_path.resolve()
    project_root = project_root.resolve()

    payload = _read_manifest(manifest_path)
    existing_samples = payload.setdefault("samples", [])
    if not isinstance(existing_samples, list):
        raise ValueError("manifest samples must be a list")

    samples_by_path = {
        _normalize_manifest_path(sample.get("input_path", ""), project_root): sample
        for sample in existing_samples
        if isinstance(sample, dict)
    }

    added = 0
    updated = 0
    merged_samples: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for asset_path in _iter_asset_files(assets_dir):
        input_path = _relative_posix(asset_path, project_root)
        existing = samples_by_path.get(input_path)
        generated = _build_sample(asset_path.relative_to(assets_dir), input_path)
        if existing is None:
            merged_samples.append(generated)
            added += 1
        else:
            before = dict(existing)
            _fill_missing_defaults(existing, generated)
            merged_samples.append(existing)
            if existing != before:
                updated += 1
        seen_paths.add(input_path)

    for sample in existing_samples:
        if not isinstance(sample, dict):
            continue
        input_path = _normalize_manifest_path(sample.get("input_path", ""), project_root)
        if input_path not in seen_paths:
            merged_samples.append(sample)

    payload["samples"] = sorted(
        merged_samples,
        key=lambda sample: str(sample.get("input_path", "")),
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"added": added, "updated": updated, "total": len(payload["samples"])}


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {"samples": []}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest top-level JSON must be an object")
    return payload


def _iter_asset_files(assets_dir: Path) -> list[Path]:
    if not assets_dir.exists():
        raise FileNotFoundError(f"assets directory does not exist: {assets_dir}")
    return sorted(
        path
        for path in assets_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _build_sample(asset_path: Path, input_path: str) -> dict[str, Any]:
    background_mode = _classify_background(asset_path)
    return {
        "name": _sample_name(asset_path),
        "input_path": input_path,
        "background_mode": background_mode,
        "quality_mode": DEFAULT_QUALITY_MODE,
        "candidate_models": list(DEFAULT_CANDIDATE_MODELS),
        "risk_ceilings": _risk_ceilings(background_mode),
    }


def _fill_missing_defaults(sample: dict[str, Any], generated: dict[str, Any]) -> None:
    for key, value in generated.items():
        sample.setdefault(key, value)


def _classify_background(asset_path: Path) -> str:
    tokens = [part.lower() for part in asset_path.parts]
    stem = asset_path.stem.lower()
    token_text = " ".join([*tokens, stem])
    if "black" in token_text:
        return "black_background"
    if "green" in token_text:
        return "green_screen"
    return "auto"


def _risk_ceilings(background_mode: str) -> dict[str, float]:
    if background_mode == "black_background":
        return {
            "background_residue": 0.2,
            "transparent_effect_loss": 0.5,
        }
    if background_mode == "green_screen":
        return {
            "background_residue": 0.2,
            "hair_edge_loss": 0.5,
        }
    return {
        "background_residue": 0.25,
        "hair_edge_loss": 0.6,
    }


def _sample_name(asset_path: Path) -> str:
    parent = asset_path.parent.name.lower()
    stem = asset_path.stem.lower()
    if parent in {"assets", "."}:
        return stem
    return f"{parent}_{stem}"


def _relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _normalize_manifest_path(value: object, project_root: Path) -> str:
    path = Path(str(value))
    if path.is_absolute():
        try:
            return path.resolve().relative_to(project_root).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auto-classify assets samples and update matting_quality manifest.json."
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("assets"),
        help="Assets directory to scan.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("tests/fixtures/matting_quality/manifest.json"),
        help="Manifest JSON path to update.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root used for relative manifest paths.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = update_manifest(
        assets_dir=args.assets_dir,
        manifest_path=args.manifest,
        project_root=args.project_root,
    )
    print(
        "Updated manifest: "
        f"added={summary['added']} updated={summary['updated']} total={summary['total']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
