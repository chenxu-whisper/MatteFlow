"""Apply targeted fixes for Ruff logic-risk diagnostics.

This script intentionally fixes only the currently known F541/E721/E722
diagnostics. It uses exact byte replacements so vendored code is not
accidentally reformatted beyond the intended lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Replacement:
    path: str
    before: bytes
    after: bytes
    description: str


REPLACEMENTS = (
    Replacement(
        path="scripts/download_models.py",
        before='print(f"[OK] CorridorKey 下载完成！")'.encode(),
        after='print("[OK] CorridorKey 下载完成！")'.encode(),
        description="Remove redundant f-string prefix for CorridorKey success message.",
    ),
    Replacement(
        path="scripts/download_models.py",
        before='print(f"[OK] RVM 下载完成！")'.encode(),
        after='print("[OK] RVM 下载完成！")'.encode(),
        description="Remove redundant f-string prefix for RVM success message.",
    ),
    Replacement(
        path="src/matteflow/utils/model_checker.py",
        before='print(f"  自动下载: 支持")'.encode(),
        after='print("  自动下载: 支持")'.encode(),
        description="Remove redundant f-string prefix for auto-download status.",
    ),
    Replacement(
        path="src/matteflow/vendor/gvm_core/gvm/pipelines/pipeline_gvm.py",
        before='f"pytorch_lora_weights.pt"'.encode(),
        after='"pytorch_lora_weights.pt"'.encode(),
        description="Remove redundant f-string prefix for LoRA checkpoint name.",
    ),
    Replacement(
        path="src/matteflow/vendor/corridorkey_module/core/model_transformer.py",
        before=b"        except:\r\n            feature_channels = [112, 224, 448, 896]",
        after=b"        except Exception:\r\n            feature_channels = [112, 224, 448, 896]",
        description="Replace bare except while preserving fallback feature channels.",
    ),
    Replacement(
        path="src/matteflow/vendor/gvm_core/gvm/pipelines/pipeline_gvm.py",
        before=b"                    except:\r\n                        num_overlap_frames = min(num_overlap_frames, latent.shape[1])",
        after=b"                    except Exception:\r\n                        num_overlap_frames = min(num_overlap_frames, latent.shape[1])",
        description="Replace bare except in overlap-frame fallback.",
    ),
    Replacement(
        path="src/matteflow/vendor/matanyone2_module/matanyone2/inference/object_info.py",
        before=b"        if type(other) == int:\r\n            return self.id == other",
        after=b"        if isinstance(other, int):\r\n            return self.id == other",
        description="Use isinstance for ObjectInfo integer comparison.",
    ),
)


def apply_replacement(root: Path, replacement: Replacement) -> bool:
    file_path = root / replacement.path
    data = file_path.read_bytes()

    if replacement.after in data and replacement.before not in data:
        print(f"SKIP already fixed: {replacement.path} - {replacement.description}")
        return False

    count = data.count(replacement.before)
    if count != 1:
        raise RuntimeError(
            f"Expected one match in {replacement.path}, found {count}: "
            f"{replacement.description}"
        )

    file_path.write_bytes(data.replace(replacement.before, replacement.after, 1))
    print(f"FIXED: {replacement.path} - {replacement.description}")
    return True


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    fixed_count = 0
    for replacement in REPLACEMENTS:
        if apply_replacement(root, replacement):
            fixed_count += 1

    print(f"\nApplied {fixed_count} targeted Ruff logic-risk fixes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
