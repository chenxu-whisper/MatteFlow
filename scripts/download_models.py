#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用方法：
    python scripts/download_models.py --model gvm
    python scripts/download_models.py --model matanyone2
    python scripts/download_models.py --model sam2
    python scripts/download_models.py --model birefnet
    python scripts/download_models.py --model corridorkey
    python scripts/download_models.py --model gvm
    python scripts/download_models.py --all
"""

import argparse
import sys
import warnings
from pathlib import Path

# 禁用 HuggingFace Hub 警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
warnings.filterwarnings("ignore", message=".*symlinks.*")

# 设置环境变量禁用符号链接警告
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from matteflow.utils.model_paths import model_file, models_root


def download_gvm():
    """下载 GVM 模型 (~6GB)"""
    print("=" * 60)
    print("[DOWNLOAD] GVM (Generative Video Matting)")
    print("=" * 60)
    print("模型大小: ~6 GB")

    try:
        from huggingface_hub import snapshot_download

        cache_dir = models_root()
        cache_dir.mkdir(parents=True, exist_ok=True)

        print("开始下载...")
        snapshot_download(
            repo_id="geyongtao/gvm",
            cache_dir=cache_dir
        )

        print("[OK] GVM 下载完成！")
        return True
    except Exception as e:
        print(f"[ERROR] GVM 下载失败: {e}")
        return False


def download_matanyone2():
    """下载 MatAnyone2 模型 (~135MB)"""
    print("=" * 60)
    print("[DOWNLOAD] MatAnyone2")
    print("=" * 60)
    print("模型大小: ~135 MB")

    try:
        import urllib.request

        cache_dir = models_root()
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_file("matanyone2.pth")
        url = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"

        print("开始下载...")
        urllib.request.urlretrieve(url, model_path)

        print("[OK] MatAnyone2 下载完成！")
        return True
    except Exception as e:
        print(f"[ERROR] MatAnyone2 下载失败: {e}")
        return False


def download_sam2():
    """下载 SAM2 模型 (~324MB)"""
    print("=" * 60)
    print("[DOWNLOAD] SAM2 (Segment Anything 2)")
    print("=" * 60)
    print("模型大小: ~324 MB")

    try:
        from huggingface_hub import snapshot_download

        cache_dir = models_root()
        cache_dir.mkdir(parents=True, exist_ok=True)

        print("开始下载...")
        snapshot_download(
            repo_id="facebook/sam2-hiera-base-plus",
            cache_dir=cache_dir
        )

        print("[OK] SAM2 下载完成！")
        return True
    except Exception as e:
        print(f"[ERROR] SAM2 下载失败: {e}")
        return False


def download_birefnet():
    """下载 BiRefNet 模型 (~940MB)"""
    print("=" * 60)
    print("[DOWNLOAD] BiRefNet")
    print("=" * 60)
    print("模型大小: ~940 MB")

    try:
        from huggingface_hub import snapshot_download

        cache_dir = models_root()
        cache_dir.mkdir(parents=True, exist_ok=True)

        print("开始下载...")
        snapshot_download(
            repo_id="ZhengPeng7/BiRefNet",
            cache_dir=cache_dir
        )

        print("[OK] BiRefNet 下载完成！")
        return True
    except Exception as e:
        print(f"[ERROR] BiRefNet 下载失败: {e}")
        return False


def download_corridorkey():
    """下载 CorridorKey 模型 (~383MB)"""
    print("=" * 60)
    print("[DOWNLOAD] CorridorKey")
    print("=" * 60)
    print("模型大小: ~383 MB")

    try:
        import urllib.request

        model_dir = models_root()
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_file("corridorkey.pth")
        url = "https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main/CorridorKey_v1.0.pth"

        print("开始下载...")
        urllib.request.urlretrieve(url, model_path)
        print(f"[OK] CorridorKey 下载完成！")
        return True
    except Exception as e:
        print(f"[ERROR] CorridorKey 下载失败: {e}")
        return False


def download_rvm():
    """下载 RVM 模型 (~15MB)"""
    print("=" * 60)
    print("[DOWNLOAD] RVM (Robust Video Matting)")
    print("=" * 60)
    print("模型大小: ~15 MB")

    try:
        import urllib.request

        model_dir = models_root()
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_file("rvm_mobilenetv3.pth")
        url = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth"

        print("开始下载...")
        urllib.request.urlretrieve(url, model_path)
        print(f"[OK] RVM 下载完成！")
        return True
    except Exception as e:
        print(f"[ERROR] RVM 下载失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="下载 MatteFlow 所需的模型")
    parser.add_argument("--model", type=str, default=None,
                        help="模型名称: gvm, matanyone2, sam2, birefnet, corridorkey, rvm")
    parser.add_argument("--all", action="store_true", help="下载所有模型")

    args = parser.parse_args()

    if not args.model and not args.all:
        print("请指定要下载的模型:")
        print("  python scripts/download_models.py --model gvm")
        print("  python scripts/download_models.py --all")
        return

    if args.all:
        print("=" * 60)
        print("[INFO] 开始下载所有模型")
        print("=" * 60)

        results = []
        results.append(("GVM", download_gvm()))
        results.append(("MatAnyone2", download_matanyone2()))
        results.append(("SAM2", download_sam2()))
        results.append(("BiRefNet", download_birefnet()))
        results.append(("CorridorKey", download_corridorkey()))
        results.append(("RVM", download_rvm()))

        print("\n" + "=" * 60)
        print("下载结果汇总:")
        print("=" * 60)
        for name, success in results:
            status = "✓ 成功" if success else "✗ 失败"
            print(f"  {name}: {status}")
    else:
        if args.model == "gvm":
            download_gvm()
        elif args.model == "matanyone2":
            download_matanyone2()
        elif args.model == "sam2":
            download_sam2()
        elif args.model == "birefnet":
            download_birefnet()
        elif args.model == "corridorkey":
            download_corridorkey()
        elif args.model == "rvm":
            download_rvm()
        else:
            print(f"未知模型: {args.model}")


if __name__ == "__main__":
    main()
