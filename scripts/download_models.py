#!/usr/bin/env python3
"""下载 MatteFlow 所需的 AI 模型

用法:
    python scripts/download_models.py --all
    python scripts/download_models.py --model gvm
    python scripts/download_models.py --model matanyone2
    python scripts/download_models.py --model sam2
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def download_gvm():
    """下载 GVM 模型 (~6GB)"""
    print("=" * 60)
    print("[DOWNLOAD] GVM (Generative Video Matting)")
    print("=" * 60)
    print("模型大小: ~6 GB")
    print("来源: https://huggingface.co/geyongtao/gvm")
    print()
    
    try:
        from huggingface_hub import snapshot_download
        
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        print("开始下载...")
        snapshot_download(
            repo_id="geyongtao/gvm",
            cache_dir=cache_dir,
            local_dir_use_symlinks=False
        )
        
        print("[OK] GVM 下载完成！")
        return True
        
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        print("请手动下载: https://huggingface.co/geyongtao/gvm")
        return False


def download_matanyone2():
    """下载 MatAnyone2 模型 (~135MB)"""
    print("=" * 60)
    print("[DOWNLOAD] MatAnyone2")
    print("=" * 60)
    print("模型大小: ~135 MB")
    print("来源: https://huggingface.co/pq-yang/MatAnyone2")
    print()
    
    try:
        from huggingface_hub import hf_hub_download
        
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        print("开始下载...")
        hf_hub_download(
            repo_id="pq-yang/MatAnyone2",
            filename="matanyone2.pth",
            cache_dir=cache_dir,
            local_dir_use_symlinks=False
        )
        
        print("[OK] MatAnyone2 下载完成！")
        return True
        
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        print("请手动下载: https://huggingface.co/pq-yang/MatAnyone2")
        return False


def download_sam2():
    """下载 SAM2 模型 (~324MB)"""
    print("=" * 60)
    print("[DOWNLOAD] SAM2 (Segment Anything 2)")
    print("=" * 60)
    print("模型大小: ~324 MB")
    print("来源: https://huggingface.co/facebook/sam2-hiera-base-plus")
    print()
    
    try:
        from huggingface_hub import snapshot_download
        
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        print("开始下载...")
        snapshot_download(
            repo_id="facebook/sam2-hiera-base-plus",
            cache_dir=cache_dir,
            local_dir_use_symlinks=False
        )
        
        print("[OK] SAM2 下载完成！")
        return True
        
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        print("请手动下载: https://huggingface.co/facebook/sam2-hiera-base-plus")
        return False


def download_birefnet():
    """下载 BiRefNet 模型 (~940MB)"""
    print("=" * 60)
    print("[DOWNLOAD] BiRefNet")
    print("=" * 60)
    print("模型大小: ~940 MB")
    print("来源: https://huggingface.co/ZhengPeng7/BiRefNet")
    print()
    
    try:
        from huggingface_hub import snapshot_download
        
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        print("开始下载...")
        snapshot_download(
            repo_id="ZhengPeng7/BiRefNet",
            cache_dir=cache_dir,
            local_dir_use_symlinks=False
        )
        
        print("[OK] BiRefNet 下载完成！")
        return True
        
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        print("请手动下载: https://huggingface.co/ZhengPeng7/BiRefNet")
        return False


def download_corridorkey():
    """下载 CorridorKey 模型 (~383MB)"""
    print("=" * 60)
    print("[DOWNLOAD] CorridorKey")
    print("=" * 60)
    print("模型大小: ~383 MB")
    print("来源: https://github.com/nikopueringer/CorridorKey")
    print()
    
    try:
        import urllib.request
        
        model_dir = Path.home() / ".cache" / "matteflow" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "corridorkey_model.pth"
        
        # GitHub 发布链接
        url = "https://github.com/nikopueringer/CorridorKey/releases/download/v1.0.0/corridorkey_model.pth"
        
        print(f"开始下载...")
        print(f"URL: {url}")
        
        urllib.request.urlretrieve(url, model_path)
        
        print(f"[OK] CorridorKey 下载完成！")
        print(f"保存位置: {model_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        print("请手动下载:")
        print("  1. 访问 https://github.com/nikopueringer/CorridorKey")
        print("  2. 下载模型文件")
        print(f"  3. 放置到: {Path.home() / '.cache' / 'matteflow' / 'models'}")
        return False


def main():
    parser = argparse.ArgumentParser(description="下载 MatteFlow AI 模型")
    parser.add_argument("--all", action="store_true", help="下载所有模型")
    parser.add_argument("--model", choices=["gvm", "matanyone2", "sam2", "birefnet", "corridorkey"],
                       help="下载指定模型")
    parser.add_argument("--list", action="store_true", help="列出可用模型")
    
    args = parser.parse_args()
    
    if args.list:
        print("=" * 60)
        print("可用模型列表")
        print("=" * 60)
        print()
        print("1. GVM (Generative Video Matting)")
        print("   大小: ~6 GB")
        print("   用途: 生成式视频抠图")
        print()
        print("2. MatAnyone2")
        print("   大小: ~135 MB")
        print("   用途: 人物视频抠图")
        print()
        print("3. SAM2 (Segment Anything 2)")
        print("   大小: ~324 MB")
        print("   用途: 视频分割跟踪")
        print()
        print("4. BiRefNet")
        print("   大小: ~940 MB")
        print("   用途: 通用图像抠图")
        print()
        print("5. CorridorKey")
        print("   大小: ~383 MB")
        print("   用途: 物理分离抠图")
        print()
        print("使用方法:")
        print("  python scripts/download_models.py --model gvm")
        print("  python scripts/download_models.py --all")
        return
    
    if args.all:
        print("=" * 60)
        print("下载所有模型")
        print("=" * 60)
        print()
        
        results = []
        results.append(("GVM", download_gvm()))
        results.append(("MatAnyone2", download_matanyone2()))
        results.append(("SAM2", download_sam2()))
        results.append(("BiRefNet", download_birefnet()))
        results.append(("CorridorKey", download_corridorkey()))
        
        print()
        print("=" * 60)
        print("下载结果")
        print("=" * 60)
        for name, success in results:
            status = "OK" if success else "FAIL"
            print(f"{name}: {status}")
    
    elif args.model:
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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
